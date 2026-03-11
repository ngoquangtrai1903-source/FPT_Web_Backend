import os
import json
import firebase_admin
from firebase_admin import credentials, firestore
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from openai import OpenAI
# ✅ Bỏ hoàn toàn google.genai — dùng chung OpenRouter cho cả chat lẫn embedding
from google.cloud.firestore_v1.vector import Vector
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure

# --- 0. LOAD BIẾN MÔI TRƯỜNG ---
load_dotenv()

# --- 1. CẤU HÌNH FASTAPI & CORS ---
app = FastAPI(title="FPTU RAG Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. KHỞI TẠO DỊCH VỤ ---
if not firebase_admin._apps:
    fb_config = os.getenv("FIREBASE_CONFIG")
    if fb_config:
        cred = credentials.Certificate(json.loads(fb_config))
    else:
        cred_path = os.getenv("FIREBASE_SERVICE_ACCOUNT", "service-account.json")
        cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred)

db = firestore.client()

# ✅ Dùng chung 1 client OpenRouter cho cả Chat lẫn Embedding
client_or = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)
MODEL_ID    = "google/gemini-2.5-flash-lite"
MODEL_EMBED = "google/gemini-embedding-001"  # ✅ gemini-embedding-001 qua OpenRouter
EMBED_DIM   = 1536  # Phải khớp với dimension đã index trong Firestore

# --- 3. MENU ĐỊNH TUYẾN ---
SEARCH_KEYS_MENU = {
    "V1":  "thi tiếng anh đầu vào, xếp lớp, ielts 6.0, miễn học dự bị, cấu trúc đề thi, writing skill",
    "V2":  "LUK, lộ trình luk global, hurricane, greenfire, heatwave, thunderbolt, debate, thuyết trình",
    "V3":  "summit 1 summit 2, top notch, progress test pt, assignment môn ent, thi seb eos",
    "V4":  "mẹo pass môn tiếng anh, cách dùng edunext fap, kiểm tra điểm danh, writing speaking assignment",
    "V5":  "học nhạc cụ dân tộc, đàn bầu, đàn tranh, sáo trúc, địa chỉ mua nhạc cụ hảo vĩnh đà nẵng",
    "V6":  "học vovinam fpt, clb vovinam vvc, thi lên đai, võ nhạc, giải khơi nguồn võ việt",
    "V7":  "kinh nghiệm đi quân sự, đồ dùng tân binh, lót giày, phấn rôm, gấp chăn bánh chưng, nội vụ",
    "V8":  "review campus fpt đà nẵng, tòa nhà alpha gamma, thư viện, fpt city ngũ hành sơn",
    "V9":  "so sánh ký túc xá và trọ, ưu nhược điểm ktx fpt, an ninh nội trú, chi phí ở trọ",
    "V10": "cẩm nang thuê trọ đà nẵng, lừa đảo tiền cọc, hợp đồng thuê nhà, tìm bạn ở ghép",
    "V11": "quán ăn ngon fpt đà nẵng, cafe học bài, zone six 24/7, cơm gà xả xệ, bún đậu 1996",
    "V12": "link, link fap flm, tải phần mềm thi seb eos, lỗi kỹ thuật, checkout e360, cài đặt phần mềm",
    "V13": "quản lý thời gian, thói quen ngủ, xem trước bài, check attendance fap, kỹ năng tự học",
}

# --- 4. CẤU TRÚC DỮ LIỆU ---
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = []


# --- 5. HELPER FUNCTIONS ---

def get_router_key(user_query: str, history: List[ChatMessage]) -> str:
    menu_str = "\n".join([f"- {k}: {v}" for k, v in SEARCH_KEYS_MENU.items()])
    recent_ctx = f"\nNgữ cảnh hội thoại gần nhất: {history[-1].content}" if history else ""

    prompt = f"""Bạn là bộ định tuyến chủ đề cho chatbot sinh viên FPTU Đà Nẵng.

DANH SÁCH CHỦ ĐỀ:
{menu_str}
{recent_ctx}

CÂU HỎI: "{user_query}"

NHIỆM VỤ: Chọn đúng 1 KEY phù hợp nhất.
QUAN TRỌNG: Chỉ trả về đúng KEY (ví dụ: V6). Không giải thích, không thêm ký tự nào khác."""

    try:
        resp = client_or.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=5,
        )
        key = resp.choices[0].message.content.strip().upper()
        if key in SEARCH_KEYS_MENU:
            return key
        print(f"⚠️ Router trả về key không hợp lệ: '{key}' → fallback câu hỏi gốc")
        return ""
    except Exception as e:
        print(f"⚠️ Lỗi Router: {e}")
        return ""


def embed_text(text: str) -> list:
    """
    ✅ Tạo vector embedding qua OpenRouter (chung API với chat).
    OpenRouter trả về response theo chuẩn OpenAI embeddings.
    """
    res = client_or.embeddings.create(
        model=MODEL_EMBED,
        input=text,
        dimensions=EMBED_DIM,  # Chỉ định dimension để khớp Firestore index
    )
#    print(res)
    return res.data[0].embedding


# --- 6. ENDPOINT CHAT ---
@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    try:
        # ── BƯỚC 1: ROUTING → KEY ─────────────────────────────────────────
        router_key = get_router_key(req.message, req.history)
#        print(req.history)
        print(f"🎯 Router key: {router_key or '(không khớp, dùng câu hỏi gốc)'}")

        embed_input = SEARCH_KEYS_MENU[router_key] if router_key else req.message
        print(f"📌 Embed: '{embed_input[:80]}...'")

        # ── BƯỚC 2: VECTOR SEARCH ─────────────────────────────────────────
        query_vector = embed_text(embed_input)

        results = db.collection("fpt_handbook_v1").find_nearest(
            vector_field="embedding",
            query_vector=Vector(query_vector),
            distance_measure=DistanceMeasure.COSINE,
            limit=1,
        ).get()

        if not results:
            return {"reply": "🤖 Xin lỗi, mình không tìm thấy dữ liệu liên quan trong cẩm nang."}

        context_chunks = []
        for r in results:
            dist = float(getattr(r, "distance", 0) or 0)
            content = r.to_dict().get("content", "")
            print(f"📊 Doc: {r.id} | Distance: {dist:.4f} | Chars: {len(content)}")
            if dist <= 0.6 and content:
                context_chunks.append(content)

        if not context_chunks:
            return {"reply": "🤖 Câu hỏi này nằm ngoài phạm vi cẩm nang FPTU Đà Nẵng của mình. Bạn thử hỏi câu khác nhé! 😊"}

        combined_context = "\n\n---\n\n".join(context_chunks)
        print(f"📝 Context: {len(combined_context)} chars | {len(context_chunks)} chunks")

        # ── BƯỚC 3: GENERATION ────────────────────────────────────────────
        system_instruction = (
            "Bạn là 'FPTU Da Nang Buddy' - một người bạn đồng hành ảo cực kỳ nhiệt tình và am hiểu về FPTU Đà Nẵng. 🍊\n\n"

            "PHONG CÁCH:\n"
            "1. Ngôn ngữ: Thân thiện, gần gũi như một 'tiền bối' đang hướng dẫn 'hậu bối'.\n"
            "2. Trình bày: Dùng bullet points cho danh sách, in đậm mốc thời gian/địa điểm quan trọng, icon phù hợp ngữ cảnh.\n\n"

            "NGUYÊN TẮC XỬ LÝ:\n"
            "1. TRUNG THỰC: Chỉ trả lời dựa trên [TÀI LIỆU]. Không tự bịa số điện thoại, link hay quy định.\n"
            "2. PARAPHRASE: Diễn đạt lại thông tin cho dễ hiểu, không copy-paste văn bản hành chính.\n"
            "3. THIẾU THÔNG TIN: Nếu [TÀI LIỆU] không có câu trả lời, xin lỗi và gợi ý liên hệ Phòng Dịch vụ sinh viên.\n"
            "4. CẤU TRÚC: Chào hỏi nhẹ → Nội dung chính → Lời nhắc/chúc cuối.\n"
            "5. CHỐNG HALLUCINATION: [TÀI LIỆU] là nguồn sự thật duy nhất. "
            "6. Nếu tin nhắn của người dùng mang tính chất chào hỏi, tâm sự và không có mục đích muốn truy xuất dữ liệu, thì bạn hãy trả lời giao lưu tự nhiên mà không cần dùng tài liệu"
            "Nếu lịch sử hội thoại mâu thuẫn với [TÀI LIỆU], hãy tin [TÀI LIỆU]. "
            "Không trộn thông tin từ chủ đề cũ sang chủ đề mới."
        )

        messages = [{"role": "system", "content": system_instruction}]

        for h in req.history:
            messages.append({"role": h.role, "content": h.content})

        messages.append({
            "role": "user",
            "content": (
                f"[TÀI LIỆU]\n{combined_context}\n\n"
                f"[CÂU HỎI]\n{req.message}"
            ),
        })

        resp = client_or.chat.completions.create(
            model=MODEL_ID,
            messages=messages,
            temperature=0.3,
        )
#        print(resp.choices[0].message.content)
        return {"reply": resp.choices[0].message.content}

    except Exception as e:
        print(f"❌ Lỗi: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)