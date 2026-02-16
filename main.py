import os
import json
import firebase_admin
from firebase_admin import credentials, firestore
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from google import genai
from google.genai import types
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

# Khởi tạo Gemini Client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL_ID = "gemini-2.5-flash"
# CHỈNH MODEL: Khớp với model bạn dùng để upload
MODEL_EMBED = "gemini-embedding-001"

# --- 3. MENU ĐỊNH TUYẾN (GIỮ NGUYÊN 100%) ---
SEARCH_KEYS_MENU = {
    "V1": "thi tiếng anh đầu vào, xếp lớp, ielts 6.0, miễn học dự bị, cấu trúc đề thi, writing skill",
    "V2": "lộ trình luk global, hurricane, greenfire, heatwave, thunderbolt, debate, thuyết trình",
    "V3": "summit 1 summit 2, top notch, progress test pt, assignment môn ent, thi seb eos",
    "V4": "mẹo pass môn tiếng anh, cách dùng edunext fap, kiểm tra điểm danh, writing speaking assignment",
    "V5": "học nhạc cụ dân tộc, đàn bầu, đàn tranh, sáo trúc, địa chỉ mua nhạc cụ hảo vĩnh đà nẵng",
    "V6": "học vovinam fpt, clb vovinam vvc, thi lên đai, võ nhạc, giải khơi nguồn võ việt",
    "V7": "kinh nghiệm đi quân sự, đồ dùng tân binh, lót giày, phấn rôm, gấp chăn bánh chưng, nội vụ",
    "V8": "review campus fpt đà nẵng, tòa nhà alpha gamma, thư viện, fpt city ngũ hành sơn",
    "V9": "so sánh ký túc xá và trọ, ưu nhược điểm ktx fpt, an ninh nội trú, chi phí ở trọ",
    "V10": "cẩm nang thuê trọ đà nẵng, lừa đảo tiền cọc, hợp đồng thuê nhà, tìm bạn ở ghép",
    "V11": "quán ăn ngon fpt đà nẵng, cafe học bài, zone six 24/7, cơm gà xả xệ, bún đậu 1996",
    "V12": "link fap flm, tải phần mềm thi seb eos, lỗi kỹ thuật, checkout e360, cài đặt phần mềm",
    "V13": "quản lý thời gian, thói quen ngủ, xem trước bài, check attendance fap, kỹ năng tự học"
}

# --- 4. CẤU TRÚC DỮ LIỆU ---
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = []

# --- 5. LOGIC XỬ LÝ CHÍNH ---

def get_semantic_search_query(user_raw_query, history):
    menu_str = "\n".join([f"- {k}: {v}" for k, v in SEARCH_KEYS_MENU.items()])
    context_recent = f"Ngữ cảnh lịch sử: {history[-1].content}" if history else ""

    prompt = f"""Bạn là bộ định tuyến dữ liệu cho sinh viên FPTU.
    Nhiệm vụ: Phân tích câu hỏi và chọn ra BỘ KEYWORD phù hợp nhất.
    DANH SÁCH KEYWORDS:
    {menu_str}
    {context_recent}
    CÂU HỎI NGƯỜI DÙNG: "{user_raw_query}"
    YÊU CẦU: CHỈ TRẢ VỀ chuỗi keyword tương ứng hoặc câu hỏi gốc. Không giải thích."""

    try:
        response = client.models.generate_content(model=MODEL_ID, contents=prompt)
        return response.text.strip()
    except:
        return user_raw_query

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    try:
        # BƯỚC 1: ROUTING (GIỮ NGUYÊN LOGIC)
        search_query = get_semantic_search_query(req.message, req.history)
        print(f"🎯 Router đã chọn: {search_query}")

        # BƯỚC 2: TRUY XUẤT VECTOR
        # CHỈNH: dimensionality=1536 để khớp với Index bạn vừa tạo
        embed_res = client.models.embed_content(
            model=MODEL_EMBED,
            contents=search_query,
            config={'output_dimensionality': 1536}
        )
        query_vector = embed_res.embeddings[0].values

        # CHỈNH: Đổi tên collection thành fpt_handbook_v1
        results = db.collection("fpt_handbook_v1").find_nearest(
            vector_field="embedding",
            query_vector=Vector(query_vector),
            distance_measure=DistanceMeasure.COSINE,
            limit=1
        ).get()

        if not results:
            return {"reply": "🤖 Bot: Xin lỗi, mình không tìm thấy dữ liệu liên quan."}

        top_result = results[0]
        context = top_result.to_dict().get('content', 'Không có nội dung')
        dist = getattr(top_result, 'distance', 0) or 0
        print(f"📊 Distance: {dist:.4f}")

        # GIỮ NGUYÊN THÔNG SỐ 0.6
        if dist > 0.6:
            return {"reply": "🤖 Bot: Câu hỏi này nằm ngoài phạm vi cẩm nang sinh viên FPTU."}

        # BƯỚC 3: GENERATION (GIỮ NGUYÊN LOGIC)
        system_instruction = "Bạn là trợ lý ảo thông minh cho sinh viên Đại học FPT Đà Nẵng. Trả lời thân thiện, ngắn gọn, có icon."

        response = client.models.generate_content(
            model=MODEL_ID,
            contents=f"THÔNG TIN CẨM NANG: {context}\n\nCÂU HỎI: {req.message}",
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.4,
            )
        )

        return {"reply": response.text}

    except Exception as e:
        print(f"Lỗi: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)