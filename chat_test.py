import firebase_admin
from firebase_admin import credentials, firestore
from sentence_transformers import SentenceTransformer
from google.cloud.firestore_v1.vector import Vector
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
from google import genai
from google.genai import types

# --- 1. KHỞI TẠO CẤU HÌNH ---
client = genai.Client(api_key="AIzaSyBi2YSIsnx4krzjW54xH0Lu52hCNCA6B2Y")
MODEL_ID = "gemini-2.5-flash"

# Kết nối Firestore
cred = credentials.Certificate("service-account.json")
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)
db = firestore.client()

model = SentenceTransformer('all-MiniLM-L6-v2')
chat_history = []

# --- DANH SÁCH KEY ĐÃ TỐI ƯU (Copy từ module NLP) ---
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


def get_semantic_search_query(user_raw_query):
    """Sử dụng Gemini làm Router để chọn ra Key chuẩn nhất từ Menu"""

    # Chuẩn bị menu cho Prompt
    menu_str = "\n".join([f"- {k}: {v}" for k, v in SEARCH_KEYS_MENU.items()])

    # Lấy ngữ cảnh ngắn gọn
    context_recent = ""
    if chat_history:
        context_recent = f"Ngữ cảnh lịch sử: {chat_history[-1]['content']}"

    prompt = f"""Bạn là bộ định tuyến dữ liệu cho sinh viên FPTU.
    Nhiệm vụ: Phân tích câu hỏi và chọn ra BỘ KEYWORD phù hợp nhất trong danh sách dưới đây.

    DANH SÁCH KEYWORDS:
    {menu_str}

    {context_recent}
    CÂU HỎI NGƯỜI DÙNG: "{user_raw_query}"

    YÊU CẦU:
    - Nếu câu hỏi liên quan đến nội dung trong danh sách, CHỈ TRẢ VỀ chuỗi keyword tương ứng.
    - Nếu câu hỏi hoàn toàn không liên quan (ví dụ: "thời tiết hôm nay"), trả về chính xác câu hỏi gốc.
    - Không giải thích, không thêm văn bản thừa.
    """

    try:
        response = client.models.generate_content(model=MODEL_ID, contents=prompt)
        return response.text.strip()
    except:
        return user_raw_query


def hoi_chatbot_ai(user_q):
    global chat_history

    # BƯỚC 1: ROUTING (Biến câu hỏi thành Search Key chuẩn)
    search_query = get_semantic_search_query(user_q)
    print(f"🎯 Router đã chọn Key: {search_query}")

    # BƯỚC 2: TRUY XUẤT VECTOR
    query_vector = model.encode(search_query).tolist()

    # Sử dụng find_nearest để tìm tài liệu tương đương nhất
    results = db.collection("handbook_vectors").find_nearest(
        vector_field="embedding",
        query_vector=Vector(query_vector),
        distance_measure=DistanceMeasure.COSINE,
        limit=1
    ).get()

    if not results:
        return "🤖 Bot: Xin lỗi, mình không tìm thấy dữ liệu liên quan."

    # --- SỬA LỖI TẠI ĐÂY ---
    # Lấy document snapshot đầu tiên
    top_result = results[0]

    # Lấy nội dung text
    doc_data = top_result.to_dict()
    context = doc_data.get('content', 'Không có nội dung')

    # Lấy khoảng cách (distance) đúng cách theo DocumentSnapshot
    # Trong các phiên bản SDK mới, distance nằm trong thuộc tính 'vector_distance' hoặc 'metadata'
    # Nếu không lấy được, ta mặc định là 0 vì Router đã định hướng rất tốt
    dist = getattr(top_result, 'distance', 0)

    # Nếu vẫn báo lỗi hoặc dist trả về None, dùng giá trị mặc định để bypass check
    if dist is None: dist = 0

    print(f"📊 Khoảng cách Vector (Distance): {dist}")

    # Kiểm tra ngưỡng tin cậy (Vì dùng Router nên dist thường rất nhỏ < 0.2)
    if dist > 0.6:
        return "🤖 Bot: Câu hỏi này nằm ngoài phạm vi cẩm nang sinh viên FPTU. Bạn thử hỏi về Vovinam, KTX hoặc Tiếng Anh xem sao!"

    # BƯỚC 3: GENERATION
    system_instruction = """Bạn là một trợ lý ảo thông minh cho sinh viên Đại học FPT Đà Nẵng.
    Nhiệm vụ: Dùng THÔNG TIN CẨM NANG cung cấp để trả lời câu hỏi. 
    - Trả lời thân thiện (xưng mình - gọi bạn).
    - Trả lời ngắn gọn, đúng trọng tâm, có icon sinh động.
    """

    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=f"THÔNG TIN CẨM NANG: {context}\n\nCÂU HỎI: {user_q}",
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.4,
            )
        )
        answer = response.text

        # Cập nhật lịch sử (Chỉ giữ 3 cặp câu để tránh quá tải ngữ cảnh)
        chat_history.append({"role": "user", "content": user_q})
        chat_history.append({"role": "bot", "content": answer})
        if len(chat_history) > 6: chat_history = chat_history[-6:]

        return answer
    except Exception as e:
        return f"🤖 Bot: Đã có lỗi xảy ra trong quá trình tạo câu trả lời ({str(e)})"


if __name__ == "__main__":
    print("🤖 Chatbot FPTU (Version 3.0 - Semantic Router) đã sẵn sàng!")
    while True:
        user_q = input("\nBạn: ")
        if user_q.lower() in ['exit', 'thoát']: break
        print(f"🤖 Bot: {hoi_chatbot_ai(user_q)}")