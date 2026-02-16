import os
from google import genai
from firebase_admin import firestore, credentials, initialize_app
from google.cloud.firestore_v1.vector import Vector
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
from dotenv import load_dotenv

load_dotenv()

# 1. Khởi tạo (Thay thế đường dẫn service-account nếu cần)
if not firebase_admin._apps:
    cred = credentials.Certificate("service-account.json")
    initialize_app(cred)
db = firestore.client()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# 2. Tạo vector mẫu để tìm kiếm
query_text = "Làm sao để pass môn tiếng Anh?"
result = client.models.embed_content(
    model="gemini-embedding-001",
    contents=query_text,
    config={'task_type': 'RETRIEVAL_QUERY', 'output_dimensionality': 1536}
)
query_vector = result.embeddings[0].values

# 3. Thực hiện truy vấn (Lệnh này sẽ gây lỗi và trả về link tạo Index)
try:
    collection_ref = db.collection("fpt_handbook_v1")
    results = collection_ref.find_nearest(
        vector_field="embedding",
        query_vector=Vector(query_vector),
        distance_measure=DistanceMeasure.COSINE,
        limit=3
    ).get()

    for doc in results:
        print(f"Tìm thấy: {doc.to_dict()['metadata']['section']}")

except Exception as e:
    print("\n--- BẠN HÃY NHẤN VÀO LINK DƯỚI ĐÂY ĐỂ TẠO INDEX ---")
    print(e)