import os
import firebase_admin
from google import genai  # Thư viện SDK mới nhất
from firebase_admin import credentials, firestore
from google.cloud.firestore_v1.vector import Vector
from dotenv import load_dotenv

load_dotenv()

# --- 1. KHỞI TẠO KẾT NỐI ---
if not firebase_admin._apps:
    cred = credentials.Certificate("service-account.json")
    firebase_admin.initialize_app(cred)
db = firestore.client()

# Khởi tạo Client Gemini mới
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

TARGET_COLLECTION = "fpt_handbook_v1" # Đặt biến này ở ngoài để dùng chung

def upload_vector_final(vector_id, full_text, search_key, chapter, section):
    rich_context = f"Question context: {search_key} | Content preview: {full_text[:200]}"
    try:
        result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=rich_context,
            config={'task_type': 'RETRIEVAL_DOCUMENT', 'title': search_key, 'output_dimensionality': 1536}
        )

        embedding_values = [float(x) for x in result.embeddings[0].values]

        data = {
            "content": full_text.strip(),
            "search_key": search_key,
            "embedding": Vector(embedding_values),
            "metadata": {
                "chapter": chapter,
                "section": section,
                "vector_id": vector_id
            }
        }

        # CHÚ Ý: Sử dụng biến TARGET_COLLECTION ở đây thay vì viết cứng tên
        db.collection(TARGET_COLLECTION).document(vector_id).set(data)
        print(f"✅ Đã tải lên thành công: {vector_id} - {section}")

    except Exception as e:
        print(f"❌ Lỗi tại {vector_id}: {str(e)}")


# --- 2. DÁN NỘI DUNG CỦA BẠN VÀO ĐÂY ---

# CHƯƠNG 1
v1_full = """CHƯƠNG I: CHƯƠNG TRÌNH HỌC TIẾNG ANH ĐẦU VÀO
1.1. Kì thi tiếng anh đầu vào cho tân sinh viên
-   Với chương trình đào tạo theo hướng quốc tế của Đại học FPT, chương trình đào tạo sẽ là 100% tiếng Anh, điều đó yêu cầu sinh viên vào trường học phải có trình độ tiếng Anh ở mức độ nhất định khoảng 6.0 IELTS. Nếu bạn chưa có chứng chỉ này, chưa đạt đến trình độ đó thì cũng đừng vội lo lắng về cơ hội học tập tại môi trường tiếng Anh quốc tế này. Để giải quyết vấn đề đó, trường sẽ tổ chức cuộc thi tiếng Anh đầu vào cho tân sinh viên, từ đó sẽ đánh giá và phân loại lớp tiếng Anh theo trình độ phù hợp. Tất nhiên, nếu bạn làm tốt việc pass thẳng lên chuyên ngành thì là điều bình thường.
-   Chuẩn bị tốt cho kì thi này là điều cần thiết cho các bạn trong những ngày nghỉ đợi nhập học. Lý do mình nói như vậy là vì đây là cơ hội để các bạn sớm học lên chuyên ngành và ra trường sớm. Trong chương trình đào tạo của Đại học FPT gồm 4 năm thì sẽ dành trọn 1 năm đầu để học tiếng Anh dự bị nếu thi đầu tiên với kết quả không tốt.
-   Cuộc thi này nhằm đánh giá trình độ tiếng Anh của bạn và xếp lớp theo đúng trình độ. Đề thi sẽ có khoảng 2 phần chính: phần thứ nhất là trắc nghiệm các cấu trúc ngữ pháp mà bạn thường học ở cấp 3, phần 2 là phần writing skill nhằm đánh giá vốn từ vựng cũng như kĩ năng viết câu và đoạn văn của bạn bằng tiếng Anh.
-   Theo kinh nghiệm của mình thì các bạn không cần quá áp lực về cuộc thi này, hãy làm hết sức mà bạn có thể và nhận được kết quả đúng với khả năng của bạn. Khi đó, bạn sẽ được xếp lớp đúng với trình độ của bạn. Điều này rất quan trọng để bạn có một môi trường học tập và phát triển tiếng Anh cùng với những người có cùng trình độ.
"""
v1_key = (
    "thi tiếng anh đầu vào, xếp lớp, ielts 6.0, miễn học dự bị, cấu trúc đề thi, writing skill, fpt university entrance test")
v2_full = """1.2. Chương trình học tiếng anh tại LUK Global(level 1- level 4)
Sau khi trải qua kì thi tiếng Anh đầu vào, bạn sẽ được chia lớp học tiếng Anh theo từng level (bao gồm 6 level) dựa vào kết quả bài thi của các bạn. Trường hợp kết quả của bạn thấp cho thấy rằng kỹ năng tiếng Anh của bạn còn yếu, trường sẽ xếp cho bạn học tại chương trình LUK Global (bắt đầu với level 1).
Chương trình LUK tại Đà Nẵng và TP. Hồ Chí Minh sẽ có 4 level và tại Hà Nội sẽ học 5 level. LUK là chương trình tiếng Anh chủ yếu tập trung vào kỹ năng giao tiếp, ngoài ra, các bạn còn được học các kỹ năng mềm khác cần thiết cho công việc sau này. Ở LUK, các bạn sẽ bắt buộc phải sử dụng tiếng Anh 100%.
Để nói kỹ hơn về chương trình đào tạo của LUK, chúng ta sẽ đi qua từng level. Mặc dù mỗi năm chương trình của LUK sẽ được cập nhật mỗi lúc một khác, nên mình sẽ chia sẻ chương trình theo từng level của LUK năm 2025 (năm mà mình học tập tại LUK), với tư duy trọng tâm là học tiếng Anh để hội nhập và giao tiếp. LUK tập trung vào kỹ năng nghe và nói. Sau đây là lộ trình 4 level mà mình học tại LUK tại Đà Nẵng:
Level 1 (Hurricane): Sinh viên sẽ bắt đầu tiếp cận với tiếng Anh cơ bản, giao tiếp cơ bản, luyện tập phát âm theo chuẩn API, trong đó các bạn sẽ học các kỹ năng mềm như tư duy khi học tiếng Anh, trang bị tư tưởng đúng đắn về lợi ích của việc học tiếng Anh cho sau này. Ngoài ra, các bạn sẽ được tập làm quen với thuyết trình bằng tiếng Anh khi đứng trước nhiều người.
Level 2 (Greenfire): Khi đến level này, các bạn sẽ đạt được kỹ năng giao tiếp tiếng Anh cơ bản trong giao tiếp hằng ngày. Các bạn sẽ bắt đầu tập viết news, đọc báo và tóm tắt lại các bảng tin. Ngoài ra, các bạn còn có các project mỗi tuần là thuyết trình theo team. Đầu tuần, các bạn sẽ nhận được chủ đề và chuẩn bị cùng nhóm để cuối tuần sẽ thuyết trình trước các lớp khác.
Level 3 (Heatwave): Lúc này, các bạn đã có vốn tiếng Anh nhất định. Các bạn sẽ được bắt đầu học về tranh luận (debate) về những vấn đề chung sôi nổi của thế giới, đọc, nghiên cứu, tìm kiếm bằng chứng để tranh luận về một motion đã cho trước. Với hình thức thi đấu giữa các lớn, chia ra 2 luồng ý kiến đồng ý và không đồng ý để phản biện.
Level 4 (Thunderbolt): Lúc này, các bạn sẽ chuyển sang một format debate mới, đó là debate for solution, tranh luận nhằm tìm ra giải pháp. LUK sẽ tổ chức giải với 2 bảng đấu tính điểm đến khi chọn 4 đội mạnh nhất vào vòng trong, sau đó sẽ tìm ra nhà vô địch.
Học ở LUK, bạn sẽ được tập trung phát triển kỹ năng nghe nói đầu tiên, nên mình khuyên các bạn nên bỏ thêm thời gian ở nhà để rèn luyện thêm về ngữ pháp cũng như từ vựng, tránh bỡ ngỡ sau này lúc lên học summit. Học ở LUK không khó để pass, chỉ cần các bạn đi học đầy đủ, nhưng hãy học nghiêm túc, các bạn sẽ phát triển nhanh và thuận lợi cho sau này.
"""
v2_key = (
    "lộ trình luk global, hurricane, greenfire, heatwave, thunderbolt, tiếng anh giao tiếp, debate, thuyết trình, 100% english")
v3_full = """1.3. chương trình học tiếng Anh tại Top Notch, Summit:
Summit 1 và Summit 2 là hai học phần đặc biệt chú trọng vào kỹ năng Reading và Writing, vì vậy các bạn cần đầu tư thời gian học và luyện tập nghiêm túc cho hai kỹ năng này.
Trong khi đó, Top Notch là level tập trung nhiều hơn vào Speaking và Listening. Vì vậy, nếu bạn đã pass tiếng Anh dự bị tại Đại học FPT, bạn sẽ có sẵn khá nhiều kỹ năng nền tảng quan trọng để theo học tốt các môn này.


Trong suốt học phần sẽ có 3 bài Progress Test (PT):
+ Mỗi bài PT bao gồm 3 unit đã học, với các phần: Vocabulary, Listening, Grammar và Reading.
+ Mỗi bài PT chiếm 6,7% tổng điểm môn Tiếng Anh.
Bên cạnh đó, mỗi kỳ học sẽ có 4 bài Assignment rất quan trọng, bao gồm: 2 Writing Assignment và 2 Speaking Assignment. Mỗi bài sẽ chiếm 5% tổng số điểm học phần của bạn.
Việc chỉ học kiến thức trên lớp là chưa đủ. Ở trường chúng ta, kỹ năng mềm (Soft Skills) cũng được đánh giá rất cao. Summit và Top Notch cũng thường xuyên tổ chức các sự kiện, hoạt động trải nghiệm để giúp sinh viên rèn luyện và phát triển những kỹ năng này. Thông thường, mỗi kỳ học tiếng Anh sẽ có một sự kiện lớn, ví dụ như Holiday Harmony, Spooky Scenes, Summer Voices…, và còn rất nhiều sự kiện thú vị khác đang chờ các bạn tham gia.
Một trong những điều quan trọng nhất khi học môn ENT là đi học đầy đủ. Mặc dù quy định cho phép nghỉ dưới 20% số buổi (tương đương 7 buổi), nhưng mình khuyến khích các bạn hạn chế nghỉ. Lý do là vì trong nhiều buổi học, giảng viên có thể tổ chức quiz nhanh hoặc hoạt động nhóm để tính điểm học phần.
Về hình thức thi của Summit, sẽ khác với LUK vì các bài thi đều thực hiện trên phần mềm:

Progress Test (PT): thi trên SEB (Safe Exam Browser).
Final Exam (FE): thi trên EOS.
Nếu chưa biết cách cài đặt hoặc sử dụng, các bạn nên xuống phòng IT tại thư viện để được hướng dẫn. Việc cài sai phần mềm có thể gây nhiều rắc rối và mất rất nhiều thời gian khi đi thi.
"""
v3_key = (
    "summit 1 summit 2, top notch, progress test pt, assignment môn ent, thi seb eos, sự kiện holiday harmony summer voices")
v4_full = """1.3.1 Một vài tips để pass môn ENT - TOP NOTCH VÀ SUMMIT (chia sẻ thêm):
Đi học đầy đủ là ưu tiên số 1. Nhiều bạn nghĩ nghỉ vài buổi không sao, nhưng thực tế có những buổi giáo viên cho quiz nhỏ, hoạt động nhóm hoặc điểm cộng, nghỉ là mất luôn cơ hội lấy điểm.

Đối với Progress Test (PT), các bạn không cần học lan man. Chỉ cần:

Học kỹ từ vựng trong sách

Hiểu cấu trúc ngữ pháp cơ bản trong từng unit. Làm được mấy phần này thì khả năng pass đã rất cao rồi.

Khi đi thi, nhất định phải đi sớm. Lý do là vì:

Có thể gặp lỗi máy

Phần mềm chưa cài đúng.

Máy không vào được SEB hoặc EOS.
Đi sớm để còn thời gian nhờ giám thị hoặc IT hỗ trợ, tránh tâm lý hoảng khi vào giờ thi.

Trường mình có nhiều nền tảng công nghệ hỗ trợ học tập, các bạn nên dùng quen ngay từ đầu:

 EduNext: dùng để nhận thông báo và làm bài tập về nhà

FAP: dùng để kiểm tra điểm danh, lịch học và điểm số

👉 Lưu ý cực kỳ quan trọng là phải kiểm tra điểm danh trên FAP mỗi ngày.
 Nếu thấy bị điểm danh sai, phải báo ngay cho giáo viên, vì sau 24 giờ hệ thống sẽ không cho chỉnh sửa nữa.

Với Writing Assignment, nên làm sớm, đừng để sát deadline. Làm sớm sẽ có thời gian:

Sửa lỗi ngữ pháp

Hỏi bạn bè hoặc giáo viên

Tránh lỗi nộp trễ bị trừ điểm

Với Speaking Assignment, đừng quá áp lực. Giáo viên thường chấm dựa trên:

Phát âm rõ.
Nói đủ ý.
Tự tin.
Không cần nói quá cao siêu, nói đơn giản nhưng rõ ràng là ổn.
"""
v4_key = (
    "mẹo pass môn tiếng anh, cách dùng edunext fap, kiểm tra điểm danh, writing speaking assignment tips, chuẩn bị máy tính đi thi")
# CHƯƠNG 2
v5_full = """CHƯƠNG II: “VĂN VÕ SONG TOÀN” : TỪ NHẠC CỤ DÂN TỘC ĐẾN ĐƯỜNG QUYỀN VOVINAM
2.1 Kinh nghiệm học nhạc cụ dân tộc:
. Nên tìm hiểu về các loại nhạc cụ dân tộc như đàn bầu, đàn tranh, đàn tỳ bà, sáo và lựa chọn nhạc cụ phù hợp với sở thích và tinh thần của bạn. Chọn nhạc cụ phù hợp với cá tính của mình. Ví dụ: Nam ưu tiên học sáo, nữ học đàn. Đăng ký khóa học: Tham gia các lớp học được tổ chức tại trường qua các link đăng ký. Hoặc bổ túc thêm bên ngoài để duy trì và phát triển kỹ năng
. Xây dựng thói quen: Đặt ra lịch trình luyện tập hàng ngày. Việc tập luyện thường xuyên rất quan trọng để cải thiện kỹ năng và sự tự tin. Ghi âm các buổi tập luyện để theo dõi sự tiến bộ và nhận diện những điểm cần cải thiện. Tham gia hoạt động nghệ thuật để nâng cao sự tự tin. Giao lưu kết nối các bạn cùng sở thích để trao đổi kinh nghiệm.
. Duy trì đam mê và kiên nhẫn
 Chấp nhận khó khăn: Học nhạc cụ là một hành trình dài, hãy kiên nhẫn và đừng nản lòng khi gặp khó khăn.
Tìm kiếm niềm vui: Luôn nhớ lý do bạn bắt đầu học và tìm những niềm vui trong mỗi buổi luyện tập.
Học nhạc cụ dân tộc tại Đại học FPT không chỉ giúp bạn phát triển kỹ năng âm nhạc mà còn gắn bó với văn hóa dân tộc.
Địa chỉ mua nhạc cụ
Nhạc cụ Hảo Vĩnh - 86 Hùng Vương, Đà Nẵng
"""
v5_key = (
    "học nhạc cụ dân tộc, đàn bầu, đàn tranh, sáo trúc, địa chỉ mua nhạc cụ hảo vĩnh đà nẵng, kinh nghiệm pass môn nhạc cụ")
v6_full = """2.2 Kinh nghiệm học Vovinam

Tại ĐH FPT, Vovinam là môn giáo dục thể chất chính khóa bắt buộc cho tất cả sinh viên năm nhất. Bạn sẽ học từ kiến thức cơ bản đến nâng cao trong suốt học kỳ. Bài học không chỉ là tập đòn thế mà còn gắn với võ đạo – tôn sư trọng đạo, kỷ luật, tính kiên trì và nhân cách.
💡Kinh nghiệm:
➡️ Đừng coi môn này chỉ là “học cho xong”. Nếu bạn chú trọng từ đầu, bạn sẽ tiến bộ nhanh và dễ lấy điểm cao hơn.
Tham gia CLB Vovinam để nâng cao kĩ năng 
FPT Vovinam Club là CLB sinh viên hoạt động sôi nổi tại trường với mục đích: trao đổi kinh nghiệm, rèn luyện kỹ thuật, chia sẻ chiến thuật và tổ chức các sự kiện gắn với Vovinam.


CLB là nơi bạn sẽ được học với bạn bè có cùng đam mê, tập thêm ngoài giờ học chính và tham gia các sự kiện build team rất thú vị.
💡 Kinh nghiệm:
➡️ Đừng ngại đăng ký CLB ngay từ đầu năm. Đây là nơi kết nối với “anh chị khóa trên”, dễ học hỏi và tập thêm kỹ năng thực chiến.
           Các hoạt động và sân chơi phong phú
ĐH FPT tổ chức nhiều hoạt động/show trình diễn võ nhạc, Vovinam Dance, giải FPT Edu Khơi Nguồn Võ Việt, tạo cơ hội cho sinh viên cọ xát, biểu diễn và thi đấu.


Đây không chỉ là nơi rèn kỹ thuật mà còn là nơi nối kết bạn bè, tăng tinh thần đồng đội và tự tin trình diễn trước đám đông.
💡 Kinh nghiệm:
➡️ Nếu có thể, tham gia thi hoặc biểu diễn dù không bắt buộc — sẽ giúp bạn tiến bộ nhanh hơn nhiều so với chỉ tập trong lớp
Tips  Từ sinh viên đi trước:
Tập đều đặn ngoài giờ học, ít nhất 2–3 buổi/tuần nếu có thể.


Làm quen kỹ thuật cơ bản thật tốt trước khi “nhảy” lên kỹ thuật khó.


Không ngại hỏi “anh chị khóa trên” hay thầy cô khi chưa hiểu.


Chuẩn bị giày dép, đồ tập riêng để thoải mái tập luyện.
"""
v6_key = (
    "học vovinam fpt, clb vovinam vvc, thi lên đai, võ nhạc, giải khơi nguồn võ việt, giáo dục thể chất bắt buộc, võ đạo")
# CHƯƠNG 3
v7_full = """CHƯƠNG III THÁNG NĂM RỰC RỠ: 4 TUẦN RÈN LUYỆN TẬP TRUNG, QUÂN SỰ
Lời nói đầu
Kỳ quân sự là “cú sốc” ban đầu nhưng sẽ trở thành kỷ niệm khó quên của thời sinh viên.
Dù mệt về thể xác, đây là khoảng thời gian tạo nên nhiều kỷ niệm đáng nhớ trong đời sinh viên.


Cuốn cẩm nang chia sẻ kinh nghiệm thực tế để giúp tân binh sống sót, thích nghi và tận hưởng 28 ngày quân sự.



3.1 Logistics tân binh – Xếp đồ thông minh
Nguyên tắc: Tối giản hành lý – tối đa tiện ích.
3.1.1. Những vật dụng “cứu cánh”
Băng vệ sinh hoặc miếng lót giày : lót giày chống đau chân, hút mồ hôi.


Phấn rôm: giữ chân khô, khử mùi, tránh bị nấm da.


Kem chống nắng SPF 50+: bảo vệ da luôn khỏe mạnh và tươi sáng khi tập ngoài trời.


3.1.2. Năng lượng & kết nối
Ổ cắm điện nối dài: giải quyết thiếu ổ điện và không có chỗ cắm, bởi vì thường thì các ổ điện có sẵn trong khu quân sự sẽ thường ở trong góc.
Sạc dự phòng dung lượng lớn: phòng những ngày bị mất điện hoặc là đi hành quân.


Sim 4G mạnh: Wi‑Fi yếu hoặc không có.


3.1.3. Đồ cá nhân cần thiết
Móc quần áo (≥10 cái), bút, sổ tay.


Quần áo thường
Đồ ngủ
Tất, vớ
Dép, giày thể thao


Đồ vệ sinh cá nhân, bột giặt


Đồ ăn vặt
Quạt mini
Tiền lẻ


3.1.4. Không nên mang
Trang sức và đồ có giá trị cao vì dễ thất lạc.


Mỹ phẩm quá nhiều, không phù hợp với môi trường tập thể.


Hành lý cồng kềnh gây khó khăn trong việc sắp xếp và di chuyển.



3.2 - 24 giờ đầu tiên – Thích nghi nhanh
Ngày đầu là ngày căng thẳng nhất, đòi hỏi phản ứng nhanh và tuân thủ kỷ luật.


Phải tập trung đúng giờ, nghe hiệu lệnh ngay khi có còi, không chần chừ.


Nhà tắm tập thể: chuẩn bị đồ gọn, tắm nhanh, phối hợp với đồng đội.
Không chú trọng ăn diện, ưu tiên mặc quân phục và đúng tác phong.



3.3 Nội vụ – Gấp chăn “bánh chưng”
Gấp chăn là yêu cầu bắt buộc và dễ bị phạt nhất.


Dùng thước, thẻ ATM để tạo nếp vuông đẹp.


Mẹo: mang chăn cá nhân để đắp, giữ chăn quân đội luôn gọn suốt 28 ngày.
"""
v7_key = (
    "kinh nghiệm đi quân sự, đồ dùng tân binh, lót giày, phấn rôm, gấp chăn bánh chưng, nội vụ, quân khu 5, sống sót 28 ngày")
# CHƯƠNG 4
v8_full = """CHƯƠNG IV: “AN CƯ LẠC NGHIỆP” - KÝ TÚC XÁ, TRỌ VÀ ẨM THỰC TẠI TRƯỜNG ĐẠI HỌC FPT ĐÀ NẴNG
Giới thiệu chương
        Trong hành trình đại học, học tập chỉ là một phần của cuộc sống sinh viên. Nơi ở, môi trường sinh hoạt và thói quen ăn uống hàng ngày mới chính là những yếu tố âm thầm nhưng có ảnh hưởng lâu dài đến sức khỏe, tinh thần và hiệu quả học tập.
        Đối với sinh viên Trường Đại học FPT Đà Nẵng – nơi có campus khép kín, hiện đại và nằm trong khu đô thị riêng – việc lựa chọn ở ký túc xá hay ở trọ bên ngoài không chỉ là câu chuyện chi phí mà còn liên quan đến lối sống, khả năng thích nghi và mức độ tự lập.
        Chương này sẽ phân tích chi tiết từng lựa chọn chỗ ở, đánh giá môi trường campus và gợi ý các giải pháp ăn uống phù hợp, giúp sinh viên xây dựng một cuộc sống “an cư” vững vàng để “lạc nghiệp” trong suốt quãng đời đại học.

4.1. Review tổng quan khuôn viên Trường Đại học FPT Đà Nẵng
        Trường Đại học FPT Đà Nẵng tọa lạc tại Khu đô thị FPT City, quận Ngũ Hành Sơn, sở hữu campus rộng rãi, hiện đại và tách biệt khỏi sự ồn ào của trung tâm thành phố.
        Campus được thiết kế theo mô hình “học tập – sinh hoạt – trải nghiệm” trong cùng một không gian, giúp sinh viên có thể học tập, nghỉ ngơi và tham gia hoạt động ngoại khóa mà không cần di chuyển xa.
4.1.1 Không gian học tập
Các tòa nhà học tập như Alpha, Gamma được trang bị phòng học hiện đại, phòng máy tính và phòng thực hành chuyên ngành.


Thư viện là nơi tự học lý tưởng, cung cấp nhiều tài liệu học tập, không gian yên tĩnh và khu học nhóm.


4.1.2 Không gian sinh hoạt
Căn tin phục vụ nhu cầu ăn uống hằng ngày của sinh viên.


Ký túc xá nằm ngay trong campus, thuận tiện cho sinh viên nội trú.


Khu thể thao và không gian xanh giúp sinh viên cân bằng giữa học tập và vận động, giảm căng thẳng sau giờ học.


        Nhìn chung, campus FPT Đà Nẵng không chỉ là nơi học mà còn là một không gian sống đúng nghĩa cho sinh viên.
"""
v8_key = (
    "review campus fpt đà nẵng, tòa nhà alpha gamma, thư viện, city city ngũ hành sơn, cơ sở vật chất, không gian tự học")
v9_full = """4.2 So sánh ở Ký túc xá và ở trọ bên ngoài
4.2.1. Ở Ký túc xá (KTX) – Cuộc sống trong campus FPT Đà Nẵng
        Ký túc xá Trường Đại học FPT Đà Nẵng được xây dựng ngay trong khuôn viên campus, tạo nên một môi trường sinh hoạt khép kín, thuận tiện và an toàn cho sinh viên, đặc biệt là sinh viên năm nhất hoặc sinh viên ở xa.
Ưu điểm
- Thứ nhất, vị trí thuận lợi tuyệt đối.
        Sinh viên ở KTX chỉ mất vài phút đi bộ để đến lớp học, thư viện, căn tin hay khu thể thao. Điều này giúp tiết kiệm đáng kể thời gian di chuyển, giảm mệt mỏi và hạn chế việc trễ giờ học – một yếu tố rất quan trọng trong môi trường học tập kỷ luật của FPT.
- Thứ hai, chi phí hợp lý và dễ kiểm soát.
        So với việc thuê trọ bên ngoài, chi phí ở KTX thường thấp hơn và ít phát sinh. Sinh viên không phải lo lắng quá nhiều về tiền điện, nước, internet hay các khoản phụ thu khác, từ đó dễ dàng quản lý tài chính cá nhân.
- Thứ ba, môi trường sinh hoạt tập thể.
        Ở KTX, sinh viên có cơ hội sống và sinh hoạt cùng bạn bè đến từ nhiều vùng miền khác nhau. Điều này giúp rèn luyện kỹ năng giao tiếp, làm việc nhóm, giải quyết mâu thuẫn và xây dựng các mối quan hệ xã hội – những kỹ năng mềm rất cần thiết cho tương lai.
- Thứ tư, an ninh được đảm bảo.
        KTX nằm trong khu campus có bảo vệ, quản lý nội trú và hệ thống kiểm soát ra vào, giúp sinh viên và phụ huynh yên tâm hơn, đặc biệt với sinh viên năm đầu xa nhà.
Nhược điểm
Hạn chế về không gian riêng tư.
        Phòng ở KTX thường là phòng chung, sinh viên phải chia sẻ không gian sinh hoạt với nhiều người. Điều này có thể gây bất tiện cho những bạn cần không gian yên tĩnh tuyệt đối để học tập hoặc nghỉ ngơi.
Giờ giấc và nội quy tương đối nghiêm.
        Sinh viên ở KTX cần tuân thủ các quy định về giờ giấc, sinh hoạt và sử dụng không gian chung. Với những bạn quen lối sống tự do, điều này đôi khi gây cảm giác gò bó.

4.2.2. Ở trọ bên ngoài – Cuộc sống tự lập và chủ động
        Bên cạnh KTX, nhiều sinh viên FPT Đà Nẵng lựa chọn thuê trọ bên ngoài, đặc biệt là những bạn mong muốn có không gian sống riêng.
Ưu điểm
- Không gian riêng tư cao.
        Ở trọ giúp sinh viên có không gian cá nhân, dễ dàng sắp xếp góc học tập, nghỉ ngơi và sinh hoạt theo thói quen của bản thân. Đây là yếu tố quan trọng với những bạn cần sự yên tĩnh hoặc học tập cường độ cao.
- Tự do về giờ giấc và sinh hoạt.
        Sinh viên ở trọ không bị ràng buộc bởi nội quy nội trú. Việc về muộn, học khuya, nấu ăn hay tiếp bạn bè đều linh hoạt hơn, giúp hình thành lối sống tự lập.
- Chủ động trong ăn uống.
        Có bếp riêng cho phép sinh viên tự nấu ăn, vừa tiết kiệm chi phí, vừa đảm bảo vệ sinh và dinh dưỡng – điều mà không phải lúc nào căn tin cũng đáp ứng đầy đủ.
Nhược điểm
- Chi phí cao và khó kiểm soát hơn.
        Ngoài tiền thuê phòng, sinh viên còn phải chi trả tiền điện, nước, internet, rác thải… Nếu không biết quản lý, tổng chi phí hàng tháng có thể cao hơn đáng kể so với ở KTX.
- Vấn đề an ninh và di chuyển.
Ở trọ bên ngoài đòi hỏi sinh viên phải tự lo an ninh cá nhân và phương tiện đi lại. Nếu trọ xa campus, việc di chuyển mỗi ngày có thể gây mệt mỏi và tốn thời gian.

4.2.3. Đánh giá tổng quát
-KTX phù hợp với sinh viên năm nhất, sinh viên ở xa hoặc những bạn muốn môi trường ổn định, tiết kiệm và an toàn.


-Ở trọ phù hợp với sinh viên đã quen nhịp sống đại học, mong muốn tự do, riêng tư và sẵn sàng tự quản lý cuộc sống.
"""
v9_key = (
    "so sánh ký túc xá và trọ, ưu nhược điểm ktx fpt, an ninh nội trú, chi phí ở trọ, giờ giấc ktx, tự lập sinh viên")
v10_full = """4.3. Cẩm nang: tất tần tật kinh nghiệm thuê trọ cho tân sinh viên
        Tìm được một "chốn an cư" lạc nghiệp giữa thành phố xa lạ là bước đệm quan trọng để bạn bắt đầu đời sinh viên rực rỡ. Để tránh những "cú lừa" và tìm được phòng ưng ý, hãy bỏ túi ngay những bí kíp sau:
1. Vị trí: Ưu tiên "Nhất cận lộ, nhị cận trường"
        Đừng đợi đến sát ngày nhập học mới tìm nhà. Ngay khi có kết quả, hãy khoanh vùng khu vực dựa trên:
• Cơ sở học tập: Kiểm tra xem sinh viên năm nhất học ở cơ sở nào để tránh thuê nhầm chỗ quá xa.
• Giao thông: Ưu tiên bán kính 1–2 km quanh trường để có thể đi bộ hoặc đạp xe. Nếu xa hơn, hãy chọn nơi gần trạm xe buýt.
• Tiện ích: Gần chợ, siêu thị tiện lợi và hiệu thuốc là một điểm cộng lớn.
2. Săn tin: Thông minh trên "thế giới ảo"
Thay vì đi bộ giữa nắng gắt, hãy bắt đầu bằng cách khảo sát giá:
• Tận dụng hội nhóm: Tham gia các group Facebook như "Tìm phòng trọ quận [X]", "Review phòng trọ [Tên trường]".
• Từ khóa tìm kiếm: Sử dụng các cụm từ cụ thể như "phòng trọ giá rẻ cho sinh viên + [Quận]".
• Cảnh giác: Cẩn thận với những bài đăng "phòng đẹp như khách sạn, giá rẻ bất ngờ" – đó thường là mồi nhử của môi giới hoặc lừa đảo tiền cọc.
3. Check-list: Kiểm tra phòng "như một chuyên gia"
Khi đi xem phòng thực tế, đừng chỉ nhìn qua loa, hãy kiểm tra 4 yếu tố:
• Cơ sở vật chất: Tường có thấm mốc không? Điện, nước có ổn định? Nhà vệ sinh có kín đáo và sạch sẽ không?
• An ninh: Tránh các hẻm quá sâu, vắng vẻ. Ưu tiên nơi có camera hoặc cổng khóa vân tay.
• Môi trường sống: Tránh gần quán nhậu, karaoke ồn ào. Hãy thử hỏi thăm những người đang ở đó về tính tình chủ nhà và tình hình an ninh khu phố.
• Chi phí ẩn: Hỏi rõ giá điện, nước, phí rác, internet và phí gửi xe (nếu có).
4. Hợp đồng: "Bút sa gà chết"
Mọi thỏa thuận miệng đều vô giá trị, tất cả phải nằm trên giấy trắng mực đen:
• Tiền cọc: Quy định rõ điều kiện để được hoàn lại tiền cọc khi chuyển đi.
• Hiện trạng vật chất: Ghi lại danh sách đồ đạc có sẵn và tình trạng hư hỏng (nếu có) vào hợp đồng để không bị đền bù oan sau này.
• Thời hạn: Làm rõ thời gian thuê tối thiểu và thời hạn báo trước khi muốn chuyển đi (thường là 30 ngày).
5. Ở ghép: Chọn bạn mà chơi, chọn người mà ở
Ở ghép giúp tiết kiệm chi phí nhưng cũng dễ phát sinh mâu thuẫn:
• Đối tượng: Ưu tiên bạn học cùng lớp hoặc người quen từ quê.
• Nguyên tắc chung: Ngay từ đầu, hãy thống nhất về giờ giấc, việc dọn dẹp và việc dẫn bạn bè về phòng.
• Cảnh giác: Nếu ở cùng người lạ, hãy bảo quản tài sản cá nhân (laptop, điện thoại, ví tiền) thật cẩn thận, ít nhất là trong thời gian đầu.
💡 Tips nhỏ cho bạn: Hãy đi xem phòng vào buổi trưa hoặc lúc trời mưa. Đó là lúc bạn biết rõ nhất phòng có bị nóng hầm hay bị ngập nước/thấm dột hay không!
"""
v10_key = (
    "cẩm nang thuê trọ đà nẵng, lừa đảo tiền cọc, hợp đồng thuê nhà, tìm bạn ở ghép, kiểm tra phòng trọ, khu vực ngũ hành sơn")
v11_full = """4.4. Ẩm thực quanh campus – Nhu cầu thiết yếu của sinh viên
4.4.1. Quán ăn giá sinh viên
Xung quanh campus và khu vực lân cận có nhiều quán ăn bình dân phục vụ sinh viên:
Canteen trường Đại học FPT: Đa dạng món, không gian "chill" và giá sinh viên.
Xôi, bánh mì Cô Phương (gần cổng trường Việt Hàn): Phục vụ xôi, bánh mì thịt nướng và bò kho với mức giá khoảng 15.000–20.000 đồng, khẩu phần đầy đặn.
Hoạ Mơ Coffee & Food (gần Trường Đại học FPT): Kết hợp cà phê và các món ăn, thuận tiện cho sinh viên vừa học vừa ăn.
Quán Cô Thống (Đối diện FPT Complex): Chuyên cơm trưa bình dân
Cơm Cao Bồi (V5.B01.35 Shophouse FPT): Chuyên cơm trưa.
Mỳ Quảng 37 (364 Trần Đại Nghĩa): Hương vị bản địa.
Quán Phở Bắc (Trần Đại Nghĩa, gần cafe Vành Đai): Ăn sáng.
Cơm gà Xả Xệ (Đối diện FPT Complex): Bán cả trưa và tối
Bánh cuốn nóng Hoa (04 Nguyễn Duy Cung):Món ăn sáng phổ biến.
Xôi xéo Hà Nội (V5.B01.12 Shophouse FPT):Bữa sáng ngon, tiện lợi.

4.4.2. Quán cà phê học bài/làm việc (khu FPT City & lân cận)
Ngoài ăn uống, sinh viên thường tìm các quán cà phê yên tĩnh để học nhóm, làm bài tập hoặc thảo luận dự án. Những quán có wifi mạnh, không gian thoáng và giá đồ uống vừa phải luôn được ưu tiên.
Trees Tea & Coffee (đường Nam Kỳ Khởi Nghĩa): Không gian rộng rãi, thoáng mát và yên tĩnh, phù hợp cho sinh viên học bài hoặc làm việc cá nhân trong thời gian dài.


Nốt Coffee (đối diện Trường Đại học FPT): Vị trí thuận tiện, dễ dàng di chuyển giữa các ca học, phù hợp cho những buổi học nhanh hoặc làm việc ngắn.


Dailly Coffee (shophouse 06 FPT Plaza 2): Thiết kế hiện đại, không gian sáng, phù hợp cho học nhóm và làm việc.


Oxy Garden Coffee (đường Trần Quốc Vượng, đối diện FPT Complex): Không gian xanh, view đẹp, tạo cảm giác thư giãn nhưng vẫn đảm bảo sự tập trung.


Zone Six Cafe (40 Trần Văn Dư): Quán hoạt động 24/7, có không gian riêng, thích hợp cho sinh viên học khuya hoặc làm việc ngoài giờ.



4.4.3. Ăn vặt
Cuối tuần là thời gian sinh viên thư giãn, tụ họp bạn bè. Việc khám phá các quán ăn vặt, quán nướng hay quán gần biển giúp cân bằng cuộc sống và tạo thêm kỷ niệm sinh viên.
Camry Quán (304 Trần Đại Nghĩa): Địa điểm quen thuộc của sinh viên, phục vụ mì cay & ăn vặt với mức giá hợp lý, phù hợp cho bữa ăn nhanh.

Bếp của Nem (SHV5.B05.38 KĐT FPT): bánh mỳ chảo đặc biệt.


Bún đậu 1996 (358 Trần Đại Nghĩa)


Bánh xèo – nem lụi Cô Mười: quán ăn vặt giá rẻ, được nhiều sinh viên lựa chọn.


Kem bơ Cô Vân (chợ Bắc Mỹ An): Món tráng miệng nổi tiếng, giá rẻ, phù hợp với sinh viên.
Hee Mang Chicken (358A Trần Đại Nghĩa)
Tiệm Bánh Nhà Kim (k230 Trần Hưng Đạo, Điện Ngọc): Chuyên bánh ngọt, bánh kem, ăn vặt.
Ông Tèo - Hải Sản Bình Dân (Lô 01 khu B3-78):Quán nhậu gần FPT, giá hợp lý, view sông.
"""
v11_key = (
    "quán ăn ngon fpt đà nẵng, cafe học bài, zone six 24/7, cơm gà xả xệ, bún đậu 1996, canteen fpt, ăn vặt trần đại nghĩa")
# CHƯƠNG 5
v12_full = """CHƯƠNG V: THÔNG TIN LIÊN HỆ CÁC PHONG BAN TẠI ĐẠI HỌC FPT VÀ LINK QUAN TRỌNG

Để có kết quả học tập thật tốt, các bạn cần quản lý và sử dụng thành thạo các trang web và ứng dụng của FPT University. Nên chapter này ở đây để giúp bạn giải quyết các vấn đề về kỹ thuật.

A. Vấn đề kỹ thuật
5.1.  FAP
Các thông báo của trường, điểm danh, lịch học, cũng như nộp tiền và các dịch vụ khác sẽ được tích hợp trên FAP.
https://fap.fpt.edu.vn/ 

5.2.  FLM
Giáo trình và slide của các môn học sẽ được tải lên trang web này.
https://flm.fpt.edu.vn/Login

5.3.  SEB và EOS

5.3.1  SEB
- SEB là gì? SEB (Safe Exam Browser) là một phần mềm dùng để làm các bài kiểm tra lớn, nhỏ.

- Đây là link để tải SEB: https://drive.google.com/drive/u/2/folders/1RmjeKAvef6BXg_qlAl6JnZx2ZkY3qj_3



5.3.2  EOS
- EOS cũng dùng để làm bài kiểm tra, nhưng là bài kiểm tra cuối kì hay FE (Final Exam). Do đó, việc cập nhật và kiểm tra EOS trước ngày thi là vô cùng quan trọng.

- Đây là đường dẫn để tải xuống EOS:
https://lmsdn.fpt.edu.vn/hd/eos/

Hoặc vào trang web: https://lmsdn.fpt.edu.vn/hd/ rồi tìm mục Software trong phần Download. 


Sau đó, chọn EOS và tải.

Lưu ý: Sau khi đã làm xong bài FE dù là ở Top Notch, Summit, hay ở chuyên ngành thì bạn phải luôn xác nhận mình đã làm bài ở link sau:
https://e360.fpt.edu.vn/checkout 

Nếu không, bài làm của bạn sẽ không được công nhận và bài FE của bạn sẽ không có điểm.
"""
v12_key = ("link fap flm, tải phần mềm thi seb eos, lỗi kỹ thuật, checkout e360, hướng dẫn cài đặt phần mềm trường fpt")
v13_full = """B. Thói quen học tập
5.1. Ngủ đủ giấc
- Ngủ đủ 7 - 8 tiếng mỗi đêm để tinh thần và não bộ luôn ở tình trạng tốt nhất.
- Kết hợp thêm với tập thể dục buổi sáng sau khi thức dậy, sẽ là một cách mở đầu ngày mới tuyệt vời.

5.2. Xem lại bài
Một điều cực kỳ quan trọng sau khi hoàn thành một ngày học ở trường là gì? Đó là kiểm tra điểm danh (check attendance) của bạn. Thế nhưng, việc cũng quan trọng không kém đó là luôn xem lại các bài học sau mỗi buổi học để nắm chắc kiến thức, để khi đến lúc kiểm tra thì chỉ cần ôn lại bài chứ không phải học lại toàn bộ.

5.3. Xem trước bài
Dành ra 10–15 phút xem trước bài giúp bạn nắm được hướng đi của bài học và hiểu bài dễ dàng khi đến lớp, nhớ sâu nội dung của bài học.

5.4. Nghỉ ngơi
- Chăm chỉ là tốt, nhưng cho bản thân nghỉ ngơi là điều nên làm. Não bộ cũng giống như cơ bắp, nó cũng cần nghỉ ngơi để phát triển và hình thành những nhóm cơ mới. 
- Vì vậy, sau 1 giờ học tập nghiêm túc hãy dành ra 5 - 10 phút thư giãn dành riêng cho bản thân, không thiết bị, không điện thoại di động và bạn có thể bắt chuyện với bạn hoặc người thân để giải tỏa căng thẳng.
"""
v13_key = (
    "quản lý thời gian, thói quen ngủ, xem trước bài, check attendance fap, kỹ năng tự học hiệu quả, sức khỏe não bộ")  # --- 3. LỆNH CHẠY UPLOAD (ĐÃ CHECK LỖI) ---
# --- 3. LỆNH CHẠY UPLOAD (ĐÃ ĐỒNG NHẤT TÊN COLLECTION) ---
if __name__ == "__main__":
    print(f"🚀 Đang làm sạch và chuẩn bị tải lên collection: {TARGET_COLLECTION}...")

    # Xóa dữ liệu cũ của collection MỚI (để tránh trùng lặp khi chạy lại nhiều lần)
    docs = db.collection(TARGET_COLLECTION).stream()
    for doc in docs:
        doc.reference.delete()

    upload_vector_final("V1", v1_full, v1_key, 1, "1.1 - Thi đầu vào")
    upload_vector_final("V2", v2_full, v2_key, 1, "1.2 - LUK Global")
    upload_vector_final("V3", v3_full, v3_key, 1, "1.3 - Summit & TopNotch")
    upload_vector_final("V4", v4_full, v4_key, 1, "1.3.1 - Tips Pass ENT")
    upload_vector_final("V5", v5_full, v5_key, 2, "2.1 - Nhạc cụ dân tộc")
    upload_vector_final("V6", v6_full, v6_key, 2, "2.2 - Vovinam")
    upload_vector_final("V7", v7_full, v7_key, 3, "3.1 - Quân sự")
    upload_vector_final("V8", v8_full, v8_key, 4, "4.1 - Campus Review")
    upload_vector_final("V9", v9_full, v9_key, 4, "4.2 - KTX vs Trọ")
    upload_vector_final("V10", v10_full, v10_key, 4, "4.3 - Cẩm nang thuê trọ")
    upload_vector_final("V11", v11_full, v11_key, 4, "4.4 - Ẩm thực ăn uống")
    upload_vector_final("V12", v12_full, v12_key, 5, "5.A - Link & Kỹ thuật")
    upload_vector_final("V13", v13_full, v13_key, 5, "5.B - Thói quen học tập")

    print("\n🚀 QUÁ TRÌNH HOÀN TẤT!")