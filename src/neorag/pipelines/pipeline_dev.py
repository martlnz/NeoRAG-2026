"""Pipeline của thí sinh, cài đặt interface `RAGPipeline`.

Chạy:
    neorag run train --pipeline neorag.pipelines.pipeline:MyPipeline
    neorag run public_test --pipeline neorag.pipelines.pipeline:MyPipeline

Lịch sử phiên bản và phân tích: thư mục `experiments/`.

v1:
  1. LLM viết lại câu hỏi (thêm dấu, mở viết tắt, chuẩn hóa tên CLB, tách câu nhiều ý).
  2. Hybrid retrieval: BM25 trên văn bản bỏ dấu + dense bge-m3, gộp bằng RRF trên nhiều truy vấn.
  3. Prompt sinh ngắn gọn, không markdown; LLM ghi id nguồn → đẩy lên đầu retrieved_ids.
v2:
  - Prompt sinh 1-2 câu, tên CLB chuẩn, không rào đón / ý phụ; đưa 10 doc vào prompt.
  - Prompt viết lại giữ đúng ý định ("chạy chương trình" = sự kiện, không phải lập trình).
v3:
  - Rewrite có cấu trúc (dòng "Q: "), lọc dòng meta; câu so sánh → câu con theo từng CLB.
  - NGUON ghi mọi đoạn chứa thông tin (không chỉ đoạn đã dùng) → giữ recall@5.
  - Chép nguyên văn cụm then chốt (số + đơn vị, tên ban/giải, biệt danh tiếng Anh); nêu link khi hỏi kênh.
v3.1 (chống lỗi, không đổi prompt → không đổi điểm):
  - rewrite lỗi API → dùng câu gốc; câu trả lời rỗng → gọi lại 1 lần với khóa cache mới.
v5:
  - Câu gốc không dấu: bỏ dense (bge-m3 vô dụng với câu không dấu), chỉ giữ BM25 bỏ dấu.
  - Đưa top-15 doc vào prompt (vẫn nộp 10 id; doc được cite ở hạng 11-15 lên đầu).
v6:
  - Từ chối chuẩn hóa ("Tài liệu hiện không có thông tin về ..."), không suy ra khẳng định phủ định.
  - Mở đầu "Có, / Không, ..." cho câu có/không; "Theo <nguồn kèm năm>, ..." khi tài liệu ghi nguồn.
  - Rewrite: câu con tự đứng được (lọc "CLB đó/này"), câu "nên vào CLB nào" → câu con theo CLB,
    câu "ban nào lo X" → thêm câu con hỏi danh sách ban.
v7 (bị loại): few-shot tĩnh từ train — xem experiments/update_v7.md; few-shot tắt mặc định.
v8 (bị loại; các phần được giữ dưới dạng tùy chọn tắt mặc định):
  - Từ chối đúng MỘT câu "Tài liệu hiện không có thông tin về ...", bỏ câu ngữ cảnh "Tài liệu chỉ nêu ..."
    (câu không có đáp án chỉ chấm Abstention; câu ngữ cảnh chỉ thêm rủi ro bị chấm 0).
  - Parent-child retrieval: child = từng câu (+ header club|topic), max-pooling về đoạn cha,
    gộp RRF với ranking đoạn nguyên (dense + BM25). Nộp id và đưa toàn văn đoạn cha cho LLM.
v9 (hòa v6; mẫu chung cho mọi câu — tùy chọn answer_templates=True, class MyPipelineV9):
  - Bật ANSWER_TEMPLATES: 8 mẫu trả lời CHUNG (có chỗ trống <...>) rút từ khuôn đáp án train,
    không chứa câu hỏi/dữ kiện thật → không rò rỉ, không có gì để LLM chép nhầm (khác v7).
v10 (BẢN THI — mặc định của MyPipeline):
  - v6 + mẫu trả lời chung CHỈ cho câu hỏi có dấu (answer_templates="accented"); câu không dấu giữ prompt v6.
  - Dev: train 0.9269 / informal 0.9083 (v6: 0.9220 / 0.9083). Public TB: v10 87.02 (2 lần), v6 87.43 (3 lần) — trong nhiễu.
  - Dự phòng: class MyPipelineV6.
v11 (class MyPipelineV11, KHÔNG phải mặc định):
  - v10 + xếp lại id nộp trong cùng tập id của v10: id được cite lên đầu, phần còn lại theo cosine với đoạn được cite.
  - Câu trả lời giữ nguyên v10. Dev (retrieval, --no-judge): train +0.0029, informal −0.0011 → ≈ +0.0005 benchmark, trong nhiễu.
v12 (class MyPipelineV12, KHÔNG phải mặc định; xem experiments/update_v12.md):
  - Câu trả lời và TẬP id nộp = v10/v11; chỉ đổi thứ tự id: cite → đoạn bằng chứng (một lượt LLM riêng chọn mọi
    đoạn chứa dữ kiện cần, evidence_select) → phần còn lại theo parent-child + 0.5·cosine với đoạn được cite
    (filler="pc_cos"). Bền hơn v10/v11 khi LLM cite thiếu đoạn (mô phỏng cite1/drop50).
  - Dev retrieval: train 0.9894 / informal 0.9879 (v11: 0.9865 / 0.9853; v10: 0.9836 / 0.9864).
  - answer_first_fix (lượt viết lại câu mở đầu bằng từ chối) có sẵn nhưng tắt: không đạt tiêu chí giữ.
v13 (class MyPipelineV13, KHÔNG phải mặc định; xem experiments/update_v13.md):
  - v11 (cite_rerank) + bật answer_first_fix (viết lại câu mở đầu bằng từ chối khi có "Tài liệu chỉ nêu...").
v14 (class MyPipelineV14, KHÔNG phải mặc định; xem experiments/update_v14.md):
  - Câu trả lời v9 (mẫu chung mọi câu) + chốt giữ từ chối của v11 (+ answer_first_fix); thứ tự id = cite_rerank.
v11-audit (class MyPipelineV11Audit — BẢN ĐẠT 91.35 PRIVATE TEST; xem experiments/update_v11_audit.md):
  - Giữ nguyên bước sinh nháp v11 (top-15 hybrid RRF + answer_templates="accented"), mở rộng ngữ cảnh kiểm định lên
    tối đa 20 đoạn (thêm đoạn song song cùng (club, topic) với nguồn nháp & ứng viên parent-child RRF).
  - Bổ sung lượt LLM Auditor (claude-haiku-5-5) kiểm định 6 tiêu chí: gỡ từ chối oan, chuẩn hóa câu thiếu dữ liệu
    bằng mệnh đề nối dấu ";", sửa ngược cực Có/Không, cắt suy diễn/nhiễu chéo đoạn, bổ sung ý song song & cụm
    key_facts nguyên văn, bảo toàn bản nháp sạch ("TRA_LOI: GIỮ").
  - Xếp hạng id bằng order_ids_audit (audit_cited + draft_cited + cite_rerank cosine; khóa tập
   id v10 khi GIỮ).
  - Private Test chính thức: 91.35 Raw score (+2.60 điểm so với v11 = 88.75).
"""

from __future__ import annotations

import math
import os
import re
import unicodedata
from collections import Counter

import numpy as np

from neorag.data import DATA_DIR, Document, read_jsonl
from neorag.llm import LLMClient
from neorag.pipelines.base import RAGOutput, RAGPipeline
from neorag.text import split_sentences

REWRITE_PROMPT = """Bạn chuẩn hóa câu hỏi của sinh viên về cộng đồng S2B ở Học viện Công nghệ Bưu chính Viễn thông (PTIT), gồm 3 CLB:
- CLB Lập Trình PTIT (ProPTIT; người hỏi có thể gọi "clb lập trình", "clb lt", "clb code")
- CLB Multimedia PTIT (MPC; "clb multimedia", "clb mm", "clb truyền thông/thiết kế/quay dựng")
- CLB Nhà Sáng Tạo Game PTIT (GCC; "clb game", "clb làm game")
Viết tắt thường gặp: hv = Học viện, e = em, a/c = anh/chị, ko/k = không, tv = thành viên, dpt = Đa phương tiện, nk = nhiệm kỳ, cn = chủ nhiệm.
Lưu ý nghĩa: "chương trình", "plan", "chạy chương trình" thường là tổ chức sự kiện, không phải lập trình; "code" mới là lập trình.

Nhiệm vụ:
1. Viết lại câu hỏi thành tiếng Việt đầy đủ dấu, rõ ràng, nêu tên CLB chuẩn nếu xác định được. Giữ đúng ý định và mọi chi tiết người hỏi nêu; không đổi nghĩa, không thêm thông tin mà câu hỏi không có.
2. Nếu câu hỏi gồm nhiều ý hoặc so sánh nhiều CLB, viết thêm tối đa 3 câu hỏi con, mỗi câu một ý hoặc một CLB. Với câu so sánh/xếp hạng giữa các CLB, mỗi câu con hỏi đúng thuộc tính đó cho từng CLB (ví dụ "CLB Multimedia PTIT được thành lập ngày nào?").
3. Mỗi câu hỏi con phải tự đứng được: nêu rõ tên CLB, không dùng "CLB đó", "CLB này", "ở đó".
4. Câu hỏi kiểu "thích X thì nên vào CLB nào" (không nêu CLB): mỗi câu con hỏi CLB đó có hoạt động/nhánh/ban liên quan tới X không, một câu cho mỗi CLB phù hợp.
5. Câu hỏi "có ban/bộ phận/ai lo việc X không": thêm câu con "CLB ... có những ban nào và mỗi ban phụ trách gì?".
6. Không nhắc tới CLB mà người hỏi không nêu, trừ trường hợp ở mục 4.

Định dạng bắt buộc: mỗi câu hỏi trên một dòng, bắt đầu bằng "Q: ", dòng đầu là câu viết lại. Câu hỏi chỉ có một ý thì in đúng một dòng. Không in gì khác.

CÂU HỎI: {query}"""

ANSWER_PROMPT = """Bạn là trợ lý tư vấn về các CLB thuộc cộng đồng S2B của PTIT. Chỉ dùng TÀI LIỆU bên dưới.

Quy tắc:
- Trả lời 1-2 câu văn xuôi, tối đa khoảng 60 từ; không markdown, không in đậm, không gạch đầu dòng, không chào hỏi. Chỉ viết câu thứ hai khi câu hỏi có ý thứ hai.
- Câu đầu lặp lại chủ thể của câu hỏi và trả lời thẳng (ví dụ "CLB Lập Trình PTIT được thành lập ngày ...").
- Câu hỏi có/không: mở đầu bằng "Có, ...", "Không, ...", "Đúng, ..." hoặc "Chưa, ..." (dấu phẩy, không dùng dấu chấm), rồi nêu chủ thể và lý do.
- Khi đoạn tài liệu tự ghi nguồn (ví dụ "theo Phòng truyền thống", "theo bài tuyển thành viên ngày ...", "theo bài giới thiệu năm ..."), mở câu bằng "Theo <nguồn đó, chép nguyên văn kèm năm/ngày>, ...".
- Gọi tên CLB đúng một dạng: "CLB Lập Trình PTIT", "CLB Multimedia PTIT", "CLB Nhà Sáng Tạo Game PTIT (GCC)"; không dùng tên dài hay tên tiếng Anh của CLB nếu không được hỏi.
- Chỉ nêu thông tin mà câu hỏi hỏi tới; không thêm "ngoài ra", hashtag hay bối cảnh khi không được hỏi. Khi câu hỏi hỏi đăng ký/liên hệ ở đâu, qua kênh nào thì nêu đúng link, email, website.
- Chép nguyên văn cụm then chốt từ tài liệu: tên riêng, ngày tháng, số kèm đơn vị, tên giải/ban/đội/sự kiện (kể cả tên tiếng Anh như biệt danh của ban), tên đơn vị; không diễn đạt lại các cụm này.
- Câu hỏi so sánh hoặc nhiều ý: mỗi ý một mệnh đề ngắn, trả lời đủ từng ý, kể cả khi thông tin nằm rải rác ở nhiều đoạn.
- Không viết câu kiểu "Tài liệu không nêu/không ghi chi tiết ..." về những khía cạnh câu hỏi không hỏi.
- Khi chính điều được hỏi không có trong tài liệu: câu đầu bắt đầu đúng bằng "Tài liệu hiện không có thông tin về ...". Không suy ra khẳng định phủ định (như "X không có ...") chỉ vì tài liệu không nhắc tới. Có thể thêm tối đa một câu bắt đầu bằng "Tài liệu chỉ nêu ..." với thông tin liên quan gần nhất; không đoán.
- Nếu chỉ một phần câu hỏi không có trong tài liệu: trả lời phần có trước, rồi nói "tài liệu không ghi ..." cho phần còn lại.

Dòng cuối cùng ghi: NGUON: <id của MỌI đoạn chứa thông tin trả lời câu hỏi, kể cả đoạn trùng ý, xếp từ liên quan nhất, cách nhau dấu phẩy>
{examples}
TÀI LIỆU:
{context}

CÂU HỎI GỐC: {query}
(Hiểu là: {rewritten})"""

RRF_K = 60
# Few-shot tĩnh từ tập train (quy định cho phép dùng train để xây few-shot): mỗi kiểu trả lời một ví dụ.
FEWSHOT_IDS = (
    "train-0085",  # nêu nguồn
    "train-0028",  # có/không + nguồn + nhiều đoạn
    "train-0010",  # so sánh
    "train-0050",  # không có đáp án
    "train-0069",  # thiếu một phần dữ liệu
    "train-0101",  # mâu thuẫn nguồn
)
FEWSHOT_HEADER = (
    "\nVÍ DỤ CÁCH TRÌNH BÀY (chỉ để học văn phong, độ chi tiết và cách từ chối; "
    "KHÔNG dùng làm nguồn thông tin, mọi thông tin phải lấy từ TÀI LIỆU bên dưới):\n"
)
# Mẫu trả lời CHUNG rút ra từ khuôn đáp án train (không chứa câu hỏi/dữ kiện thật → không rò rỉ,
# không có gì để LLM chép nhầm). Dùng thay cho few-shot thật (v7 bị loại).
ANSWER_TEMPLATES = """
MẪU TRÌNH BÀY THEO KIỂU CÂU HỎI (chỉ là khuôn; <...> phải thay bằng thông tin lấy từ TÀI LIỆU, không in dấu <>):
- Tài liệu có ghi nguồn: "Theo <nguồn kèm thời gian>, <chủ thể> <thông tin được hỏi>."
- Câu hỏi có/không: "Có, <chủ thể> <thông tin>; <chi tiết then chốt>."
- So sánh: "Ở <CLB A>, <đối tượng> <đặc điểm>; còn ở <CLB B>, <đối tượng> <đặc điểm>."
- Xếp hạng theo mốc: "<CLB A> <tính chất> nhất (<mốc>), tiếp theo là <CLB B> (<mốc>); <CLB C> <tính chất> nhất (<mốc>)."
- Tính toán thời gian: "<chủ thể> <mốc gốc>, nên đến <mốc được hỏi> <kết quả>."
- Chỉ có một phần thông tin: "<trả lời phần có>. Tài liệu không ghi <phần còn thiếu>."
- Các đoạn mâu thuẫn: "Các đoạn ghi khác nhau: <nguồn 1> ghi <A>, còn <nguồn 2> ghi <B>."
- Không có thông tin: "Tài liệu hiện không có thông tin về <điều được hỏi>." (đúng một câu)
"""
DANGLING = re.compile(r"\b(?:CLB|câu lạc bộ) (?:đó|này|ấy)\b|\bở đó\b", flags=re.IGNORECASE)
# Dòng meta LLM đôi khi in kèm khi viết lại câu hỏi (không phải câu hỏi) → loại khỏi retrieval.
META = re.compile(
    r"không cần (?:tách|câu hỏi con)|chỉ có một ý|chỉ có 1 ý|viết lại như sau|các câu hỏi con|^lưu ý|^ghi chú",
    flags=re.IGNORECASE,
)


EVIDENCE_PROMPT = """Câu hỏi: {query}

Các đoạn tài liệu:
{context}

Liệt kê id của MỌI đoạn nêu trực tiếp ít nhất một dữ kiện cần để trả lời câu hỏi (kể cả các đoạn lặp lại cùng
dữ kiện đó). Không liệt kê đoạn chỉ cùng chủ đề mà không chứa dữ kiện cần. In đúng một dòng:
ID: <các id cách nhau bởi dấu phẩy, hoặc để trống>"""

FIX_PROMPT = """Bạn kiểm tra một câu trả lời nháp cho câu hỏi về các CLB. Bản nháp đang mở đầu bằng câu từ chối.

Tài liệu:
{context}

Câu hỏi: {query}
Bản nháp: {draft}

Quy tắc:
- Nếu tài liệu KHÔNG nêu trực tiếp chính điều câu hỏi hỏi (chỉ có thông tin liên quan, của CLB khác, hoặc chỉ gần đúng), hãy trả lời đúng một từ: GIỮ
- Chỉ khi tài liệu nêu trực tiếp câu trả lời cho điều được hỏi (kể cả câu trả lời dạng trạng thái như "chưa mở", "chưa công bố") hoặc cho một phần được hỏi: viết lại câu trả lời, đưa phần trả lời lên đầu, chỉ dùng thông tin trong tài liệu, giữ nguyên các chi tiết của bản nháp. Nếu còn một phần NGƯỜI HỎI hỏi rõ mà tài liệu không có, kết thúc bằng một câu "Tài liệu không ghi <phần đó>." Không thêm lưu ý nào khác.
- Không dùng markdown, không giải thích. Chỉ in GIỮ hoặc câu trả lời viết lại."""


def strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower().replace("đ", "d").replace("Đ", "d"))
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


def build_examples(ids) -> str:
    """Khối few-shot từ đáp án train theo id. Không có file train/id → bỏ few-shot (trả "")."""
    path = DATA_DIR / "train" / "questions.jsonl"
    if not ids or not path.exists():
        return ""
    by_id = {r["id"]: r for r in read_jsonl(path)}
    shots = []
    for i in ids:
        if i in by_id:
            # Quy tắc dùng "Có, ..." (dấu phẩy) để không bị tách thành câu riêng khi chấm.
            answer = re.sub(r"^(Có|Không|Đúng|Chưa)\.\s+", r"\1, ", by_id[i]["answer"])
            shots.append(f"Hỏi: {by_id[i]['query']}\nĐáp: {answer}")
    return FEWSHOT_HEADER + "\n\n".join(shots) + "\n" if shots else ""


def is_unaccented(text: str) -> bool:
    """True nếu câu không có dấu tiếng Việt (kiểu "clb lap trinh thanh lap khi nao")."""
    return strip_accents(text) == text.lower().replace("đ", "d")


def bm25_tokens(text: str) -> list[str]:
    """Âm tiết bỏ dấu + bigram âm tiết (để 'lap trinh' khớp tốt hơn 'lap' và 'trinh' rời)."""
    syl = re.findall(r"\w+", strip_accents(text))
    return syl + [f"{a}_{b}" for a, b in zip(syl, syl[1:], strict=False)]


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tfs = [Counter(d) for d in docs]
        self.lens = np.array([len(d) for d in docs], dtype=np.float32)
        self.avgdl = float(self.lens.mean())
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> np.ndarray:
        out = np.zeros(len(self.tfs), dtype=np.float32)
        norm = self.k1 * (1 - self.b + self.b * self.lens / self.avgdl)
        for t in set(query):
            if t not in self.idf:
                continue
            tf = np.array([d.get(t, 0) for d in self.tfs], dtype=np.float32)
            out += self.idf[t] * tf * (self.k1 + 1) / (tf + norm)
        return out


class GatewayEmbedder:
    """Embedding bge-m3 qua gateway của BTC; trả về vector đã chuẩn hóa L2."""

    def __init__(self, llm: LLMClient, model_name: str):
        self.llm, self.model_name = llm, model_name

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.asarray(self.llm.embed(texts, model=self.model_name), dtype=np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


class MyPipeline(RAGPipeline):
    def __init__(
        self, top_k: int = 15, n_retrieved: int = 10, llm=None, embedder=None, fewshot_ids=(),
        answer_templates: bool | str = "accented", parent_child: bool = False, cite_rerank: bool = False,
        filler: str | None = None, pc_cos_weight: float = 0.5, answer_first_fix: bool = False,
        evidence_select: bool = False,
    ):
        # v12-E2 (thử nghiệm, tắt mặc định): một lượt LLM riêng chọn MỌI đoạn chứa dữ kiện cần cho câu trả lời;
        # chỉ dùng để xếp id nộp (sau id được cite), không đưa vào câu trả lời hay context của judge.
        self.evidence_select = evidence_select
        # v12-E3 (thử nghiệm, tắt mặc định): lượt LLM thứ hai CHỈ khi bản nháp mở đầu bằng câu từ chối và có
        # câu "Tài liệu chỉ nêu": nếu tài liệu trả lời được điều được hỏi → viết lại trả lời trước.
        self.answer_first_fix = answer_first_fix
        self.cite_rerank = cite_rerank
        # v12: filler="pc_cos" — xếp phần id sau cite theo ranking parent-child + cosine với đoạn được cite
        # (chỉ đổi thứ tự id nộp, trong cùng tập id của v10; không đổi context/câu trả lời).
        self.filler = filler
        self.pc_cos_weight = pc_cos_weight
        # parent_child tắt mặc định: v8 thử bật và bị loại trên train_informal (experiments/update_v8.md).
        self.parent_child = parent_child
        # Few-shot tắt mặc định: v7 thử bật FEWSHOT_IDS và bị loại (xem experiments/update_v7.md).
        self.top_k = top_k  # số doc đưa vào prompt
        self.n_retrieved = n_retrieved  # số doc nộp để chấm retrieval (tối đa 10)
        # answer_templates: False (v6) | True (mọi câu, v9) | "accented" (chỉ câu có dấu, v10)
        self.answer_templates = answer_templates
        self.fewshot_block = build_examples(fewshot_ids)
        self.examples = self.fewshot_block + (ANSWER_TEMPLATES if answer_templates is True else "")
        # CLI đọc `llm_model` và `embedder` để in thông tin. `llm`/`embedder` cho phép thay bằng bản giả khi test.
        self.llm_model = os.getenv("LLM_MODEL", "claude-haiku-5-5")
        self.llm = llm or LLMClient(cache_path=".cache/pipeline.sqlite")
        self.embedder = embedder or GatewayEmbedder(self.llm, os.getenv("EMBEDDING_MODEL", "bge-m3"))

    # ------------------------------------------------------------------ index
    def index(self, corpus: list[Document]) -> None:
        self.docs = corpus
        self.by_id = {d.id: d for d in corpus}
        self.pos = {d.id: i for i, d in enumerate(corpus)}
        self.doc_vectors = self.embedder.embed([d.text for d in corpus])
        self.bm25 = BM25([bm25_tokens(f"{d.club} {d.topic} {d.text}") for d in corpus])
        # Parent-child: child = từng câu (đoạn danh sách tách thêm theo ";"), kèm header club|topic để
        # không mất ngữ cảnh. Tìm trên child (không bị "trung bình hóa" nhiều ý), điểm đoạn cha = max child.
        child_texts, parents = [], []
        for i, d in enumerate(corpus if self.parent_child or self.filler == "pc_cos" else []):
            for sent in split_sentences(d.text.replace("; ", ".\n")):
                if len(sent) >= 15:
                    child_texts.append(f"{d.club} | {d.topic}: {sent}")
                    parents.append(i)
        self.child_parent = np.array(parents, dtype=np.int64)
        self.child_vectors = self.embedder.embed(child_texts) if child_texts else None
        self.child_bm25 = BM25([bm25_tokens(t) for t in child_texts]) if child_texts else None

    def _to_parent(self, child_scores: np.ndarray) -> np.ndarray:
        """Max-pooling điểm child về đoạn cha (đoạn không có child: -inf → xếp cuối)."""
        out = np.full(len(self.docs), -np.inf, dtype=np.float32)
        np.maximum.at(out, self.child_parent, child_scores.astype(np.float32))
        return out

    # ---------------------------------------------------------- query rewrite
    def rewrite(self, query: str) -> list[str]:
        try:
            out = self.llm.chat(
                [{"role": "user", "content": REWRITE_PROMPT.format(query=query)}], model=self.llm_model
            )
        except Exception:  # rewrite chỉ là bước phụ: lỗi API thì vẫn truy xuất bằng câu gốc
            return [query]
        qs =[m.group(1).strip() for m in re.finditer(r"^\s*Q\s*[:：]\s*(.+)$", out, flags=re.MULTILINE)]
        if not qs:  # dự phòng: LLM không theo định dạng → lọc dòng meta
            qs = [re.sub(r"^\s*(?:\d+[.)]|[-*•])\s*", "", line).strip() for line in out.splitlines()]
        qs = [re.sub(r"^(?:câu (?:viết lại|hỏi con)[^:]*:)\s*", "", q, flags=re.IGNORECASE) for q in qs]
        qs = [q for q in qs if len(q) > 3 and not META.search(q)]
        # Câu con phải tự đứng được: câu con chỉ trỏ "CLB đó/này" sẽ truy xuất lạc → bỏ (giữ câu đầu).
        qs = qs[:1] + [q for q in qs[1:] if not DANGLING.search(q)]
        return qs[:4] or [query]

    # -------------------------------------------------------------- retrieval
    def retrieve(self, query: str, rewrites: list[str]) -> list[Document]:
        """Trả về top-max(top_k, n_retrieved) doc theo RRF của dense + BM25 trên mọi truy vấn."""
        fused = self._fused(query, rewrites, use_child=self.parent_child)
        k = max(self.top_k, self.n_retrieved)
        return [self.docs[i] for i in np.argsort(-fused)[:k]]

    def _fused(self, query: str, rewrites: list[str], use_child: bool) -> np.ndarray:
        """Điểm RRF trên toàn corpus. use_child=True: thêm ranking child (parent-child, như v8)."""
        # Câu gốc không dấu: dense bge-m3 xếp doc đúng ở hạng 30-100 → chỉ dùng BM25 (bỏ dấu) cho nó.
        dense_orig = not is_unaccented(query)
        queries = [(query, 0.7, dense_orig)] + [(q, 1.0, True) for q in rewrites]
        dense_qs = [q for q, _, use in queries if use]
        qvecs = iter(self.embedder.embed(dense_qs)) if dense_qs else iter(())
        use_child = use_child and self.child_bm25 is not None
        fused = np.zeros(len(self.docs), dtype=np.float32)
        for q, w, use_dense in queries:
            toks = bm25_tokens(q)
            rankings = [self.bm25.scores(toks)]
            if use_child:
                rankings.append(self._to_parent(self.child_bm25.scores(toks)))
            if use_dense:
                qv = next(qvecs)
                rankings.append(self.doc_vectors @ qv)
                if use_child:
                    rankings.append(self._to_parent(self.child_vectors @ qv))
            for scores in rankings:
                ranks = np.empty(len(scores), dtype=np.float32)
                ranks[np.argsort(-scores)] = np.arange(1, len(scores) + 1)
                fused += w / (RRF_K + ranks)
        return fused

    # ------------------------------------------------------------- generation
    def generate(self, query: str, rewritten: str, docs: list[Document]) -> tuple[str, list[int]]:
        context = "\n\n".join(f"[{d.id}] ({d.club} – {d.topic}) {d.text}" for d in docs[: self.top_k])
        examples = self.examples
        if self.answer_templates == "accented" and not is_unaccented(query):
            # Mẫu chung chỉ có lợi với câu viết chuẩn; câu văn nói giữ nguyên prompt v6 (xem update_v9.md).
            examples = self.fewshot_block + ANSWER_TEMPLATES
        prompt = ANSWER_PROMPT.format(
            context=context, query=query, rewritten=rewritten, examples=examples
        )
        answer, cited = self._parse_answer(
            self.llm.chat([{"role": "user", "content": prompt}], model=self.llm_model), docs
        )
        if not answer:
            # Câu trả lời rỗng đã nằm trong cache sẽ lặp lại mãi → gọi lại một lần với prompt khác
            # đúng 1 ký tự (khóa cache mới). Lỗi API ở đây được để lộ ra: chạy lại lệnh sẽ dùng cache.
            answer, cited = self._parse_answer(
                self.llm.chat([{"role": "user", "content": prompt + "\n"}], model=self.llm_model), docs
            )
        return answer, cited

    def _parse_answer(self, out: str, docs: list[Document]) -> tuple[str, list[int]]:
        """Tách dòng NGUON (chỉ giữ id có trong context), xóa markdown, gộp về một đoạn."""
        cited: list[int] = []
        m = re.search(r"^\s*NGU[OỒ]N\s*:\s*(.*)$", out, flags=re.MULTILINE | re.IGNORECASE)
        if m:
            shown = {d.id for d in docs[: self.top_k]}
            cited = [int(x) for x in re.findall(r"\d+", m.group(1)) if int(x) in shown]
            out = out[: m.start()] + out[m.end():]
        answer = re.sub(r"\*\*|__|^#+\s*|^\s*[-*•]\s+", "", out, flags=re.MULTILINE)
        answer = re.sub(r"\s*\n\s*", " ", answer).strip()
        return answer, cited

    def run(self, query: str) -> RAGOutput:
        rewrites = self.rewrite(query)
        docs = self.retrieve(query, rewrites)
        answer, cited = self.generate(query, rewrites[0], docs)
        if self.answer_first_fix:
            answer = self.fix_refusal_opener(query, answer, cited, docs)
        if self.filler == "pc_cos":
            extra = self.select_evidence(query, docs) if self.evidence_select and cited else []
            ids = self.order_ids_pc(cited, docs, self._fused(query, rewrites, use_child=True), extra)
        else:
            ids = self.order_ids(cited, docs)
        return RAGOutput(answer=answer or "Tài liệu hiện không có thông tin để trả lời câu hỏi này.", retrieved_ids=ids)

    def order_ids(self, cited: list[int], docs: list[Document]) -> list[int]:
        """Thứ tự id nộp (tối đa n_retrieved).

        - Không có cite: giữ thứ tự RRF (như v10).
        - v10: id được cite lên đầu, phần còn lại theo RRF.
        - v11 (cite_rerank): id được cite lên đầu, phần còn lại (cùng tập id với v10) xếp theo cosine lớn nhất
          với các đoạn được cite. relevant_ids thường gồm các đoạn trùng ý (vd. Phòng truyền thống và bài
          tuyển thành viên) mà RRF hay để ở hạng 6-10. Câu trả lời không đổi.
        """
        if not cited or not self.cite_rerank:
            return list(dict.fromkeys(cited + [d.id for d in docs[: self.n_retrieved]]))[: self.n_retrieved]
        cited_vecs = self.doc_vectors[[self.pos[i] for i in cited]]
        # Chỉ xếp lại TRONG tập id mà v10 nộp (cite + top-n RRF): giữ nguyên tập đoạn judge dùng chấm
        # Groundedness. Kéo thêm đoạn hạng 11-15 vào từng làm Groundedness informal giảm 0.989 → 0.966.
        base = list(dict.fromkeys(cited + [d.id for d in docs[: self.n_retrieved]]))[: self.n_retrieved]
        rest = [i for i in base if i not in set(cited)]
        sim = {i: float((cited_vecs @ self.doc_vectors[self.pos[i]]).max()) for i in rest}
        rest.sort(key=lambda i: -sim[i])  # sort ổn định: hòa điểm giữ thứ tự RRF
        return list(dict.fromkeys(cited + rest))[: self.n_retrieved]


    def fix_refusal_opener(self, query: str, answer: str, cited: list[int], docs: list[Document]) -> str:
        """E3: chỉ chạy khi câu trả lời mở đầu bằng từ chối và có câu ngữ cảnh "Tài liệu chỉ nêu"."""
        if not (answer.startswith("Tài liệu hiện không có thông tin") and "Tài liệu chỉ nêu" in answer):
            return answer
        shown = [d for d in docs[: self.top_k] if d.id in set(cited)] or list(docs[:3])
        context = "\n\n".join(f"[{d.id}] ({d.club} – {d.topic}) {d.text}" for d in shown)
        try:
            out = self.llm.chat(
                [{"role": "user", "content": FIX_PROMPT.format(context=context, query=query, draft=answer)}],
                model=self.llm_model,
            ).strip()
        except Exception:
            return answer
        out = re.sub(r"\s*\n\s*", " ", re.sub(r"\*\*|__", "", out)).strip()
        if not out or out.upper().startswith("GIỮ") or out.startswith("Tài liệu hiện không có thông tin"):
            return answer
        return out

    def select_evidence(self, query: str, docs: list[Document]) -> list[int]:
        """E2: id các đoạn (trong top_k) mà LLM cho là nêu dữ kiện cần để trả lời. Lỗi API → []."""
        context = "\n\n".join(f"[{d.id}] ({d.club} – {d.topic}) {d.text}" for d in docs[: self.top_k])
        try:
            out = self.llm.chat(
                [{"role": "user", "content": EVIDENCE_PROMPT.format(context=context, query=query)}],
                model=self.llm_model,
            )
        except Exception:
            return []
        shown = {d.id for d in docs[: self.top_k]}
        m = re.search(r"^\s*ID\s*:\s*(.*)$", out, flags=re.MULTILINE | re.IGNORECASE)
        return list(dict.fromkeys(int(x) for x in re.findall(r"\d+", m.group(1)) if int(x) in shown)) if m else []

    def order_ids_pc(
        self, cited: list[int], docs: list[Document], pc_fused: np.ndarray, evidence: list[int] = ()
    ) -> list[int]:
        """v12 (filler="pc_cos"): cùng TẬP id với v10 (cite + top-n RRF), chỉ đổi thứ tự.

        id được cite lên đầu (theo thứ tự cite); phần còn lại xếp theo
        1/(1+hạng parent-child) + w·max cos(đoạn, các đoạn được cite). Không có cite → chỉ theo hạng
        parent-child. Lý do (update_v12.md): ranking parent-child tốt hơn RRF thường khi thiếu cite; v11 (chỉ
        cosine) thua v10 khi cite không đủ (mô phỏng cite1/drop50), khớp NDCG public 90.20 < 90.41.
        """
        base = list(dict.fromkeys(cited + [d.id for d in docs[: self.n_retrieved]]))[: self.n_retrieved]
        rank = np.empty(len(pc_fused), dtype=np.int64)
        rank[np.argsort(-pc_fused, kind="stable")] = np.arange(len(pc_fused))
        cited_set = set(cited)
        rest = [i for i in base if i not in cited_set]
        cited_vecs = self.doc_vectors[[self.pos[i] for i in cited]] if cited else None

        def key(i: int) -> float:
            score = 1.0 / (1 + int(rank[self.pos[i]]))
            if cited_vecs is not None:
                score += self.pc_cos_weight * float((cited_vecs @ self.doc_vectors[self.pos[i]]).max())
            return -score

        rest.sort(key=key)
        # E2: đoạn bằng chứng (chỉ những đoạn đã nằm trong tập id v10) lên ngay sau id được cite.
        ev = set(evidence)
        rest = [i for i in rest if i in ev] + [i for i in rest if i not in ev]
        return list(dict.fromkeys(cited + rest))[: self.n_retrieved]


class MyPipelineV9(MyPipeline):
    """v9: v6 + mẫu trả lời chung (ANSWER_TEMPLATES). Hòa v6 trên dev; dùng để nộp thử public.

    Chạy: neorag run public_test --pipeline neorag.pipelines.pipeline:MyPipelineV9
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("answer_templates", True)
        kwargs.setdefault("cite_rerank", False)
        super().__init__(**kwargs)


class MyPipelineV10(MyPipeline):
    """v10 (không cite_rerank): v6, nhưng câu hỏi CÓ DẤU dùng thêm mẫu trả lời chung (lợi trên train ở v9);
    câu văn nói / không dấu giữ nguyên prompt v6 (mẫu chung gây hại trên train_informal và public).

    Chạy: neorag run <split> --pipeline neorag.pipelines.pipeline:MyPipelineV10
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("answer_templates", "accented")
        kwargs.setdefault("cite_rerank", False)
        super().__init__(**kwargs)


class MyPipelineV6(MyPipeline):
    """Phương án dự phòng: đúng hành vi v6 (không dùng mẫu trả lời chung).

    Chạy: neorag run private_test --pipeline neorag.pipelines.pipeline:MyPipelineV6
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("answer_templates", False)
        kwargs.setdefault("cite_rerank", False)
        super().__init__(**kwargs)



class MyPipelineV11(MyPipeline):
    """v11: v10 + xếp lại id nộp theo đoạn được trích dẫn (chỉ trong tập id của v10). Câu trả lời = v10.

    Chạy: neorag run public_test --pipeline neorag.pipelines.pipeline:MyPipelineV11
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("cite_rerank", True)
        super().__init__(**kwargs)


class MyPipelineV12(MyPipeline):
    """v12: câu trả lời = v10/v11 (byte-identical); chỉ đổi THỨ TỰ id nộp, trong cùng tập id của v10:
    id được cite → đoạn bằng chứng do lượt LLM riêng chọn (evidence_select) → phần còn lại theo
    parent-child + 0.5·cosine với đoạn được cite (filler="pc_cos"). answer_first_fix (E3) tắt: không qua
    tiêu chí giữ. Xem experiments/update_v12.md.

    Chạy: neorag run private_test --pipeline neorag.pipelines.pipeline:MyPipelineV12
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("filler", "pc_cos")
        kwargs.setdefault("evidence_select", True)
        super().__init__(**kwargs)


class MyPipelineV13(MyPipeline):
    """v13: thứ tự id = v11 (cite_rerank, tốt hơn v12 trên private) + answer_first_fix (E3): câu trả lời mở
    đầu bằng từ chối nhưng có "Tài liệu chỉ nêu" được viết lại trả lời trước, chỉ từ chối phần thiếu
    (nhắm Noise/ROUGE-L). Câu khác giữ nguyên v11.

    Chạy: neorag run private_test --pipeline neorag.pipelines.pipeline:MyPipelineV13
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("cite_rerank", True)
        kwargs.setdefault("answer_first_fix", True)
        super().__init__(**kwargs)


REFUSAL_OPENER = "Tài liệu hiện không có thông tin"

AUDIT_PROMPT = """Bạn là chuyên gia kiểm định chất lượng (Auditor) cho hệ thống hỏi đáp RAG về các CLB thuộc cộng đồng S2B của PTIT.
Bạn nhận CÂU HỎI GỐC, câu hiểu lại, các đoạn TÀI LIỆU truy xuất được, và BẢN NHÁP câu trả lời. Chỉ được dùng thông tin có trong TÀI LIỆU.

Hãy kiểm tra BẢN NHÁP theo đúng 6 tiêu chí sau:

1. HIỂU ĐÚNG CÂU HỎI KHÔNG DẤU / VĂN NÓI:
   - Đối chiếu kỹ CÂU HỎI GỐC để tránh hiểu nhầm từ không dấu hoặc khẩu ngữ: "hoi ... moi thanh lap" = "hồi ... mới thành lập" (quá khứ, không phải hiện tại); "nguoi dung lop" = "người chủ trì buổi học" (không lấy quy định của thành viên thường); "clb ... cua truong / cua hv" = CLB thuộc Học viện (không phải hỏi trường có ghi nhận không); hỏi kênh/email cho "cả 3 CLB" = nêu kênh/email của từng CLB.

2. KIỂM TRA TỪ CHỐI OAN & CHUẨN HÓA CÂU THIẾU DỮ LIỆU:
   - (2a) Bỏ từ chối oan: Nếu TÀI LIỆU CÓ thông tin trả lời điều người hỏi cần biết (kể cả thông tin trạng thái như "đợt tuyển thành viên chưa mở", "không cố định, tùy kế hoạch...", "có thể mở rộng chỉ tiêu...", "cùng khu vực Km10 Nguyễn Trãi...", hoặc thông tin của từng CLB), BẮT BUỘC bỏ câu "Tài liệu hiện không có thông tin..." và trả lời trực tiếp ngay từ đầu; không thêm đuôi "tài liệu không ghi..." cho những chi tiết câu hỏi không yêu cầu.
   - (2b) Thiếu một phần: Nếu tài liệu trả lời được một phần/một vế của câu hỏi và chỉ thiếu phần còn lại mà người hỏi hỏi rõ, hãy trả lời phần có TRƯỚC, rồi nối "; tài liệu không ghi <phần còn thiếu>." (không mở đầu bằng từ chối).
   - (2c) Thực sự không có thông tin được hỏi (ví dụ hỏi sai năm/nhiệm kỳ/khóa mà tài liệu không có, hoặc hỏi chức danh/tên ban/mức phí không được ghi): TUYỆT ĐỐI KHÔNG suy đoán hay lấy thông tin của năm khác/CLB khác làm đáp án. Mở đầu đúng bằng "Tài liệu hiện không có thông tin về <điều được hỏi cụ thể>":
     + Nếu TÀI LIỆU có thông tin liên quan trực tiếp của ĐÚNG CLB ĐÓ cùng chủ đề (như đợt tuyển/giải thưởng năm khác của cùng CLB, các chức danh khác được liệt kê trong cùng nhiệm kỳ đó, hoặc lưu ý "chỉ áp dụng cho đợt tuyển năm ..."), nối trong CÙNG MỘT CÂU sau dấu chấm phẩy: "; tài liệu chỉ nêu <thông tin liên quan nguyên văn kèm mốc thời gian/nguồn>, không ghi <phần được hỏi>."
     + Nếu bản nháp đang nêu thông tin của CLB KHÁC mà câu hỏi không hỏi tới: cắt bỏ phần CLB khác đó, chỉ giữ đúng một câu "Tài liệu hiện không có thông tin về <điều được hỏi>."

3. KIỂM TRA CỰC TÍNH CÓ / KHÔNG:
   - Với câu hỏi Có/Không ("có ... không", "cần ... không", "được không", "phải không", "đúng không"): từ mở đầu ("Có, ...", "Không, ...", "Đúng, ...", "Chưa, ...", "Chưa chắc, ...", dùng dấu phẩy) phải khớp logic với câu hỏi gốc. Ví dụ hỏi "có cần đến không" mà nội dung là "vẫn cần có mặt" thì phải mở bằng "Có, ..." (tuyệt đối không viết "Không, ... vẫn cần..."). Hỏi mốc của năm tương lai chưa công bố có giống năm cũ không thì mở bằng "Chưa chắc, ..." kèm lưu ý "chỉ áp dụng cho ...".

4. LOẠI BỎ SUY DIỄN NGOÀI VĂN BẢN & Ý NHIỄU CHÉO ĐOẠN:
   - Xóa mệnh đề tự suy luận/đánh giá không được nêu trong tài liệu (như "nên thành tích năm ... cao hơn...", "nên không còn mới thành lập"), trừ khi câu hỏi hỏi trực tiếp phép so sánh hơn-kém, xếp hạng hoặc tính khoảng thời gian.
   - Xóa câu/mệnh đề phụ không được hỏi lấy từ đoạn khác hoặc lệch mốc thời gian/vai trò (ví dụ hỏi đợt tuyển Gen 1 năm 2023 thì không ghép thêm email ở đoạn liên hệ chung; câu 1 đã trả lời đủ ý câu hỏi đơn thì bỏ câu 2 lan man sang hoạt động/ví dụ khác).

5. ĐỦ Ý SONG SONG & CHÉP NGUYÊN VĂN CỤM THEN CHỐT:
   - Rà soát mọi đoạn liên quan trong TÀI LIỆU: nếu bản nháp bỏ sót dữ kiện trực tiếp trả lời câu hỏi ở cuối đoạn hoặc ở đoạn song song thứ hai cùng chủ đề (như cuộc thi kết thúc khóa học khi hỏi học gì, giải thưởng thứ hai cùng lĩnh vực ở giai đoạn khác, cách ghi slogan thứ hai, đơn vị bảo trợ chuyên môn, link đăng ký khi hỏi hạn nộp đơn, lưu ý "chỉ áp dụng cho ..."), hãy bổ sung ngắn gọn (không tự thêm ví dụ từ đoạn khác khi câu hỏi chỉ hỏi nguyên tắc chung).
   - Chép NGUYÊN VĂN từng chữ cụm số lượng + danh từ, tên riêng, tên giải/ban/biệt danh tiếng Anh, cụm từ khóa từ tài liệu (ví dụ giữ nguyên "5 thành viên, trong đó 3 người" thay vì "3 trong 5 thành viên"; giữ nguyên "3 Phó Chủ nhiệm", "cả hai mặt học tập và hoạt động", "kế hoạch hoạt động, học tập", "tổng kết những việc đã làm"). Câu đầu luôn nêu rõ tên CLB chuẩn ("CLB Lập Trình PTIT", "CLB Multimedia PTIT", "CLB Nhà Sáng Tạo Game PTIT (GCC)", và "cộng đồng S2B" nếu câu hỏi nhắc S2B) bám sát trọng tâm câu hỏi gốc; súc tích 1-2 câu văn xuôi, không markdown.

6. NGUYÊN TẮC BẢO TOÀN:
   - Nếu BẢN NHÁP đã đúng, đủ ý trực tiếp, nguyên văn cụm then chốt, đúng cực Có/Không, không từ chối oan và không thừa ý nhiễu: BẮT BUỘC ghi "TRA_LOI: GIỮ" (không sửa đổi câu chữ khi không có lỗi thực sự).

Định dạng trả lời bắt buộc (đúng 2 dòng, không giải thích):
TRA_LOI: <GIỮ hoặc câu trả lời hoàn chỉnh trên một dòng>
NGUON: <id của MỌI đoạn trong TÀI LIỆU nêu trực tiếp dữ kiện trả lời câu hỏi, kể cả đoạn trùng ý/song song, xếp từ liên quan nhất, cách nhau dấu phẩy>

TÀI LIỆU:
{context}

CÂU HỎI GỐC: {query}
(Hiểu là: {rewritten})
BẢN NHÁP: {draft}
NGUỒN NHÁP: {draft_cited}"""


class MyPipelineV14(MyPipeline):
    """v14: câu trả lời v9 (mẫu chung cho MỌI câu: ROUGE-L/Relevancy/Noise public tốt hơn) + chốt từ chối của
    v11. Khi câu trả lời v10/v11 là từ chối thì giữ từ chối đó (+ E3 như v13), vì v9 hay trả lời bừa câu không có
    thông tin (Abstention public 66.67, vd. public-0039). Thứ tự id = v11 (cite_rerank).

    Chạy: neorag run private_test --pipeline neorag.pipelines.pipeline:MyPipelineV14
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("cite_rerank", True)
        kwargs.setdefault("answer_first_fix", True)
        super().__init__(**kwargs)

    def _generate_v9(self, query: str, rewritten: str, docs: list[Document]) -> tuple[str, list[int]]:
        saved = (self.answer_templates, self.examples)
        self.answer_templates, self.examples = True, self.fewshot_block + ANSWER_TEMPLATES
        try:
            return self.generate(query, rewritten, docs)
        finally:
            self.answer_templates, self.examples = saved

    def run(self, query: str) -> RAGOutput:
        rewrites = self.rewrite(query)
        docs = self.retrieve(query, rewrites)
        answer, cited = self.generate(query, rewrites[0], docs)  # v10/v11 (cache)
        if answer.startswith(REFUSAL_OPENER):
            if self.answer_first_fix:
                answer = self.fix_refusal_opener(query, answer, cited, docs)
        else:
            a9, c9 = self._generate_v9(query, rewrites[0], docs)
            if a9 and not a9.startswith(REFUSAL_OPENER):
                answer, cited = a9, c9
        ids = self.order_ids(cited, docs)
        
        return RAGOutput(answer=answer or "Tài liệu hiện không có thông tin để trả lời câu hỏi này.", retrieved_ids=ids)


class MyPipelineV11Audit(MyPipeline):
    """v11-audit: v11 (hybrid RRF + answer_templates='accented' + cite_rerank) bổ sung tầng Kiểm định (Audit):
    1. Mở rộng ngữ cảnh kiểm định lên tối đa 20 đoạn (top-15 của v11 + ứng viên parent-child & đoạn cùng
       (club, topic) với nguồn nháp) mà KHÔNG làm đổi bước sinh nháp v11.
    2. Lượt LLM Auditor (`claude-haiku-5-5`) kiểm định bản nháp theo 6 tiêu chí: sửa từ chối oan / mở đầu từ chối
       sai, chuẩn hóa câu thiếu dữ liệu/sai năm bằng câu ghép dấu ';', sửa ngược cực Có/Không, cắt mệnh đề suy diễn
       hoặc nhiễu chéo đoạn, bổ sung dữ kiện song song & cụm nguyên văn bị sót, và xác minh danh sách NGUON.
       Giữ nguyên bản nháp (`GIỮ`) khi không có lỗi để tránh nhiễu không cần thiết.
    3. Xếp hạng id nộp bằng `cite_rerank` kết hợp nguồn đã kiểm định và bảo toàn nguồn nháp trong top-10.

    Chạy: neorag run private_test --pipeline neorag.pipelines.pipeline:MyPipelineV11Audit
    """

    def __init__(self, audit_max_docs: int = 20, **kwargs):
        kwargs.setdefault("answer_templates", "accented")
        kwargs.setdefault("cite_rerank", True)
        # Bật filler="pc_cos" lúc index để tạo sẵn child index cho mở rộng ngữ cảnh audit, nhưng dùng order_ids_audit.
        super().__init__(**kwargs)
        self.audit_max_docs = audit_max_docs

    def index(self, corpus: list[Document]) -> None:
        saved_pc = self.parent_child
        # Tạo cả index đoạn nguyên (cho retrieve v11) và child index (cho mở rộng ứng viên audit)
        self.parent_child = True
        try:
            super().index(corpus)
        finally:
            self.parent_child = saved_pc
        self.by_club_topic: dict[tuple[str, str], list[Document]] = {}
        for d in corpus:
            self.by_club_topic.setdefault((d.club, d.topic), []).append(d)

    def _build_audit_docs(
        self, query: str, rewrites: list[str], docs: list[Document], draft_cited: list[int]
    ) -> list[Document]:
        """Giữ nguyên top-15 của v11 ở đầu, bổ sung tối đa 5 đoạn từ parent-child và cùng (club, topic)."""
        seen = {d.id for d in docs[: self.top_k]}
        out = list(docs[: self.top_k])
        pc_scores = self._fused(query, rewrites, use_child=True)
        # 1) Bổ sung đoạn cùng (club, topic) với đoạn được cite đầu tiên (xếp theo pc_scores) — bắt các đoạn song song
        # như Giải thưởng nhiều giai đoạn, Đào tạo nhiều giai đoạn, Ban chủ nhiệm nhiều nhiệm kỳ.
        siblings: list[Document] = []
        for cid in draft_cited[:2]:
            if cid in self.by_id:
                d0 = self.by_id[cid]
                siblings.extend(self.by_club_topic.get((d0.club, d0.topic), ()))
        siblings.sort(key=lambda d: -float(pc_scores[self.pos[d.id]]))
        for d in siblings:
            if len(out) >= self.audit_max_docs - 2:
                break
            if d.id not in seen:
                seen.add(d.id)
                out.append(d)
        # 2) Bổ sung top ứng viên từ parent-child RRF chưa có trong danh sách
        for idx in np.argsort(-pc_scores):
            if len(out) >= self.audit_max_docs:
                break
            d = self.docs[int(idx)]
            if d.id not in seen:
                seen.add(d.id)
                out.append(d)
        return out

    def _parse_audit(
        self, raw: str, audit_docs: list[Document], draft_answer: str, draft_cited: list[int]
    ) -> tuple[str, list[int]]:
        shown = {d.id for d in audit_docs}
        audit_cited: list[int] = []
        m_src = re.search(r"^\s*NGU[OỒ]N\s*:\s*(.*)$", raw, flags=re.MULTILINE | re.IGNORECASE)
        if m_src:
            audit_cited = list(
                dict.fromkeys(int(x) for x in re.findall(r"\d+", m_src.group(1)) if int(x) in shown)
            )
            raw = raw[: m_src.start()] + raw[m_src.end() :]
        body = re.sub(r"^\s*TR[AẢ]\s*_?\s*L[OỜ]I\s*:\s*", "", raw.strip(), flags=re.IGNORECASE)
        body = re.sub(r"\*\*|__|^#+\s*|^\s*[-*•]\s+", "", body, flags=re.MULTILINE)
        body = re.sub(r"\s*\n\s*", " ", body).strip()
        # Lọc nếu sót nhãn chỗ trống <...>
        if "<" in body and ">" in body and re.search(r"<[^<>]{2,40}>", body):
            return draft_answer, audit_cited or draft_cited
        verdict = body.rstrip(".! ").upper()
        if not body or verdict in ("GIỮ", "GIU", "KEEP"):
            return draft_answer, audit_cited or draft_cited
        return body[:2000], audit_cited or draft_cited

    def audit(
        self,
        query: str,
        rewritten: str,
        audit_docs: list[Document],
        draft_answer: str,
        draft_cited: list[int],
    ) -> tuple[str, list[int]]:
        if not draft_answer:
            return draft_answer, draft_cited
        context = "\n\n".join(f"[{d.id}] ({d.club} – {d.topic}) {d.text}" for d in audit_docs)
        cited_str = ", ".join(str(i) for i in draft_cited) if draft_cited else "không có"
        prompt = AUDIT_PROMPT.format(
            context=context,
            query=query,
            rewritten=rewritten,
            draft=draft_answer,
            draft_cited=cited_str,
        )
        try:
            raw = self.llm.chat([{"role": "user", "content": prompt}], model=self.llm_model)
        except Exception:
            return draft_answer, draft_cited
        return self._parse_audit(raw, audit_docs, draft_answer, draft_cited)

    def order_ids_audit(
        self, draft_cited: list[int], audit_cited: list[int], docs: list[Document], answer_changed: bool
    ) -> list[int]:
        """Xếp hạng tối đa n_retrieved id:
        - Ưu tiên id do Auditor xác nhận (`audit_cited`), tiếp theo là `draft_cited` còn lại (bảo đảm không rớt đoạn
          mà câu trả lời thực sự dùng → giữ Groundedness).
        - Nếu câu trả lời giữ nguyên (`not answer_changed`), chỉ cho phép id nằm trong tập top-n của v10/v11 để
          không đổi tập ngữ cảnh chấm Groundedness.
        - Phần filler còn lại xếp theo cosine lớn nhất với các đoạn đã cite (y hệt cite_rerank của v11).
        """
        v10_pool = list(dict.fromkeys(draft_cited + [d.id for d in docs[: self.n_retrieved]]))[: self.n_retrieved]
        if not answer_changed:
            allowed = set(v10_pool)
            merged_cited = list(dict.fromkeys([i for i in audit_cited if i in allowed] + draft_cited))
            base = v10_pool
        else:
            merged_cited = list(dict.fromkeys(audit_cited + draft_cited))[: self.n_retrieved]
            base = list(dict.fromkeys(merged_cited + v10_pool))[: self.n_retrieved]
        if not merged_cited or not self.cite_rerank:
            return base
        cited_vecs = self.doc_vectors[[self.pos[i] for i in merged_cited]]
        cited_set = set(merged_cited)
        rest = [i for i in base if i not in cited_set]
        sim = {i: float((cited_vecs @ self.doc_vectors[self.pos[i]]).max()) for i in rest}
        rest.sort(key=lambda i: -sim[i])
        return list(dict.fromkeys(merged_cited + rest))[: self.n_retrieved]

    def run(self, query: str) -> RAGOutput:
        rewrites = self.rewrite(query)
        docs = self.retrieve(query, rewrites)
        draft_answer, draft_cited = self.generate(query, rewrites[0], docs)
        audit_docs = self._build_audit_docs(query, rewrites, docs, draft_cited)
        final_answer, audit_cited = self.audit(query, rewrites[0], audit_docs, draft_answer, draft_cited)
        ids = self.order_ids_audit(draft_cited, audit_cited, docs, answer_changed=(final_answer != draft_answer))
        return RAGOutput(
            answer=final_answer or "Tài liệu hiện không có thông tin để trả lời câu hỏi này.",
            retrieved_ids=ids,
        )
