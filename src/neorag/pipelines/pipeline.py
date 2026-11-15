"""Pipeline phục vụ phiên bản v11-audit (Bản tối ưu cao nhất - 91.35 điểm Private Test).

Kiến trúc:
1. Query Rewriting: Chuẩn hóa câu hỏi tiếng Việt có dấu, mở từ viết tắt, phân tách so sánh.
2. Hybrid Retrieval: BM25 trên âm tiết bỏ dấu + Dense bge-m3, hợp nhất RRF (k=60).
   (Tự động tắt Dense trên câu gốc nếu câu không dấu để tránh nhiễu vector).
3. Draft Generation: Sinh câu trả lời nháp từ top-15 tài liệu bằng Claude-3.5-Haiku.
   Áp dụng Generic Answer Templates (<...>) phân tầng cho câu có dấu.
4. Context Expansion: Mở rộng ngữ cảnh kiểm định lên 20 đoạn bổ sung các đoạn song song
   cùng (club, topic) với nguồn nháp và ứng viên Parent-Child RRF.
5. LLM Quality Auditor: Kiểm định bản nháp theo 6 tiêu chí (chống từ chối oan, chuẩn hóa
   câu thiếu dữ liệu bằng liên kết dấu ';', sửa cực tính Có/Không, loại bỏ suy diễn/nhiễu
   chéo đoạn, bổ sung số liệu song song, bảo toàn bản nháp sạch bằng 'GIỮ').
6. Audit ID Reranking: Xếp hạng ID nộp bài kết hợp bằng chứng kiểm định và độ tương đồng Cosine.

Chạy:
    neorag run private_test --pipeline neorag.pipelines.pipeline:MyPipeline
    (hoặc neorag run private_test --pipeline neorag.pipelines.pipeline:MyPipelineV11Audit)
"""

from __future__ import annotations

import math
import os
import re
import unicodedata
from collections import Counter

import numpy as np

from neorag.data import Document
from neorag.llm import LLMClient
from neorag.pipelines.base import RAGOutput, RAGPipeline
from neorag.text import split_sentences

RRF_K = 60

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

DANGLING = re.compile(r"\b(?:CLB|câu lạc bộ) (?:đó|này|ấy)\b|\bở đó\b", flags=re.IGNORECASE)
META = re.compile(
    r"không cần (?:tách|câu hỏi con)|chỉ có một ý|chỉ có 1 ý|viết lại như sau|các câu hỏi con|^lưu ý|^ghi chú",
    flags=re.IGNORECASE,
)


def strip_accents(text: str) -> str:
    """Loại bỏ dấu tiếng Việt, chuyển về dạng chuẩn không dấu."""
    text = unicodedata.normalize("NFD", text.lower().replace("đ", "d").replace("Đ", "d"))
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


def is_unaccented(text: str) -> bool:
    """True nếu câu hoàn toàn không có dấu tiếng Việt."""
    return strip_accents(text) == text.lower().replace("đ", "d")


def bm25_tokens(text: str) -> list[str]:
    """Tách âm tiết bỏ dấu kèm bigram âm tiết kế tiếp."""
    syl = re.findall(r"\w+", strip_accents(text))
    return syl + [f"{a}_{b}" for a, b in zip(syl, syl[1:], strict=False)]


class BM25:
    """Cài đặt chuẩn Okapi BM25 với k1=1.2, b=0.75."""

    def __init__(self, docs: list[list[str]], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tfs = [Counter(d) for d in docs]
        self.lens = np.array([len(d) for d in docs], dtype=np.float32)
        self.avgdl = float(self.lens.mean()) if len(docs) > 0 else 1.0
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
    """Embedding bge-m3 qua gateway của BTC với chuẩn hóa L2."""

    def __init__(self, llm: LLMClient, model_name: str):
        self.llm, self.model_name = llm, model_name

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.asarray(self.llm.embed(texts, model=self.model_name), dtype=np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


class MyPipeline(RAGPipeline):
    """Pipeline hoàn chỉnh cho phiên bản v11-audit đạt 91.35 điểm Private Test."""

    def __init__(
        self,
        top_k: int = 15,
        n_retrieved: int = 10,
        audit_max_docs: int = 20,
        llm=None,
        embedder=None,
        answer_templates: bool | str = "accented",
        cite_rerank: bool = True,
    ):
        self.top_k = top_k
        self.n_retrieved = n_retrieved
        self.audit_max_docs = audit_max_docs
        self.answer_templates = answer_templates
        self.cite_rerank = cite_rerank
        self.llm_model = os.getenv("LLM_MODEL", "claude-haiku-5-5")
        self.llm = llm or LLMClient(cache_path=".cache/pipeline.sqlite")
        self.embedder = embedder or GatewayEmbedder(self.llm, os.getenv("EMBEDDING_MODEL", "bge-m3"))

    def index(self, corpus: list[Document]) -> None:
        self.docs = corpus
        self.by_id = {d.id: d for d in corpus}
        self.pos = {d.id: i for i, d in enumerate(corpus)}
        self.doc_vectors = self.embedder.embed([d.text for d in corpus])
        self.bm25 = BM25([bm25_tokens(f"{d.club} {d.topic} {d.text}") for d in corpus])

        # Phân loại tài liệu theo (club, topic) phục vụ kéo đoạn song song trong bước audit
        self.by_club_topic: dict[tuple[str, str], list[Document]] = {}
        for d in corpus:
            self.by_club_topic.setdefault((d.club, d.topic), []).append(d)

        # Index Parent-Child cho bước mở rộng ngữ cảnh kiểm định
        child_texts, parents = [], []
        for i, d in enumerate(corpus):
            for sent in split_sentences(d.text.replace("; ", ".\n")):
                if len(sent) >= 15:
                    child_texts.append(f"{d.club} | {d.topic}: {sent}")
                    parents.append(i)
        self.child_parent = np.array(parents, dtype=np.int64)
        self.child_vectors = self.embedder.embed(child_texts) if child_texts else None
        self.child_bm25 = BM25([bm25_tokens(t) for t in child_texts]) if child_texts else None

    def _to_parent(self, child_scores: np.ndarray) -> np.ndarray:
        """Max-pooling điểm số của các đoạn con (child) về đoạn văn bản cha."""
        out = np.full(len(self.docs), -np.inf, dtype=np.float32)
        np.maximum.at(out, self.child_parent, child_scores.astype(np.float32))
        return out

    def rewrite(self, query: str) -> list[str]:
        """Viết lại câu hỏi thành tiếng Việt chuẩn có dấu và phân tách các ý con."""
        try:
            out = self.llm.chat(
                [{"role": "user", "content": REWRITE_PROMPT.format(query=query)}],
                model=self.llm_model,
            )
        except Exception:
            return [query]
        qs = [m.group(1).strip() for m in re.finditer(r"^\s*Q\s*[:：]\s*(.+)$", out, flags=re.MULTILINE)]
        if not qs:
            qs = [re.sub(r"^\s*(?:\d+[.)]|[-*•])\s*", "", line).strip() for line in out.splitlines()]
        qs = [re.sub(r"^(?:câu (?:viết lại|hỏi con)[^:]*:)\s*", "", q, flags=re.IGNORECASE) for q in qs]
        qs = [q for q in qs if len(q) > 3 and not META.search(q)]
        qs = qs[:1] + [q for q in qs[1:] if not DANGLING.search(q)]
        return qs[:4] or [query]

    def _fused(self, query: str, rewrites: list[str], use_child: bool = False) -> np.ndarray:
        """Tính điểm Reciprocal Rank Fusion (RRF, k=60) kết hợp BM25 và Dense bge-m3."""
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

    def retrieve(self, query: str, rewrites: list[str]) -> list[Document]:
        """Truy xuất danh sách tài liệu ứng viên top-max(top_k, n_retrieved) theo RRF."""
        fused = self._fused(query, rewrites, use_child=False)
        k = max(self.top_k, self.n_retrieved)
        return [self.docs[i] for i in np.argsort(-fused)[:k]]

    def generate(self, query: str, rewritten: str, docs: list[Document]) -> tuple[str, list[int]]:
        """Sinh câu trả lời nháp từ top_k tài liệu truy xuất được."""
        context = "\n\n".join(f"[{d.id}] ({d.club} – {d.topic}) {d.text}" for d in docs[: self.top_k])
        examples = ""
        if self.answer_templates == "accented" and not is_unaccented(query):
            examples = ANSWER_TEMPLATES
        elif self.answer_templates is True:
            examples = ANSWER_TEMPLATES

        prompt = ANSWER_PROMPT.format(
            context=context, query=query, rewritten=rewritten, examples=examples
        )
        answer, cited = self._parse_answer(
            self.llm.chat([{"role": "user", "content": prompt}], model=self.llm_model), docs
        )
        if not answer:
            answer, cited = self._parse_answer(
                self.llm.chat([{"role": "user", "content": prompt + "\n"}], model=self.llm_model),
                docs,
            )
        return answer, cited

    def _parse_answer(self, out: str, docs: list[Document]) -> tuple[str, list[int]]:
        """Tách dòng NGUON, lọc id trong context, loại bỏ markdown."""
        cited: list[int] = []
        m = re.search(r"^\s*NGU[OỒ]N\s*:\s*(.*)$", out, flags=re.MULTILINE | re.IGNORECASE)
        if m:
            shown = {d.id for d in docs[: self.top_k]}
            cited = [int(x) for x in re.findall(r"\d+", m.group(1)) if int(x) in shown]
            out = out[: m.start()] + out[m.end() :]
        answer = re.sub(r"\*\*|__|^#+\s*|^\s*[-*•]\s+", "", out, flags=re.MULTILINE)
        answer = re.sub(r"\s*\n\s*", " ", answer).strip()
        return answer, cited

    def _build_audit_docs(
        self, query: str, rewrites: list[str], docs: list[Document], draft_cited: list[int]
    ) -> list[Document]:
        """Mở rộng ngữ cảnh kiểm định lên tối đa 20 đoạn, bổ sung đoạn song song cùng (club, topic)."""
        seen = {d.id for d in docs[: self.top_k]}
        out = list(docs[: self.top_k])
        pc_scores = self._fused(query, rewrites, use_child=True)

        # 1. Bổ sung các đoạn cùng (club, topic) với các đoạn được cite đầu tiên
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

        # 2. Bổ sung top ứng viên parent-child RRF còn lại
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
        """Phân tích phản hồi của Auditor, kích hoạt chốt an toàn bảo toàn bản nháp."""
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

        # Kiểm tra chốt an toàn: nếu sót nhãn <...> thì fallback về bản nháp
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
        """Thực hiện kiểm định bản nháp qua Claude-3.5-Haiku với 6 tiêu chí."""
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
        self,
        draft_cited: list[int],
        audit_cited: list[int],
        docs: list[Document],
        answer_changed: bool,
    ) -> list[int]:
        """Tái sắp xếp ID nộp bài sau kiểm định:
        - Nếu giữ nguyên câu trả lời: khóa chặt trong tập top-10 ban đầu để bảo toàn Groundedness.
        - Nếu cập nhật câu trả lời: ưu tiên audit_cited và draft_cited lên đầu danh sách.
        - Các vị trí còn lại xếp theo Cosine Similarity lớn nhất với các đoạn đã trích dẫn.
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

    def order_ids(self, cited: list[int], docs: list[Document]) -> list[int]:
        """Sắp xếp ID theo cite_rerank (tương thích backward)."""
        if not cited or not self.cite_rerank:
            return list(dict.fromkeys(cited + [d.id for d in docs[: self.n_retrieved]]))[: self.n_retrieved]
        cited_vecs = self.doc_vectors[[self.pos[i] for i in cited]]
        base = list(dict.fromkeys(cited + [d.id for d in docs[: self.n_retrieved]]))[: self.n_retrieved]
        rest = [i for i in base if i not in set(cited)]
        sim = {i: float((cited_vecs @ self.doc_vectors[self.pos[i]]).max()) for i in rest}
        rest.sort(key=lambda i: -sim[i])
        return list(dict.fromkeys(cited + rest))[: self.n_retrieved]

    def run(self, query: str) -> RAGOutput:
        """Thực thi pipeline hoàn chỉnh từ câu hỏi đến câu trả lời và ID nộp bài."""
        rewrites = self.rewrite(query)
        docs = self.retrieve(query, rewrites)
        draft_answer, draft_cited = self.generate(query, rewrites[0], docs)
        audit_docs = self._build_audit_docs(query, rewrites, docs, draft_cited)
        final_answer, audit_cited = self.audit(query, rewrites[0], audit_docs, draft_answer, draft_cited)
        ids = self.order_ids_audit(
            draft_cited, audit_cited, docs, answer_changed=(final_answer != draft_answer)
        )
        return RAGOutput(
            answer=final_answer or "Tài liệu hiện không có thông tin để trả lời câu hỏi này.",
            retrieved_ids=ids,
        )


# Alias cho CLI — cả hai tên đều trỏ về cùng class
MyPipelineV11Audit = MyPipeline
