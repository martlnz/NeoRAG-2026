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
        answer_templates: bool | str = "accented", parent_child: bool = False,
    ):
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
        self.doc_vectors = self.embedder.embed([d.text for d in corpus])
        self.bm25 = BM25([bm25_tokens(f"{d.club} {d.topic} {d.text}") for d in corpus])
        # Parent-child: child = từng câu (đoạn danh sách tách thêm theo ";"), kèm header club|topic để
        # không mất ngữ cảnh. Tìm trên child (không bị "trung bình hóa" nhiều ý), điểm đoạn cha = max child.
        child_texts, parents = [], []
        for i, d in enumerate(corpus if self.parent_child else []):
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
        # Câu gốc không dấu: dense bge-m3 xếp doc đúng ở hạng 30-100 → chỉ dùng BM25 (bỏ dấu) cho nó.
        dense_orig = not is_unaccented(query)
        queries = [(query, 0.7, dense_orig)] + [(q, 1.0, True) for q in rewrites]
        dense_qs = [q for q, _, use in queries if use]
        qvecs = iter(self.embedder.embed(dense_qs)) if dense_qs else iter(())
        fused = np.zeros(len(self.docs), dtype=np.float32)
        for q, w, use_dense in queries:
            toks = bm25_tokens(q)
            rankings = [self.bm25.scores(toks)]
            if self.child_bm25 is not None:
                rankings.append(self._to_parent(self.child_bm25.scores(toks)))
            if use_dense:
                qv = next(qvecs)
                rankings.append(self.doc_vectors @ qv)
                if self.child_vectors is not None:
                    rankings.append(self._to_parent(self.child_vectors @ qv))
            for scores in rankings:
                ranks = np.empty(len(scores), dtype=np.float32)
                ranks[np.argsort(-scores)] = np.arange(1, len(scores) + 1)
                fused += w / (RRF_K + ranks)
        k = max(self.top_k, self.n_retrieved)
        return [self.docs[i] for i in np.argsort(-fused)[:k]]

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
        # id được cite (có thể ở hạng 11-15 của context) lên đầu, rồi phần còn lại theo RRF; nộp tối đa 10.
        ids = list(dict.fromkeys(cited + [d.id for d in docs[: self.n_retrieved]]))[: self.n_retrieved]
        return RAGOutput(answer=answer or "Tài liệu hiện không có thông tin để trả lời câu hỏi này.", retrieved_ids=ids)


class MyPipelineV9(MyPipeline):
    """v9: v6 + mẫu trả lời chung (ANSWER_TEMPLATES). Hòa v6 trên dev; dùng để nộp thử public.

    Chạy: neorag run public_test --pipeline neorag.pipelines.pipeline:MyPipelineV9
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("answer_templates", True)
        super().__init__(**kwargs)


class MyPipelineV10(MyPipeline):
    """v10: v6, nhưng câu hỏi CÓ DẤU dùng thêm mẫu trả lời chung (lợi trên train ở v9);
    câu văn nói / không dấu giữ nguyên prompt v6 (mẫu chung gây hại trên train_informal và public).

    Chạy: neorag run <split> --pipeline neorag.pipelines.pipeline:MyPipelineV10
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("answer_templates", "accented")
        super().__init__(**kwargs)


class MyPipelineV6(MyPipeline):
    """Phương án dự phòng: đúng hành vi v6 (không dùng mẫu trả lời chung).

    Chạy: neorag run private_test --pipeline neorag.pipelines.pipeline:MyPipelineV6
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("answer_templates", False)
        super().__init__(**kwargs)
