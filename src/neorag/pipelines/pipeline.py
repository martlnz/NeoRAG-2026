"""Pipeline của thí sinh, cài đặt interface `RAGPipeline`.

Chạy:
    neorag run train --pipeline neorag.pipelines.pipeline:MyPipeline
    neorag run public_test --pipeline neorag.pipelines.pipeline:MyPipeline

Lịch sử phiên bản và phân tích: thư mục `experiments/`.

v1:
  1. LLM viết lại câu hỏi (thêm dấu, mở viết tắt, chuẩn hóa tên CLB, tách câu nhiều ý).
  2. Hybrid retrieval: BM25 trên văn bản bỏ dấu + dense bge-m3, gộp bằng RRF trên nhiều truy vấn.
  3. Prompt sinh ngắn gọn, không markdown; LLM ghi id nguồn → đẩy lên đầu retrieved_ids.
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

REWRITE_PROMPT = """Bạn chuẩn hóa câu hỏi của sinh viên về cộng đồng S2B ở Học viện Công nghệ Bưu chính Viễn thông (PTIT), gồm 3 CLB:
- CLB Lập Trình PTIT (ProPTIT; người hỏi có thể gọi "clb lập trình", "clb lt", "clb code")
- CLB Multimedia PTIT (MPC; "clb multimedia", "clb mm", "clb truyền thông/thiết kế/quay dựng")
- CLB Nhà Sáng Tạo Game PTIT (GCC; "clb game", "clb làm game")
Viết tắt thường gặp: hv = Học viện, e = em, a/c = anh/chị, ko/k = không, tv = thành viên, dpt = Đa phương tiện, nk = nhiệm kỳ, cn = chủ nhiệm.

Nhiệm vụ:
1. Viết lại câu hỏi thành tiếng Việt đầy đủ dấu, rõ ràng, nêu tên CLB chuẩn nếu xác định được. Không thêm thông tin mà câu hỏi không có.
2. Nếu câu hỏi gồm nhiều ý hoặc so sánh nhiều CLB, viết thêm tối đa 3 câu hỏi con, mỗi câu một ý hoặc một CLB.

Chỉ in các câu hỏi, mỗi câu một dòng, dòng đầu là câu viết lại. Không đánh số, không giải thích.

CÂU HỎI: {query}"""

ANSWER_PROMPT = """Bạn là trợ lý tư vấn về các CLB thuộc cộng đồng S2B của PTIT. Chỉ dùng TÀI LIỆU bên dưới.

Quy tắc:
- Trả lời ngắn gọn 1-3 câu văn xuôi; không markdown, không in đậm, không gạch đầu dòng, không chào hỏi.
- Câu đầu trả lời thẳng vào câu hỏi và nêu tên CLB đầy đủ.
- Giữ nguyên tên riêng, ngày tháng, số liệu đúng như trong tài liệu.
- Câu hỏi so sánh hoặc nhiều ý: trả lời đủ từng ý.
- Nếu tài liệu chỉ có một phần thông tin, trả lời phần đó và nói ngắn gọn phần còn thiếu.
- Chỉ khi không đoạn nào liên quan mới trả lời "Tài liệu hiện không có thông tin về ..." và không đoán.

Dòng cuối cùng ghi: NGUON: <id các đoạn đã dùng, cách nhau dấu phẩy>

TÀI LIỆU:
{context}

CÂU HỎI GỐC: {query}
(Hiểu là: {rewritten})"""

RRF_K = 60


def strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower().replace("đ", "d").replace("Đ", "d"))
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


def bm25_tokens(text: str) -> list[str]:
    """Âm tiết bỏ dấu + bigram âm tiết (để 'lap trinh' khớp tốt hơn 'lap' và 'trinh' rời)."""
    syl = re.findall(r"\w+", strip_accents(text))
    return syl + [f"{a}_{b}" for a, b in zip(syl, syl[1:])]


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
    def __init__(self, top_k: int = 8, n_retrieved: int = 10):
        self.top_k = top_k  # số doc đưa vào prompt
        self.n_retrieved = n_retrieved  # số doc nộp để chấm retrieval (tối đa 10)
        # CLI đọc `llm_model` và `embedder` để in thông tin.
        self.llm_model = os.getenv("LLM_MODEL", "claude-haiku-5-5")
        self.llm = LLMClient(cache_path=".cache/pipeline.sqlite")
        self.embedder = GatewayEmbedder(self.llm, os.getenv("EMBEDDING_MODEL", "bge-m3"))

    # ------------------------------------------------------------------ index
    def index(self, corpus: list[Document]) -> None:
        self.docs = corpus
        self.by_id = {d.id: d for d in corpus}
        self.doc_vectors = self.embedder.embed([d.text for d in corpus])
        self.bm25 = BM25([bm25_tokens(f"{d.club} {d.topic} {d.text}") for d in corpus])

    # ---------------------------------------------------------- query rewrite
    def rewrite(self, query: str) -> list[str]:
        out = self.llm.chat(
            [{"role": "user", "content": REWRITE_PROMPT.format(query=query)}], model=self.llm_model
        )
        lines = [re.sub(r"^\s*(?:\d+[.)]|[-*•])\s*", "", l).strip() for l in out.splitlines()]
        return [l for l in lines if len(l) > 3][:4] or [query]

    # -------------------------------------------------------------- retrieval
    def retrieve(self, query: str, rewrites: list[str]) -> list[Document]:
        queries = [(query, 0.7)] + [(q, 1.0) for q in rewrites]
        qvecs = self.embedder.embed([q for q, _ in queries])
        fused = np.zeros(len(self.docs), dtype=np.float32)
        for (q, w), qv in zip(queries, qvecs):
            for scores in (self.doc_vectors @ qv, self.bm25.scores(bm25_tokens(q))):
                ranks = np.empty(len(scores), dtype=np.float32)
                ranks[np.argsort(-scores)] = np.arange(1, len(scores) + 1)
                fused += w / (RRF_K + ranks)
        return [self.docs[i] for i in np.argsort(-fused)[: self.n_retrieved]]

    # ------------------------------------------------------------- generation
    def generate(self, query: str, rewritten: str, docs: list[Document]) -> tuple[str, list[int]]:
        context = "\n\n".join(f"[{d.id}] ({d.club} – {d.topic}) {d.text}" for d in docs[: self.top_k])
        out = self.llm.chat(
            [{"role": "user", "content": ANSWER_PROMPT.format(context=context, query=query, rewritten=rewritten)}],
            model=self.llm_model,
        )
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
        ids = list(dict.fromkeys(cited + [d.id for d in docs]))[: self.n_retrieved]
        return RAGOutput(answer=answer or "Tài liệu hiện không có thông tin để trả lời câu hỏi này.", retrieved_ids=ids)
