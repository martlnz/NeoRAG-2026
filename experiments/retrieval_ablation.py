"""Thử nghiệm các biến thể retrieval trên train và train_informal (chỉ đo retrieval, không judge).

Đo trên ranking thô (trước khi LLM trích dẫn đẩy doc lên đầu): recall@5/10/15, MRR@5, NDCG@5
và retrieval score = mean(recall@5, MRR@5, NDCG@5) — đúng công thức trong configs/eval.toml.

Câu viết lại lấy từ cache của pipeline (.cache/pipeline.sqlite) → gần như không tốn ngân sách.
HyDE cần 1 lời gọi Haiku ngắn/câu (cũng được cache). Cross-encoder chạy local (CPU/GPU).

Chạy:
    python experiments/retrieval_ablation.py                    # biến thể không tốn tiền
    python experiments/retrieval_ablation.py --rerank --hyde    # thêm cross-encoder và HyDE
"""

from __future__ import annotations

import argparse
import statistics as st
import time

import numpy as np

from neorag.data import DATA_DIR, load_corpus, load_queries
from neorag.metrics.retrieval import score_query
from neorag.pipelines.pipeline import BM25, RRF_K, MyPipeline, bm25_tokens, is_unaccented
from neorag.text import split_sentences

HYDE_PROMPT = """Viết một đoạn văn ngắn (2-3 câu, tiếng Việt có dấu) như thể trích từ tài liệu giới thiệu \
các CLB của PTIT (CLB Lập Trình PTIT, CLB Multimedia PTIT, CLB Nhà Sáng Tạo Game PTIT) để trả lời câu hỏi sau. \
Chỉ in đoạn văn.

CÂU HỎI: {query}"""

K_VALUES = [5, 10, 15]


def rrf(rankings: list[tuple[np.ndarray, float]], n: int) -> np.ndarray:
    fused = np.zeros(n, dtype=np.float32)
    for scores, w in rankings:
        ranks = np.empty(n, dtype=np.float32)
        ranks[np.argsort(-scores)] = np.arange(1, n + 1)
        fused += w / (RRF_K + ranks)
    return fused


class Ablation:
    def __init__(self, rerank: bool, hyde: bool, m3: bool = False):
        self.p = MyPipeline()
        self.corpus = load_corpus()
        self.p.index(self.corpus)
        self.ids = np.array([d.id for d in self.corpus])
        texts = [f"{d.club} | {d.topic}: {d.text}" for d in self.corpus]
        self.header_vectors = self.p.embedder.embed(texts)
        # Chunk nhỏ: mỗi câu (đoạn danh sách tách thêm theo ";") + header club|topic để giữ ngữ cảnh.
        self.chunk_parent: list[int] = []  # chỉ số đoạn cha trong corpus
        chunk_texts: list[str] = []
        for i, d in enumerate(self.corpus):
            for sent in split_sentences(d.text.replace("; ", ".\n")):
                if len(sent) >= 15:
                    chunk_texts.append(f"{d.club} | {d.topic}: {sent}")
                    self.chunk_parent.append(i)
        self.chunk_parent_arr = np.array(self.chunk_parent)
        self.chunk_vectors = self.p.embedder.embed(chunk_texts)
        self.chunk_bm25 = BM25([bm25_tokens(t) for t in chunk_texts])
        print(f"{len(self.corpus)} đoạn → {len(chunk_texts)} chunk câu")
        self.hyde = hyde
        self.m3 = None
        if m3:
            # bge-m3 local: multi-vector (ColBERT, MaxSim theo token) + sparse (trọng số từ vựng học được).
            from FlagEmbedding import BGEM3FlagModel

            self.m3 = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)
            enc = self.m3.encode(
                [d.text for d in self.corpus], batch_size=8, max_length=512,
                return_dense=False, return_sparse=True, return_colbert_vecs=True,
            )
            self.m3_doc_colbert = enc["colbert_vecs"]
            self.m3_doc_sparse = enc["lexical_weights"]
            print("bge-m3 local: đã encode multi-vector + sparse cho", len(self.corpus), "đoạn")
        self.reranker = None
        if rerank:
            from sentence_transformers import CrossEncoder

            self.reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)

    # -------------------------------------------------------------- variants
    def rankings(self, query: str) -> dict[str, list[int]]:
        p, n = self.p, len(self.corpus)
        rewrites = p.rewrite(query)
        qs = [(query, 0.7, not is_unaccented(query))] + [(q, 1.0, True) for q in rewrites]
        vecs = {q: v for q, v in zip([q for q, _, _ in qs], p.embedder.embed([q for q, _, _ in qs]), strict=True)}
        bm = {q: p.bm25.scores(bm25_tokens(q)) for q, _, _ in qs}

        def fuse(use_dense=True, use_bm25=True, doc_vecs=None, extra=()):
            doc_vecs = p.doc_vectors if doc_vecs is None else doc_vecs
            parts = []
            for q, w, dense_ok in qs:
                if use_bm25:
                    parts.append((bm[q], w))
                if use_dense and dense_ok:
                    parts.append((doc_vecs @ vecs[q], w))
            parts.extend(extra)
            return rrf(parts, n)

        def to_parent(chunk_scores: np.ndarray) -> np.ndarray:
            """Max-pooling điểm chunk về đoạn cha (đoạn không có chunk nào: -inf)."""
            doc = np.full(n, -np.inf, dtype=np.float32)
            np.maximum.at(doc, self.chunk_parent_arr, chunk_scores)
            return doc

        def chunk_parts():
            parts = []
            for q, w, dense_ok in qs:
                parts.append((to_parent(self.chunk_bm25.scores(bm25_tokens(q))), w))
                if dense_ok:
                    parts.append((to_parent(self.chunk_vectors @ vecs[q]), w))
            return parts

        out: dict[str, np.ndarray] = {
            "chunk câu (max về đoạn cha)": rrf(chunk_parts(), n),
            "base + chunk câu (RRF)": fuse(extra=chunk_parts()),
            "v5_base (RRF dense+BM25)": fuse(),
            "dense only": fuse(use_bm25=False),
            "bm25 only": fuse(use_dense=False),
            "dense trên header club|topic": fuse(doc_vecs=self.header_vectors),
            "base + dense header (cả hai)": fuse(extra=[(self.header_vectors @ vecs[q], w) for q, w, ok in qs if ok]),
        }
        if self.m3 is not None:
            q_texts = [q for q, _, _ in qs]
            qenc = self.m3.encode(q_texts, max_length=128, return_dense=False, return_sparse=True,
                                  return_colbert_vecs=True)
            colbert_parts, sparse_parts = [], []
            for (_, w, _), qc, qs_ in zip(qs, qenc["colbert_vecs"], qenc["lexical_weights"], strict=True):
                colbert_parts.append((np.array([float(self.m3.colbert_score(qc, dc)) for dc in self.m3_doc_colbert]), w))
                sparse_parts.append((np.array([float(self.m3.compute_lexical_matching_score(qs_, ds))
                                               for ds in self.m3_doc_sparse]), w))
            child = chunk_parts()
            out["v8 (base + child)"] = fuse(extra=child)
            out["v8 + m3 multi-vector"] = fuse(extra=child + colbert_parts)
            out["v8 + m3 sparse"] = fuse(extra=child + sparse_parts)
            out["v8 + m3 multi-vector + sparse"] = fuse(extra=child + colbert_parts + sparse_parts)
            out["m3 multi-vector only"] = rrf(colbert_parts, n)
        if self.hyde:
            fake = p.llm.chat([{"role": "user", "content": HYDE_PROMPT.format(query=rewrites[0])}], model=p.llm_model)
            hv = p.embedder.embed([fake])[0]
            out["base + HyDE"] = fuse(extra=[(p.doc_vectors @ hv, 1.0)])
        ranked = {k: [int(self.ids[i]) for i in np.argsort(-v)] for k, v in out.items()}
        if self.reranker is not None:
            cand = ranked["v5_base (RRF dense+BM25)"][:20]
            by_id = {d.id: d for d in self.corpus}
            scores = self.reranker.predict([(rewrites[0], f"{by_id[i].club}: {by_id[i].text}") for i in cand])
            order = [cand[i] for i in np.argsort(-scores)]
            ranked["base top20 → cross-encoder"] = order + ranked["v5_base (RRF dense+BM25)"][20:]
            # Gộp RRF giữa ranking gốc và ranking reranker (giữ tín hiệu BM25 cho câu không dấu)
            pos = {i: r for r, i in enumerate(order)}
            fused = sorted(cand, key=lambda i: -(1 / (RRF_K + cand.index(i) + 1) + 1 / (RRF_K + pos[i] + 1)))
            ranked["RRF(base, cross-encoder)"] = fused + ranked["v5_base (RRF dense+BM25)"][20:]
        return ranked

    # ---------------------------------------------------------------- report
    def evaluate(self, split: str) -> None:
        queries = [q for q in load_queries(DATA_DIR / split / "questions.jsonl") if q.relevant_ids]
        rows: dict[str, list[dict]] = {}
        t0 = time.perf_counter()
        for q in queries:
            for name, ranking in self.rankings(q.query).items():
                rows.setdefault(name, []).append(score_query(ranking[:15], set(q.relevant_ids), K_VALUES))
        print(f"\n### {split} ({len(queries)} câu có relevant_ids, {time.perf_counter() - t0:.0f}s)\n")
        print("| Biến thể | retrieval score | R@5 | MRR@5 | NDCG@5 | R@10 | R@15 |")
        print("|---|---|---|---|---|---|---|")
        for name, rs in rows.items():
            m = {k: st.mean(r[k] for r in rs) for k in ("recall@5", "mrr@5", "ndcg@5", "recall@10", "recall@15")}
            score = (m["recall@5"] + m["mrr@5"] + m["ndcg@5"]) / 3
            print(
                f"| {name} | **{score:.4f}** | {m['recall@5']:.4f} | {m['mrr@5']:.4f} | "
                f"{m['ndcg@5']:.4f} | {m['recall@10']:.4f} | {m['recall@15']:.4f} |"
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "train_informal"])
    ap.add_argument("--rerank", action="store_true", help="cross-encoder bge-reranker-v2-m3 (local)")
    ap.add_argument("--hyde", action="store_true", help="HyDE: +1 lời gọi Haiku/câu (có cache)")
    ap.add_argument("--m3", action="store_true", help="bge-m3 local: multi-vector (ColBERT) + sparse")
    ap.add_argument("--only-m3", action="store_true", help="chỉ in các biến thể v8/m3")
    args = ap.parse_args()
    ab = Ablation(rerank=args.rerank, hyde=args.hyde, m3=args.m3 or args.only_m3)
    for split in args.splits:
        ab.evaluate(split)


if __name__ == "__main__":
    main()
