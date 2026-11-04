"""Pipeline của thí sinh, cài đặt interface `RAGPipeline`.

Chạy:
    neorag run train --pipeline neorag.pipelines.pipeline:MyPipeline
    neorag run public_test --pipeline neorag.pipelines.pipeline:MyPipeline

Lịch sử phiên bản và phân tích: thư mục `experiments/`.
"""

from __future__ import annotations

import os

import numpy as np

from neorag.data import Document
from neorag.llm import LLMClient
from neorag.pipelines.base import RAGOutput, RAGPipeline

PROMPT = """Dựa vào các tài liệu dưới đây, hãy trả lời câu hỏi.

TÀI LIỆU:
{context}

CÂU HỎI: {query}"""


class GatewayEmbedder:
    """Embedding bge-m3 qua gateway của BTC; trả về vector đã chuẩn hóa L2."""

    def __init__(self, llm: LLMClient, model_name: str):
        self.llm, self.model_name = llm, model_name

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.asarray(self.llm.embed(texts, model=self.model_name), dtype=np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


class MyPipeline(RAGPipeline):
    """v0 (init): dense retrieval bge-m3 trên nguyên đoạn văn + một lần gọi LLM."""

    def __init__(self, top_k: int = 5, n_retrieved: int = 10):
        self.top_k = top_k  # số doc đưa vào prompt
        self.n_retrieved = n_retrieved  # số doc trả về để chấm retrieval (tối đa 10)
        # CLI đọc `llm_model` và `embedder` để in thông tin.
        self.llm_model = os.getenv("LLM_MODEL", "claude-haiku-5-5")
        self.llm = LLMClient(cache_path=".cache/pipeline.sqlite")
        self.embedder = GatewayEmbedder(self.llm, os.getenv("EMBEDDING_MODEL", "bge-m3"))

    def index(self, corpus: list[Document]) -> None:
        self.docs = corpus
        self.doc_vectors = self.embedder.embed([d.text for d in corpus])

    def retrieve(self, query: str) -> list[Document]:
        q = self.embedder.embed([query])[0]
        ranking = np.argsort(-(self.doc_vectors @ q))[: self.n_retrieved]
        return [self.docs[i] for i in ranking]

    def generate(self, query: str, docs: list[Document]) -> str:
        context = "\n\n".join(d.text for d in docs[: self.top_k])
        return self.llm.chat(
            [{"role": "user", "content": PROMPT.format(context=context, query=query)}],
            model=self.llm_model,
        )

    def run(self, query: str) -> RAGOutput:
        docs = self.retrieve(query)
        return RAGOutput(answer=self.generate(query, docs), retrieved_ids=[d.id for d in docs])
