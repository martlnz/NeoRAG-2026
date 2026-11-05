"""Test MyPipeline (src/neorag/pipelines/pipeline.py) bằng LLM/embedding giả — không gọi API."""

import pytest

from conftest import FakeEmbedder, FakeLLM
from neorag.data import Document, Query
from neorag.pipelines import RAGPipeline, load_pipeline
from neorag.pipelines.pipeline import BM25, MyPipeline, bm25_tokens, is_unaccented, strip_accents
from neorag.predict import predict


def make_llm(rewrite: str, answer: str) -> FakeLLM:
    """Lời gọi rewrite trả `rewrite`, lời gọi sinh câu trả lời trả `answer`."""
    return FakeLLM(lambda m: answer if "TÀI LIỆU:" in m[0]["content"] else rewrite)


def make_pipeline(corpus, rewrite: str, answer: str) -> MyPipeline:
    p = MyPipeline(llm=make_llm(rewrite, answer), embedder=FakeEmbedder())
    p.index(corpus)
    return p


# ------------------------------------------------------------------ text / BM25
def test_strip_accents_handles_d_and_tones():
    assert strip_accents("Đà Nẵng Lập Trình") == "da nang lap trinh"


def test_bm25_tokens_include_bigrams():
    toks = bm25_tokens("Lập Trình PTIT")
    assert {"lap", "trinh", "ptit", "lap_trinh", "trinh_ptit"} <= set(toks)


def test_bm25_matches_unaccented_query(corpus):
    bm = BM25([bm25_tokens(d.text) for d in corpus])
    scores = bm.scores(bm25_tokens("clb lap trinh thanh lap ngay nao"))
    assert int(scores.argmax()) == 0  # doc ProPTIT


def test_bm25_unknown_terms_score_zero(corpus):
    bm = BM25([bm25_tokens(d.text) for d in corpus])
    assert bm.scores(["khongtontai"]).sum() == 0


# ----------------------------------------------------------------------- rewrite
@pytest.mark.parametrize(
    "out, expected",
    [
        ("Q: CLB A thành lập khi nào?\nQ: CLB B thành lập khi nào?",
         ["CLB A thành lập khi nào?", "CLB B thành lập khi nào?"]),
        # LLM không theo định dạng, in kèm dòng meta → lọc bỏ
        ("Câu viết lại: CLB A có phải CLB chính thức không?\n"
         "Câu hỏi này chỉ có một ý, nên không cần câu hỏi con.",
         ["CLB A có phải CLB chính thức không?"]),
        ("1. Câu hỏi X?\n- Câu hỏi Y?", ["Câu hỏi X?", "Câu hỏi Y?"]),
        ("", ["câu gốc"]),  # rỗng → dùng câu gốc
    ],
)
def test_rewrite_parsing(corpus, out, expected):
    p = make_pipeline(corpus, rewrite=out, answer="")
    assert p.rewrite("câu gốc") == expected


def test_rewrite_caps_at_four_queries(corpus):
    p = make_pipeline(corpus, rewrite="\n".join(f"Q: câu hỏi số {i}" for i in range(6)), answer="")
    assert len(p.rewrite("x")) == 4


# ------------------------------------------------------------- generate / run
def test_cited_ids_move_to_front_and_line_is_removed(corpus):
    p = make_pipeline(
        corpus,
        rewrite="Q: CLB Nhà Sáng Tạo Game PTIT ra đời năm nào?",
        answer="CLB Nhà Sáng Tạo Game PTIT (GCC) ra đời năm 2023.\nNGUON: 3",
    )
    out = p.run("gcc ra doi nam nao")
    assert out.retrieved_ids[0] == 3
    assert sorted(out.retrieved_ids) == [1, 2, 3]  # đủ doc, không trùng
    assert "NGUON" not in out.answer
    assert out.answer == "CLB Nhà Sáng Tạo Game PTIT (GCC) ra đời năm 2023."


def test_hallucinated_citation_is_ignored(corpus):
    p = make_pipeline(corpus, rewrite="Q: x", answer="Trả lời.\nNGUON: 999, 2")
    out = p.run("x")
    assert 999 not in out.retrieved_ids
    assert out.retrieved_ids[0] == 2


def test_markdown_is_stripped(corpus):
    p = make_pipeline(corpus, rewrite="Q: x", answer="**CLB A** có:\n- ý một\n- ý hai\nNGUON: 1")
    assert p.run("x").answer == "CLB A có: ý một ý hai"


def test_empty_answer_falls_back_to_abstention(corpus):
    p = make_pipeline(corpus, rewrite="Q: x", answer="NGUON: 1")
    assert p.run("x").answer.startswith("Tài liệu hiện không có thông tin")


def test_two_llm_calls_per_question(corpus):
    llm = make_llm("Q: x", "Trả lời.\nNGUON: 1")
    p = MyPipeline(llm=llm, embedder=FakeEmbedder())
    p.index(corpus)
    p.run("x")
    assert len(llm.calls) == 2


# ------------------------------------------------------------- robustness (v3.1)
def test_empty_answer_is_retried_once(corpus):
    """Câu trả lời rỗng → gọi lại một lần với prompt khác (khóa cache mới) và dùng kết quả đó."""
    answers = iter(["NGUON: 1", "Ngày 09/10/2011.\nNGUON: 1"])
    llm = FakeLLM(lambda m: next(answers) if "TÀI LIỆU:" in m[0]["content"] else "Q: x")
    p = MyPipeline(llm=llm, embedder=FakeEmbedder())
    p.index(corpus)
    out = p.run("x")
    assert out.answer == "Ngày 09/10/2011."
    assert len(llm.calls) == 3  # rewrite + answer rỗng + gọi lại
    assert llm.calls[2][0]["content"] != llm.calls[1][0]["content"]


def test_rewrite_api_error_falls_back_to_original_query(corpus):
    def responder(messages):
        if "TÀI LIỆU:" not in messages[0]["content"]:
            raise RuntimeError("gateway 500")
        return "Trả lời.\nNGUON: 1"

    p = MyPipeline(llm=FakeLLM(responder), embedder=FakeEmbedder())
    p.index(corpus)
    assert p.rewrite("câu gốc") == ["câu gốc"]
    assert p.run("câu gốc").answer == "Trả lời."


def test_answer_api_error_is_not_silently_swallowed(corpus):
    """Lỗi ở bước sinh câu trả lời phải lộ ra (chạy lại dùng cache), không được nộp câu từ chối giả."""
    def responder(messages):
        if "TÀI LIỆU:" in messages[0]["content"]:
            raise RuntimeError("gateway 500")
        return "Q: x"

    p = MyPipeline(llm=FakeLLM(responder), embedder=FakeEmbedder())
    p.index(corpus)
    with pytest.raises(RuntimeError):
        p.run("x")


# --------------------------------------------------------------------- v5
def test_is_unaccented():
    assert is_unaccented("clb lap trinh thanh lap khi nao a")
    assert not is_unaccented("CLB Lập Trình PTIT thành lập khi nào?")
    assert not is_unaccented("đi học")


class RecordingEmbedder(FakeEmbedder):
    def __init__(self):
        self.seen: list[str] = []

    def embed(self, texts):
        self.seen.extend(texts)
        return super().embed(texts)


def test_unaccented_original_query_skips_dense(corpus):
    emb = RecordingEmbedder()
    p = MyPipeline(llm=make_llm("Q: CLB Lập Trình PTIT thành lập khi nào?", "x\nNGUON: 1"), embedder=emb)
    p.index(corpus)
    emb.seen.clear()
    p.run("clb lap trinh thanh lap khi nao")
    assert emb.seen == ["CLB Lập Trình PTIT thành lập khi nào?"]  # chỉ embed câu viết lại


def test_accented_original_query_still_uses_dense(corpus):
    emb = RecordingEmbedder()
    p = MyPipeline(llm=make_llm("Q: viết lại", "x\nNGUON: 1"), embedder=emb)
    p.index(corpus)
    emb.seen.clear()
    p.run("CLB Lập Trình PTIT thành lập khi nào?")
    assert emb.seen[0] == "CLB Lập Trình PTIT thành lập khi nào?"


def test_context_has_15_docs_and_late_citation_moves_to_front():
    docs = [Document(i, "CLB Lập Trình PTIT", "t", f"đoạn văn số {i} nói về chủ đề {i}") for i in range(1, 21)]
    llm = make_llm("Q: x", "")
    p = MyPipeline(llm=llm, embedder=FakeEmbedder())
    p.index(docs)
    ranked = p.retrieve("x", ["x"])
    assert len(ranked) == 15
    late = ranked[12].id  # hạng 13: ngoài 10 id nộp nhưng vẫn có trong context
    llm.responder = lambda m: f"Trả lời.\nNGUON: {late}" if "TÀI LIỆU:" in m[0]["content"] else "Q: x"
    out = p.run("x")
    prompt = llm.calls[-1][0]["content"]
    assert prompt.count("(CLB Lập Trình PTIT – t)") == 15
    assert out.retrieved_ids[0] == late and len(out.retrieved_ids) == 10


# --------------------------------------------------------------------- v6
def test_dangling_subqueries_are_dropped_but_first_line_kept(corpus):
    out = (
        "Q: Em thích vẽ và làm game thì nên tham gia CLB nào?\n"
        "Q: CLB Nhà Sáng Tạo Game PTIT (GCC) có nhánh vẽ không?\n"
        "Q: Khi tham gia CLB đó em sẽ được học những phần mềm gì?"
    )
    p = make_pipeline(corpus, rewrite=out, answer="")
    assert p.rewrite("x") == [
        "Em thích vẽ và làm game thì nên tham gia CLB nào?",
        "CLB Nhà Sáng Tạo Game PTIT (GCC) có nhánh vẽ không?",
    ]


def test_answer_prompt_has_v6_rules():
    from neorag.pipelines.pipeline import ANSWER_PROMPT, REWRITE_PROMPT

    assert "Tài liệu hiện không có thông tin về" in ANSWER_PROMPT
    assert "Không suy ra khẳng định phủ định" in ANSWER_PROMPT
    assert '"Có, ..."' in ANSWER_PROMPT and "Theo <nguồn" in ANSWER_PROMPT
    assert "tự đứng được" in REWRITE_PROMPT and "có những ban nào" in REWRITE_PROMPT


def test_yes_no_opening_is_preserved(corpus):
    p = make_pipeline(corpus, rewrite="Q: x", answer="Có, CLB Lập Trình PTIT nhận sinh viên năm nhất.\nNGUON: 1")
    assert p.run("x").answer == "Có, CLB Lập Trình PTIT nhận sinh viên năm nhất."


def test_end_to_end_with_runner(corpus):
    p = make_pipeline(corpus, rewrite="Q: ProPTIT thành lập khi nào?", answer="Ngày 09/10/2011.\nNGUON: 1")
    preds = predict(p, [Query("q1", "proptit thanh lap khi nao")], max_retrieved=10)
    assert preds[0].retrieved_ids[0] == 1
    assert preds[0].answer == "Ngày 09/10/2011."


def test_loadable_via_cli_spec():
    """Chuỗi truyền cho --pipeline phải nạp được MyPipeline (kwargs giả để không tạo client thật)."""
    p = load_pipeline(
        "neorag.pipelines.pipeline:MyPipeline", llm=FakeLLM(), embedder=FakeEmbedder()
    )
    assert isinstance(p, MyPipeline) and isinstance(p, RAGPipeline)
