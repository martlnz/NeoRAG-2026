"""Test MyPipeline (src/neorag/pipelines/pipeline.py) bằng LLM/embedding giả — không gọi API."""

import re

import numpy as np
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
    # v8 (từ chối đúng một câu) bị loại → quay về quy tắc v6
    assert "Tài liệu chỉ nêu" in ANSWER_PROMPT
    assert '"Có, ..."' in ANSWER_PROMPT and "Theo <nguồn" in ANSWER_PROMPT
    assert "tự đứng được" in REWRITE_PROMPT and "có những ban nào" in REWRITE_PROMPT


def test_yes_no_opening_is_preserved(corpus):
    p = make_pipeline(corpus, rewrite="Q: x", answer="Có, CLB Lập Trình PTIT nhận sinh viên năm nhất.\nNGUON: 1")
    assert p.run("x").answer == "Có, CLB Lập Trình PTIT nhận sinh viên năm nhất."


# --------------------------------------------------------------------- v7
def test_fewshot_examples_built_from_train():
    from neorag.pipelines.pipeline import FEWSHOT_IDS, build_examples

    block = build_examples(FEWSHOT_IDS)
    assert block.count("Hỏi: ") == len(FEWSHOT_IDS)
    assert "KHÔNG dùng làm nguồn thông tin" in block
    assert "Đáp: Có, Theo Phòng truyền thống" in block  # "Có." trong đáp án train → "Có,"
    assert build_examples(()) == ""


def test_fewshot_is_off_by_default(corpus):
    llm = make_llm("Q: x", "Trả lời.\nNGUON: 1")
    p = MyPipeline(llm=llm, embedder=FakeEmbedder())
    p.index(corpus)
    p.run("x")
    assert "VÍ DỤ CÁCH TRÌNH BÀY" not in llm.calls[1][0]["content"]


def test_fewshot_block_is_in_answer_prompt_only(corpus):
    from neorag.pipelines.pipeline import FEWSHOT_IDS

    llm = make_llm("Q: x", "Trả lời.\nNGUON: 1")
    p = MyPipeline(llm=llm, embedder=FakeEmbedder(), fewshot_ids=FEWSHOT_IDS)
    p.index(corpus)
    p.run("x")
    rewrite_prompt, answer_prompt = llm.calls[0][0]["content"], llm.calls[1][0]["content"]
    assert "VÍ DỤ CÁCH TRÌNH BÀY" in answer_prompt and "VÍ DỤ CÁCH TRÌNH BÀY" not in rewrite_prompt
    assert answer_prompt.index("VÍ DỤ CÁCH TRÌNH BÀY") < answer_prompt.index("TÀI LIỆU:\n")


# --------------------------------------------------------------------- v8
def test_parent_child_index_and_max_pooling():
    docs = [
        Document(1, "CLB Lập Trình PTIT", "Giải thưởng", "Giải Nhất ICPC PTIT 2022; Giải Ba Chuyên Tin 2022; Huy chương Vàng Olympic Toán 2023."),
        Document(2, "CLB Multimedia PTIT", "Cơ cấu", "Các ban gồm Ban Nội dung, Ban Đối ngoại và Ban Sự kiện."),
    ]
    p = MyPipeline(llm=make_llm("Q: x", "x\nNGUON: 1"), embedder=FakeEmbedder(), parent_child=True)
    p.index(docs)
    assert list(p.child_parent) == [0, 0, 0, 1]  # đoạn danh sách tách theo ";" → 3 child
    pooled = p._to_parent(np.array([0.1, 0.9, 0.2, 0.5], dtype=np.float32))
    assert pooled.tolist() == pytest.approx([0.9, 0.5])  # điểm đoạn cha = child tốt nhất


def test_child_match_ranks_its_parent_first():
    """Câu hỏi chỉ khớp một ý nhỏ trong đoạn danh sách dài vẫn kéo được đoạn cha lên đầu."""
    docs = [
        Document(1, "CLB Lập Trình PTIT", "Giải thưởng",
                 "Giải Nhì Huawei ICT 2025; Giải Ba P-Innovation 2025; Quán quân Coding Fest 2025; Giải Nhất Tech4Good."),
        Document(2, "CLB Lập Trình PTIT", "Sự kiện", "Coding Fest là sự kiện thường niên của trường dành cho sinh viên."),
        Document(3, "CLB Multimedia PTIT", "Giới thiệu", "CLB Multimedia PTIT thành lập ngày 10/11/2012."),
    ]
    p = MyPipeline(llm=make_llm("Q: Ai là quán quân Coding Fest 2025?", "x"), embedder=FakeEmbedder(), parent_child=True)
    p.index(docs)
    ranked = p.retrieve("quan quan coding fest 2025", ["Ai là quán quân Coding Fest 2025?"])
    assert ranked[0].id == 1


def test_parent_child_off_by_default(corpus):
    p = MyPipeline(llm=make_llm("Q: x", "x"), embedder=FakeEmbedder())
    p.index(corpus)
    assert p.child_vectors is None and p.child_bm25 is None


# --------------------------------------------------------------------- v9
def test_generic_templates_opt_in_and_fact_free(corpus):
    from neorag.pipelines.pipeline import ANSWER_TEMPLATES

    llm = make_llm("Q: x", "Trả lời.\nNGUON: 1")
    p = MyPipeline(llm=llm, embedder=FakeEmbedder(), answer_templates=True)
    p.index(corpus)
    p.run("x")
    prompt = llm.calls[1][0]["content"]
    assert "MẪU TRÌNH BÀY THEO KIỂU CÂU HỎI" in prompt
    assert prompt.index("MẪU TRÌNH BÀY") < prompt.index("TÀI LIỆU:\n")
    # Mẫu chung không chứa dữ kiện thật (ngày, số liệu, tên CLB cụ thể) → không có gì để chép nhầm.
    assert not re.search(r"\d", re.sub(r"<[^>]*>", "", ANSWER_TEMPLATES))  # bỏ nhãn chỗ trống <nguồn 1>
    assert "Lập Trình" not in ANSWER_TEMPLATES and "Multimedia" not in ANSWER_TEMPLATES


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


# --------------------------------------------------------------------- v10
def test_v10_templates_only_for_accented_queries(corpus):
    from neorag.pipelines.pipeline import MyPipelineV10

    llm = make_llm("Q: x", "Trả lời.\nNGUON: 1")
    p = MyPipelineV10(llm=llm, embedder=FakeEmbedder())
    p.index(corpus)
    p.run("CLB Lập Trình PTIT thành lập khi nào?")
    p.run("clb lap trinh thanh lap khi nao")
    accented, unaccented = llm.calls[1][0]["content"], llm.calls[3][0]["content"]
    assert "MẪU TRÌNH BÀY" in accented
    assert "MẪU TRÌNH BÀY" not in unaccented


def test_exam_default_is_v10_and_v6_fallback_exists(corpus):
    """Khóa cấu hình bản thi Private: MyPipeline = v10; MyPipelineV6 = dự phòng không có mẫu chung."""
    from neorag.pipelines.pipeline import MyPipelineV6

    for cls, expect_templates in ((MyPipeline, True), (MyPipelineV6, False)):
        llm = make_llm("Q: x", "Trả lời.\nNGUON: 1")
        p = cls(llm=llm, embedder=FakeEmbedder())
        p.index(corpus)
        p.run("CLB Lập Trình PTIT thành lập khi nào?")
        assert ("MẪU TRÌNH BÀY" in llm.calls[1][0]["content"]) is expect_templates


# --------------------------------------------------------------------- v11
class TableEmbedder(FakeEmbedder):
    """Embedding cố định theo nội dung, để biết chính xác đoạn nào giống đoạn nào."""

    TABLE = {"A": [1, 0, 0], "B": [0.9, 0.1, 0], "C": [0, 1, 0], "D": [0, 0, 1]}

    def embed(self, texts):
        out = []
        for t in texts:
            key = next((k for k in self.TABLE if t.startswith(k)), None)
            out.append(self.TABLE[key] if key else [0.5, 0.5, 0.5])
        v = np.asarray(out, dtype=np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def _v11_docs():
    return [Document(i, "CLB Lập Trình PTIT", "t", f"{k} đoạn {k}") for i, k in enumerate("ABCD", start=1)]


def test_v11_cited_first_then_most_similar():
    from neorag.pipelines.pipeline import MyPipelineV11

    p = MyPipelineV11(llm=make_llm("Q: x", "x\nNGUON: 1"), embedder=TableEmbedder())
    p.index(_v11_docs())
    docs = [p.by_id[i] for i in (3, 4, 1, 2)]  # thứ tự RRF giả định: C, D, A, B
    assert p.order_ids([1], docs) == [1, 2, 3, 4]  # A được cite → B (giống A nhất) lên ngay sau


def test_v11_without_citation_keeps_rrf_order_and_v10_unchanged():
    from neorag.pipelines.pipeline import MyPipelineV10, MyPipelineV11

    docs_ids = (3, 4, 1, 2)
    p = MyPipelineV11(llm=make_llm("Q: x", "x"), embedder=TableEmbedder())
    p.index(_v11_docs())
    docs = [p.by_id[i] for i in docs_ids]
    assert p.order_ids([], docs) == list(docs_ids)
    v10 = MyPipelineV10(llm=make_llm("Q: x", "x"), embedder=TableEmbedder())
    v10.index(_v11_docs())
    assert v10.order_ids([1], [v10.by_id[i] for i in docs_ids]) == [1, 3, 4, 2]


def test_v11_submits_same_id_set_as_v10():
    """v11 chỉ đổi THỨ TỰ, không đổi TẬP id → context judge dùng chấm Groundedness không đổi."""
    from neorag.pipelines.pipeline import MyPipelineV10, MyPipelineV11

    docs = [Document(i, "CLB", "t", f"đoạn văn số {i} về chủ đề {i % 4}") for i in range(1, 21)]
    a = MyPipelineV11(llm=make_llm("Q: x", "x"), embedder=FakeEmbedder())
    b = MyPipelineV10(llm=make_llm("Q: x", "x"), embedder=FakeEmbedder())
    a.index(docs)
    b.index(docs)
    ranked = a.retrieve("x", ["x"])
    for cited in ([ranked[12].id], [ranked[0].id, ranked[5].id], []):
        assert set(a.order_ids(cited, ranked)) == set(b.order_ids(cited, ranked))


# --------------------------------------------------------------------- v12
def _v12_llm(answer: str, evidence: str = "ID:", fix: str = "GIỮ") -> FakeLLM:
    def respond(m):
        c = m[0]["content"]
        if "TÀI LIỆU:" in c:
            return answer
        if c.startswith("Câu hỏi:") and "Liệt kê id" in c:
            return evidence
        if "Bản nháp:" in c:
            return fix
        return "Q: x"

    return FakeLLM(respond)


def test_v12_pc_filler_cited_first_then_pc_rank_plus_cosine():
    from neorag.pipelines.pipeline import MyPipelineV12

    p = MyPipelineV12(llm=_v12_llm("x"), embedder=TableEmbedder())
    p.index(_v11_docs())
    docs = [p.by_id[i] for i in (3, 4, 1, 2)]  # tập id v10: C, D, A, B
    pc = np.array([0.1, 0.2, 0.9, 0.5], dtype=np.float32)  # hạng parent-child: C(3), D(4), B(2), A(1)
    # Không cite → theo hạng parent-child.
    assert p.order_ids_pc([], docs, pc) == [3, 4, 2, 1]
    # Cite A: C (hạng 1 parent-child: 1/1) vẫn đầu; B (1/3 + 0.5·cos≈0.99) vượt D (1/2 + 0).
    assert p.order_ids_pc([1], docs, pc) == [1, 3, 2, 4]
    # Bằng chứng (evidence) lên ngay sau id được cite; id ngoài tập v10 bị bỏ qua.
    assert p.order_ids_pc([1], docs, pc, evidence=[4, 99]) == [1, 4, 3, 2]


def test_v12_same_answer_and_id_set_as_v10_and_main_retrieval_unchanged():
    from neorag.pipelines.pipeline import MyPipelineV10, MyPipelineV12

    docs = [Document(i, "CLB", "t", f"đoạn văn số {i} về chủ đề {i % 4}. Câu thứ hai dài hơn {i}.")
            for i in range(1, 21)]
    a = MyPipelineV12(llm=_v12_llm("Trả lời.\nNGUON: 5", evidence="ID: 7, 5, 999"), embedder=FakeEmbedder())
    b = MyPipelineV10(llm=_v12_llm("Trả lời.\nNGUON: 5"), embedder=FakeEmbedder())
    a.index(docs)
    b.index(docs)
    assert [d.id for d in a.retrieve("x", ["x"])] == [d.id for d in b.retrieve("x", ["x"])]
    ra, rb = a.run("đoạn văn số 5"), b.run("đoạn văn số 5")
    assert ra.answer == rb.answer
    assert set(ra.retrieved_ids) == set(rb.retrieved_ids)
    assert ra.retrieved_ids[0] == 5
    if 7 in rb.retrieved_ids:
        assert ra.retrieved_ids[1] == 7


def test_v12_evidence_call_skipped_without_citation():
    from neorag.pipelines.pipeline import MyPipelineV12

    llm = _v12_llm("Tài liệu hiện không có thông tin về X.")
    p = MyPipelineV12(llm=llm, embedder=FakeEmbedder())
    p.index(_v11_docs())
    p.run("x")
    assert not any("Liệt kê id" in c[0]["content"] for c in llm.calls)


def test_answer_first_fix_off_by_default_and_gated():
    from neorag.pipelines.pipeline import MyPipelineV12

    assert MyPipeline(llm=FakeLLM(), embedder=FakeEmbedder()).answer_first_fix is False
    assert MyPipelineV12(llm=FakeLLM(), embedder=FakeEmbedder()).answer_first_fix is False
    llm = _v12_llm("x", fix="Đợt tuyển chưa mở.")
    p = MyPipeline(llm=llm, embedder=FakeEmbedder(), answer_first_fix=True)
    p.index(_v11_docs())
    docs = list(p.docs)
    # Không mở đầu bằng từ chối → không gọi LLM.
    assert p.fix_refusal_opener("q", "Có, CLB có.", [1], docs) == "Có, CLB có."
    # Từ chối không có câu "Tài liệu chỉ nêu" → giữ nguyên.
    refusal = "Tài liệu hiện không có thông tin về X."
    assert p.fix_refusal_opener("q", refusal, [1], docs) == refusal
    assert llm.calls == []
    draft = "Tài liệu hiện không có thông tin về lịch. Tài liệu chỉ nêu đợt tuyển chưa mở."
    assert p.fix_refusal_opener("q", draft, [1], docs) == "Đợt tuyển chưa mở."
    keep = MyPipeline(llm=_v12_llm("x", fix="GIỮ"), embedder=FakeEmbedder(), answer_first_fix=True)
    assert keep.fix_refusal_opener("q", draft, [1], docs) == draft


# ---------------------------------------------------------------- v11-audit
def _audit_llm(rewrite: str, draft: str, audit_out: str) -> FakeLLM:
    def respond(m):
        c = m[0]["content"]
        if "BẢN NHÁP:" in c:
            return audit_out
        if "TÀI LIỆU:" in c:
            return draft
        return rewrite

    return FakeLLM(respond)


def test_v11_audit_loadable_via_cli_spec():
    from neorag.pipelines.pipeline import MyPipelineV11Audit

    p = load_pipeline(
        "neorag.pipelines.pipeline:MyPipelineV11Audit", llm=FakeLLM(), embedder=FakeEmbedder()
    )
    assert isinstance(p, MyPipelineV11Audit) and isinstance(p, RAGPipeline)


def test_v11_audit_keeps_draft_and_id_pool_when_giu(corpus):
    from neorag.pipelines.pipeline import MyPipelineV11, MyPipelineV11Audit

    draft = "CLB Lập Trình PTIT được thành lập ngày 09/10/2011.\nNGUON: 1"
    p_audit = MyPipelineV11Audit(
        llm=_audit_llm("Q: CLB thành lập khi nào?", draft, "TRA_LOI: GIỮ\nNGUON: 1"),
        embedder=TableEmbedder(),
    )
    p_v11 = MyPipelineV11(
        llm=make_llm("Q: CLB thành lập khi nào?", draft),
        embedder=TableEmbedder(),
    )
    p_audit.index(corpus)
    p_v11.index(corpus)
    assert [d.id for d in p_audit.retrieve("clb thanh lap khi nao", ["CLB thành lập khi nào?"])] == [
        d.id for d in p_v11.retrieve("clb thanh lap khi nao", ["CLB thành lập khi nào?"])
    ]
    out_a = p_audit.run("clb thanh lap khi nao")
    out_11 = p_v11.run("clb thanh lap khi nao")
    assert out_a.answer == out_11.answer
    assert set(out_a.retrieved_ids) == set(out_11.retrieved_ids)


def test_v11_audit_fixes_draft_and_promotes_sibling_citation():
    from neorag.pipelines.pipeline import MyPipelineV11Audit

    docs = [
        Document(i, "CLB Lập Trình PTIT", "Khác", f"đoạn văn phụ số {i} dài hơn mười lăm ký tự.")
        for i in range(1, 18)
    ] + [
        Document(18, "CLB Lập Trình PTIT", "Giải thưởng", "Giải Nhì AICOVIDVN-115M Challenge năm 2021."),
        Document(19, "CLB Lập Trình PTIT", "Giải thưởng", "Giải Nhì Kalapa Challenge Medical Question Answering 2023."),
    ]
    llm = _audit_llm(
        "Q: CLB đạt giải y tế nào?",
        "Có, CLB Lập Trình PTIT đạt Giải Nhì AICOVIDVN-115M Challenge.\nNGUON: 18",
        "TRA_LOI: Có, CLB Lập Trình PTIT đạt Giải Nhì AICOVIDVN-115M Challenge và Giải Nhì Kalapa Challenge Medical Question Answering 2023.\nNGUON: 18, 19",
    )
    p = MyPipelineV11Audit(llm=llm, embedder=FakeEmbedder())
    p.index(docs)
    # Giả lập retrieve trả 15 đoạn có 18 nhưng thiếu 19 → _build_audit_docs kéo 19 vào nhờ cùng (club, topic)
    audit_docs = p._build_audit_docs("y te", ["CLB đạt giải y tế nào?"], docs[:14] + [docs[17]], [18])
    assert 19 in {d.id for d in audit_docs}
    ans, cited = p.audit("y te", "CLB đạt giải y tế nào?", audit_docs, "Nháp.", [18])
    assert "Kalapa Challenge" in ans and cited == [18, 19]


def test_v11_audit_api_error_and_placeholder_guard_fallback_to_draft(corpus):
    from neorag.pipelines.pipeline import MyPipelineV11Audit

    def broken_audit(m):
        c = m[0]["content"]
        if "BẢN NHÁP:" in c:
            raise RuntimeError("gateway timeout")
        if "TÀI LIỆU:" in c:
            return "Bản nháp chuẩn.\nNGUON: 1"
        return "Q: x"

    p = MyPipelineV11Audit(llm=FakeLLM(broken_audit), embedder=FakeEmbedder())
    p.index(corpus)
    assert p.run("x").answer == "Bản nháp chuẩn."

    # Sót <...> trong câu trả lời audit → giữ nguyên bản nháp
    ans, cited = p._parse_audit("TRA_LOI: Theo <nguồn kèm thời gian>, CLB A.\nNGUON: 2", corpus, "Nháp gốc.", [1])
    assert ans == "Nháp gốc." and cited == [2]
