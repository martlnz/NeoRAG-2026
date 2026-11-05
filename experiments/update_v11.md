# update_v11 — Xếp lại id nộp theo đoạn được trích dẫn (câu trả lời giữ nguyên v10)

- **Class:** `MyPipelineV11` trong `src/neorag/pipelines/pipeline.py` (`cite_rerank=True`). **`MyPipeline` mặc định vẫn là v10.**
- **Lệnh:** `neorag run <split> --pipeline neorag.pipelines.pipeline:MyPipelineV11`
- **File nộp Public Test:** `experiments/submissions/v11_public_test_submission.zip`. Câu trả lời và tập id giống hệt v10; cả 42/42 câu đều đổi thứ tự id.
- **Kiểm tra:** `pytest` 80/80 đạt (thêm 3 test cho v11), `ruff check` sạch
- **Chi phí:** gần như 0. Câu trả lời lấy từ cache; chỉ một lần chấm judge thử ở bản nháp đầu.

## 1. Đã làm gì
Người dùng yêu cầu: *"làm bất cứ cách nào lên được điểm, làm bản v11 để tôi nộp"*. Trong đánh giá kỹ thuật ở `update_v10.md` mục 6, đây là ứng viên duy nhất **không đổi câu trả lời**.

- Ý tưởng: `relevant_ids` thường gồm nhiều đoạn trùng ý (ví dụ Phòng truyền thống và bài tuyển thành viên). Khi LLM chỉ trích dẫn một đoạn, đoạn trùng ý còn lại hay nằm ở hạng 6–10.
- Cách làm: id được trích dẫn đứng đầu, các id còn lại xếp theo **cosine lớn nhất (bge-m3) với các đoạn đã được trích dẫn**, thay vì theo thứ tự RRF. Câu không có trích dẫn thì giữ thứ tự như v10.

### Hai bản thử
| Bản | Tập ứng viên | train | informal |
|---|---|---|---|
| Nháp 1 (đã chấm judge đầy đủ) | top-15 context | **0.9290** (+0.0021) | 0.9071 (**−0.0012**) |
| **Bản cuối** (chỉ đo retrieval bằng `--no-judge`) | **chỉ tập 10 id của v10** | retrieval 0.9836 → **0.9865** (≈ +0.0015 benchmark) | retrieval 0.9864 → 0.9853 (≈ −0.0006 benchmark) |

- Bản nháp 1 làm **Groundedness informal giảm từ 0.989 xuống 0.966**. Judge chấm Groundedness trên các đoạn mình nộp, mà việc kéo đoạn hạng 11–15 vào đã đẩy ra các đoạn câu trả lời có dùng nhưng LLM không ghi trong NGUON.
- Bản cuối giới hạn việc xếp lại **bên trong đúng tập id của v10**, có test kiểm tra. Tập đoạn judge thấy không đổi, nên Groundedness không bị ảnh hưởng.

## 2. Đánh giá
- Lợi ước tính trung bình hai bộ chỉ khoảng **+0.0005 benchmark (+0.05 điểm)**.
- Trên câu văn nói, retrieval vẫn giảm nhẹ: vài đoạn đúng bị đẩy từ top-5 xuống hạng 6–10, vì đoạn "giống đoạn được trích dẫn nhất" không phải lúc nào cũng là đoạn đúng.
- Mức này **nằm trong biên nhiễu**: cùng một file nộp public lệch khoảng ±3 điểm.

### Điểm Public Test (trang thi aichallenge)
_(chờ nộp)_

## 3. Kết luận
- Đã cung cấp v11 để nộp theo yêu cầu.
- **Bản thi Private vẫn là `MyPipeline` (v10)**, vì v11 không thắng rõ trên bộ văn nói, mà public và private chủ yếu là câu văn nói.
- Nếu điểm public của v11 (lấy trung bình nhiều lần nộp) cao hơn v10 rõ rệt, khoảng trên 1.5 điểm, có thể chuyển. Lệnh chuyển: `--pipeline neorag.pipelines.pipeline:MyPipelineV11`.
- Kết luận chung sau v6–v11: **không còn cách hợp lệ nào cho mức tăng chắc chắn**. Các metric chính đã sát trần, và phần còn hở (Response Relevancy của câu văn nói) chỉ đẩy được bằng cách tác động judge, điều quy định cấm.
