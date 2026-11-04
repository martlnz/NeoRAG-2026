# update_v4 — (v3.1) Chống lỗi cho ngày Private Test, không đổi prompt nên không đổi điểm

- **File:** `src/neorag/pipelines/pipeline.py` · class `MyPipeline(RAGPipeline)`
- **Lệnh chấm:** `neorag run train --pipeline neorag.pipelines.pipeline:MyPipeline`
- **File nộp Public Test:** `experiments/submissions/v4_public_test_submission.zip`. Nội dung (câu trả lời và `retrieved_ids`) **giống hệt v3**; chỉ khác `latency_ms`.
- **Kiểm tra:** `pytest` 61/61 đạt (thêm 3 test chống lỗi), `ruff check` sạch.

## 1. Đã làm gì (so với v3)

Triển khai mục 1–2 trong đề xuất của chuyên gia ở `update_v3.md`. Theo yêu cầu, mục 3 (`run_stream` cho chatbot) để lại sau, ưu tiên phần RAG chấm điểm.

1. **Câu trả lời rỗng thì gọi lại một lần.**
   - Trước đây, nếu LLM trả về chuỗi rỗng (hoặc chỉ có dòng NGUON), câu rỗng đó bị lưu vào cache. Pipeline sẽ nộp câu từ chối dự phòng, tức từ chối oan, và chạy lại cũng không sửa được.
   - Giờ pipeline gọi lại một lần với prompt khác đúng 1 ký tự (`\n`) để có khóa cache mới.
   - Lời gọi chính vẫn giữ nguyên, nên cache hiện có vẫn trúng.
2. **Bước rewrite gặp lỗi API thì dùng câu gốc.** Rewrite chỉ là bước phụ, nên không còn làm dừng cả lượt chạy.
3. **Lỗi API ở bước sinh câu trả lời được cố ý để lộ ra**, không bị thay ngầm bằng câu từ chối. Khi đó, chạy lại lệnh sẽ lấy các câu đã xong từ cache và chỉ gọi lại câu bị lỗi.
4. Tách phần đọc NGUON và làm sạch markdown thành hàm riêng `_parse_answer`.

Test mới trong `tests/test_my_pipeline.py`:
- `test_empty_answer_is_retried_once`: câu rỗng được gọi lại, tổng cộng 3 lời gọi, prompt lần gọi lại khác lần đầu.
- `test_rewrite_api_error_falls_back_to_original_query`
- `test_answer_api_error_is_not_silently_swallowed`

## 2. Điểm số (copy từ `outputs/eval_train_pred/report.md`)

- Retrieval score: **0.9807**
- Generation score: **0.8664**
- Benchmark score: **0.9235**

Điểm không đổi so với v3. Lần chạy lấy 100% từ cache (12.5 giây cho train, 5.9 giây cho public), nên không tốn ngân sách. Bảng metric chi tiết giống `update_v3.md`.

## 3. Phân tích

### Đã làm được
- Đóng được 2 rủi ro có thể làm mất điểm khi code đã đóng băng: câu rỗng nằm trong cache, và lỗi API ở bước rewrite.
- Xác nhận bằng so sánh trực tiếp: file public của v4 giống hệt v3 trên cả 42 câu.

### Chưa làm được và giới hạn
- Lỗi API ở bước sinh câu trả lời (sau 5 lần retry của `LLMClient`) vẫn làm dừng cả lượt. Cách xử lý là chạy lại lệnh.
- Dư địa tăng điểm còn khoảng +0.5–0.8 điểm, nhỏ hơn dao động ±1 của judge (xem `update_v3.md`).

## 4. Checklist ngày Private Test (08/11)

1. Kéo `data/private_test/questions.jsonl` từ repo BTC về đúng đường dẫn đó.
2. Chạy `neorag run private_test --pipeline neorag.pipelines.pipeline:MyPipeline`.
3. Nếu gặp lỗi API: **chạy lại đúng lệnh trên, không xóa `.cache/`**.
4. Kiểm tra dòng `Pipeline` trong khung thông tin là `neorag.pipelines.pipeline:MyPipeline`, và dòng "hợp lệ" ở bước ghi file.
5. Nộp `outputs/private_test_submission.zip`. Không sửa code trong ngày thi.
