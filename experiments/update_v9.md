# update_v9 — Mẫu trả lời chung (generic templates) thay cho few-shot thật

- **Class:** `MyPipelineV9` trong `src/neorag/pipelines/pipeline.py`, tức v6 cộng `answer_templates=True`.
  - `MyPipeline` mặc định vẫn là **v6**.
- **Lệnh:** `neorag run <split> --pipeline neorag.pipelines.pipeline:MyPipelineV9`
- **File nộp Public Test:** `experiments/submissions/v9_public_test_submission.zip` (42/42 hợp lệ, không lọt dấu `<…>`)
- **Kiểm tra:** `pytest` 75/75 đạt, `ruff check` sạch
- **Trạng thái:** ⚖️ **Hòa v6 trên dev.** Người dùng yêu cầu nộp thử public để so sánh.

## 1. Đã làm gì (so với v6)

Theo đề xuất của người dùng: *"dựa vào các mẫu trả lời đặc trưng để tạo ra các mẫu chung, không sử dụng lại nguyên văn"*.

- Thống kê khuôn đáp án train:
  - 15 đáp án mở bằng "Tài liệu…", 14 bằng "Theo Phòng truyền thống…", 12 bằng "Có. …".
  - So sánh dùng "Ở X…; còn …".
  - Mâu thuẫn dùng "Các đoạn ghi khác nhau: …".
- Từ đó viết **8 mẫu chung có chỗ trống `<…>`**: nêu nguồn, có/không, so sánh, xếp hạng theo mốc, tính toán thời gian, thiếu một phần, mâu thuẫn nguồn, không có thông tin (đúng một câu).
- Mẫu **không chứa câu hỏi hay dữ kiện thật nào** (có test kiểm tra: không có số liệu, không có tên CLB). Vì vậy:
  - không rò rỉ đáp án, nên chấm trên **toàn bộ** train vẫn công bằng;
  - không có dữ kiện để LLM chép nhầm. Đây là nguyên nhân làm hỏng v7: với few-shot thật, Noise và Groundedness bị ảnh hưởng.

## 2. Điểm số (copy từ `outputs/eval_*/report.md`)

### train
| string_presence | rouge_l | bleu_4 | groundedness | response_relevancy | noise_sensitivity | abstention |
|---|---|---|---|---|---|---|
| 0.9193 | 0.6724 | 0.5226 | 0.9913 | 0.6659 | 0.0261 | 1.0000 |

- Retrieval score: **0.9837**
- Generation score: **0.8705**
- Benchmark score: **0.9271**

### train_informal
| string_presence | rouge_l | bleu_4 | groundedness | response_relevancy | noise_sensitivity | abstention |
|---|---|---|---|---|---|---|
| 0.8904 | 0.6377 | 0.5017 | 0.9831 | 0.4709 | 0.0674 | 1.0000 |

- Retrieval score: **0.9871**
- Generation score: **0.8191**
- Benchmark score: **0.9031**

### Điểm Public Test (trang thi aichallenge)

_(chờ nộp)_

### So với v6

| | v6 | v9 | Δ |
|---|---|---|---|
| train | 0.9220 | **0.9271** | +0.0051 |
| train_informal | **0.9083** | 0.9031 | −0.0052 |
| **Trung bình hai bộ** | **0.9152** | 0.9151 | ≈ 0 |
| train: String Presence / ROUGE-L | 0.910 / 0.662 | **0.919 / 0.672** | +0.009 / +0.010 |
| train: Noise ↓ | 0.048 | **0.026** | tốt hơn rõ |
| informal: String Presence / ROUGE-L | **0.905 / 0.660** | 0.890 / 0.638 | −0.015 / −0.022 |
| informal: Noise ↓ | **0.045** | 0.067 | tệ hơn |

## 3. Phân tích

### Đã làm được
- **Với câu viết chuẩn, mẫu chung làm đúng điều kỳ vọng.** String Presence và ROUGE-L tăng, Noise giảm gần một nửa (0.048 → 0.026). Abstention vẫn đạt 1.0. Đây là cách đúng để truyền "văn phong đáp án" mà không gặp tác dụng phụ của few-shot thật.

### Chưa làm được
- **Với câu văn nói, mẫu chung làm hại**: String Presence, ROUGE và Noise đều xấu đi. Giả thuyết: khi câu hỏi mơ hồ hoặc không dấu, LLM khó nhận ra câu hỏi thuộc kiểu nào, nên áp sai khuôn (ví dụ dùng khuôn "So sánh" hay "Theo <nguồn>" khi không phù hợp) và thêm ý không được hỏi.
- Tăng và giảm triệt tiêu nhau, nằm trong biên nhiễu, nên **không có bằng chứng v9 tốt hơn v6**.

## 4. Quyết định
- Giữ **v6 làm bản thi Private** (`MyPipeline` mặc định).
- Nộp thử `v9_public_test_submission.zip` theo yêu cầu, để có thêm một điểm dữ liệu public. Lưu ý mỗi lần nộp nhiễu khoảng ±3 điểm.
- Hướng tiếp theo nếu còn ngân sách: **chỉ bật mẫu chung khi câu hỏi gốc có dấu**, tức câu không phải văn nói (dùng `is_unaccented`). Cách này giữ được phần lợi trên train mà tránh phần hại trên informal. Có thể kiểm tra **gần như miễn phí**, vì câu trả lời của cả hai nhánh đều đã có trong cache.
