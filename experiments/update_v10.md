# update_v10 — v6 + mẫu trả lời chung CHỈ cho câu hỏi có dấu

- **Class:** `MyPipelineV10` trong `src/neorag/pipelines/pipeline.py` (`answer_templates="accented"`).
- **Lệnh:** `neorag run <split> --pipeline neorag.pipelines.pipeline:MyPipelineV10`
- **File nộp Public Test:** `experiments/submissions/v10_public_test_submission.zip`
- **Kiểm tra:** `pytest` 76/76 đạt (thêm test: câu có dấu nhận mẫu, câu không dấu giữ prompt v6), `ruff check` sạch
- **Chi phí:** gần như **0**. Mọi câu trả lời và điểm judge đều lấy từ cache: train 2.7 giây, informal 2.1 giây, public 4.4 giây.

## 1. Vì sao làm v10: "càng sửa càng tệ" và cách sửa

- v7, v8, v9 mỗi bản đổi một thứ **áp dụng cho mọi câu hỏi**. Thay đổi đó có lợi cho câu viết chuẩn nhưng gây hại cho **câu văn nói / không dấu**, mà nhóm này chiếm phần lớn bộ public.
- Qua ba bản, `train_informal` dự đoán đúng chiều điểm public: bản nào thua v6 trên informal thì cũng thua trên public (v8 88.15, v9 86.05, so với v6 88.43).
- **Nguyên tắc của v10:** giữ nguyên v6 cho nhóm câu mà v6 đang tốt nhất, và chỉ đổi ở nhóm câu mà thay đổi đã **đo được là có lợi**.
  - Câu hỏi gốc **không dấu** (`is_unaccented`): prompt **y hệt v6**.
  - Câu hỏi gốc **có dấu**: thêm 8 mẫu trả lời chung của v9. Trên train, nhóm này được lợi rõ: String Presence +0.009, ROUGE +0.010, Noise 0.048 → 0.026.

## 2. Điểm số (copy từ `outputs/eval_*/report.md`)

### train
- Retrieval score: **0.9836**
- Generation score: **0.8701**
- Benchmark score: **0.9269**

### train_informal
- Retrieval score: **0.9864**
- Generation score: **0.8301**
- Benchmark score: **0.9083** (giống hệt v6 từng câu, vì mọi câu đều không dấu)

### Điểm Public Test (trang thi aichallenge)

**85.53** (v6: 88.43, v9: 86.05, v8: 88.15).

### So với các bản trước

| | v6 | v9 | **v10** |
|---|---|---|---|
| train | 0.9220 | **0.9271** | 0.9269 |
| train_informal | **0.9083** | 0.9031 | **0.9083** |
| Trung bình hai bộ | 0.9152 | 0.9151 | **0.9176** |
| Public | **88.43** | 86.05 | 85.53 |

Cách ghép trên public, đã kiểm tra bằng code: **22/42 câu có dấu** dùng đúng câu trả lời của v9, **20/42 câu không dấu** dùng đúng câu trả lời của v6.

## 3. Phân tích

### Đã làm được
- **Không có nhóm câu nào bị kém đi so với v6** trên dev: informal giữ nguyên, train tăng +0.0049. Đây là cải tiến "chỉ cộng, không trừ".
- Không tốn ngân sách, vì tái sử dụng toàn bộ kết quả đã cache của v6 và v9.

### Rủi ro / giới hạn
- Public của v9 thấp hơn v6 2.4 điểm. Phần giảm này có thể đến một phần từ 22 câu có dấu chứ không chỉ từ câu văn nói, nhưng không đo riêng được vì judge public chấm tổng và nhiễu ±3. Kỳ vọng v10 trên public **ngang hoặc nhỉnh hơn v6**, không chắc chắn.
- Ranh giới "có dấu / không dấu" là xấp xỉ cho "câu viết chuẩn / văn nói". Câu có dấu nhưng mơ hồ vẫn nhận mẫu chung.

## 4. Quyết định: ❌ không dùng v10, **chốt v6 cho Private**

- v10 thấp hơn v6 khoảng 2.9 điểm trên public, dù v10 tốt hơn hoặc bằng v6 trên mọi bộ dev.
- Hai bản có mẫu trả lời chung (v9 86.05, v10 85.53) đều thấp hơn v6 khoảng 2.5–3 điểm. Đây là xu hướng lặp lại, không chỉ là một lần judge dao động.
- Kết luận: **mẫu chung gây hại cả với câu có dấu của public.** Câu có dấu trên public vẫn mang giọng hội thoại ("Em chưa có máy tính…", "…không ạ?"), khác với câu viết chuẩn trong train. Vì vậy bộ dev, kể cả `train_informal`, không đại diện đủ cho phần này.

### Tổng kết sau v6 (v7–v10)

| Bản | Thay đổi so với v6 | Dev (train / informal) | Public |
|---|---|---|---|
| **v6** | — | 0.9220 / 0.9083 | **88.43** |
| v7 | few-shot thật | thấp hơn (sau khi loại câu few-shot) | — |
| v8 | parent-child + từ chối 1 câu | 0.9229 / 0.8945 | 88.15 |
| v9 | mẫu chung mọi câu | 0.9271 / 0.9031 | 86.05 |
| v10 | mẫu chung cho câu có dấu | 0.9269 / 0.9083 | 85.53 |

**Bài học:**
1. Khi dev đã sát trần (khoảng 0.92), những mức tăng 0.005 trên dev không đáng tin để chuyển sang public. Phân phối câu hỏi public khác dev, và judge public dao động khoảng ±3.
2. Các thay đổi về *văn phong câu trả lời* (few-shot, mẫu chung) đặc biệt rủi ro, vì judge chấm Noise và Relevancy theo câu.
3. Đã đến lúc **đóng băng v6**, giữ ngân sách (khoảng 1.8 USD) cho ngày 08/11, và dồn thời gian cho Demo và pitching.
