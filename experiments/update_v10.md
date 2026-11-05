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

Hai lần nộp **cùng một file**: **85.53**, rồi **88.51** (lệch 2.98). Trung bình **87.02**. (v6: 88.43 với 1 lần nộp, v9: 86.05, v8: 88.15.)

### So với các bản trước

| | v6 | v9 | **v10** |
|---|---|---|---|
| train | 0.9220 | **0.9271** | 0.9269 |
| train_informal | **0.9083** | 0.9031 | **0.9083** |
| Trung bình hai bộ | 0.9152 | 0.9151 | **0.9176** |
| Public (từng lần nộp) | 88.43 | 86.05 | 85.53, 88.51 (TB 87.02) |

Cách ghép trên public, đã kiểm tra bằng code: **22/42 câu có dấu** dùng đúng câu trả lời của v9, **20/42 câu không dấu** dùng đúng câu trả lời của v6.

## 3. Phân tích

### Đã làm được
- **Không có nhóm câu nào bị kém đi so với v6** trên dev: informal giữ nguyên, train tăng +0.0049. Đây là cải tiến "chỉ cộng, không trừ".
- Không tốn ngân sách, vì tái sử dụng toàn bộ kết quả đã cache của v6 và v9.

### Rủi ro / giới hạn
- Public của v9 thấp hơn v6 2.4 điểm. Phần giảm này có thể đến một phần từ 22 câu có dấu chứ không chỉ từ câu văn nói, nhưng không đo riêng được vì judge public chấm tổng và nhiễu ±3. Kỳ vọng v10 trên public **ngang hoặc nhỉnh hơn v6**, không chắc chắn.
- Ranh giới "có dấu / không dấu" là xấp xỉ cho "câu viết chuẩn / văn nói". Câu có dấu nhưng mơ hồ vẫn nhận mẫu chung.

## 4. Quyết định (đã sửa sau lần nộp lại)

> **Đính chính:** sau lần nộp thứ nhất (85.53), mình đã kết luận "v9 và v10 thấp hơn v6 là xu hướng lặp lại" rồi chốt v6. Nộp lại **đúng file v10** được **88.51**. Kết luận trước đó **sai**: nó dựa trên một lần nộp duy nhất.

### Độ nhiễu thực tế của judge trên public (cùng một file)
| File | Các lần nộp | Lệch |
|---|---|---|
| v2 | 87.13, 84.23 | 2.90 |
| v10 | 85.53, 88.51 | 2.98 |

- Lệch khoảng **3 điểm giữa các lần nộp cùng một file**. Nguyên nhân chính: public chỉ có khoảng 3–4 câu không có đáp án, mỗi câu bị judge đổi nhãn là khoảng 2 điểm. Ngoài ra Noise và Groundedness được chấm theo từng câu.
- **Một lần nộp public không phân biệt được hai bản chênh nhau dưới khoảng 3 điểm.** Mọi so sánh v6–v10 trên public từ trước tới giờ đều nằm trong biên này.

### Quyết định
- Thước đo chính là **dev (train + train_informal)**. Trên dev, v10 tốt hơn hoặc bằng v6: train 0.9269 so với 0.9220, informal bằng nhau ở 0.9083.
- Để quyết định cho chắc: nộp **v6 thêm 2 lần** và **v10 thêm 1 lần**, rồi so trung bình mỗi bản (khoảng 3 lần nộp).
  - Nếu chênh lệch dưới khoảng 1.5 điểm: chọn **v10**, vì dev nghiêng về v10.
  - Nếu v6 hơn rõ: giữ v6.
- Lệnh chuyển bản thi sang v10 nếu được chọn: dùng `--pipeline neorag.pipelines.pipeline:MyPipelineV10`, hoặc đổi mặc định `answer_templates="accented"` trong `MyPipeline`.

### Bài học
1. **Không kết luận từ một lần nộp public.** Cần lấy trung bình nhiều lần, hoặc dựa vào dev.
2. Bộ dev có nhiều câu hơn (226 so với 42) và cache judge cố định, nên ổn định hơn public để so sánh các bản.
