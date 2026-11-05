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

## 5. Kết quả nộp nhiều lần và quyết định cuối cùng

| Bản | Các lần nộp public | Trung bình | Số lần |
|---|---|---|---|
| v6 | 88.43, 88.39, 85.48 | **87.43** | 3 |
| v10 | 85.53, 88.51 | **87.02** | 2 |
| v9 | 86.05 | 86.05 | 1 |

- Chênh lệch v6 và v10 là **0.41 điểm**, nhỏ hơn rất nhiều so với độ nhiễu (cùng một file lệch tới 3 điểm). Bản thân v6 cũng có lần chỉ được 85.48. Trên public, hai bản **không phân biệt được**.
- Theo quy tắc đã đặt trước (chênh dưới 1.5 điểm thì theo dev): **chọn v10**, vì trên dev v10 tốt hơn hoặc bằng v6.
- **Đã đặt v10 làm mặc định của `MyPipeline`** để lệnh thi không đổi. Đã kiểm tra lại từ cache:
  - train 0.9269, informal 0.9083;
  - file public **giống hệt** `v10_public_test_submission.zip`.
- **Phương án dự phòng:** `--pipeline neorag.pipelines.pipeline:MyPipelineV6`.
- Có test khóa cấu hình bản thi: `test_exam_default_is_v10_and_v6_fallback_exists` (77/77 test đạt).

## 6. Có nên làm v11? Chuyên gia đánh giá kỹ thuật nâng cao (đề xuất của người dùng)

Chuyên gia tái tạo được **nhãn judge cho từng câu của v10** từ cache, đủ 226 câu hỏi, và tính lại khớp report. Sau đó mô phỏng từng kỹ thuật **không gọi API**.

**Hiện trạng v10 theo nhãn từng câu:**

| | train | informal |
|---|---|---|
| Số câu văn bị chấm `incorrect` (Noise) | 5/163 | 8/148 |
| Số câu văn không `supported` (Groundedness) | 2 | 2 |
| Abstention | 11/11 | 11/11 |
| Câu hỏi thiếu key fact (SP < 1) | 19 | 17 |

**Mô phỏng các kỹ thuật:**

| Kỹ thuật | Ước lượng benchmark (train / informal) | Kết luận |
|---|---|---|
| Tự kiểm và cắt câu không có căn cứ (+1 lời gọi LLM) | trần dùng nhãn gold +0.0009 / +0.0017; SP tụt | Không làm |
| Cắt câu "Tài liệu không ghi/chỉ nêu…" ở cuối | generation −0.0002 / −0.0025 | Không làm |
| Kiểm tra riêng trước khi từ chối | 0 (đã 11/11) | Không làm |
| Định tuyến / chọn đáp án v6 hay v9 cho từng câu | −0.0010 / +0.0013 (tín hiệu gần như tung đồng xu) | Không làm |
| Sparse / multi-vector, top_k thích ứng | trần +0.008 (retrieval đã 0.984–0.986) | Không làm |
| Trích nguyên văn / liệt kê đủ | trần khoảng +0.003; rủi ro cao (v7) | Không làm |
| **Xếp lại id nộp**: id được cite đứng trước, sau đó các đoạn còn lại xếp theo độ giống với đoạn được cite | **+0.0014 / +0.0008** (retrieval 0.9864 / 0.9879) | Ứng viên duy nhất chỉ cộng không trừ; lợi khoảng +0.1 điểm |

**Kết luận: đóng băng v10.**
- Kỹ thuật tốt nhất chỉ thêm khoảng **+0.1 điểm**, trong khi cùng một file nộp lại lệch tới ±3 điểm.
- Các thay đổi về văn phong câu trả lời đều từng làm hại câu văn nói.
- Giữ ngân sách (khoảng 1.8 USD) cho ngày Private.

**Quy tắc ngày Private (`docs/submission.md`): lần nộp gần nhất được dùng để tính điểm.** Không thể chọn lần cao nhất trong nhiều lần nộp. Vì vậy:
- Chạy một lần, kiểm tra file (hợp lệ, 42 câu, không có câu rỗng, dòng `Pipeline` đúng), rồi nộp một lần.
- Các lượt còn lại chỉ dùng khi lần đầu bị Failed.
