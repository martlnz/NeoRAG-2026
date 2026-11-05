# update_v6 — Từ chối chuẩn hóa + mở đầu "Có, … / Theo …" + rewrite có câu con tự đứng được

- **File:** `src/neorag/pipelines/pipeline.py` · class `MyPipeline(RAGPipeline)`
- **Lệnh chấm:** `neorag run train` và `neorag run train_informal` (cùng `--pipeline neorag.pipelines.pipeline:MyPipeline`)
- **File nộp Public Test:** `experiments/submissions/v6_public_test_submission.zip`
- **Kiểm tra:** `pytest` 68/68 đạt (thêm 3 test cho v6), `ruff check` sạch

## 1. Đã làm gì (so với v5)

Triển khai cả 3 mục trong spec v6 của chuyên gia (`update_v5.md`, mục 4). Mục tiêu do người dùng đặt: public trên 90.

1. **Định dạng từ chối**:
   - Khi chính điều được hỏi không có trong tài liệu, câu đầu bắt đầu đúng bằng "Tài liệu hiện không có thông tin về …".
   - **Không suy ra khẳng định phủ định** chỉ vì tài liệu không nhắc tới. Đây là lỗi ở train-0103 ("S2B không có fanpage") và public-0002.
   - Có thể thêm tối đa một câu "Tài liệu chỉ nêu …".
   - Nếu chỉ một phần không có thì trả lời phần có trước, rồi nói "tài liệu không ghi …".
2. **Cách mở đầu câu trả lời**:
   - Câu hỏi có/không: "Có, … / Không, … / Đúng, … / Chưa, …". Dùng dấu phẩy, không dùng dấu chấm, để không bị tách thành một câu riêng khi chấm.
   - Khi tài liệu tự ghi nguồn: "Theo <nguồn kèm năm/ngày, chép nguyên văn>, …". Có 30% đáp án chuẩn mở đầu kiểu này.
   - Câu thứ hai chỉ viết khi câu hỏi có ý thứ hai.
3. **Rewrite**:
   - Câu con phải tự đứng được, nêu tên CLB. Câu con chứa "CLB đó / này / ấy / ở đó" bị lọc bằng regex `DANGLING`; câu đầu luôn được giữ.
   - Câu "thích X thì nên vào CLB nào": sinh một câu con cho mỗi CLB phù hợp.
   - Câu "có ban nào lo việc X": thêm câu con "CLB … có những ban nào và mỗi ban phụ trách gì?". Mục tiêu là public-0003 (doc 94).

Sự cố trong lúc chạy: lần chấm `train_informal` đầu tiên bị dừng vì **không kết nối được gateway** (lỗi mạng tạm thời, gateway vẫn hoạt động ngay sau đó). Chạy lại đúng lệnh thì các câu đã xong được lấy từ cache. Sự cố này xác nhận checklist ngày thi: lỗi gateway thì chạy lại, không xóa `.cache/`.

## 2. Điểm số (copy từ `outputs/eval_*/report.md`)

### train

| K | hit@k | precision@k | recall@k | f1@k | map@k | mrr@k | ndcg@k |
|---|---|---|---|---|---|---|---|
| 5 | 1.0000 | 0.3033 | 0.9679 | 0.4461 | 0.9584 | 1.0000 | 0.9716 |

| string_presence | rouge_l | bleu_4 | groundedness | response_relevancy | noise_sensitivity | abstention |
|---|---|---|---|---|---|---|
| 0.9099 | 0.6624 | 0.5294 | 0.9957 | 0.6651 | 0.0478 | 1.0000 |

- Retrieval score: **0.9798**
- Generation score: **0.8642**
- Benchmark score: **0.9220**

### train_informal

| K | hit@k | precision@k | recall@k | f1@k | map@k | mrr@k | ndcg@k |
|---|---|---|---|---|---|---|---|
| 5 | 1.0000 | 0.3250 | 0.9783 | 0.4715 | 0.9714 | 1.0000 | 0.9809 |

| string_presence | rouge_l | bleu_4 | groundedness | response_relevancy | noise_sensitivity | abstention |
|---|---|---|---|---|---|---|
| 0.9051 | 0.6602 | 0.5084 | 0.9888 | 0.4717 | 0.0449 | 1.0000 |

- Retrieval score: **0.9864**
- Generation score: **0.8301**
- Benchmark score: **0.9083**

### Điểm Public Test (trang thi aichallenge)

_(chờ nộp)_

### So với các bản trước

| | v4 | v5 | v6 | Δ v6−v5 |
|---|---|---|---|---|
| **train** | 0.9235 | 0.9173 | **0.9220** | +0.0047 |
| **train_informal** | 0.8823 | 0.9026 | **0.9083** | +0.0057 |
| Trung bình hai bộ | 0.9029 | 0.9100 | **0.9152** | +0.0052 |
| Public | 84.94 | ≈87 | _chờ_ | |
| Retrieval train / informal | 0.981 / 0.962 | 0.984 / 0.980 | 0.980 / **0.986** | |
| ROUGE-L train / informal | 0.644 / 0.629 | 0.644 / 0.624 | **0.662 / 0.660** | +0.018 / +0.036 |
| String Presence train / informal | 0.935 / 0.890 | 0.936 / 0.911 | 0.910 / 0.905 | **−0.026 / −0.006** |
| Abstention train / informal | 1.0 / 0.909 | 0.909 / 1.0 | **1.0 / 1.0** | |
| Noise ↓ train / informal | 0.044 / 0.062 | 0.046 / 0.056 | 0.048 / **0.045** | |

## 3. Phân tích

### Theo loại câu hỏi (train)

| Loại câu hỏi | n | recall@5 | mrr@5 | ndcg@5 | string_presence | rouge_l | groundedness | response_relevancy | noise_sensitivity | abstention |
|---|---|---|---|---|---|---|---|---|---|---|
| không dấu/viết tắt | 26 | 0.962 | 1.000 | 0.966 | 0.962 | 0.719 | 1.000 | 0.471 | 0.077 | — |
| diễn đạt khác | 22 | 0.962 | 1.000 | 0.969 | 0.955 | 0.685 | 1.000 | 0.685 | 0.045 | — |
| nhiều đoạn | 20 | 0.938 | 1.000 | 0.946 | 0.840 | 0.609 | 1.000 | 0.688 | 0.075 | — |
| so sánh | 20 | 0.983 | 1.000 | 0.990 | 0.908 | 0.663 | 1.000 | 0.743 | 0.025 | — |
| thời gian | 12 | 1.000 | 1.000 | 1.000 | 0.833 | 0.555 | 0.958 | 0.712 | 0.042 | — |
| không có đáp án | 11 | 1.000 | 1.000 | 0.967 | — | — | — | — | — | 1.000 |
| trực tiếp | 8 | 1.000 | 1.000 | 1.000 | 0.958 | 0.805 | 1.000 | 0.858 | 0.000 | — |
| thiếu dữ liệu | 5 | 1.000 | 1.000 | 1.000 | 0.800 | 0.527 | 1.000 | 0.714 | 0.000 | — |
| mâu thuẫn nguồn | 2 | 0.750 | 1.000 | 0.807 | 1.000 | 0.616 | 1.000 | 0.781 | 0.000 | — |

### Đã làm được
1. **Abstention = 1.0 trên cả hai bộ.** Định dạng từ chối mới sửa được train-0103: không còn khẳng định "S2B không có fanpage".
2. **ROUGE-L tăng rõ nhất từ v2 đến giờ** (train +0.018, informal +0.036), nhờ mở đầu "Có, …" và "Theo <nguồn>", giống cách viết của đáp án chuẩn. Nhóm không dấu đạt ROUGE 0.719.
3. **Retrieval trên câu văn nói đạt cao nhất (0.986, MRR@5 = 1.0)**: câu con tự đứng được và câu con theo từng CLB giúp kéo đúng đoạn.
4. Tăng trên **cả hai bộ**, nên không phải do judge dao động trên một bộ.

### Chưa làm được và nguyên nhân
1. **String Presence giảm** (train 0.936 → 0.910). Có 20 câu SP < 1, nguyên nhân chính:
   - **Diễn đạt lại thay vì chép nguyên văn:** "3 thành viên" thay vì "3 người" (train-0025); "không ưu tiên" thay vì "không giúp ứng viên được ưu tiên" (train-0005); thiếu "tri ân" (train-0018).
   - **Câu nhiều ý bị rút gọn:** quy tắc "câu thứ hai chỉ khi có ý thứ hai" và giới hạn 60 từ làm mất các mục trong danh sách. Ví dụ train-0003 thiếu ProJApp và ProPixelGameJam; nhóm "nhiều đoạn" có SP 0.840.
   - Nhóm "thời gian" (0.833) và "thiếu dữ liệu" (0.800) thấp nhất.
2. Response Relevancy vẫn ở mức trần cố hữu (informal khoảng 0.47).

## 4. Hướng tiếp theo

**Theo đề xuất của người dùng:**
- (a) **Thử các phương pháp nâng cao** từng bị gạt vì "chưa thấy lợi trên train", vì chúng có thể bền hơn trên đề thật:
  - chunk theo câu (gom về đoạn cha bằng max-pooling);
  - cross-encoder `bge-reranker-v2-m3`;
  - HyDE;
  - embedding có header `club | topic`.
  Đo bằng `experiments/retrieval_ablation.py` trên cả hai bộ.
- (b) **v7: few-shot từ đáp án train** để học cách diễn đạt và mức độ chi tiết (chép nguyên văn, "Có. Theo …", "Dữ liệu gốc chỉ nêu …", "Các đoạn ghi khác nhau: …"):
  - 6 ví dụ tĩnh, mỗi kiểu trả lời một câu: train-0085, 0028, 0010, 0050, 0069, 0101.
  - Để đo trung thực, báo thêm điểm **sau khi loại 6 câu ví dụ và 6 bản informal tương ứng**.
  - Mục tiêu: lấy lại String Presence mà vẫn giữ ROUGE-L.
