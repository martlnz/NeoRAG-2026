# update_v3 — Rewrite có cấu trúc + NGUON đầy đủ + chép nguyên văn cụm then chốt

- **File:** `src/neorag/pipelines/pipeline.py` · class `MyPipeline(RAGPipeline)`
- **Lệnh chấm:** `neorag run train --pipeline neorag.pipelines.pipeline:MyPipeline`
- **File nộp Public Test:** `experiments/submissions/v3_public_test_submission.zip`
- **Kiểm tra trước khi chấm:** `pytest` (58/58 đạt, gồm 16 test mới trong `tests/test_my_pipeline.py`) và `ruff check` (sạch)

## 1. Đã làm gì (so với v2)

Triển khai mục 1–3 trong kế hoạch của chuyên gia ở `update_v2.md`:

1. **Rewrite có cấu trúc**:
   - Mỗi câu hỏi in trên một dòng bắt đầu bằng `Q: `; pipeline chỉ đọc những dòng này.
   - Lớp dự phòng: regex `META` lọc các dòng meta ("chỉ có một ý", "không cần câu hỏi con", ...) và bỏ tiền tố "Câu viết lại:". Không còn dòng hợp lệ nào thì dùng câu gốc.
   - Câu so sánh hoặc xếp hạng nhiều CLB: mỗi câu con hỏi đúng thuộc tính đó cho từng CLB.
   - Không nhắc tới CLB mà người hỏi không nêu.
2. **NGUON ghi mọi đoạn chứa thông tin trả lời** (kể cả đoạn trùng ý, xếp theo mức liên quan), thay vì chỉ ghi đoạn đã dùng.
3. **Prompt sinh câu trả lời**:
   - Chép nguyên văn cụm then chốt: số kèm đơn vị, tên giải / ban / đội / sự kiện, biệt danh tiếng Anh của ban.
   - Nêu link / email / website khi câu hỏi hỏi kênh đăng ký hoặc liên hệ.
   - Quy tắc "không tên tiếng Anh" giờ chỉ áp dụng cho tên CLB.
4. **Chất lượng code**:
   - `MyPipeline` nhận thêm tham số tùy chọn `llm` và `embedder` để test bằng bản giả.
   - Thêm 16 unit test và đưa script phân tích vào `experiments/analyze.py`.
   - Sửa toàn bộ lỗi ruff.

## 2. Điểm số (copy từ `outputs/eval_train_pred/report.md`)

**126 câu hỏi** (115 có đáp án) · judge: `claude-haiku-5-5`

### Retrieval

| K | hit@k | precision@k | recall@k | f1@k | map@k | mrr@k | ndcg@k |
|---|---|---|---|---|---|---|---|
| 3 | 1.0000 | 0.4918 | 0.9529 | 0.6231 | 0.9563 | 1.0000 | 0.9676 |
| 5 | 1.0000 | 0.3033 | 0.9693 | 0.4467 | 0.9600 | 1.0000 | 0.9727 |
| 7 | 1.0000 | 0.2190 | 0.9727 | 0.3470 | 0.9614 | 1.0000 | 0.9741 |

### Generation

| string_presence | rouge_l | bleu_4 | groundedness | response_relevancy | noise_sensitivity | abstention |
|---|---|---|---|---|---|---|
| 0.9345 | 0.6442 | 0.5134 | 1.0000 | 0.6630 | 0.0435 | 1.0000 |

### Điểm tổng hợp

- Retrieval score: **0.9807**
- Generation score: **0.8664**
- Benchmark score: **0.9235**

### Điểm Public Test (trang thi aichallenge)

_(chờ nộp)_

### So với các bản trước

| | v0 | v1 | v2 | v3 | Δ v3−v2 |
|---|---|---|---|---|---|
| Retrieval | 0.7219 | 0.9771 | 0.9675 | **0.9807** | +0.0132 |
| Generation | 0.7540 | 0.8306 | 0.8507 | **0.8664** | +0.0157 |
| **Benchmark train** | 0.7379 | 0.9038 | 0.9091 | **0.9235** | **+0.0144** |
| **Public** | — | 83.36 | 87.13 | _chờ_ | |
| Recall@5 | 0.782 | 0.969 | 0.956 | 0.969 | +0.014 |
| MRR@5 | 0.693 | 0.992 | 0.988 | **1.000** | +0.012 |
| NDCG@5 | 0.691 | 0.970 | 0.959 | 0.973 | +0.014 |
| String Presence | 0.733 | 0.886 | 0.869 | **0.935** | +0.066 |
| ROUGE-L | 0.425 | 0.555 | 0.627 | **0.644** | +0.018 |
| Groundedness | 0.935 | 0.974 | 0.983 | **1.000** | +0.017 |
| Response Relevancy | 0.692 | 0.672 | 0.665 | 0.663 | −0.002 |
| Noise Sensitivity ↓ | 0.261 | 0.103 | 0.039 | 0.044 | +0.004 |
| Abstention | 1.000 | 1.000 | 1.000 | 1.000 | 0 |
| Độ dài trung vị (ký tự) | 405 | 284 | 180 | 202 | (đáp án chuẩn: 196) |

## 3. Phân tích theo loại câu hỏi (`python experiments/analyze.py`)

| Loại câu hỏi | n | recall@5 | mrr@5 | ndcg@5 | string_presence | rouge_l | groundedness | response_relevancy | noise_sensitivity | abstention |
|---|---|---|---|---|---|---|---|---|---|---|
| không dấu/viết tắt | 26 | 0.962 | 1.000 | 0.966 | 0.962 | 0.665 | 1.000 | 0.474 | 0.096 | — |
| diễn đạt khác | 22 | 1.000 | 1.000 | 0.995 | 0.977 | 0.654 | 1.000 | 0.686 | 0.045 | — |
| nhiều đoạn | 20 | 0.912 | 1.000 | 0.931 | 0.898 | 0.644 | 1.000 | 0.688 | 0.050 | — |
| so sánh | 20 | 0.975 | 1.000 | 0.983 | 0.925 | 0.616 | 1.000 | 0.720 | 0.025 | — |
| thời gian | 12 | 1.000 | 1.000 | 1.000 | 0.875 | 0.566 | 1.000 | 0.713 | 0.000 | — |
| không có đáp án | 11 | 1.000 | 1.000 | 0.967 | — | — | — | — | — | 1.000 |
| trực tiếp | 8 | 1.000 | 1.000 | 1.000 | 1.000 | 0.777 | 1.000 | 0.884 | 0.000 | — |
| thiếu dữ liệu | 5 | 1.000 | 1.000 | 1.000 | 0.800 | 0.602 | 1.000 | 0.722 | 0.000 | — |
| mâu thuẫn nguồn | 2 | 0.750 | 1.000 | 0.807 | 1.000 | 0.600 | 1.000 | 0.713 | 0.000 | — |

Có 8/115 câu có recall@5 < 1: train-0003, 0015, 0028, 0044, 0059, 0099, 0113, 0117.

### Đã làm được
1. **Sửa được lỗi định dạng ở bước rewrite**: train-0042 (v2 trả lời "không có thông tin") và train-0017 (câu so sánh năm thành lập) đều đã truy xuất đúng. Retrieval quay lại mức của v1 (0.981), và **MRR@5 = 1.000**: doc đúng luôn nằm ở vị trí 1.
2. **String Presence tăng mạnh (0.869 → 0.935)**, nhờ chép nguyên văn cụm then chốt và cho phép nêu link khi được hỏi kênh. Nhóm "diễn đạt khác" tăng từ 0.852 lên 0.977, nhóm "không dấu" từ 0.910 lên 0.962.
3. **Groundedness = 1.000** ở mọi nhóm, và nhóm "nhiều đoạn" tăng từ 0.900 lên 1.000 nhờ NGUON đầy đủ.
4. Độ dài trung vị 202 ký tự, sát đáp án chuẩn (196). ROUGE-L tiếp tục tăng lên 0.644.
5. Abstention vẫn đạt 1.0. Code có test và lint sạch.

### Chưa làm được và nguyên nhân
1. **Noise nhóm không dấu tăng lại (0.038 → 0.096)**. Ví dụ train-0125: câu trả lời thêm "trên website proptit.com", vốn không có trong đáp án chuẩn, nên bị chấm `incorrect` dù String Presence vẫn = 1. Đây là tác dụng phụ của quy tắc "nêu link khi hỏi kênh".
2. **8 câu nhiều đoạn vẫn trượt retrieval**: các câu cần 3–6 đoạn (0003, 0044) hoặc dùng từ vựng khác hẳn tài liệu (0028 "AI y tế" ↔ "Medical Question Answering", 0113 "ngoài thuật toán").
3. **Response Relevancy giữ ở khoảng 0.66**: trần cố hữu của câu không dấu (khoảng 0.53) không thay đổi.
4. **Latency tăng nhẹ (trung vị 5.9 giây)** vì câu trả lời và danh sách NGUON dài hơn.

### Quan sát trên Public Test (không có đáp án)
- Độ dài trung vị 203.5 ký tự. Câu rào đón "không nêu/không ghi" chỉ còn **1/42** (public-0039).
- Có 7/42 câu chứa cụm "không có thông tin": 0002, 0003, 0015, 0016, 0026, 0027, 0036.
  - 0002, 0016, 0027 là câu mà corpus không có thông tin, nên từ chối là đúng.
  - 0003, 0015, 0026, 0036 là **trả lời một phần**: trả lời đúng ý chính rồi nói rõ phần tài liệu không có. Ví dụ public-0015 nói về quỹ CLB hằng tháng và mục đích sử dụng, nhưng không có số tiền cụ thể.
- public-0003 giờ trả lời đúng ý: CLB Multimedia PTIT phù hợp với người thích tổ chức sự kiện.

## 4. Hướng tiếp theo (chuyên gia đề xuất): dừng tối ưu điểm, làm v3.1 để chống lỗi

**Dư địa còn lại (thang 100)**: mỗi metric retrieval chiếm 1/6 điểm tổng, mỗi metric generation chiếm 1/12.

| Metric | v3 | Tăng tối đa | Tăng thực tế |
|---|---|---|---|
| Recall@5 / NDCG@5 | 0.969 / 0.973 | +0.51 / +0.46 | +0.1–0.2 |
| MRR, Groundedness, Abstention | 1.0 | 0 | 0 |
| String Presence | 0.935 | +0.55 | +0.2 |
| ROUGE-L | 0.644 | khoảng +0.45 | +0.1–0.2 |
| Response Relevancy | 0.663 | bị trần ở nhóm không dấu | ~0 |
| Noise | 0.044 | +0.36 | ~0 (phần lớn do judge dao động, ví dụ train-0125) |

- Có 5/8 câu trượt retrieval mà đoạn đúng không có trong top-10, nên LLM không thể trích dẫn để đẩy chúng lên. Riêng train-0044 cần 6 đoạn nên recall@5 không bao giờ đạt 1.
- **Tổng dư địa thực tế khoảng +0.5–0.8 điểm, nhỏ hơn dao động ±1 của judge**, nên không đo được cải thiện một cách đáng tin cậy.

**Về Public**:
- Các câu từ chối của v2 trên public đều có vẻ đúng: 0002, 0016, 0027 không có thông tin trong corpus.
- Bước rewrite của v3 sạch hoàn toàn trên train: 0/126 câu có dòng ngoài định dạng `Q:`. Đây là thay đổi về độ ổn định, nên khả năng mang sang public tốt nhất.
- Quy tắc chọn bản: nếu public của v3 ≥ 86 thì đóng băng v3; nếu thấp hơn thì so lại với v2.
- Đã cân nhắc và **không nên làm**: mở đầu câu trả lời bằng "Có."/"Không.". Mức tăng ROUGE rất nhỏ, mà "Có." sẽ bị tách thành một câu riêng để judge chấm.

**Rủi ro ngày Private Test (code đóng băng)**:
1. Câu trả lời rỗng bị lưu vào cache, khiến câu đó bị từ chối oan và chạy lại cũng không sửa được.
2. `predict` chạy tuần tự và không bắt exception, nên một lỗi API sau 5 lần retry sẽ dừng cả lượt. **Cách xử lý: chạy lại `neorag run private_test`, không xóa `.cache`**, vì các câu đã xong sẽ được lấy từ cache.
3. Lỗi ở bước rewrite cũng làm dừng cả lượt chạy.

**Spec v3.1** (không đổi prompt, nên không đổi điểm):
1. `generate`: nếu câu trả lời rỗng thì gọi lại một lần, thay đổi nhẹ message để không trúng cache. Vẫn rỗng mới dùng câu từ chối dự phòng.
2. `rewrite`: bọc trong `try/except`, lỗi thì dùng câu gốc.
3. (Cho Demo) Thêm `run_stream()`: embed câu gốc song song với bước rewrite và stream câu trả lời. `run()` giữ nguyên để kết quả chấm không đổi.

Sau đó chạy lại train từ cache để xác nhận điểm không đổi, đóng băng code, rồi dồn thời gian cho Demo và pitching.
