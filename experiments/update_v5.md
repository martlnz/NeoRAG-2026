# update_v5 — Bộ dev văn nói + bỏ dense cho câu không dấu + 15 doc trong context (bản cuối)

- **File:** `src/neorag/pipelines/pipeline.py` · class `MyPipeline(RAGPipeline)`
- **Lệnh chấm:** `neorag run train --pipeline neorag.pipelines.pipeline:MyPipeline` và `neorag run train_informal --pipeline neorag.pipelines.pipeline:MyPipeline`
- **File nộp Public Test:** `experiments/submissions/v5_public_test_submission.zip`
- **Kiểm tra:** `pytest` 65/65 đạt (thêm 4 test cho v5), `ruff check` sạch
- **Đây là bản cuối**: theo yêu cầu, dừng tối ưu sau v5.

## 1. Đã làm gì (so với v4)

### 1a. Thước đo mới: bộ dev `train_informal`
Public Test quá nhiễu để so sánh các bản: cùng một file v2 được chấm 87.13 rồi 84.23 (xem `update_v4.md`). Train thì chỉ có 26/126 câu văn nói, trong khi phần lớn câu public là văn nói. Vì vậy:
- `experiments/make_informal.py` chuyển 100 câu train viết chuẩn sang phong cách public. Script tất định (cố định seed) và không gọi LLM:
  - bỏ dấu, viết thường;
  - viết tắt: hv, tv, ko, dc, j, e, dk, v;
  - gọi CLB bằng tên gián tiếp: "clb lap trinh cua hv", "clb code", "clb mm", "clb da phuong tien", "clb game", ...
- Kết quả là `data/train_informal/questions.jsonl`, giữ nguyên đáp án chuẩn, chấm được bằng `neorag run train_informal`.
- `experiments/analyze.py <split>` phân tích được mọi bộ.
- **v4 trên `train_informal`: 0.8823**, thấp hơn train (0.9235) **−0.041**. Chuyên gia phân tích từ cache:
  - Khoảng 0.022 trong phần chênh này là **trần cố hữu của Response Relevancy**. bge-m3 coi câu không dấu và câu có dấu là xa nhau: cos(câu informal, câu gốc có dấu) chỉ 0.54. Trong khi đó các câu judge sinh ngược vẫn khớp ý hỏi ở mức 0.79, nên câu trả lời không có vấn đề.
  - Phần sửa được nằm ở **retrieval**: dense trên câu gốc không dấu xếp đoạn đúng ở hạng 30–107, và 5/10 câu trượt có đoạn đúng nằm ở hạng 11–14, ngay ngoài top-10 mà LLM được thấy.

### 1b. Thay đổi pipeline (mục A và B trong đề xuất của chuyên gia)
- **A. Câu gốc không dấu thì bỏ embedding dense.**
  - Hàm `is_unaccented(q)` nhận biết câu không dấu. Khi đó câu gốc chỉ dùng BM25 bỏ dấu, với trọng số 0.7.
  - Các câu viết lại (có dấu) vẫn dùng cả dense lẫn BM25.
  - Câu gốc có dấu giữ nguyên hành vi như v4.
- **B. Đưa top-15 doc vào prompt** (trước là 10).
  - Vẫn nộp 10 id. Đoạn được cite, kể cả khi nằm ở hạng 11–15, được đẩy lên đầu `retrieved_ids`.
- Mục C (rewrite có câu con tự đứng được, sửa nhóm lỗi như inf-0080) **không làm**, vì đã dừng sau v5.

## 2. Điểm số (copy từ `outputs/eval_*/report.md`)

### train (126 câu)

| K | hit@k | precision@k | recall@k | f1@k | map@k | mrr@k | ndcg@k |
|---|---|---|---|---|---|---|---|
| 3 | 1.0000 | 0.4918 | 0.9542 | 0.6237 | 0.9567 | 1.0000 | 0.9684 |
| 5 | 1.0000 | 0.3066 | 0.9747 | 0.4505 | 0.9641 | 1.0000 | 0.9763 |
| 7 | 1.0000 | 0.2213 | 0.9795 | 0.3505 | 0.9656 | 1.0000 | 0.9780 |

| string_presence | rouge_l | bleu_4 | groundedness | response_relevancy | noise_sensitivity | abstention |
|---|---|---|---|---|---|---|
| 0.9357 | 0.6443 | 0.5065 | 0.9971 | 0.6653 | 0.0464 | 0.9091 |

- Retrieval score: **0.9837**
- Generation score: **0.8508**
- Benchmark score: **0.9173**

### train_informal (100 câu)

| K | hit@k | precision@k | recall@k | f1@k | map@k | mrr@k | ndcg@k |
|---|---|---|---|---|---|---|---|
| 3 | 1.0000 | 0.5174 | 0.9523 | 0.6447 | 0.9514 | 0.9948 | 0.9639 |
| 5 | 1.0000 | 0.3229 | 0.9740 | 0.4686 | 0.9592 | 0.9948 | 0.9720 |
| 7 | 1.0000 | 0.2307 | 0.9740 | 0.3623 | 0.9578 | 0.9948 | 0.9711 |

| string_presence | rouge_l | bleu_4 | groundedness | response_relevancy | noise_sensitivity | abstention |
|---|---|---|---|---|---|---|
| 0.9112 | 0.6242 | 0.4976 | 0.9944 | 0.4766 | 0.0562 | 1.0000 |

- Retrieval score: **0.9803**
- Generation score: **0.8250**
- Benchmark score: **0.9026**

### Điểm Public Test (trang thi aichallenge)

**≈ 87** (người dùng báo). Các bản trước: v2 87.13 / 84.23, v3 84.81, v4 84.94. Đây là mức cao nhất nếu tính trung bình, nhưng vẫn nằm trong biên nhiễu khoảng ±3 của judge.

### So với v4

| | v4 train | v5 train | v4 informal | v5 informal |
|---|---|---|---|---|
| Retrieval | 0.9807 | **0.9837** | 0.9615 | **0.9803** |
| Generation | 0.8664 | 0.8508 | 0.8031 | **0.8250** |
| **Benchmark** | **0.9235** | 0.9173 (−0.0062) | 0.8823 | **0.9026 (+0.0203)** |
| Recall@5 | 0.969 | 0.975 | 0.949 | **0.974** |
| MRR@5 | 1.000 | 1.000 | 0.984 | 0.995 |
| String Presence | 0.935 | 0.936 | 0.890 | **0.911** |
| Abstention | 1.000 | **0.909** | 0.909 | 1.000 |

## 3. Phân tích

### Đã làm được
1. **Retrieval với câu văn nói tăng rõ**: R@5 tăng từ 0.949 lên **0.974**, NDCG từ 0.951 lên 0.972. Bỏ dense trên câu gốc không dấu đã loại được đoạn nhiễu, còn 15 doc trong context giúp LLM thấy và cite được các đoạn đúng ở hạng 11–15. Retrieval chấm bằng code nên phần tăng này không bị judge làm nhiễu.
2. **String Presence trên bộ văn nói tăng từ 0.890 lên 0.911**, vì các câu trước đây trượt retrieval (nhóm 0017, 0052, 0130) nay đã có đủ đoạn.
3. Retrieval trên train cũng nhích lên (0.9807 → 0.9837), nên câu có dấu không bị ảnh hưởng xấu.

### Vì sao vẫn chọn v5 dù benchmark train giảm 0.0062
- Phần giảm đến gần như hoàn toàn từ **một câu Abstention** (train-0103), tương đương khoảng −0.0076 benchmark.
- Câu trả lời của v5 cho câu này, "Cộng đồng S2B không có fanpage hay website riêng trong tài liệu; ... chỉ có kênh của từng CLB thành viên", **trùng ý với đáp án chuẩn**. Vậy đây là judge chấm nhiễu, ngược chiều với inf-0072 ở v4.
- Bỏ câu nhiễu đó đi thì train của v5 xấp xỉ v4 (khoảng +0.001), còn bộ văn nói, giống phần lớn bộ public, tăng +0.020.

### Chưa làm được và giới hạn
1. **Response Relevancy với câu văn nói vẫn khoảng 0.48**. Đây là trần cố hữu, khoảng 0.54. Mọi cách đẩy thêm, như lặp lại cách viết không dấu của người hỏi, đều là thao túng judge, nên không làm.
2. Mục C chưa làm: rewrite có câu con không nêu tên CLB ("CLB đó..."), ví dụ inf-0080.
3. Latency tăng nhẹ vì prompt dài hơn (15 doc).
4. Judge trên public dao động khoảng ±3 điểm giữa các lần nộp cùng một file, nên điểm public của một lần nộp không đủ để kết luận bản nào hơn.

## 4. Kết luận và bản thi Private

**Chốt v5 làm bản thi Private Test.** Lý do: retrieval cao nhất trên cả hai bộ, tốt nhất với câu văn nói, và vẫn giữ toàn bộ phần chống lỗi của v4. Checklist ngày 08/11 vẫn như trong `update_v4.md`:
- Chạy `neorag run private_test --pipeline neorag.pipelines.pipeline:MyPipeline`.
- Nếu gặp lỗi API thì chạy lại lệnh đó, **không xóa `.cache/`**.
- Không sửa code trong ngày thi.
