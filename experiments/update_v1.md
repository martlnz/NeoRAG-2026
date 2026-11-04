# update_v1 — Viết lại câu hỏi + Hybrid BM25/Dense + Prompt ngắn gọn có trích nguồn

- **File:** `src/neorag/pipelines/pipeline.py` · class `MyPipeline(RAGPipeline)`
- **Lệnh chấm:** `neorag run train --pipeline neorag.pipelines.pipeline:MyPipeline`
- **File nộp Public Test:** `experiments/submissions/v1_public_test_submission.zip`

## 1. Đã làm gì (so với v0)

Triển khai các mục 1–6 trong kế hoạch của chuyên gia ở `initPipeline.md`:

1. **Viết lại câu hỏi bằng LLM** (thêm 1 lời gọi Haiku cho mỗi câu): thêm dấu, mở viết tắt (`hv`, `e`, `ko`, `tv`, `dpt`, `nk`...), chuẩn hóa tên gọi tắt của 3 CLB (ProPTIT / MPC / GCC, "clb lt", "clb game"...), và tách câu nhiều ý hoặc so sánh thành tối đa 3 câu hỏi con.
2. **Hybrid retrieval:**
   - BM25 (k1 = 1.2, b = 0.75) tự viết bằng numpy, chạy trên `club + topic + text` đã **bỏ dấu**, gồm âm tiết đơn và bigram âm tiết.
   - Dense `bge-m3` như v0.
   - Gộp bằng **RRF** (k = 60) trên mọi truy vấn: query gốc trọng số 0.7, câu viết lại và các câu hỏi con trọng số 1.0. Lấy top-10.
3. **Prompt sinh mới:**
   - 1–3 câu văn xuôi, không markdown, câu đầu trả lời thẳng và nêu tên CLB.
   - Giữ nguyên số liệu và tên riêng; câu nhiều ý thì trả lời đủ từng ý.
   - Có thông tin một phần thì trả lời phần đó; chỉ từ chối khi không đoạn nào liên quan.
   - Context có 8 doc, mỗi doc có tiền tố `[id] (club – topic)`.
4. **Trích nguồn:** LLM ghi dòng `NGUON: id,...`. Chỉ giữ những id có trong context, đẩy chúng lên đầu `retrieved_ids`, rồi xóa dòng này khỏi câu trả lời. Dấu `**`, `#` và gạch đầu dòng còn sót cũng được xóa.

Chi phí: 2 lời gọi LLM mỗi câu (viết lại + sinh câu trả lời). Latency trung vị khoảng 5.2 giây/câu.

## 2. Điểm số (copy từ `outputs/eval_train_pred/report.md`)

**126 câu hỏi** (115 có đáp án) · judge: `claude-haiku-5-5`

### Retrieval

| K | hit@k | precision@k | recall@k | f1@k | map@k | mrr@k | ndcg@k |
|---|---|---|---|---|---|---|---|
| 3 | 1.0000 | 0.4863 | 0.9474 | 0.6177 | 0.9492 | 0.9918 | 0.9613 |
| 5 | 1.0000 | 0.3033 | 0.9693 | 0.4467 | 0.9582 | 0.9918 | 0.9703 |
| 7 | 1.0000 | 0.2178 | 0.9720 | 0.3459 | 0.9586 | 0.9918 | 0.9709 |

### Generation

| string_presence | rouge_l | bleu_4 | groundedness | response_relevancy | noise_sensitivity | abstention |
|---|---|---|---|---|---|---|
| 0.8855 | 0.5554 | 0.3967 | 0.9739 | 0.6715 | 0.1029 | 1.0000 |

### Điểm tổng hợp

- Retrieval score: **0.9771**
- Generation score: **0.8306**
- Benchmark score: **0.9038**

### So với v0

| | v0 | v1 | Δ |
|---|---|---|---|
| Retrieval | 0.7219 | **0.9771** | +0.2552 |
| Generation | 0.7540 | **0.8306** | +0.0766 |
| **Benchmark** | 0.7379 | **0.9038** | **+0.1659** |
| Recall@5 / MRR@5 / NDCG@5 | 0.782 / 0.693 / 0.691 | 0.969 / 0.992 / 0.970 | |
| String Presence | 0.733 | 0.886 | +0.152 |
| ROUGE-L | 0.425 | 0.555 | +0.131 |
| Groundedness | 0.935 | 0.974 | +0.039 |
| Response Relevancy | 0.692 | 0.672 | **−0.020** |
| Noise Sensitivity ↓ | 0.261 | 0.103 | −0.158 (tốt hơn) |
| Abstention | 1.000 | 1.000 | 0 |

## 3. Phân tích theo loại câu hỏi

| Loại câu hỏi | n | recall@5 | mrr@5 | ndcg@5 | string_presence | rouge_l | groundedness | response_relevancy | noise_sensitivity | abstention |
|---|---|---|---|---|---|---|---|---|---|---|
| không dấu/viết tắt | 26 | 0.962 | 1.000 | 0.966 | 0.885 | 0.592 | 0.968 | 0.489 | 0.090 | — |
| diễn đạt khác | 22 | 1.000 | 1.000 | 0.997 | 0.795 | 0.533 | 1.000 | 0.698 | 0.083 | — |
| nhiều đoạn | 20 | 0.887 | 0.950 | 0.888 | 0.908 | 0.499 | 0.975 | 0.692 | 0.100 | — |
| so sánh | 20 | 0.975 | 1.000 | 0.985 | 0.933 | 0.574 | 0.958 | 0.750 | 0.142 | — |
| thời gian | 12 | 1.000 | 1.000 | 1.000 | 0.917 | 0.507 | 0.972 | 0.702 | 0.111 | — |
| không có đáp án | 11 | 1.000 | 1.000 | 0.982 | — | — | — | — | — | 1.000 |
| trực tiếp | 8 | 1.000 | 1.000 | 1.000 | 1.000 | 0.689 | 1.000 | 0.846 | 0.062 | — |
| thiếu dữ liệu | 5 | 1.000 | 1.000 | 1.000 | 0.700 | 0.512 | 0.900 | 0.753 | 0.200 | — |
| mâu thuẫn nguồn | 2 | 1.000 | 1.000 | 1.000 | 1.000 | 0.562 | 1.000 | 0.679 | 0.000 | — |

- Độ dài câu trả lời: trung vị **284** ký tự (v0: 405; đáp án chuẩn: 196).
- Có 8/115 câu có recall@5 < 1 (v0: 34).

### Đã làm được
1. **Đã xử lý được câu không dấu/viết tắt**: recall@5 tăng từ 0.436 lên **0.962**, MRR từ 0.359 lên **1.000**. Bước viết lại câu hỏi cùng BM25 bỏ dấu đã giải quyết đúng nguyên nhân gốc của v0. Đây là thay đổi quan trọng nhất cho Public Test, vì phần lớn câu public thuộc kiểu này.
2. **MRR@5 = 0.992**: doc đúng gần như luôn nằm ở vị trí 1, nhờ RRF và nhờ LLM trích nguồn để đẩy doc đã dùng lên đầu.
3. **Không còn từ chối oan**: noise giảm từ 0.261 xuống 0.103, String Presence tăng từ 0.733 lên 0.886.
4. **Câu trả lời gọn hơn, không markdown**: ROUGE-L tăng từ 0.425 lên 0.555.
5. Abstention vẫn giữ 1.0 dù quy tắc từ chối đã hẹp hơn.

### Chưa làm được và nguyên nhân
1. **Response Relevancy giảm nhẹ (0.692 → 0.672), thấp nhất ở nhóm không dấu (0.489).** Judge sinh ngược câu hỏi từ câu trả lời, sau đó so cosine với **câu hỏi gốc không dấu**. Câu do judge sinh ra luôn có dấu và đúng chuẩn, nên cosine bị giới hạn bởi chính kiểu viết của câu hỏi gốc. Chuyên gia đo được trần của nhóm này chỉ khoảng 0.53, và v1 đã đạt 93% trần (xem mục 4).
2. **Câu trả lời thường kèm câu rào đón** kiểu "Tài liệu không nêu/không ghi ..." (train-0021, 0066, 0069, 0118). Judge gán `incorrect` cho các câu này, khiến noise tăng và relevancy giảm. Đây là tác dụng phụ của quy tắc "nói ngắn gọn phần còn thiếu".
3. **Câu nhiều đoạn vẫn trượt retrieval** (recall@5 0.887). Câu cần 3–6 đoạn (train-0003, 0044) chỉ lấy được một phần. Riêng train-0118 lấy sai doc về Ban Đối nội của MPC, nên LLM trả lời "không nêu rõ".
4. **Câu trả lời vẫn dài hơn đáp án chuẩn** (284 so với 196 ký tự). Một số câu lặp lại tên CLB dài, ví dụ "CLB Sáng tạo Đa phương tiện MULTIMEDIA - PTIT (CLB Multimedia PTIT)", làm giảm ROUGE-L.
5. Latency tăng lên khoảng 5 giây/câu vì có 2 lời gọi LLM nối tiếp nhau (public: trung vị 6.1 giây). Điều này ảnh hưởng tới tiêu chí Demo.

### Quan sát trên Public Test (không có đáp án, đọc bằng mắt)
- Có **8/42 câu trả lời "không có thông tin"**: public-0002, 0003, 0016, 0021, 0023, 0024, 0028, 0036. Một số câu có thể thật sự không có đáp án.
- **public-0003** ("vào clb mm có đất diễn không ... lên plan chạy chương trình") là trường hợp truy xuất trượt rõ ràng: lấy về các đoạn 26, 8, 23, 25 của ProPTIT thay vì các đoạn về ban của MPC. Cần kiểm tra lại bước rewrite với tên gọi "clb mm".
- **public-0021** (so sánh vòng tuyển của cả 3 CLB) lấy được đoạn của MPC và GCC nhưng lại trả lời "không có thông tin". Câu so sánh nhiều CLB vẫn là điểm yếu.

## 4. Hướng cải thiện cho v2 (chuyên gia đề xuất)

Chuyên gia chỉ phân tích lại từ cache (`judge.sqlite`, `pipeline.sqlite`), không gọi API: dựng lại 3 câu hỏi mà judge sinh ngược cho từng câu trả lời, kèm embedding của chúng.

**Chẩn đoán**

a) **Response Relevancy của nhóm không dấu gần như đã chạm trần.** "Trần" ở đây là cosine giữa câu hỏi gốc và bản viết lại có dấu: dù judge sinh lại đúng câu hỏi chuẩn thì điểm cũng chỉ tới mức này.

| Loại | n | RR thực tế | Trần (gốc ~ viết lại) |
|---|---|---|---|
| không dấu/viết tắt | 26 | 0.489 | **0.530** |
| trực tiếp | 8 | 0.846 | 0.914 |
| diễn đạt khác | 22 | 0.698 | 0.880 |
| so sánh | 20 | 0.750 | 0.889 |
| nhiều đoạn | 20 | 0.692 | 0.837 |

- Nhóm không dấu đạt 93% trần. Dù làm hoàn hảo thì RR của cả bộ cũng chỉ tăng khoảng +0.009.
- Các nhóm có dấu mới đạt khoảng 84% trần. Nguyên nhân: judge sinh câu hỏi từ các ý phụ (hashtag, "ngoài ra...") và từ câu rào đón.
- Câu trả lời dưới 200 ký tự có RR 0.86 và ROUGE 0.65; câu từ 300 ký tự trở lên chỉ đạt 0.70 và 0.51.

b) **Noise do tự thêm thông tin.** Khoảng 12 câu kết thúc bằng câu rào đón "Tài liệu không nêu...". Các câu có rào đón có noise trung bình 0.14, các câu không rào đón chỉ 0.085. Số còn lại là các câu "ngoài ra/còn..." thêm ý ngoài câu hỏi.

c) **ROUGE-L**: câu trả lời dài, và tên CLB không theo dạng của đáp án chuẩn. Đáp án chuẩn dùng "CLB Lập Trình PTIT", "CLB Multimedia PTIT", "CLB Nhà Sáng Tạo Game PTIT (GCC)".

d) **Retrieval còn 8 câu trượt**, chủ yếu là câu nhiều đoạn. Một số đoạn đúng nằm ở hạng 6–10 (ví dụ đoạn 34 ở train-0059).

**Các thay đổi theo thứ tự ưu tiên**
1. **Prompt mới**:
   - 1–2 câu, khoảng 60 từ.
   - Câu đầu lặp lại chủ thể của câu hỏi.
   - Dùng tên CLB chuẩn.
   - Không thêm ý phụ, không viết câu rào đón về những khía cạnh câu hỏi không hỏi.
   - Dự kiến: ROUGE-L +0.04–0.06, noise −0.04, RR +0.02–0.04, benchmark tăng khoảng +0.8–1.2 điểm.
2. **`top_k` từ 8 lên 10**: LLM thấy được và trích dẫn được các đoạn đúng đang nằm ở hạng 6–10. Benchmark tăng khoảng +0.1–0.2.
3. (Tùy chọn, dễ overfit) Mở rộng sang các đoạn kề nhau cùng club và topic.
4. **Cross-encoder reranker: chưa đáng làm**, vì MRR đã 0.99.

**Không nên thay đổi**
- Không cố đẩy RR cho câu không dấu, ví dụ bằng cách chép lại cách viết của người hỏi. Lợi ích rất nhỏ, lại giáp ranh vi phạm quy tắc "tác động judge".
- Giữ nguyên bước rewrite, BM25 bỏ dấu và `n_retrieved = 10`.
- Không tối ưu riêng theo từng id của train, vì judge tự dao động khoảng ±1 điểm.
