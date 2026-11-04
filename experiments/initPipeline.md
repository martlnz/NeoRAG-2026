# initPipeline — v0 (bản khởi đầu)

- **File:** `src/neorag/pipelines/pipeline.py` · class `MyPipeline(RAGPipeline)`
- **Lệnh chấm:** `neorag run train --pipeline neorag.pipelines.pipeline:MyPipeline`
- **File nộp Public Test:** `experiments/submissions/v0_init_public_test_submission.zip`

## 1. Pipeline v0 làm gì

Cài đặt trực tiếp interface `RAGPipeline`, logic tương đương baseline để có mốc so sánh:

1. `index()`: embed nguyên từng đoạn văn (118 đoạn, không chunking) bằng `bge-m3` qua gateway, lưu ma trận numpy trong RAM.
2. `retrieve()`: embed câu hỏi, cosine similarity, lấy top-10 làm `retrieved_ids`.
3. `generate()`: đưa top-5 đoạn vào prompt ngắn ("Dựa vào các tài liệu dưới đây, hãy trả lời câu hỏi"), gọi `claude-haiku-5-5` một lần.

## 2. Điểm số (copy từ `outputs/eval_train_pred/report.md`)

**126 câu hỏi** (115 có đáp án) · judge: `claude-haiku-5-5`

### Retrieval

| K | hit@k | precision@k | recall@k | f1@k | map@k | mrr@k | ndcg@k |
|---|---|---|---|---|---|---|---|
| 3 | 0.8197 | 0.3634 | 0.7322 | 0.4677 | 0.6207 | 0.6872 | 0.6676 |
| 5 | 0.8443 | 0.2393 | 0.7821 | 0.3554 | 0.6392 | 0.6925 | 0.6910 |
| 7 | 0.8443 | 0.1756 | 0.7951 | 0.2803 | 0.6442 | 0.6925 | 0.6967 |

### Generation

| string_presence | rouge_l | bleu_4 | groundedness | response_relevancy | noise_sensitivity | abstention |
|---|---|---|---|---|---|---|
| 0.7333 | 0.4248 | 0.2682 | 0.9349 | 0.6919 | 0.2612 | 1.0000 |

### Điểm tổng hợp

- Retrieval score: **0.7219**
- Generation score: **0.7540**
- Benchmark score: **0.7379**

Public Test: chưa nộp. Pipeline này tương đương baseline, mà baseline đạt 52.40 trên bảng xếp hạng, dưới ngưỡng 53.40, nên dự kiến nhận **0 điểm** xếp hạng.

## 3. Phân tích theo loại câu hỏi (từ `per_query.jsonl`)

| Loại câu hỏi | n | recall@5 | mrr@5 | ndcg@5 | string_presence | rouge_l | groundedness | response_relevancy | noise_sensitivity | abstention |
|---|---|---|---|---|---|---|---|---|---|---|
| không dấu/viết tắt | 26 | 0.436 | 0.359 | 0.360 | 0.452 | 0.366 | 0.916 | 0.496 | 0.502 | — |
| diễn đạt khác | 22 | 0.864 | 0.735 | 0.754 | 0.830 | 0.414 | 0.952 | 0.698 | 0.239 | — |
| nhiều đoạn | 20 | 0.812 | 0.712 | 0.698 | 0.825 | 0.413 | 0.978 | 0.720 | 0.175 | — |
| so sánh | 20 | 0.792 | 0.779 | 0.727 | 0.758 | 0.375 | 0.861 | 0.778 | 0.280 | — |
| thời gian | 12 | 1.000 | 0.861 | 0.897 | 0.917 | 0.461 | 0.972 | 0.752 | 0.078 | — |
| không có đáp án | 11 | 0.929 | 0.762 | 0.777 | — | — | — | — | — | 1.000 |
| trực tiếp | 8 | 1.000 | 1.000 | 1.000 | 0.771 | 0.727 | 0.938 | 0.893 | 0.000 | — |
| thiếu dữ liệu | 5 | 0.900 | 0.700 | 0.730 | 0.700 | 0.429 | 1.000 | 0.732 | 0.343 | — |
| mâu thuẫn nguồn | 2 | 1.000 | 1.000 | 1.000 | 1.000 | 0.478 | 0.900 | 0.757 | 0.000 | — |

- Độ dài câu trả lời: trung vị **405** ký tự, đáp án chuẩn trung vị **196** ký tự.
- Có 34/115 câu có recall@5 < 1.

### Đã làm được
- **Câu hỏi viết chuẩn, có dấu** (trực tiếp, thời gian, mâu thuẫn nguồn): retrieval gần như hoàn hảo, recall@5 = 1.0. Lý do là corpus đã chia đoạn tốt và mỗi đoạn đều nêu tên CLB.
- **Abstention = 1.0**: Haiku tự từ chối đúng cả 11 câu không có đáp án.
- **Groundedness 0.93**: câu trả lời bám sát context, hầu như không bịa thêm.

### Chưa làm được và nguyên nhân
1. **Câu không dấu/viết tắt là điểm yếu lớn nhất**: recall@5 0.436, MRR 0.359. `bge-m3` không khớp được "truong ban su kien proptit", "clb lt ptit", "hv"... với văn bản có dấu. Public Test chủ yếu là kiểu câu hỏi này, nên đây là lý do chính khiến baseline chỉ đạt 52.40 trên public dù đạt 73.79 trên train.
2. **Truy xuất trượt kéo theo từ chối nhầm**: khi không lấy được đoạn đúng, LLM trả lời "tài liệu không có thông tin". Kết quả là noise = 1 và String Presence = 0, gây phạt kép ở cả retrieval lẫn generation. Ví dụ: train-0045, 0076, 0087.
3. **Câu so sánh và nhiều đoạn**: một lần truy xuất dense không phủ được nhiều CLB. Ví dụ, train-0017 cần các đoạn 1, 85, 97 nhưng lại lấy về toàn các đoạn chung của S2B.
4. **Câu trả lời dài dòng**, có markdown in đậm và gạch đầu dòng, dài gấp khoảng 2 lần đáp án chuẩn. Hậu quả là ROUGE-L chỉ 0.42 và Response Relevancy 0.69, đồng thời có thêm câu thừa dễ bị chấm `incorrect`.
5. Prompt không yêu cầu giữ nguyên số liệu và tên riêng, và không hướng dẫn cách trả lời câu hỏi nhiều ý.

## 4. Hướng cải thiện cho v1 (chuyên gia đề xuất)

Một agent chuyên gia RAG đã đọc mã nguồn bộ chấm (`evaluate.py`, `judge.py`, `eval.toml`), `per_query.jsonl` và quy định, rồi đề xuất như sau.

**Chẩn đoán**
1. Dense bge-m3 không khớp được câu không dấu và tên gọi tắt ("clb game", "clb lt", "hv", "dpt").
2. Một vector truy vấn duy nhất không phủ được câu so sánh nhiều CLB.
3. Truy xuất trượt khiến LLM từ chối, dẫn tới noise = 1 và SP = 0.
4. Câu trả lời dài và có markdown. Mỗi gạch đầu dòng bị tách thành một "câu" để chấm, nên càng nhiều câu thì càng dễ có câu bị gán `incorrect`.
5. Groundedness được chấm trên **toàn bộ** `retrieved_ids` (đã xác minh tại `evaluate.py:72`), không chỉ trên các doc đưa vào prompt.

**Các thay đổi theo thứ tự ưu tiên**

| # | Thay đổi | Metric hưởng lợi | Chi phí/câu |
|---|---|---|---|
| 1 | LLM viết lại câu hỏi: thêm dấu, mở viết tắt, chuẩn hóa tên CLB, tách câu so sánh thành tối đa 3 sub-query | Recall/MRR, kéo theo SP và noise | +1 lời gọi ngắn |
| 2 | Hybrid BM25 trên văn bản bỏ dấu kết hợp dense, gộp bằng RRF trên các truy vấn: gốc, viết lại, sub-query | Recall, NDCG | 0 |
| 3 | Prompt sinh 1–3 câu văn xuôi, không markdown, chép nguyên số liệu/tên riêng | ROUGE-L, Relevancy, noise | Giảm token đầu ra |
| 4 | LLM ghi dòng `NGUON: id,...`, các id này được đẩy lên đầu `retrieved_ids` | MRR, NDCG | 0 |
| 5 | Nộp đủ 10 id, đưa top-8 doc vào prompt | Groundedness, tránh từ chối oan | +~600 token |
| 6 | Quy tắc từ chối hẹp: có thông tin một phần thì trả lời phần đó | noise, SP | 0 |
| 7 | (v2) Cross-encoder `bge-reranker-v2-m3` chạy cục bộ | MRR | 0 USD, chậm hơn |

**Mục tiêu v1:** recall@5 nhóm không dấu ≥ 0.70, ROUGE-L ≥ 0.50, độ dài trung vị câu trả lời ≤ 250 ký tự, giữ Abstention = 1.0.

**Rủi ro:**
- Bước viết lại có thể đoán sai tên CLB. Giữ query gốc trong RRF là để chống lỗi này.
- Câu trả lời quá ngắn có thể làm mất key fact ở câu nhiều ý.
