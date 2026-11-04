# update_v2 — Prompt sinh gọn (tên CLB chuẩn, bỏ rào đón) + rewrite giữ ý định + top_k 10

- **File:** `src/neorag/pipelines/pipeline.py` · class `MyPipeline(RAGPipeline)`
- **Lệnh chấm:** `neorag run train --pipeline neorag.pipelines.pipeline:MyPipeline`
- **File nộp Public Test:** `experiments/submissions/v2_public_test_submission.zip`

## 1. Đã làm gì (so với v1)

Triển khai mục 1–2 trong kế hoạch của chuyên gia ở `update_v1.md`, kèm một sửa lỗi phát hiện trên Public Test:

1. **Prompt sinh câu trả lời mới:**
   - 1–2 câu, khoảng 60 từ.
   - Câu đầu lặp lại chủ thể của câu hỏi.
   - Tên CLB theo một dạng chuẩn: "CLB Lập Trình PTIT", "CLB Multimedia PTIT", "CLB Nhà Sáng Tạo Game PTIT (GCC)".
   - Chỉ nêu điều được hỏi: không "ngoài ra", không hashtag, link hay bối cảnh thừa.
   - Không viết câu rào đón "Tài liệu không nêu..." về những khía cạnh câu hỏi không hỏi.
2. **`top_k` từ 8 lên 10**: LLM thấy cả 10 doc nên có thể trích dẫn đoạn đúng đang nằm ở hạng 9–10.
3. **Prompt rewrite giữ đúng ý định:** thêm lưu ý "chương trình / plan / chạy chương trình" là tổ chức sự kiện, không phải lập trình, và yêu cầu không đổi nghĩa câu hỏi. Lỗi này phát hiện ở public-0003: v1 viết lại "thích lên plan chạy chương trình" thành "thích lập trình", nên truy xuất sang đoạn của ProPTIT.

Chi phí vẫn là 2 lời gọi LLM mỗi câu. Latency trung vị khoảng 5.0 giây/câu.

## 2. Điểm số (copy từ `outputs/eval_train_pred/report.md`)

**126 câu hỏi** (115 có đáp án) · judge: `claude-haiku-5-5`

### Retrieval

| K | hit@k | precision@k | recall@k | f1@k | map@k | mrr@k | ndcg@k |
|---|---|---|---|---|---|---|---|
| 3 | 0.9918 | 0.4809 | 0.9392 | 0.6111 | 0.9413 | 0.9877 | 0.9536 |
| 5 | 0.9918 | 0.2967 | 0.9556 | 0.4379 | 0.9458 | 0.9877 | 0.9591 |
| 7 | 0.9918 | 0.2155 | 0.9617 | 0.3417 | 0.9486 | 0.9877 | 0.9618 |

### Generation

| string_presence | rouge_l | bleu_4 | groundedness | response_relevancy | noise_sensitivity | abstention |
|---|---|---|---|---|---|---|
| 0.8688 | 0.6265 | 0.4722 | 0.9826 | 0.6653 | 0.0391 | 1.0000 |

### Điểm tổng hợp

- Retrieval score: **0.9675**
- Generation score: **0.8507**
- Benchmark score: **0.9091**

### Điểm Public Test (trang thi aichallenge)

**87.13** (v1: 83.36 → **+3.77**; baseline 52.40, ngưỡng 53.40)

Trên public, điểm tăng mạnh hơn hẳn trên train (+0.53). Các thay đổi của v2 hợp với câu hỏi văn nói của public:
- Câu trả lời ngắn, không rào đón, nên noise và ROUGE-L tốt hơn.
- Rewrite giữ đúng ý định, nên giảm số câu từ chối oan (từ 8 xuống 6).

Điều này cho thấy train không phản ánh đầy đủ public, và nên dùng điểm public làm thước đo chính để quyết định bản nộp.

### So với các bản trước

| | v0 | v1 | v2 | Δ v2−v1 |
|---|---|---|---|---|
| Retrieval | 0.7219 | 0.9771 | 0.9675 | −0.0096 |
| Generation | 0.7540 | 0.8306 | **0.8507** | +0.0201 |
| **Benchmark** | 0.7379 | 0.9038 | **0.9091** | **+0.0053** |
| Recall@5 | 0.782 | 0.969 | 0.956 | −0.014 |
| MRR@5 | 0.693 | 0.992 | 0.988 | −0.004 |
| NDCG@5 | 0.691 | 0.970 | 0.959 | −0.011 |
| String Presence | 0.733 | 0.886 | 0.869 | −0.017 |
| ROUGE-L | 0.425 | 0.555 | **0.627** | +0.071 |
| Groundedness | 0.935 | 0.974 | **0.983** | +0.009 |
| Response Relevancy | 0.692 | 0.672 | 0.665 | −0.006 |
| Noise Sensitivity ↓ | 0.261 | 0.103 | **0.039** | −0.064 (tốt hơn) |
| Abstention | 1.000 | 1.000 | 1.000 | 0 |
| Độ dài trung vị (ký tự) | 405 | 284 | **180** | (đáp án chuẩn: 196) |

## 3. Phân tích theo loại câu hỏi

| Loại câu hỏi | n | recall@5 | mrr@5 | ndcg@5 | string_presence | rouge_l | groundedness | response_relevancy | noise_sensitivity | abstention |
|---|---|---|---|---|---|---|---|---|---|---|
| không dấu/viết tắt | 26 | 0.962 | 1.000 | 0.966 | 0.910 | 0.665 | 1.000 | 0.471 | 0.038 | — |
| diễn đạt khác | 22 | 1.000 | 1.000 | 0.995 | 0.852 | 0.655 | 1.000 | 0.688 | 0.023 | — |
| nhiều đoạn | 20 | 0.863 | 0.950 | 0.882 | 0.842 | 0.617 | 0.900 | 0.654 | 0.100 | — |
| so sánh | 20 | 0.942 | 0.975 | 0.949 | 0.850 | 0.615 | 1.000 | 0.748 | 0.050 | — |
| thời gian | 12 | 1.000 | 1.000 | 1.000 | 0.875 | 0.511 | 1.000 | 0.731 | 0.000 | — |
| không có đáp án | 11 | 1.000 | 1.000 | 0.967 | — | — | — | — | — | 1.000 |
| trực tiếp | 8 | 1.000 | 1.000 | 1.000 | 0.958 | 0.759 | 1.000 | 0.866 | 0.000 | — |
| thiếu dữ liệu | 5 | 1.000 | 1.000 | 1.000 | 0.700 | 0.457 | 1.000 | 0.786 | 0.000 | — |
| mâu thuẫn nguồn | 2 | 0.750 | 1.000 | 0.807 | 1.000 | 0.606 | 1.000 | 0.735 | 0.000 | — |

Có 10/115 câu có recall@5 < 1 (v1: 8): train-0003, 0015, 0017, 0028, 0042, 0044, 0059, 0099, 0113, 0117.

### Đã làm được
1. **Generation tăng rõ (+0.020)** nhờ prompt mới:
   - ROUGE-L tăng từ 0.555 lên **0.627**.
   - Noise giảm từ 0.103 xuống **0.039**: bỏ câu rào đón và ý phụ đúng như chẩn đoán.
   - Groundedness 0.983; ở nhóm không dấu, diễn đạt khác, so sánh và thời gian đạt 1.0.
2. **Độ dài câu trả lời khớp đáp án chuẩn**: trung vị 180 ký tự so với 196. Câu ngắn còn có lợi cho latency và trải nghiệm Demo.
3. Nhóm không dấu vẫn giữ recall 0.962 và MRR 1.0. ROUGE-L của nhóm này tăng từ 0.592 lên 0.665, và noise giảm từ 0.090 xuống 0.038.
4. Abstention vẫn đạt 1.0.

### Chưa làm được và nguyên nhân
1. **Retrieval giảm nhẹ (0.977 → 0.968)**, tập trung ở câu **nhiều đoạn** (recall từ 0.887 xuống 0.863) và **so sánh** (từ 0.975 xuống 0.942). Haiku không cho chỉnh temperature nên bước rewrite không ổn định, và prompt rewrite mới cũng khiến các câu hỏi con thay đổi. Chênh lệch nhỏ, cùng bậc với độ dao động ±1 điểm.
2. **Truy xuất trượt dẫn tới "không có thông tin" sai**:
   - train-0042: hỏi MPC có phải CLB chính thức không, cần đoạn [85, 96] nhưng lấy về [86, 8, 19, ...].
   - train-0017: câu so sánh năm thành lập cần đoạn [1, 85, 97] của 3 CLB, nhưng chỉ có đoạn 97.
   Câu so sánh nhiều CLB chưa có cơ chế bảo đảm mỗi CLB có ít nhất một đoạn trong kết quả.
3. **String Presence giảm nhẹ (0.886 → 0.869)**: câu trả lời 1–2 câu đôi khi bỏ sót một key fact ở câu nhiều ý.
4. **Response Relevancy không đổi (0.665)**: nhóm không dấu vẫn bị trần khoảng 0.53 (xem `update_v1.md`).
5. Latency khoảng 5 giây/câu vì 2 lời gọi LLM nối tiếp.

### Quan sát trên Public Test (không có đáp án)
- Số câu trả lời chứa "không có thông tin" giảm từ **8 xuống 6/42**: public-0002, 0003, 0016, 0024, 0027, 0039. Các câu 0021, 0023, 0028, 0036 nay đã trả lời được.
- **public-0003** đã được sửa: v2 hiểu đúng "lên plan chạy chương trình" là tổ chức sự kiện, truy xuất đúng sang MPC và trả lời rằng CLB Multimedia PTIT phù hợp với người thích tổ chức sự kiện. Câu trả lời vẫn còn một ý phụ "tài liệu không nêu ...".

## 4. Hướng cải thiện cho v3 (chuyên gia đề xuất)

Chuyên gia đã chạy lại cả v1 lẫn v2 chỉ từ cache (126/126 câu đều có sẵn, không gọi API) để so sánh kết quả của từng bước.

**Chẩn đoán**
1. **Retrieval thô (RRF) không giảm**: R@5 0.916 → 0.920, MRR 0.853 → 0.862. MRR cuối đạt 0.99 là nhờ bước đẩy các đoạn được trích dẫn lên đầu. R@5 cuối giảm (−0.013) do hai nguyên nhân:
   - **Lỗi định dạng của bước rewrite** (khoảng −0.016):
     - LLM in thêm dòng meta như "Câu hỏi này chỉ có một ý, nên không cần câu hỏi con." hay "Câu viết lại: ...". Các dòng này vào retrieval với trọng số 1.0 và có thể thành phần "Hiểu là" trong prompt.
     - Xảy ra ở khoảng 10/126 câu train, và trên public ở public-0002, 0025.
     - train-0042 vì lỗi này mà trả lời "không có thông tin".
     - train-0017 thì bị tách thành "lâu đời nhất / trẻ nhất" thay vì hỏi ngày thành lập của từng CLB.
   - **LLM trích dẫn ít hơn** (khoảng −0.008): trung bình 1.67 → 1.43 nguồn/câu. Prompt "ngắn gọn" khiến LLM chỉ ghi đoạn thực sự dùng, nên số câu có đoạn đúng mà không được trích tăng từ 8 lên 15.
2. **String Presence giảm** (12 câu mất điểm, 6 câu được điểm), do ba nguyên nhân:
   - Lỗi rewrite (0017, 0042).
   - LLM viết lại cụm từ thay vì chép nguyên văn: "25 bạn" → "25 thành viên", "Giải Khuyến khích - Top 6" → "(Top 6)".
   - Chính các quy tắc mới: quy tắc "không tên tiếng Anh" làm mất "The Bards", quy tắc "không link" làm mất "mpc.proptit.com".
3. **Các hướng không đáng làm**:
   - Cross-encoder: MRR đã 0.99.
   - Mở rộng sang đoạn lân cận: chỉ top-5 được tính điểm.
   - Bỏ bước rewrite cho câu đã có dấu: R@10 giảm 3 điểm.
4. **Dư địa còn lại**: mỗi +0.01 của một chỉ số generation chỉ được khoảng +0.0008 benchmark. Trần thực tế khoảng **0.915–0.92**, tức nhỏ hơn dao động của judge (±1 điểm).

**Spec v3**
1. **Rewrite có cấu trúc**:
   - Mỗi câu hỏi in trên một dòng bắt đầu bằng `Q: `, và chỉ đọc những dòng đó.
   - Lọc bằng regex các dòng meta còn sót.
   - Nếu không có dòng hợp lệ thì dùng câu gốc.
   - Câu so sánh nhiều CLB: mỗi câu con hỏi đúng thuộc tính đó cho từng CLB.
   - Không nhắc tới CLB mà người hỏi không nêu.
2. **NGUON ghi mọi đoạn chứa thông tin trả lời** (kể cả đoạn trùng ý), xếp theo mức liên quan, thay vì chỉ ghi đoạn đã dùng.
3. **Tinh chỉnh prompt cho String Presence**:
   - Chép nguyên văn cụm then chốt: số kèm đơn vị, tên giải / ban / đội / sự kiện, kể cả tên tiếng Anh như The Bards.
   - Nêu link khi câu hỏi hỏi đăng ký ở đâu hay kênh nào.
   - Quy tắc "không tên tiếng Anh" chỉ áp dụng cho tên CLB.
4. (Tùy chọn) Chừa chỗ cố định cho mỗi CLB trong câu so sánh: chưa làm vì sửa được 0017 nhưng làm hỏng 0052.

**Khuyến nghị**: làm v3 với mục 1–3, chủ yếu để tăng độ ổn định cho private test. Sau đó **dừng tối ưu điểm**, vì phần lợi còn lại nằm trong dao động của judge. Nên dành ngân sách và thời gian cho Demo: stream câu trả lời và gọi embedding song song với bước rewrite.
