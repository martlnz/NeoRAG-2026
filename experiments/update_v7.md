# update_v7 — Few-shot từ đáp án train (BỊ LOẠI, quay về v6)

- **Trạng thái:** ❌ **Không dùng.** Code v7 được giữ trong commit lịch sử, còn `src/neorag/pipelines/pipeline.py` được khôi phục về v6 ở commit kế tiếp.
- **Lệnh chấm:** `neorag run train` / `neorag run train_informal`, sau đó `python experiments/heldout_score.py`
- **Public Test:** không nộp, để tiết kiệm ngân sách vì bản này đã bị loại trên dev.

## 1. Đã làm gì (so với v6)

Theo gợi ý của người dùng ("cách diễn đạt, độ chi tiết, cách trả lời sát thông tin cũng quan trọng"):

1. **Few-shot tĩnh 6 ví dụ** đọc từ `data/train/questions.jsonl`, mỗi kiểu trả lời một ví dụ:

   | Kiểu | Câu |
   |---|---|
   | Nêu nguồn | train-0085 |
   | Có/không + nguồn + nhiều đoạn | train-0028 |
   | So sánh | train-0010 |
   | Không có đáp án | train-0050 |
   | Thiếu một phần | train-0069 |
   | Mâu thuẫn nguồn | train-0101 |

   - Khối ví dụ có ghi chú "chỉ học văn phong, KHÔNG dùng làm nguồn thông tin".
   - "Có." trong đáp án mẫu được đổi thành "Có," cho khớp quy tắc dùng dấu phẩy.
   - Quy định cho phép: *"được dùng tập train để … xây dựng few-shot"*.
2. **Câu hỏi hỏi danh sách: liệt kê đủ mọi mục**, được phép dài hơn 60 từ. Mục tiêu là các câu mất String Presence do bị cắt (train-0003).
3. **Đo trung thực bằng `experiments/heldout_score.py`**:
   - Tính lại benchmark sau khi **loại 6 câu few-shot và 6 bản informal tương ứng**, theo đúng công thức trong `configs/eval.toml`.
   - Chạy trên toàn bộ thì script ra đúng y điểm chính thức (v6: 0.9220 / 0.9083).

## 2. Điểm số

### Theo `report.md`

| | train v6 | train v7 | informal v6 | informal v7 |
|---|---|---|---|---|
| Recall@5 | 0.968 | 0.981 | 0.978 | 0.984 |
| MRR@5 / NDCG@5 | 1.000 / 0.972 | 0.996 / 0.979 | 1.000 / 0.981 | 1.000 / 0.980 |
| String Presence | 0.910 | 0.910 | 0.905 | 0.885 |
| ROUGE-L | 0.662 | 0.644 | 0.660 | 0.645 |
| Groundedness | 0.996 | 0.987 | 0.989 | 0.972 |
| Response Relevancy | 0.665 | 0.663 | 0.472 | 0.476 |
| Noise ↓ | 0.048 | **0.068** | 0.045 | 0.062 |
| Abstention | 1.000 | 1.000 | 1.000 | **0.818** |
| Retrieval score | 0.9798 | **0.9854** | 0.9864 | **0.9880** |
| Generation score | **0.8642** | 0.8560 | **0.8301** | 0.7890 |
| **Benchmark** | **0.9220** | 0.9207 | **0.9083** | 0.8885 |

### Sau khi loại câu few-shot (`heldout_score.py`), dùng để quyết định

| Bộ | v6 | v7 | Δ |
|---|---|---|---|
| train (120 câu) | **0.9206** | 0.9186 | −0.0020 |
| train_informal (94 câu) | **0.9091** | 0.8870 | **−0.0221** |

## 3. Phân tích

### Đã làm được
- **Retrieval tăng nhẹ** (train 0.980 → 0.985). LLM trích nguồn đầy đủ hơn khi thấy ví dụ nhiều đoạn (train-0028).

### Vì sao bị loại
1. **Abstention trên informal giảm từ 1.0 xuống 0.818** (2/11 câu):
   - inf-0049: "Tài liệu hiện không có thông tin về thời gian kéo dài của vòng Training… *Theo các bài đăng, vòng phỏng vấn được tổng kết ngày 21/09/2026…*"
   - inf-0093: "Tài liệu hiện không có thông tin về mức quỹ… của GCC. *Tài liệu chỉ nêu CLB Lập Trình PTIT thu một khoản nhỏ hằng tháng…*"
   - Cả hai đều từ chối đúng, nhưng **câu ngữ cảnh kèm theo chứa thông tin cụ thể**, nên judge coi là "đưa ra thông tin như thể đang trả lời". Ví dụ few-shot train-0050 ("Tài liệu chỉ cho biết …") đã khuyến khích viết thêm câu này.
2. **Câu trả lời dài và "đầy đủ" hơn**, theo văn phong đáp án mẫu và quy tắc liệt kê đủ. Kết quả là noise tăng (train 0.048 → 0.068), groundedness giảm, và ROUGE-L giảm khoảng 0.02. String Presence không tăng như kỳ vọng.
3. **Bài học:** Haiku bắt chước *độ dài và độ đầy đủ* của ví dụ mạnh hơn là *cách chọn chữ*. Với bộ chấm này, câu trả lời ngắn và đúng trọng tâm (v6) vẫn có lợi hơn.

### Phát hiện dùng được tiếp
- **Câu ngữ cảnh "Tài liệu chỉ nêu …" sau lời từ chối là rủi ro cho Abstention**, kể cả trong v6, vì quy tắc v6 cho phép tối đa một câu như vậy. Trên public, nơi chỉ có khoảng 3–4 câu không có đáp án, mỗi câu như vậy là khoảng 2 điểm.
- **Hướng v8:** khi chính điều được hỏi không có trong tài liệu, chỉ trả lời **một câu** "Tài liệu hiện không có thông tin về …", không thêm câu ngữ cảnh. Phần còn lại giữ nguyên như v6.

## 4. Quyết định
Khôi phục pipeline về **v6** (public 88.43). Giữ lại các công cụ: `experiments/heldout_score.py`, hàm `build_examples` và test của chúng, có thể dùng cho thử nghiệm sau.
