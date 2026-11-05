# update_v8 — Từ chối đúng một câu + Parent-child retrieval (ĐANG DỞ)

- **Trạng thái:** ⏸️ Đang dở. Đã dừng vì người dùng cần tắt máy. Bộ train đã chấm xong, **train_informal và public chưa chấm**.
- **Bản nộp an toàn hiện tại vẫn là v6:** `experiments/submissions/v6_public_test_submission.zip` (public 88.43).

## 1. Đã làm gì (so với v6)
1. **Từ chối đúng một câu:** khi điều được hỏi không có trong tài liệu, chỉ trả lời "Tài liệu hiện không có thông tin về …", bỏ câu ngữ cảnh "Tài liệu chỉ nêu …".
   - Lý do: câu không có đáp án chỉ được chấm Abstention, nên câu ngữ cảnh không thêm điểm mà chỉ thêm rủi ro.
   - Bằng chứng: v7 có 2 câu informal (inf-0049, inf-0093) bị chấm Abstention 0 vì câu ngữ cảnh.
2. **Parent-child retrieval** (đề xuất của người dùng):
   - Child = từng câu của đoạn (đoạn danh sách tách thêm theo ";"), kèm header `club | topic`. Có 401 child từ 118 đoạn.
   - Dense và BM25 chạy trên child, điểm của đoạn cha = điểm child tốt nhất (max-pooling).
   - Kết quả child được gộp RRF với ranking đoạn nguyên. Nộp id đoạn cha và đưa toàn văn đoạn cha cho LLM.
3. **Chuẩn bị cho v9** (đã có trong code nhưng **tắt mặc định**, nên không ảnh hưởng v8):
   - `ANSWER_TEMPLATES`: 8 mẫu trả lời chung có chỗ trống `<…>`, rút ra từ khuôn đáp án train, không chứa dữ kiện thật. Bật bằng `answer_templates=True`.
   - Bộ thử nghiệm có thêm cờ `--m3`: multi-vector (ColBERT) và sparse của bge-m3 chạy local qua `FlagEmbedding` (đã cài 1.4.2).

## 2. Kết quả thử nghiệm retrieval (`experiments/retrieval_ablation.py`, xếp hạng thô trước khi LLM trích nguồn)

| Biến thể | train | train_informal |
|---|---|---|
| v5/v6 (đoạn nguyên, dense + BM25) | 0.8743 | 0.8614 |
| Parent-child (chunk câu → max về đoạn cha) | 0.8863 | **0.8933** |
| **Đoạn nguyên + parent-child (RRF)**, dùng trong v8 | **0.8949** | 0.8831 |
| Đoạn nguyên + dense có header | 0.8959 | 0.8823 |
| HyDE | 0.8780 | 0.8586 (không có lợi) |
| Chỉ dense / chỉ BM25 | 0.830 / 0.804 | 0.807 / 0.800 |

Cross-encoder `bge-reranker-v2-m3` chạy trên CPU quá chậm: hơn 15 phút vẫn chưa xong bộ train. Thử nghiệm này đã dừng; reranker không khả thi cho Demo và ngày thi.

## 3. Điểm v8 (copy từ `outputs/eval_train_pred/report.md`)

### train
- Retrieval score: **0.9836** (v6: 0.9798)
- Generation score: **0.8622** (v6: 0.8642)
- Benchmark score: **0.9229** (v6: 0.9220)

### train_informal
_(chưa chấm)_

## 4. Việc cần làm khi mở máy lại
```powershell
cd D:\NeoRAG-2026
.venv\Scripts\neorag.exe run train_informal --pipeline neorag.pipelines.pipeline:MyPipeline   # so với v6: 0.9083
.venv\Scripts\neorag.exe run public_test --pipeline neorag.pipelines.pipeline:MyPipeline       # tạo zip v8
.venv\Scripts\python.exe experiments\retrieval_ablation.py --m3                               # multi-vector + sparse
```
- Các câu đã chạy nằm trong `.cache/`, nên chạy lại không tốn tiền cho phần đó. **Không xóa `.cache/`.**
- Giữ v8 nếu train_informal ≥ 0.9083. Nếu thấp hơn thì nộp v6.
