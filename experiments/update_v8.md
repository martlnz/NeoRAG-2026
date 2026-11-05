# update_v8 — Từ chối đúng một câu + Parent-child retrieval (BỊ LOẠI, quay về v6)

- **Trạng thái:** ❌ **Không dùng.** Sau khi mở máy lại đã chấm xong train_informal: v8 kém v6. Code khôi phục về hành vi v6; parent-child giữ lại dưới dạng tùy chọn `parent_child=True` (tắt mặc định).
- File public v8 (chỉ để lưu hồ sơ, **không nộp**): `experiments/submissions/v8_public_test_submission.zip`
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
| string_presence | rouge_l | bleu_4 | groundedness | response_relevancy | noise_sensitivity | abstention |
|---|---|---|---|---|---|---|
| 0.8876 | 0.6319 | 0.4914 | 1.0000 | 0.4704 | 0.0393 | 0.9091 |

- Retrieval score: **0.9790** (v6: 0.9864)
- Generation score: **0.8100** (v6: 0.8301)
- Benchmark score: **0.8945** (v6: **0.9083**, tức −0.0138)

### Kết luận
| | v6 | v8 |
|---|---|---|
| train | 0.9220 | 0.9229 (ngang) |
| **train_informal** | **0.9083** | 0.8945 (−0.0138) |
| **Public (trang thi)** | **88.43** | 88.15 (người dùng nộp thử) |

Điểm public của v8 thấp hơn v6 một chút (−0.28, nằm trong biên nhiễu ±3), nhất quán với kết luận trên dev. Quyết định giữ v6 được xác nhận.

1. **Parent-child có lợi ở xếp hạng thô** (+0.02 retrieval score), nhưng **không chuyển thành điểm cuối**. R@15 không đổi (khoảng 0.985), và bước LLM trích nguồn vốn đã đẩy doc đúng lên đầu, nên phần thắng ở MRR/NDCG thô bị san bằng. Retrieval cuối trên informal còn giảm nhẹ (0.986 → 0.979). Child có header đổi thứ tự 15 doc đưa vào prompt, khiến LLM trích nguồn khác đi.
2. **Abstention trên informal lại trượt ở inf-0049** ("vòng Training MPC Gen 14 kéo dài bao lâu"). Lần này LLM coi đây là câu thiếu một phần: trả lời ngày bắt đầu rồi nói "không ghi thời lượng". Đáp án chuẩn coi đây là câu không có đáp án. Đây là câu ranh giới, đã trượt cả ở v7, nên quy tắc từ chối đúng một câu không giải quyết được.
3. String Presence trên informal giảm (0.905 → 0.888). Noise trên train thì tốt hơn rõ (0.048 → 0.022).
4. **Bài học:** cải tiến ở xếp hạng thô phải được xác nhận bằng điểm end-to-end. Khi retrieval cuối đã khoảng 0.98, nút thắt không còn ở retrieval.

## 4. (Đã làm) Việc cần làm khi mở máy lại
```powershell
cd D:\NeoRAG-2026
.venv\Scripts\neorag.exe run train_informal --pipeline neorag.pipelines.pipeline:MyPipeline   # so với v6: 0.9083
.venv\Scripts\neorag.exe run public_test --pipeline neorag.pipelines.pipeline:MyPipeline       # tạo zip v8
.venv\Scripts\python.exe experiments\retrieval_ablation.py --m3                               # multi-vector + sparse
```
- Các câu đã chạy nằm trong `.cache/`, nên chạy lại không tốn tiền cho phần đó. **Không xóa `.cache/`.**
- Giữ v8 nếu train_informal ≥ 0.9083. Nếu thấp hơn thì nộp v6.
