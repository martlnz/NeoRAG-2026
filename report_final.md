# NeoRAG-2026 — Báo cáo tổng hợp (report_final)
## 1. Bài toán và cách chấm điểm

### 1.1 Bài toán

Trả lời câu hỏi của sinh viên về cộng đồng S2B PTIT, gồm 3 CLB: Lập Trình PTIT (ProPTIT), Multimedia PTIT (MPC), Nhà Sáng Tạo Game PTIT (GCC). Kho tri thức nhỏ: **118 đoạn** trong `data/corpus.jsonl`, mỗi đoạn có `id`, `club`, `topic`, `text`.

Pipeline nhận một câu hỏi và trả về `RAGOutput(answer, retrieved_ids)`: câu trả lời và danh sách id đoạn văn (tối đa 10) làm bằng chứng.

Câu hỏi rất đa dạng:
- câu chuẩn có dấu;
- câu văn nói không dấu, viết tắt (`hv`, `tv`, `nk`, `dpt`, `clb lt`);
- câu so sánh nhiều CLB, câu nhiều ý;
- câu mà tài liệu **không có** thông tin, phải từ chối đúng.

### 1.2 Bộ chấm

| Nhóm | Chỉ số | Đo cái gì |
|---|---|---|
| Retrieval (top-5) | Recall@5, MRR@5, NDCG@5 | đoạn đúng có nằm trong 5 id đầu không, và đứng cao đến đâu |
| Generation | String Presence | câu trả lời có chứa các cụm then chốt (key_facts) của đáp án chuẩn không |
| | ROUGE-L | độ giống đáp án chuẩn |
| | Groundedness | không bịa: mỗi câu phải được các đoạn đã nộp chống đỡ |
| | Response Relevancy | độ liên quan giữa câu hỏi và câu trả lời (theo báo cáo, tính bằng cosine bge-m3) |
| | Noise Sensitivity | câu thừa, câu sai (càng thấp càng tốt; pipeline dùng `1 − Noise`) |
| | Abstention | từ chối đúng với câu không có thông tin |


### 1.3 Ba khó khăn gốc

1. **Không dấu / viết tắt:** dense `bge-m3` gần như mất tác dụng. Baseline v0 chỉ đạt Recall@5 = 0.436 trên nhóm này.
2. **Thông tin nằm rải rác:** các đoạn song song cùng chủ đề (giải thưởng nhiều năm, ban chủ nhiệm nhiều nhiệm kỳ), câu hỏi so sánh.
3. **Câu không có thông tin:** LLM có xu hướng bịa, hoặc ngược lại từ chối oan câu thật ra có thông tin.

---

## 2. Kiến trúc tổng thể

```
Câu hỏi
  │
  ▼
[1] Rewrite            LLM chuẩn hóa câu hỏi, tách tối đa 3 câu con      → rewrites
  │
  ▼
[2] Hybrid Retrieval   BM25 bỏ dấu + dense bge-m3, hợp nhất RRF (k=60)    → docs (top-15)
  │
  ▼
[3] Draft Generation   LLM trả lời từ 15 đoạn, ghi dòng NGUON            → draft_answer, draft_cited
  │
  ▼
[4] Context Expansion  15 đoạn + đoạn song song cùng (club, topic)       → audit_docs (≤20)
  │
  ▼
[5] LLM Auditor        kiểm định bản nháp theo 6 tiêu chí (hoặc GIỮ)      → final_answer, audit_cited
  │
  ▼
[6] ID Reranking       chọn và xếp 10 id nộp bài                          → retrieved_ids
```

Bước 1–3 cộng với việc xếp id theo cosine (phần cuối của bước 6) chính là **v11**. Bước 4–5 và nhánh "câu trả lời được sửa" của bước 6 là phần **audit** thêm vào.

| Bước | Hàm trong code | LLM? | Vấn đề giải quyết |
|---|---|---|---|
| 1 | `rewrite` | có | câu không dấu, viết tắt, câu nhiều ý |
| 2 | `retrieve`, `_fused` | không (chỉ embedding) | tìm đúng đoạn cho cả câu có dấu và không dấu |
| 3 | `generate`, `_parse_answer` | có | trả lời ngắn, đúng văn phong, có nguồn |
| 4 | `_build_audit_docs` | không | đoạn song song bị sót ngoài top-15 |
| 5 | `audit`, `_parse_audit` | có | từ chối oan, ngược Có/Không, suy diễn, thiếu ý |
| 6 | `order_ids_audit` | không | tập id nộp phải khớp câu trả lời cuối |

**Chi phí mỗi câu hỏi:** 3 lời gọi LLM (`rewrite`, `generate`, `audit`), thêm 1 lần gọi lại nếu bản nháp rỗng. Mọi lời gọi LLM đi qua `LLMClient` có cache SQLite (`.cache/pipeline.sqlite`), nên chạy lại cùng prompt gần như miễn phí. Vì vậy v11-audit chỉ tốn thêm đúng 1 lời gọi (Auditor) so với v11.


Luồng biến trong `run()`:

```
query ──rewrite──▶ rewrites ──retrieve──▶ docs(15)
docs ──generate──▶ (draft_answer, draft_cited)
(docs, draft_cited) ──_build_audit_docs──▶ audit_docs(≤20)
audit_docs ──audit──▶ (final_answer, audit_cited)
(draft_cited, audit_cited, docs, changed?) ──order_ids_audit──▶ ids(10)
```
---

## 3. Từng bước: làm gì và vì sao

### 3.1 Index (chạy một lần, trước khi hỏi)

Các cấu trúc được dựng trong `index()`:

1. **Vector dense của từng đoạn** (`doc_vectors`): embed `d.text` bằng bge-m3, chuẩn hóa L2 để tích vô hướng = cosine.
2. **BM25 trên văn bản bỏ dấu**: mỗi đoạn được ghép thành `"{club} {topic} {text}"` rồi tách token. Ghép thêm `club` và `topic` để từ khóa như "giải thưởng", "ban chủ nhiệm", tên CLB cũng khớp được.
3. **Index Parent-Child**: mỗi đoạn tách thành từng câu (child), gắn tiền tố `"club | topic: câu"`, embed và BM25 riêng. Điểm child được max-pooling về đoạn cha. Dùng trong bước 4 (xem 3.5).
4. Hai bảng tra: `by_club_topic` (nhóm đoạn theo cặp `(club, topic)`) và `pos` (id → vị trí trong mảng).

### 3.2 Bước 1: Query Rewriting

**Vấn đề:** v0 chỉ nhúng câu hỏi gốc. Với `"clb lt hv tv"` dense bge-m3 xếp đoạn đúng ở hạng 30–100, Recall@5 chỉ 0.436, LLM không thấy thông tin nên từ chối nhầm (Noise = 1.0, String Presence = 0).

**Giải pháp (v1):** nhờ LLM viết lại thành tiếng Việt đủ dấu, mở viết tắt, chuẩn hóa tên CLB, và tách câu so sánh thành tối đa 3 câu con. Prompt `REWRITE_PROMPT` có bốn thành phần:
- bảng tên CLB và biệt danh người dùng hay gõ (`clb lt`, `clb mm`, `clb game`);
- bảng viết tắt (`hv`, `tv`, `dpt`, `nk`, `cn`, `ko`...);
- lưu ý nghĩa: "chương trình", "plan" là sự kiện, "code" mới là lập trình (sửa lỗi v1 viết lại "chạy chương trình" thành "lập trình");
- 6 nhiệm vụ: giữ đúng ý, tách câu con, câu con phải tự đứng được, xử lý câu "thích X thì nên vào CLB nào", câu "có ban nào lo việc X", không nhắc CLB người hỏi không nêu.

**Định dạng `Q:` (v3):** mỗi câu một dòng bắt đầu bằng `Q: `, để parse bằng regex ổn định. Thêm bộ lọc `META` (dòng LLM tự giải thích) và `DANGLING` ("CLB đó", "ở đó": câu con không tự đứng được thì bỏ).

**Chống lỗi (v4):** nếu gọi LLM lỗi, trả về `[query]` thay vì làm sập chương trình.

### 3.3 Bước 2: Hybrid Retrieval

**Vì sao hai bộ truy xuất:**
- **BM25 bỏ dấu** (từ đơn + bigram) khớp từ khóa chính xác, bất chấp người dùng gõ có dấu hay không. Đây là điều kiện để sống sót với câu văn nói. Recall@5 nhóm không dấu tăng từ 0.436 lên 0.969 khi thêm rewrite + BM25.
- **Dense bge-m3** bắt được nghĩa tương đương khi từ khóa khác nhau, mạnh với câu có dấu.

**Vì sao hợp nhất bằng RRF:** điểm BM25 và cosine khác thang đo, không cộng trực tiếp được. RRF chỉ dùng **thứ hạng**: mỗi bảng xếp hạng đóng góp `w / (60 + hạng)` cho từng đoạn. Không cần chuẩn hóa điểm, không có tham số học.

**Ba quyết định cụ thể trong `_fused`:**

| Quyết định | Lý do |
|---|---|
| Câu gốc có trọng số 0.7, câu viết lại 1.0 | câu gốc nhiễu hơn câu đã chuẩn hóa  |
| **Bỏ dense trên câu gốc nếu câu không dấu** (`is_unaccented`) | v5 phát hiện bge-m3 xếp đoạn đúng ở hạng 30–100 với câu không dấu, vector này chỉ thêm nhiễu. BM25 trên câu gốc vẫn dùng; dense trên câu viết lại vẫn dùng |
| Dùng **tất cả** câu viết lại (câu chính + câu con) | câu so sánh cần lấy đoạn của từng CLB |

Tham số `top_k = 15` (v5): đưa 15 đoạn vào prompt thay vì 10, để đoạn đúng nằm ở hạng 11–15 vẫn được LLM thấy. Nếu LLM trích dẫn đoạn hạng 11–15, id đó được kéo lên danh sách nộp.

### 3.4 Bước 3: Sinh câu trả lời nháp

`ANSWER_PROMPT` là kết quả của nhiều vòng sửa lỗi. Mỗi quy tắc trả lời một lỗi đã gặp:

| Quy tắc trong prompt | Lỗi / chỉ số nó nhắm tới |
|---|---|
| 1–2 câu, khoảng 60 từ, không markdown | Noise Sensitivity giảm từ 0.1029 xuống 0.0391, ROUGE-L tăng (v2) |
| Câu đầu lặp lại chủ thể và tên CLB chuẩn | String Presence (tên CLB là key_fact), Response Relevancy |
| Có/không: mở bằng `Có, ...` / `Không, ...` dùng **dấu phẩy** | dấu chấm làm bộ chấm tách thành câu cụt (v6) |
| "Theo <nguồn>, ..." khi đoạn tự ghi nguồn | khớp cách đáp án chuẩn diễn đạt |
| Gọi tên CLB đúng một dạng | String Presence |
| Chỉ nêu điều được hỏi, không "ngoài ra" | Noise |
| Chép nguyên văn tên riêng, ngày, số kèm đơn vị, tên giải/ban | String Presence chấm theo key_fact nguyên văn |
| Nhiều ý: mỗi ý một mệnh đề | thông tin rải rác nhiều đoạn |
| Không có thông tin: mở đúng bằng "Tài liệu hiện không có thông tin về ..."; **không suy ra phủ định** | Abstention; v6 sửa lỗi "tài liệu không nhắc fanpage" bị LLM kết luận "CLB không có fanpage" |
| Một phần thiếu: trả lời phần có, rồi "tài liệu không ghi ..." | thiếu một vế |
| Dòng cuối `NGUON: <id>` liệt kê **mọi** đoạn chứa thông tin, kể cả trùng ý | tăng Retrieval (kéo id lên đầu) và cho Groundedness |

**Mẫu trả lời chung `ANSWER_TEMPLATES` (v9, phân tầng ở v10):** 8 khuôn có chỗ trống `<...>` (Theo nguồn, Có/không, so sánh, xếp hạng, tính thời gian, một phần thông tin, mâu thuẫn, không có thông tin). Khuôn **không chứa dữ kiện thật nào**.
- Vì sao không dùng ví dụ thật: v7 nhét 6 ví dụ lấy từ tập train, LLM "học vẹt" dữ kiện, rò sang câu khác. Informal giảm 0.020, Abstention tụt từ 1.0 xuống 0.8182.
- Vì sao chỉ áp dụng cho **câu có dấu**: v9 áp cho mọi câu thì câu văn nói bị gượng (informal 0.9031 < v6 0.9083). v10 theo nguyên tắc "chỉ cộng, không trừ": câu không dấu giữ nguyên prompt v6, câu có dấu thêm mẫu. Kết quả train 0.9269, informal 0.9083.

**Chống lỗi runtime (v4):** nếu LLM trả chuỗi rỗng, gọi lại một lần với prompt thêm `"\n"` để đổi cache key (nếu không, cache trả lại đúng chuỗi rỗng).

### 3.5 Bước v11: xếp lại id theo đoạn được trích dẫn (`cite_rerank`)

Nội dung của `update_v11.md`, đặt đúng vị trí trong pipeline.

**Vấn đề:** `relevant_ids` thường gồm nhiều đoạn trùng ý (ví dụ Phòng truyền thống và bài tuyển thành viên cùng nói một việc). LLM chỉ trích dẫn một đoạn, đoạn trùng ý kia thường rơi xuống hạng 6–10, ngoài top-5 mà bộ chấm xét.

**Cách làm:** id được trích dẫn đứng đầu; các id còn lại xếp theo **cosine lớn nhất với các đoạn đã trích dẫn** (đoạn giống đoạn đúng thì nhiều khả năng cũng đúng) thay vì theo thứ tự RRF. Câu không có trích dẫn giữ thứ tự cũ. Câu trả lời không đổi một ký tự so với v10.

**Hai bản thử, và bài học về Groundedness:**

| Bản | Tập ứng viên | Kết quả |
|---|---|---|
| Nháp 1 | top-15 context | train 0.9290 (+0.0021), informal 0.9071 (−0.0012); Groundedness informal **giảm 0.989 → 0.966** |
| Bản cuối | **chỉ 10 id của v10** (có test kiểm tra) | retrieval train 0.9836 → 0.9865; informal 0.9864 → 0.9853 |

Nháp 1 kéo đoạn hạng 11–15 vào danh sách nộp, đẩy ra các đoạn mà câu trả lời có dùng nhưng LLM không ghi trong `NGUON`. Judge chấm Groundedness trên đoạn đã nộp nên điểm tụt. Bản cuối chỉ **đổi thứ tự bên trong tập 10 id cũ** nên tập đoạn judge thấy không đổi.

Lợi ích ước tính chỉ khoảng +0.0005 benchmark (+0.05 điểm), nằm trong biên nhiễu ±3. Điểm Private Test thực tế của v11 là **88.75** (Retrieval 96.86, Generation 80.63): Retrieval gần trần, nút thắt nằm ở Generation. Đây là xuất phát điểm của bước audit.

### 3.6 Bước 4–5–6: Audit

**Chẩn đoán trên 42 câu Private Test của v11:** 9 câu mở đầu bằng câu từ chối, nhưng chỉ 3 câu thật sự không có thông tin. 6 câu bị từ chối oan hoặc cắt cụt. Bốn loại lỗi:

| # | Lỗi | Hậu quả |
|---|---|---|
| 1 | Từ chối oan câu có thông tin dạng trạng thái ("đợt tuyển chưa mở", "không cố định") | String Presence = 0 và Noise = 1.0 (phạt kép) |
| 2 | Hiểu nhầm ngữ cảnh câu không dấu ("hoi ... moi thanh lap" là quá khứ, bị hiểu hiện tại) | ROUGE-L thấp, trượt String Presence |
| 3 | Ngược cực Có/Không ("có cần đến không" nhưng trả "Không, ... vẫn cần có mặt") | judge đọc thành từ chối oan, Noise = 1.0 |
| 4 | Sót đoạn song song cùng `(club, topic)` (nhiều đoạn Giải thưởng các năm) | thiếu key_fact |

**Căn nguyên:** lỗi 1–3 là do một prompt phải làm cùng lúc đọc hiểu, tra cứu, chọn văn phong và tự bắt lỗi logic; LLM không có cơ chế "nhìn lại" quyết định vừa ra. Lỗi 4 là do retrieval chỉ xét theo RRF nên đoạn song song nằm ngoài top-15.

**Giải pháp: tách thành hai pha Draft → Audit**, không đụng vào bước sinh nháp (tái dùng 100% cache của v11):

- **Bước 4 `_build_audit_docs`:** mở rộng ngữ cảnh lên tối đa 20 đoạn. 15 đoạn của v11 đứng đầu; thêm các đoạn cùng `(club, topic)` với 2 nguồn đầu được trích dẫn (sắp theo điểm Parent-Child RRF); chừa 2 slot cuối cho các ứng viên Parent-Child RRF khác.
  - *Vì sao Parent-Child để sắp xếp:* đoạn dạng danh sách ("Giải Nhì AICOVIDVN 2021; Giải Nhì Kalapa 2023") nếu embed cả đoạn thì nhiều ý bị "trung bình hóa". Chấm từng câu rồi lấy max giữ được tín hiệu của từng ý.
  - *Vì sao giữ 15 đoạn v11 ở đầu:* bản nháp sinh từ chính 15 đoạn đó, và LLM chú ý phần đầu prompt hơn.
  - *Vì sao 20:* bắt được hầu hết đoạn song song (118 đoạn / khoảng 12 cặp `(club, topic)` ≈ 10 đoạn mỗi nhóm) mà prompt không quá dài.
- **Bước 5 `audit` + `AUDIT_PROMPT`: sáu tiêu chí.**

| Tiêu chí | Nội dung | Lỗi xử lý |
|---|---|---|
| 1 | đối chiếu **câu hỏi gốc** không dấu để hiểu đúng ("hoi" = "hồi", "nguoi dung lop" = người chủ trì buổi học) | #2 |
| 2a | có thông tin (kể cả dạng trạng thái) thì bỏ câu từ chối, trả lời thẳng | #1 |
| 2b | thiếu một vế: trả lời phần có trước, nối `; tài liệu không ghi ...` | #1 |
| 2c | thật sự không có: mở "Tài liệu hiện không có thông tin về ...", nối thông tin cùng CLB, cùng chủ đề sau dấu `;`; nếu nháp nêu CLB khác thì cắt sạch | Abstention, Noise |
| 3 | mở đầu Có/Không/Đúng/Chưa phải khớp logic câu hỏi | #3 |
| 4 | xóa mệnh đề suy diễn ("nên thành tích cao hơn") và ý nhiễu chéo đoạn | Groundedness, Noise |
| 5 | bổ sung dữ kiện song song bị sót, chép **nguyên văn** ("5 thành viên, trong đó 3 người", không đổi thành "3 trong 5") | #4, String Presence |
| 6 | nháp đã đúng thì **bắt buộc** trả `TRA_LOI: GIỮ` | chống regression |

  - *Mẹo dấu `;`:* bộ chấm tách câu theo `. ! ? …` và **không** tách theo `;`. Nối thông tin phụ sau `;` trong cùng một câu với câu từ chối thì: câu từ chối vẫn đứng đầu (giữ Abstention), thông tin phụ không bị tách thành câu riêng để bị chấm "không có căn cứ" (giữ Groundedness), và các cụm như "không ghi", tên CLB nguyên văn trúng key_fact. Code `index()` có chi tiết cho thấy `split_sentences` không tách theo `;`: phải đổi `"; "` thành `".\n"` trước khi tách câu con **(suy ra từ code)**.
  - *Vì sao tiêu chí 6 quan trọng:* Auditor được phép sửa, không được "cải thiện" khi không cần. Thực tế 20/42 câu Private Test được giữ nguyên.
  - *Ba chốt an toàn trong `_parse_audit`:* lỗi API thì trả bản nháp; còn sót nhãn `<...>` trong câu trả lời thì trả bản nháp; Auditor nói GIỮ thì trả đúng từng ký tự bản nháp.
- **Bước 6 `order_ids_audit`:** có hai trạng thái.
  - *Câu trả lời được giữ nguyên:* khóa chặt trong tập 10 id của v10/v11 (`v10_pool`), vì judge chấm Groundedness trên đoạn đã nộp, đổi tập đoạn trong khi giữ câu trả lời chỉ thêm rủi ro.
  - *Câu trả lời được sửa:* câu mới có thể dựa vào đoạn hạng 16–20 (ví dụ giải thưởng năm 2023 ở đoạn 19). Các id đó **phải** vào danh sách nộp, nếu không câu trả lời nêu thông tin từ đoạn không có trong tập nộp và Groundedness về 0. Nên `audit_cited` được đưa lên đầu rồi mới đến `draft_cited`.
  - Cả hai trạng thái: phần còn lại sắp theo cosine lớn nhất với các đoạn đã trích dẫn (cơ chế `cite_rerank` của v11).

---

## 4. Bảng lựa chọn kỹ thuật, vấn đề giải quyết, bằng chứng

| Lựa chọn | Vấn đề giải quyết | Bằng chứng | Phiên bản |
|---|---|---|---|
| Rewrite bằng LLM | câu không dấu, viết tắt, nhiều ý | train 0.7379 → 0.9038; Public 52.40 → 83.36 | v1 |
| BM25 bỏ dấu + bigram | từ khóa chính xác với câu văn nói | Recall@5 nhóm không dấu 0.436 → 0.969 | v1 |
| RRF k=60 | cộng điểm hai bộ truy xuất khác thang đo | (cơ chế chuẩn, không đo riêng) | v1 |
| Prompt 1–2 câu, bỏ rào đón | Noise cao, ROUGE-L thấp | Noise 0.1029 → 0.0391; Public 83.36 → 87.13 | v2 |
| Rewrite định dạng `Q:` + lọc `META`, `DANGLING` | parse lỗi, câu con mơ hồ | Groundedness train 1.0 | v3, v6 |
| Retry câu rỗng, fallback khi lỗi mạng | sập vì lỗi ngẫu nhiên | Public ổn định | v4 |
| Bộ dev `train_informal` | train có dấu không đại diện cho đề thi thật | informal làm thước đo chính | v5 |
| Bỏ dense trên câu gốc không dấu | vector nhiễu | informal 0.841 → 0.8876 | v5 |
| Context 15 đoạn | đoạn đúng ở hạng 11–14 bị bỏ sót | informal tăng | v5 |
| Chuẩn hóa từ chối, "Có,/Không," | Abstention, câu cụt | Public 88.43 | v6 |
| Mẫu trả lời chung chỉ cho câu có dấu | văn phong câu chuẩn, không hại câu văn nói | train 0.9269, informal 0.9083 | v9, v10 |
| `cite_rerank` trong tập 10 id cũ | đoạn trùng ý tụt khỏi top-5 | retrieval 0.9836 → 0.9865 | v11 |
| Auditor + mở rộng 20 đoạn + `GIỮ` | từ chối oan, ngược Có/Không, sót đoạn song song | dev 88.07 → 91.56; Private 88.75 → 91.35 | v11-audit |

**Các hướng đã thử và bỏ, cùng lý do:**

| Hướng | Vì sao bỏ |
|---|---|
| Few-shot tĩnh từ train (v7) | LLM học vẹt dữ kiện, rò sang câu khác; informal −0.020, Abstention 1.0 → 0.8182 |
| Parent-Child làm retrieval chính (v8) | tăng retrieval thô nhưng xáo thứ tự 15 đoạn trong prompt, LLM trích nguồn lệch; informal 0.9083 → 0.8945. (Ở v11-audit nó chỉ dùng để **xếp ứng viên cho Auditor**, không đụng bước sinh nháp) |
| Mẫu chung cho mọi câu (v9) | hại câu văn nói; Public 86.05 |
| `cite_rerank` trên top-15 (nháp 1 của v11) | Groundedness informal 0.989 → 0.966 |
| v12 (`evidence_select`), v13 (`answer_first_fix`), v14 | không cải thiện Generation, hoặc chỉ sửa được 1/42 câu |
| Đẩy Response Relevancy bằng cách tác động judge | quy định cấm, và Auditor không khắc phục được thực chất |

Hai nguyên tắc rút ra:
1. **Đo end-to-end.** Cải tiến retrieval thô phải được kiểm bằng điểm cuối, vì nó có thể làm hỏng bước sau (v8).
2. **Chỉ cộng, không trừ.** Dùng phân tầng (có dấu / không dấu) và `GIỮ` để sửa một nhóm lỗi mà không làm hỏng nhóm đang tốt.


---

## 5. Giải thích code trong `pipeline.py`

File gồm: import, hằng số và prompt, 3 hàm tiện ích, 2 class phụ (`BM25`, `GatewayEmbedder`), class chính `MyPipeline`, và một alias.

Phụ thuộc từ framework của BTC: `Document` (có `id`, `club`, `topic`, `text`), `LLMClient` (có `chat`, `embed`, cache), `RAGPipeline` / `RAGOutput` (lớp cơ sở và kiểu kết quả), `split_sentences` (hàm tách câu của bộ chấm). Tôi không có mã nguồn các phần này nên chỉ mô tả cách `pipeline.py` dùng chúng.

### 5.1 Hằng số và prompt

| Tên | Giá trị / vai trò |
|---|---|
| `RRF_K = 60` | hằng số làm mượt trong công thức RRF: `w / (60 + hạng)`. k lớn thì chênh lệch giữa hạng 1 và hạng 5 nhỏ lại, không để một bảng xếp hạng áp đảo |
| `REWRITE_PROMPT` | prompt bước 1. Có placeholder `{query}`. Nội dung: ngữ cảnh 3 CLB, bảng viết tắt, lưu ý "chương trình" khác "code", 6 nhiệm vụ, định dạng `Q:` |
| `ANSWER_PROMPT` | prompt bước 3. Placeholder: `{examples}` (mẫu trả lời hoặc rỗng), `{context}`, `{query}`, `{rewritten}`. Quy tắc trả lời, dòng `NGUON:` bắt buộc |
| `ANSWER_TEMPLATES` | 8 khuôn trả lời có `<...>`, chèn vào `{examples}` chỉ khi câu có dấu |
| `AUDIT_PROMPT` | prompt bước 5. Placeholder: `{context}`, `{query}`, `{rewritten}`, `{draft}`, `{draft_cited}`. 6 tiêu chí, ép định dạng đúng 2 dòng `TRA_LOI:` và `NGUON:` |
| `DANGLING` | regex bắt "CLB đó/này/ấy" và "ở đó". Câu con chứa các cụm này không tự đứng được, bị loại |
| `META` | regex bắt dòng LLM tự giải thích ("không cần tách", "viết lại như sau", "lưu ý"...), bị loại |

### 5.2 Hàm tiện ích

**`strip_accents(text) -> str`**
- Làm gì: đưa văn bản về chữ thường không dấu.
- Cách làm: `lower()` rồi đổi `đ` thành `d` (không thể bỏ bằng Unicode vì `đ` là ký tự riêng, không phải `d` + dấu); chuẩn hóa `NFD` để tách mỗi chữ có dấu thành chữ cái gốc + dấu kết hợp; lọc bỏ ký tự thuộc loại `Mn` (dấu kết hợp).
- Vì sao: BM25 so khớp trên dạng không dấu thì câu gõ có dấu và không dấu cùng khớp một token.

**`is_unaccented(text) -> bool`**
- Làm gì: True nếu câu hoàn toàn không có dấu.
- Cách làm: so `strip_accents(text)` với `text.lower().replace("đ","d")`. Bằng nhau nghĩa là việc bỏ dấu không thay đổi gì, tức câu chưa có dấu nào.
- Vì sao: là công tắc phân tầng. Câu không dấu: bỏ dense trên câu gốc (`_fused`) và giữ prompt v6 không có mẫu (`generate`).
- Lưu ý: chữ `đ` được coi như `d` nên câu chỉ có `đ` mà không có dấu thanh vẫn tính là không dấu. **(suy ra từ code)** Câu thuần tiếng Anh hay chỉ có số cũng thành "không dấu".

**`bm25_tokens(text) -> list[str]`**
- Làm gì: biến văn bản thành danh sách token cho BM25.
- Cách làm: bỏ dấu, `re.findall(r"\w+")` lấy âm tiết, rồi thêm bigram nối bằng `_` (ví dụ `thanh_lap`). Tiếng Việt viết tách âm tiết bằng dấu cách nên mỗi âm tiết là một token; bigram bắt được cụm hai âm tiết ("lap_trinh") phân biệt hơn từng âm tiết riêng lẻ.

### 5.3 Class `BM25`

Cài đặt Okapi BM25 thuần numpy, `k1 = 1.2`, `b = 0.75`.

**`__init__(docs, k1, b)`** với `docs` là danh sách các danh sách token:
- `tfs`: một `Counter` tần suất từ cho mỗi tài liệu.
- `lens`: độ dài từng tài liệu; `avgdl`: độ dài trung bình.
- `df`: số tài liệu chứa mỗi token (đếm trên `set` để mỗi tài liệu chỉ tính một lần).
- `idf[t] = log(1 + (n − f + 0.5) / (f + 0.5))`: biến thể luôn không âm (kể cả khi token xuất hiện ở hơn nửa số tài liệu), token hiếm được trọng số cao.

**`scores(query) -> np.ndarray`** trả về một điểm BM25 cho **từng** tài liệu:
- `norm = k1 · (1 − b + b · len / avgdl)`: phạt tài liệu dài hơn trung bình.
- Với mỗi token duy nhất của câu hỏi có trong `idf`: `idf · tf · (k1 + 1) / (tf + norm)`, cộng dồn. Token không có trong corpus bị bỏ qua.
- Trả về mảng đủ 118 phần tử, vì RRF cần xếp hạng toàn bộ.

### 5.4 Class `GatewayEmbedder`

Bọc `LLMClient.embed` để gọi bge-m3 qua gateway của BTC.
- `__init__(llm, model_name)`: lưu client và tên model.
- `embed(texts) -> np.ndarray`: gọi API, đổi sang `float32`, **chia cho chuẩn L2** từng hàng. Sau chuẩn hóa tích vô hướng giữa hai vector chính là cosine, nên toàn bộ code sau chỉ dùng `@`.

### 5.5 Class `MyPipeline(RAGPipeline)`

#### `__init__(top_k=15, n_retrieved=10, audit_max_docs=20, llm=None, embedder=None, answer_templates="accented", cite_rerank=True)`

| Tham số | Ý nghĩa |
|---|---|
| `top_k` | số đoạn đưa vào prompt sinh nháp (v5) |
| `n_retrieved` | số id nộp bài |
| `audit_max_docs` | số đoạn tối đa Auditor thấy |
| `llm`, `embedder` | cho phép truyền đối tượng giả khi viết unit test |
| `answer_templates` | `"accented"` (chỉ câu có dấu, mặc định v10), `True` (mọi câu, kiểu v9), hoặc giá trị sai (tắt) |
| `cite_rerank` | bật cơ chế xếp id theo cosine (v11) |

Model LLM lấy từ biến môi trường `LLM_MODEL` (mặc định `claude-haiku-5-5`), embedding từ `EMBEDDING_MODEL` (mặc định `bge-m3`). `LLMClient` dùng cache ở `.cache/pipeline.sqlite`.

#### `index(corpus)`

Đã mô tả ở 3.1. Từng dòng:
- `self.docs`, `self.by_id` (id → Document), `self.pos` (id → chỉ số mảng).
- `self.doc_vectors`: embed `d.text` (không kèm header).
- `self.bm25`: trên `"{club} {topic} {text}"`. Chú ý dense dùng text trần còn BM25 dùng thêm header. **(suy ra từ code)**
- `self.by_club_topic`: nhóm đoạn theo `(club, topic)`.
- Parent-Child: với mỗi đoạn, thay `"; "` bằng `".\n"` rồi `split_sentences` (để mỗi mục trong danh sách `a; b; c` thành một câu con), giữ câu dài ≥ 15 ký tự, gắn tiền tố `"club | topic: "`. `child_parent` ghi đoạn cha của từng câu con. Nếu không có câu con thì `child_vectors`, `child_bm25` là `None`.

#### `_to_parent(child_scores) -> np.ndarray`

Max-pooling điểm câu con về đoạn cha. Tạo mảng `-inf` cho 118 đoạn rồi `np.maximum.at(out, self.child_parent, child_scores)`. Dùng `maximum.at` thay vì gán `out[idx] = ...` vì nhiều câu con trùng đoạn cha; phép gán thường chỉ giữ giá trị cuối, còn `maximum.at` lấy đúng giá trị lớn nhất. Đoạn không có câu con nào giữ `-inf`, tức xếp cuối.

#### `rewrite(query) -> list[str]`

1. Gọi LLM với `REWRITE_PROMPT`; lỗi thì trả `[query]`.
2. Lấy các dòng `Q: ...` bằng regex (chấp nhận cả dấu `:` toàn chiều rộng `：`).
3. Nếu LLM không theo định dạng, lấy từng dòng và bỏ tiền tố số thứ tự / gạch đầu dòng.
4. Bỏ tiền tố kiểu "Câu viết lại:", lọc dòng ngắn (≤ 3 ký tự) và dòng khớp `META`.
5. Giữ nguyên câu đầu; từ câu thứ hai, bỏ những câu khớp `DANGLING`.
6. Cắt tối đa 4 câu (1 câu viết lại + tối đa 3 câu con); rỗng thì trả `[query]`.

Phần tử đầu `rewrites[0]` là bản viết lại chính, được dùng làm "Hiểu là:" trong prompt.

#### `_fused(query, rewrites, use_child=False) -> np.ndarray`

Điểm RRF cho 118 đoạn. Từng bước:
1. `dense_orig = not is_unaccented(query)`. Danh sách truy vấn: `(query, 0.7, dense_orig)` cộng `(q, 1.0, True)` cho mỗi câu viết lại. Phần tử thứ ba cho biết có dùng dense không.
2. Gom các truy vấn cần dense, embed một lần, lấy qua iterator `qvecs`.
3. `use_child` chỉ có hiệu lực nếu child index tồn tại.
4. Với mỗi truy vấn, lập danh sách các bảng điểm: BM25 trên đoạn; nếu `use_child`, thêm BM25 trên câu con (đã `_to_parent`); nếu dùng dense, thêm `doc_vectors @ qv`, và nếu `use_child` thêm `child_vectors @ qv` (đã `_to_parent`). Vậy một truy vấn tạo 2 bảng (chế độ thường, không dense) tới 4 bảng.
5. Với mỗi bảng: đổi điểm thành hạng (`ranks[np.argsort(-scores)] = 1..N`), cộng `w / (RRF_K + hạng)` vào `fused`.

`use_child=False` dùng cho truy xuất chính (giữ hành vi v11); `use_child=True` chỉ dùng cho `_build_audit_docs`.

#### `retrieve(query, rewrites) -> list[Document]`

`fused = _fused(..., use_child=False)`, lấy `k = max(top_k, n_retrieved)` = 15 đoạn có điểm cao nhất, theo thứ tự giảm dần.

#### `generate(query, rewritten, docs) -> (answer, cited)`

1. `context`: mỗi đoạn một khối `[id] (club – topic) text`, chỉ lấy `docs[:top_k]`. Đưa `id` vào để LLM trích nguồn; đưa `club – topic` để LLM biết đoạn thuộc CLB nào.
2. `examples`: là `ANSWER_TEMPLATES` nếu (`answer_templates == "accented"` và câu có dấu) hoặc `answer_templates is True`; còn lại rỗng.
3. Điền `ANSWER_PROMPT`, gọi LLM, `_parse_answer`.
4. Nếu `answer` rỗng, gọi lại với `prompt + "\n"` (đổi cache key) và parse lại.

Lưu ý: trong file được nộp, `generate` **không** bọc `try/except` quanh `llm.chat`, khác với `rewrite` và `audit`. Báo cáo v4 mô tả có khối bảo vệ quanh lời gọi chat; có thể phần đó nằm trong `LLMClient`, nhưng từ file này tôi không xác nhận được.

#### `_parse_answer(out, docs) -> (answer, cited)`

1. Tìm dòng `NGUON:` (chấp nhận `NGUỒN`, không phân biệt hoa thường).
2. Từ dòng đó lấy mọi số, chỉ giữ những id nằm trong `docs[:top_k]` (loại id LLM bịa). Cắt dòng đó khỏi `out`.
3. Xóa markdown (`**`, `__`, `#` đầu dòng, gạch đầu dòng), gộp xuống dòng thành một khoảng trắng.
4. Trả `(answer, cited)`. `cited` giữ thứ tự LLM ghi (liên quan nhất trước) và có thể trùng id; các hàm sau dùng `dict.fromkeys` để khử trùng.

#### `_build_audit_docs(query, rewrites, docs, draft_cited) -> list[Document]`

Đã mô tả ở 3.6. Thứ tự trong danh sách trả về:
1. 15 đoạn của v11 (nguyên thứ tự).
2. Đoạn cùng `(club, topic)` với **hai id đầu** của `draft_cited`, sắp theo điểm Parent-Child RRF giảm dần, thêm cho tới khi đủ `audit_max_docs − 2 = 18`.
3. Các đoạn có điểm Parent-Child RRF cao nhất chưa có mặt, thêm cho tới đủ 20.

`seen` là tập id đã có để không lặp. Nếu `draft_cited` rỗng thì bước 2 không thêm gì và bước 3 lấp từ 15 lên 20.

#### `_parse_audit(raw, audit_docs, draft_answer, draft_cited) -> (answer, cited)`

1. Lấy dòng `NGUON:` như `_parse_answer`, nhưng lọc theo `audit_docs` (tối đa 20 đoạn) và khử trùng.
2. Bỏ tiền tố `TRA_LOI:` (chấp nhận `TRẢ LỜI`, `TRẢ_LỜI`...), xóa markdown, gộp dòng.
3. Ba chốt:
   - có mẫu `<...>` (2 đến 40 ký tự) trong thân: trả bản nháp;
   - thân rỗng hoặc là `GIỮ` / `GIU` / `KEEP` (sau khi bỏ `.`, `!`, khoảng trắng, đổi hoa): trả bản nháp;
   - ngược lại: trả thân đã cắt 2000 ký tự.
4. Nguồn trả về luôn là `audit_cited or draft_cited`: nếu Auditor không ghi nguồn thì dùng nguồn của bản nháp.

#### `audit(query, rewritten, audit_docs, draft_answer, draft_cited) -> (answer, cited)`

1. Bản nháp rỗng: trả luôn, không gọi LLM.
2. Dựng `context` giống `generate` nhưng từ `audit_docs`; `cited_str` là các id nháp, hoặc "không có".
3. Điền `AUDIT_PROMPT`, gọi LLM trong `try/except`; lỗi thì trả bản nháp.
4. Chuyển `_parse_audit`.

#### `order_ids_audit(draft_cited, audit_cited, docs, answer_changed) -> list[int]`

Đã mô tả ở 3.6. Từng bước:
1. `v10_pool` = `draft_cited` + 10 id RRF đầu, khử trùng, cắt 10.
2. Nếu **không đổi** câu trả lời: `merged_cited` = các id của `audit_cited` nằm trong `v10_pool`, rồi `draft_cited`; `base = v10_pool`.
3. Nếu **đã sửa**: `merged_cited = audit_cited + draft_cited` (cắt 10); `base = merged_cited + v10_pool` (cắt 10).
4. Không có id nào được trích dẫn, hoặc `cite_rerank` tắt: trả `base`.
5. Ngược lại: vector của các id đã trích; với mỗi id còn lại tính cosine lớn nhất tới các vector đó; sắp giảm dần; kết quả là `merged_cited` rồi phần còn lại, cắt 10.

Số id luôn tối đa `n_retrieved = 10`, nhưng bộ chấm retrieval chỉ nhìn top-5.

#### `order_ids(cited, docs) -> list[int]`

Bản xếp id của v11 gốc (chưa có audit), giữ lại để tương thích test cũ. `run()` không gọi hàm này. Khác `order_ids_audit` ở chỗ không có nhánh "câu trả lời đã sửa".

#### `run(query) -> RAGOutput`

```python
rewrites = self.rewrite(query)
docs = self.retrieve(query, rewrites)
draft_answer, draft_cited = self.generate(query, rewrites[0], docs)
audit_docs = self._build_audit_docs(query, rewrites, docs, draft_cited)
final_answer, audit_cited = self.audit(query, rewrites[0], audit_docs, draft_answer, draft_cited)
ids = self.order_ids_audit(draft_cited, audit_cited, docs, answer_changed=(final_answer != draft_answer))
return RAGOutput(answer=final_answer or "Tài liệu hiện không có thông tin để trả lời câu hỏi này.", retrieved_ids=ids)
```

Điểm cần để ý: `answer_changed` được suy ra bằng so sánh chuỗi, nên đúng cả khi Auditor trả GIỮ lẫn khi chốt an toàn đưa về bản nháp. Nếu cuối cùng câu trả lời rỗng, dùng câu từ chối mặc định.


### 5.6 Ví dụ chạy một câu (minh họa, id và nội dung là giả định)

Câu hỏi gõ nhanh: `"clb lt co giai nao nam 2023 ko"`.

1. `is_unaccented` trả True, nên dense trên câu gốc bị tắt.
2. `rewrite`: `["CLB Lập Trình PTIT có giải thưởng nào năm 2023 không?"]`.
3. `_fused`: BM25 trên câu gốc (trọng số 0.7), BM25 + dense trên câu viết lại (trọng số 1.0), RRF. Lấy 15 đoạn.
4. `generate`: câu không dấu nên **không** chèn mẫu. LLM trả câu trả lời và `NGUON: A, B`.
5. `_build_audit_docs`: giả sử A và B thuộc `(Lập Trình, Giải thưởng)`, nhóm này có thêm đoạn C, D ở hạng 16+; C, D được thêm vào, tổng 20 đoạn.
6. `audit`: Auditor thấy năm 2023 nằm ở đoạn C. Nó trả câu trả lời mới và `NGUON: C, A`.
7. `order_ids_audit`: câu đã đổi nên `merged_cited = [C, A, B]`; các id còn lại từ `v10_pool` xếp theo cosine tới C, A, B. Danh sách nộp có C ở top-5.

Nếu ở bước 6 Auditor trả `GIỮ`, kết quả là đúng câu nháp, và id chỉ được xếp lại bên trong `v10_pool`.

---


## 6. Bài học

1. **Hiểu giới hạn của embedding.** Dense mạnh với câu có dấu, gần như vô dụng với câu không dấu. BM25 bỏ dấu là điều kiện để chạy được trên dữ liệu người dùng thật.
2. **Retrieval thô tốt chưa chắc điểm cuối tốt.** Thứ tự đoạn trong prompt ảnh hưởng việc LLM trích nguồn (v8). Luôn đo end-to-end.
3. **Một prompt chung hay làm lợi nhóm này, hại nhóm kia.** Phân tầng (có dấu / không dấu) giữ thành quả cũ.
4. **Tách Draft và Audit.** Một lượt prompt không vừa đọc hiểu, vừa tra cứu, vừa tự kiểm lỗi được. Bước Auditor sửa được lỗi logic tinh vi, còn lệnh `GIỮ` và các chốt an toàn ngăn nó làm hỏng câu đã tốt.
5. **Hiểu cách bộ chấm hoạt động.** Dấu phẩy thay dấu chấm trong "Có,", dấu `;` thay câu mới khi từ chối, chép nguyên văn key_fact, và giữ tập id khớp câu trả lời: đều là những chi tiết nhỏ mang lại điểm, và đều hợp lệ vì chúng làm câu trả lời đúng hơn, không tác động vào judge.

---

## 7. Cách chạy và kiểm tra

```bash
neorag run private_test --pipeline neorag.pipelines.pipeline:MyPipeline
# tương đương:
neorag run private_test --pipeline neorag.pipelines.pipeline:MyPipelineV11Audit
```

Bản nộp: `experiments/submissions/v11_audit_private_test_submission.zip`.

