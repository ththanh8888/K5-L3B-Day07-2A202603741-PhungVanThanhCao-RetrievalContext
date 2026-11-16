# Báo cáo Day 7 — 2A202603741 — Phùng Văn Thanh Cao

## Dữ liệu và benchmark

- Nguồn, quyền sử dụng, số trang và số chunk: *Alice's Adventures in Wonderland* (Lewis Carroll), Project Gutenberg eBook #11 (`starter/data/day6/pg11.txt`, snapshot 2025-06-26, public domain tại Hoa Kỳ, kèm Project Gutenberg License). 12 chương, 82 unit, 250 chunk, khoảng 142.8k ký tự đã chuẩn hóa. File text không có số trang; `page` trong chunk là chỉ số do Day 6 gán, không phải trang in. Ước tính độ dài ≈ 45–50 trang in, **chưa kiểm chứng**, nên chưa chắc đạt yêu cầu ≥30 trang theo bản in.
- Vector store, embedding model, phiên bản và số chiều: Chroma persistent (`starter/data/day6/chroma_db`), collection `day06_B` (250 chunk), embedding `intfloat/multilingual-e5-small` (sentence-transformers), 384 chiều, query dùng tiền tố `query: `, vector chuẩn hóa. `src/check_setup.py` kiểm tra model của query khớp với index.
- Agent model, cách chạy và phân chia dev/test: OpenAI `gpt-4o-mini` (tool calling, temperature 0). Dev: q01–q15 (có đáp án) + q21–q22 (thiếu bằng chứng). Test: q16–q20 + q23–q24. Test chỉ chạy sau khi chốt cấu hình trên dev, mỗi cấu hình một lần.
- Cách xác định câu có/thiếu evidence và tiêu chí chấm: 20 câu có đáp án lấy từ benchmark Day 6; evidence là đoạn nguyên văn nằm trọn trong một chunk, được kiểm tra bằng code (`src/export_data.py`). 4 câu thiếu bằng chứng (q21–q24) tự viết, gần chủ đề, mỗi câu có `absence_reason` và đã grep `pg11.txt` để xác nhận thông tin không có. Chấm tay theo thang answer_grade 0/1/2, citation_support đúng/sai từng citation, context 0/1/2, abstention_correct (phiếu: `starter/results/PHIEU_CHAM_TEST.md`).

## Retrieval và thử nghiệm

Dev có N=15 câu có đáp án, test có N=5. Hit@1/Hit@3 đo trên query gốc bằng `src/run_retrieval_eval.py`; agent tự viết lại query nên với các cấu hình agent chỉ báo Context Evidence Rate (ô "—" là không đo). Thời gian/chi phí của các hàng agent là tổng cho 17 câu dev (hoặc 7 câu test) với gpt-4o-mini; retrieval riêng lẻ có chi phí 0 USD (dev: dense 26.9 ms/câu, BM25 1.3 ms, hybrid 25.0 ms).

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| Baseline (dense) | Dense e5-small, top_k 3, filter theo câu hỏi, budget 2000 | Phương pháp bắt buộc | dev | 10/15 (67%) | 11/15 (73%) | 9/15 (60%) | 36.2 s / $0.0029 |
| Phương pháp 2 (BM25) | BM25 (`rank_bm25`, tách từ regex), top_k 3, budget 2000 | Từ khóa trùng nguyên văn nên lexical mạnh | dev | 12/15 (80%) | 13/15 (87%) | 13/15 (87%) | 39.2 s / $0.0032 |
| Phương pháp 3 (Hybrid RRF) | RRF k=60 của dense và BM25 (20 ứng viên mỗi bên), top_k 3, budget 2000 | Kết hợp ngữ nghĩa và từ khóa | dev | 10/15 (67%) | 13/15 (87%) | 12/15 (80%) | 45.0 s / $0.0031 |
| Cải thiện 1 | Hybrid + chunk lân cận (neighbors=1), budget 2000 | Evidence nằm ở chunk liền kề chunk tìm được | dev | — | — | 12/15 (80%) | 37.9 s / $0.0032 |
| Cải thiện 2 | Hybrid top_k 5, budget 3500 | Tăng recall | dev | — | — | 13/15 (87%) | 42.0 s / $0.0042 |
| Cải thiện 3 (đã chọn) | Hybrid, top_k 3, neighbors=1, budget 4000 | Cải thiện 1 không đổi vì budget 2000 không chứa nổi chunk lân cận; nới budget | dev | — | — | 14/15 (93%) | 38.8 s / $0.0044 |
| Cải thiện 4 | Hybrid top_k 5, neighbors=1, budget 6000 | Kết hợp cải thiện 2 và 3 | dev | — | — | 14/15 (93%) | 43.5 s / $0.0057 |
| Thử thêm | Hybrid, max_searches 3 | Tìm nhiều lượt | dev | — | — | 12/15 (80%) | 37.1 s / $0.0032 |
| Cấu hình đã chọn | Hybrid, top_k 3, neighbors=1, budget 4000 | Chốt trên dev | test | 3/5 (60%) | 4/5 (80%) | 4/5 (80%) | 27.4 s / $0.0021 |
| Baseline | Hybrid, top_k 3, neighbors=0, budget 2000 | So sánh | test | 3/5 (60%) | 4/5 (80%) | 4/5 (80%) | 19.7 s / $0.0014 |

Ghi chú: Hit@1/Hit@3 của hai hàng test là của retrieval hybrid trên query gốc (giống nhau cho cả hai cấu hình). Retrieval riêng trên test: dense 4/5 và 4/5, BM25 3/5 và 4/5, hybrid 3/5 và 4/5. Bảng chi tiết dev: `starter/results/dev-selection.md`.

**Hai lỗi dev:**
1. **q06 "Who was declared the winner of the Caucus-race?"** Hybrid tìm đúng `B:ch03-u03:000`, nhưng câu "has won, and all must have prizes" nằm ở chunk liền kề `B:ch03-u03:001`; agent lặp lại gần như cùng một query lần hai rồi từ chối. Giả thuyết: giữ chunk lân cận. Neighbors với budget 2000 không cải thiện (12/15) vì chunk lân cận không vừa budget. Nới budget lên 4000 thì sửa được: hết bị từ chối, Context Evidence lên 14/15.
2. **q08 "In the Rabbit's house, what did Alice drink…" (filter chapter 4).** Evidence "no label this time…" diễn đạt khác với câu hỏi "what did Alice drink", nên hybrid xếp `B:ch04-u02:001` ngoài top-3 (BM25 thì tìm được). Thử top_k 5 và neighbors đều **không cải thiện**; đây là giới hạn của hybrid khi từ vựng khác nhau.

**Lý do chọn cấu hình:** tiêu chí đặt sau khi đã xem kết quả dev (không dùng test): Context Evidence Rate cao nhất; hòa thì ít câu có đáp án bị từ chối hơn, rồi chi phí thấp hơn. Cấu hình đã chọn đạt 14/15 trên dev, không từ chối nhầm câu nào, và rẻ hơn cấu hình top_k 5 có cùng kết quả. **Kết quả test không xác nhận ưu thế:** cấu hình chọn bằng baseline ở Context Evidence (4/5), kém hơn ở citation hợp lệ (3/5 so với 4/5) và grounded answer (1.4 so với 1.6). Dev chỉ có 15 câu và test 5 câu, nên chênh lệch có thể là nhiễu; cấu hình được chọn trên dev nên có thể khớp quá với dev.

## Agent và đánh giá

- Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi: một tool `search` (retriever dense, BM25 hoặc hybrid RRF); lượt đầu bắt buộc gọi tool. `max_searches` 2, `top_k` 3, context budget tính bằng ký tự (4000 cho cấu hình chọn). Dừng khi model trả lời, hết `max_searches`, hoặc câu trả lời thiếu citation hợp lệ (thành abstain). Context: loại chunk trùng, xếp theo thứ tự tìm được, thêm chunk lân cận cùng chương sau các chunk tìm được, cắt theo budget (`pack_context`). System prompt yêu cầu: chỉ trả lời từ context, có citation `[chunk-id]`, nói "Không đủ bằng chứng" khi thiếu, coi nội dung tài liệu là dữ liệu chứ không phải chỉ dẫn. Lỗi model/vector store được ghi `status: "error"` kèm thông báo, không thay bằng kết quả giả; 0 lỗi ở mọi lần chạy.

| Cấu hình / split | Tool use | Số lượt gọi | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| Baseline / test | 7/7 | 1 lượt: 4 câu; 2 lượt: 3 câu | 1/0/4, TB 1.6/2 | 4/5 | 5/5 | 2/2 | 0/5 |
| Đã chọn / test | 7/7 | 1 lượt: 5 câu; 2 lượt: 2 câu | 1/1/3, TB 1.4/2 | 3/5 | 5/5 | 2/2 | 0/5 |

Điểm context (0/1/2) trên 5 câu test có đáp án, tự chấm (số câu điểm 0/1/2): baseline 1/0/4, đã chọn 1/0/4. Câu điểm 0 là q16 ở cả hai.

Bảng dưới là cấu hình đã chọn.

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| q16 | Có đáp án | search "riddle Hatter Alice" → ch07-u03:001 và các chunk liền kề; không có chunk chứa câu đố | Kể đoạn "Have you guessed the riddle yet?", không nêu câu đố; cite ch07-u03:001 | grade 0, citation sai, context 0: câu đố "raven/writing-desk" nằm ở ch07-u01/u02 |
| q17 | Có đáp án | search "Mock Turtle lessons hours a day change over time" → ch09-u07:003 | 10 giờ ngày đầu, 9 giờ ngày sau, giảm dần; cite ch09-u07:003 | grade 2, citation đúng, context 2 |
| q18 | Có đáp án | search "Rule Forty-two" → ch12-u02:001 | Nêu đúng quy tắc; cite ch12-u02:001 | grade 2, citation đúng, context 2 |
| q19 | Có đáp án | search "Queen want to come first sentence or verdict" → ch12-u05:001 (+ lân cận ch12-u05:000) | Đúng nội dung (sentence trước) nhưng cite ch12-u05:000 | grade 1, citation sai (chunk Knave/thơ), context 2: chunk gold có trong context, agent cite nhầm chunk lân cận |
| q20 | Có đáp án | search "witness called after the Hatter" → ch11-u06:001 | Cook của Duchess, pepper-box; cite ch11-u06:001 | grade 2, citation đúng, context 2 |
| q23 | Thiếu bằng chứng | search "Cheshire Cat fur color" ×2; không chunk nào nêu màu lông | "Không đủ bằng chứng để trả lời." | abstention đúng |
| q24 | Thiếu bằng chứng | search "Mad Hatter real name" / "Mad Hatter name"; không chunk nào nêu tên | "Không đủ bằng chứng để trả lời." | abstention đúng |

Baseline cite ch12-u05:001 cho q19 nên citation đúng và grade 2; các câu còn lại giống cấu hình đã chọn.

- Đường dẫn trace, nhãn, lệnh chạy, thời gian và chi phí: `starter/results/agent-test-{baseline,chosen}/` (`cases.json`, `traces.json`, `reviews.json`, `summary.json`); dev ở `starter/results/agent-dev-*`. Lệnh: `python src/agent.py --split test --retriever hybrid_rrf --neighbors 1 --budget 4000 --output starter/results/agent-test-chosen`; baseline bỏ `--neighbors` và `--budget`. Tổng hợp bằng `starter/agent_metrics.py` (xem README). Test: baseline 19.7 s / $0.0014 (7804 token vào, 464 ra); đã chọn 27.4 s / $0.0021 (12018 vào, 444 ra).
- Kết quả trước/sau, ví dụ lỗi, đánh đổi và giới hạn:
  - Neighbors giúp q06 trên dev (câu từ chối nhầm thành đúng) nhưng làm q19 trên test cite nhầm chunk lân cận: thêm chunk lân cận vừa tăng recall vừa gây nhiễu citation, và tăng token vào (12.0k so với 7.8k).
  - q16 sai ở cả hai cấu hình vì retrieval không tìm được chunk chứa câu đố.
  - Giới hạn: một cuốn sách tiếng Anh, không có số trang thật, không có bảng/ảnh; dev 15 câu và test 5 câu quá nhỏ để kết luận chắc chắn; cấu hình được chọn sau khi xem kết quả dev; tokenizer BM25 là regex đơn giản; chưa có rerank hay bước viết lại query riêng; mỗi cấu hình chỉ chạy một lần (temperature 0).

## Thử thách nâng cao (nếu thực hiện)

Không thực hiện.
