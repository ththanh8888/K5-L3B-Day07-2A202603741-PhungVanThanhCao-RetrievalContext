# Day 7 — Agentic RAG

**Bài cá nhân · Python 3.11+ · tiếp nối Day 6**

Dùng embedding và vector store từ Day 6 để xây một agent biết tìm bằng chứng, trả lời có nguồn và nhận biết khi không đủ thông tin. Bạn tự chọn kiến trúc, số tool và cách tổ chức retrieval.

> **Tùy chọn:** [Gợi ý dùng AI để lập trình](AGENT_SETUP.md) giúp bạn làm lab; phần này không bắt buộc và không tính điểm.

## Bài cần làm

![Agent tìm tài liệu trong kho Day 6 và trả lời có nguồn](assets/retrieval-pipeline.svg)

1. Kết nối dữ liệu, embedding và vector store Day 6.
2. Xây agent dùng model thật và ít nhất một khả năng truy xuất dữ liệu. Agent có thể dùng nhiều tool và nhiều lượt gọi.
3. Thử **ít nhất 3 phương pháp retrieval khác nhau**, trong đó có dense retrieval dùng vector store Day 6. Tự chọn các phương pháp còn lại; xem [gợi ý](GUIDE.md#2-thử-các-phương-pháp-retrieval).
4. Thử nhiều cấu hình, benchmark, phân tích lỗi và chọn cấu hình tốt trên **dev**; sau đó đánh giá trên **test**.

Benchmark tối thiểu có **24 câu**: dev gồm 15 câu có đáp án và 2 câu thiếu bằng chứng; test gồm 5 câu có đáp án và 2 câu thiếu bằng chứng. Xem [benchmark](BENCHMARK.md) và [rubric](RUBRIC.md).

## Thử thách nâng cao · +15 điểm

Index một tài liệu thực tế từ **100 trang** có bảng, ảnh và biểu đồ; xây agent truy xuất và trả lời các câu hỏi về nội dung đó. Hoàn thành [thử thách nâng cao](ADVANCED.md) và đạt **ít nhất 90% benchmark test** để nhận 15 điểm bonus. Phần cơ bản tối đa 85 điểm; toàn bài tối đa 100 điểm.

## Bắt đầu

1. Đọc [hướng dẫn](GUIDE.md), [rubric](RUBRIC.md) và [quy tắc](RULES.md).
2. Tạo repo cá nhân theo [cách nộp bài](SUBMISSION.md), chuẩn bị đầu ra Day 6 và model hỗ trợ tool calling.
3. Dùng hoặc thay thế [starter](starter/README.md); bạn cần kết nối model và vector store thật.
4. Chạy `python -m unittest discover -s tests -v`. Test mock chỉ kiểm tra code, không phải benchmark agent thật.

![Các bước thử nghiệm trên dev và đánh giá trên test](assets/experiment-loop.svg)

Nộp code, cách chạy, cấu hình, benchmark, trace và [báo cáo](report/individual/MSSV_HoTen.md). Thời gian ở [checkpoints](CHECKPOINTS.md) là gợi ý, có thể điều chỉnh theo lớp và phạm vi thử nghiệm.


---

## Bài làm: cách chạy và tái tạo (2A202603741 — Phùng Văn Thanh Cao)

**Môi trường:** Python 3.13, `pip install -r requirements.txt`. Tạo `.env` từ `.env.example` (không commit `.env`). Trên Windows đặt `PYTHONUTF8=1`.

**Model và phiên bản:**
- LLM: OpenAI `gpt-4o-mini` (tool calling), `LLM_PROVIDER=openai`.
- Embedding: `intfloat/multilingual-e5-small`, 384 chiều.
- Vector store: Chroma `starter/data/day6/chroma_db`, collection `day06_B` (250 chunk, từ Day 6).
- Dữ liệu: Project Gutenberg #11 (*Alice's Adventures in Wonderland*), public domain tại Hoa Kỳ.

**Tái tạo dữ liệu và kết quả:**
```bash
python src/check_setup.py                       # kiểm tra Chroma, embedding, LLM tool calling
python src/export_data.py                       # -> data/chunks.json, data/questions.json (đọc data/absent_questions.json)
python src/run_retrieval_eval.py --split dev --output starter/results/retrieval-dev   # dense, BM25, hybrid RRF
python src/agent.py --split dev --retriever hybrid_rrf --neighbors 1 --budget 4000 --output starter/results/agent-dev-exp4-neighbors-b4000
python src/eval_traces.py starter/results/agent-dev-*   # Context Evidence Rate, chi phí
python src/agent.py --split test --retriever hybrid_rrf --neighbors 1 --budget 4000 --output starter/results/agent-test-chosen
python starter/agent_metrics.py --cases starter/results/agent-test-chosen/cases.json --traces starter/results/agent-test-chosen/traces.json --reviews starter/results/agent-test-chosen/reviews.json --split test --output starter/results/agent-test-chosen/summary.json
```
`starter/baseline.py` chỉ chấp nhận câu có `gold`, nên khi dùng cần lọc bỏ câu thiếu bằng chứng (`answerable: false`).

**Mã nguồn:** `src/retrieval.py` (dense, BM25, hybrid RRF), `src/agent.py` (agent và chọn context), `src/export_data.py`, `src/run_retrieval_eval.py`, `src/eval_traces.py`. **Báo cáo:** [report/individual/2A202603741_PhungVanThanhCao.md](report/individual/2A202603741_PhungVanThanhCao.md). Kết quả và phiếu chấm nằm ở `starter/results/` (dùng `git add -f` để thêm khi nộp).
