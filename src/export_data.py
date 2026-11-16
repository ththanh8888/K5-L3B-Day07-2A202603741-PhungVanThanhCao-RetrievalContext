"""Xuất data/chunks.json và data/questions.json từ Day 6 (Chroma day06_B + questions.json)."""
import json, os
from pathlib import Path
import chromadb
from dotenv import load_dotenv

load_dotenv()
D6 = Path("starter/data/day6")
col = chromadb.PersistentClient(path=os.environ["CHROMA_DIR"]).get_collection(os.environ["CHROMA_COLLECTION"])
r = col.get(include=["documents", "metadatas"])
chunks = []
for cid, text, m in zip(r["ids"], r["documents"], r["metadatas"]):
    m = dict(m)
    chunks.append({"id": cid, "text": text, "source_id": m.pop("document_id"), "source": m.pop("source"),
                   "page": m.pop("page"), "metadata": m})
chunks.sort(key=lambda c: (c["metadata"]["chapter"], c["metadata"]["unit_id"], c["metadata"]["chunk_index"]))

# Câu có đáp án: tìm chunk chứa trọn evidence, lấy page của chunk đó
questions = []
for q in json.loads((D6 / "questions.json").read_text(encoding="utf-8")):
    ev = q["evidence_quote"]
    hits = [c for c in chunks if c["source_id"] == q["gold_source_id"] and c["metadata"]["unit_id"] in q["gold_page"] and ev in c["text"]]
    if not hits:
        raise SystemExit(f"{q['id']}: evidence không nằm trọn trong một chunk: {ev!r}")
    item = {"id": q["id"], "split": q["split"], "question": q["question"], "expected_answer": q["expected_answer"],
            "gold": [{"source_id": h["source_id"], "page": h["page"], "evidence": ev} for h in hits[:1]]}
    if q.get("filter"):
        item["filter"] = q["filter"]
    questions.append(item)

# Câu thiếu bằng chứng
questions += json.loads(Path("data/absent_questions.json").read_text(encoding="utf-8"))
Path("data/chunks.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
Path("data/questions.json").write_text(json.dumps(questions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(len(chunks), "chunks;", len(questions), "questions")
