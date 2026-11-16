"""Chạy 3 phương pháp trên một split, báo Hit@1, Evidence Hit@3 (chỉ câu có đáp án).

    python src/run_retrieval_eval.py --split dev --output starter/results/retrieval-dev
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from retrieval import build_all, load_chunks


def hit(c, gold):
    return any(c["source_id"] == g["source_id"] and c["page"] == g["page"] and g["evidence"] in c["text"] for g in gold)


ap = argparse.ArgumentParser()
ap.add_argument("--split", required=True, choices=("dev", "test"))
ap.add_argument("--output", required=True, type=Path)
a = ap.parse_args()
qs = [q for q in json.loads(Path("data/questions.json").read_text(encoding="utf-8"))
      if q["split"] == a.split and q.get("answerable") is not False]
retrievers = build_all()
a.output.mkdir(parents=True, exist_ok=True)
summary, traces = {}, []
for name, r in retrievers.items():
    h1 = h3 = 0
    t0 = time.perf_counter()
    for q in qs:
        res = r.search(q["question"], filters=q.get("filter"), top_k=3)
        a1, a3 = any(hit(c, q["gold"]) for c in res[:1]), any(hit(c, q["gold"]) for c in res[:3])
        h1 += a1; h3 += a3
        traces.append({"strategy": name, "query_id": q["id"], "hit1": a1, "hit3": a3,
                       "results": [{k: c[k] for k in ("id", "score", "rank", "source", "page")} for c in res]})
    ms = (time.perf_counter() - t0) * 1000 / len(qs)
    summary[name] = {"n": len(qs), "hit1": f"{h1}/{len(qs)} ({h1/len(qs):.0%})",
                     "hit3": f"{h3}/{len(qs)} ({h3/len(qs):.0%})", "avg_ms": round(ms, 1), "cost_usd": 0}
(a.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
(a.output / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
