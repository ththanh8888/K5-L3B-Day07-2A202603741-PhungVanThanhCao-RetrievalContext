"""Tính Context Evidence Rate, abstain, thời gian, chi phí từ traces của agent trên dev/test.

    python src/eval_traces.py starter/results/agent-dev-* [--split dev]
"""
import json
import sys
from pathlib import Path

qs = {q["id"]: q for q in json.loads(Path("data/questions.json").read_text(encoding="utf-8"))}
chunks = {c["id"]: c for c in json.loads(Path("data/chunks.json").read_text(encoding="utf-8"))}


def ev_in(ids, gold):
    return any(chunks[i]["source_id"] == g["source_id"] and chunks[i]["page"] == g["page"] and g["evidence"] in chunks[i]["text"]
               for i in ids for g in gold)


for d in sys.argv[1:]:
    tr = json.loads((Path(d) / "traces.json").read_text(encoding="utf-8"))
    ans = [t for t in tr if qs[t["query_id"]].get("answerable") is not False]
    una = [t for t in tr if qs[t["query_id"]].get("answerable") is False]
    cer = [t["query_id"] for t in ans if ev_in(t["final_context_ids"], qs[t["query_id"]]["gold"])]
    cited = [t["query_id"] for t in ans if t["status"] == "answered" and ev_in(t["citations"], qs[t["query_id"]]["gold"])]
    out = {"dir": Path(d).name, "config": tr[0]["retriever"] + " " + json.dumps(tr[0]["config"]),
           "context_evidence": f"{len(cer)}/{len(ans)} ({len(cer)/len(ans):.0%})",
           "cited_gold": f"{len(cited)}/{len(ans)}",
           "answerable_abstained": [t["query_id"] for t in ans if t["status"] == "abstain"],
           "ctx_miss": [t["query_id"] for t in ans if t["query_id"] not in cer],
           "unanswerable_abstained": f"{sum(t['status']=='abstain' for t in una)}/{len(una)}",
           "errors": sum(t["status"] == "error" for t in tr),
           "avg_searches": round(sum(t["n_searches"] for t in tr) / len(tr), 2),
           "seconds": round(sum(t["seconds"] for t in tr), 1), "cost_usd": round(sum(t["cost_usd"] for t in tr), 4)}
    print(json.dumps(out, ensure_ascii=False))
