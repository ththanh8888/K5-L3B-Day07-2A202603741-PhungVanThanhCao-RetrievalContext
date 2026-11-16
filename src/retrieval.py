"""Ba phương pháp retrieval: dense (Chroma Day 6), BM25, hybrid RRF.

Giao diện chung (khớp starter/agentic.py::Retriever):
    search(query, *, filters=None, top_k=3) -> list[{id,text,source_id,source,page,metadata,score,rank,strategy}]
Lỗi kết nối được ném ra (RetrievalError), không thay bằng kết quả giả.
"""
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


class RetrievalError(RuntimeError):
    pass


def _match(meta, filters):
    return all(meta.get(k) == v for k, v in (filters or {}).items())


def _out(chunk, score, rank, strategy):
    return {**chunk, "score": float(score), "rank": rank, "strategy": strategy}


class DenseRetriever:
    name = "dense"

    def __init__(self, chunks):
        import chromadb
        from sentence_transformers import SentenceTransformer
        try:
            self.model_name = os.environ["EMBED_MODEL"]
            self.col = chromadb.PersistentClient(path=os.environ["CHROMA_DIR"]).get_collection(os.environ["CHROMA_COLLECTION"])
            meta = self.col.metadata or {}
            if meta.get("embedding_model") != self.model_name:
                raise RetrievalError(f"EMBED_MODEL={self.model_name} khác model của index ({meta.get('embedding_model')})")
            self.model = SentenceTransformer(self.model_name)
        except RetrievalError:
            raise
        except Exception as e:
            raise RetrievalError(f"Không kết nối được vector store/embedding: {e}") from e
        self.by_id = {c["id"]: c for c in chunks}

    def search(self, query, *, filters=None, top_k=3):
        if not query or not query.strip():
            return []
        try:
            q = self.model.encode(["query: " + query.strip()], normalize_embeddings=True).tolist()
            res = self.col.query(query_embeddings=q, n_results=top_k, include=["distances"],
                                 where=filters or None)
        except Exception as e:
            raise RetrievalError(f"Truy vấn dense lỗi: {e}") from e
        return [_out(self.by_id[i], 1 - d, r, self.name)
                for r, (i, d) in enumerate(zip(res["ids"][0], res["distances"][0]), 1)]


def tokenize(text):
    return re.findall(r"\w+", text.lower())


class BM25Retriever:
    name = "bm25"

    def __init__(self, chunks):
        from rank_bm25 import BM25Okapi
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])

    def search(self, query, *, filters=None, top_k=3):
        toks = tokenize(query or "")
        if not toks:
            return []
        scores = self.bm25.get_scores(toks)
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], self.chunks[i]["id"]))
        hits = [i for i in order if scores[i] > 0 and _match(self.chunks[i]["metadata"], filters)][:top_k]
        return [_out(self.chunks[i], scores[i], r, self.name) for r, i in enumerate(hits, 1)]


class HybridRetriever:
    """Reciprocal Rank Fusion của dense và BM25: score = sum 1/(k + rank)."""
    name = "hybrid_rrf"

    def __init__(self, dense, bm25, rrf_k=60, pool=20):
        self.parts, self.rrf_k, self.pool = (dense, bm25), rrf_k, pool

    def search(self, query, *, filters=None, top_k=3):
        fused, store = {}, {}
        for p in self.parts:
            for c in p.search(query, filters=filters, top_k=self.pool):
                fused[c["id"]] = fused.get(c["id"], 0) + 1 / (self.rrf_k + c["rank"])
                store[c["id"]] = c
        ranked = sorted(fused, key=lambda i: (-fused[i], i))[:top_k]
        return [_out({k: v for k, v in store[i].items() if k not in ("score", "rank", "strategy")}, fused[i], r, self.name)
                for r, i in enumerate(ranked, 1)]


def load_chunks(path="data/chunks.json"):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_all(chunks_path="data/chunks.json"):
    chunks = load_chunks(chunks_path)
    dense, bm25 = DenseRetriever(chunks), BM25Retriever(chunks)
    return {"dense": dense, "bm25": bm25, "hybrid_rrf": HybridRetriever(dense, bm25)}
