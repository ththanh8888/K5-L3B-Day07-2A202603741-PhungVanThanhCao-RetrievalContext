"""Smoke test: Day 6 Chroma store + e5 query embedding + LLM tool calling.

Run from repo root:  python src/check_setup.py
"""
import os
import sys

from dotenv import load_dotenv

load_dotenv()
ok = True

# 1) Vector store + query embedding (must match the Day 6 index)
import chromadb
from sentence_transformers import SentenceTransformer

model_name = os.environ["EMBED_MODEL"]
col = chromadb.PersistentClient(path=os.environ["CHROMA_DIR"]).get_collection(os.environ["CHROMA_COLLECTION"])
meta = col.metadata or {}
print(f"[store] {col.name}: {col.count()} chunks | index model={meta.get('embedding_model')} dim={meta.get('dimensions')}")
if meta.get("embedding_model") != model_name:
    print(f"[store] LỖI: EMBED_MODEL={model_name} khác model của index")
    ok = False

model = SentenceTransformer(model_name)
q = model.encode(["query: Alice follows a rabbit and falls down a deep hole"], normalize_embeddings=True)
print(f"[embed] query dim = {q.shape[1]}")
if q.shape[1] != meta.get("dimensions"):
    print("[embed] LỖI: số chiều query khác index")
    ok = False
res = col.query(query_embeddings=q.tolist(), n_results=3, include=["metadatas", "distances"])
for cid, m, d in zip(res["ids"][0], res["metadatas"][0], res["distances"][0]):
    print(f"  {cid}  score={1 - d:.3f}  chapter={m.get('chapter')} page={m.get('page')}")

# 2) LLM with tool calling
provider = os.getenv("LLM_PROVIDER", "gemini").lower()
prompt = "Use the search tool to find what the White Rabbit took out of its waistcoat-pocket."
if provider == "openai":
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        print("[llm] LỖI: OPENAI_API_KEY trống trong .env")
        sys.exit(1)
    import json
    from openai import OpenAI

    resp = OpenAI(api_key=key).chat.completions.create(
        model=os.environ["LLM_MODEL"],
        messages=[{"role": "user", "content": prompt}],
        tools=[{"type": "function", "function": {
            "name": "search", "description": "Search the Alice in Wonderland corpus",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}])
    calls = [(c.function.name, json.loads(c.function.arguments)) for c in (resp.choices[0].message.tool_calls or [])]
else:
    key = os.getenv("GEMINI_API_KEY", "")
    if not key:
        print("[llm] LỖI: GEMINI_API_KEY trống trong .env (đã lưu file chưa?)")
        sys.exit(1)
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=key)
    tool = types.Tool(function_declarations=[types.FunctionDeclaration(
        name="search", description="Search the Alice in Wonderland corpus",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]})])
    resp = client.models.generate_content(
        model=os.environ["LLM_MODEL"],
        contents=prompt,
        config=types.GenerateContentConfig(tools=[tool]))
    calls = [(c.name, dict(c.args)) for c in (resp.function_calls or [])]
print(f"[llm] {provider}/{os.environ['LLM_MODEL']} tool calls: {calls}")
if not calls:
    print("[llm] LỖI: model không gọi tool")
    ok = False

print("SETUP OK" if ok else "SETUP CÓ LỖI")
sys.exit(0 if ok else 1)
