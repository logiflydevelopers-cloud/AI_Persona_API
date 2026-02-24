from urllib.parse import urlparse
from starlette.concurrency import run_in_threadpool
from openai import AsyncOpenAI

try:
    from pinecone.grpc import PineconeGRPC as Pinecone
except Exception:
    from pinecone import Pinecone  # fallback

from app.core.settings import settings
from app.services.prompt_service import LENGTH_TO_CONTEXT, LENGTH_TO_TOPK

TEXT_META_KEYS = ["text", "content", "chunk", "page_content", "body"]
URL_META_KEYS = ["url", "source_url", "page_url", "source"]

# USD rates (same as you)
CHAT_IN_RATE = 0.15 / 1_000_000
CHAT_OUT_RATE = 0.60 / 1_000_000
EMB_RATE = 0.02 / 1_000_000

openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

pc = Pinecone(api_key=settings.PINECONE_API_KEY)
pinecone_index = pc.Index(settings.PINECONE_INDEX_NAME)

def calc_cost(emb_tokens=0, chat_in=0, chat_out=0) -> float:
    return (emb_tokens or 0) * EMB_RATE + (chat_in or 0) * CHAT_IN_RATE + (chat_out or 0) * CHAT_OUT_RATE

def guess_base_url(urls: list[str]) -> str | None:
    for u in urls:
        if not u:
            continue
        u = u.strip()
        if u.startswith("http://") or u.startswith("https://"):
            p = urlparse(u)
            if p.scheme and p.netloc:
                return f"{p.scheme}://{p.netloc}"
    return None

def extract_text_from_metadata(md: dict) -> str:
    if not md:
        return ""
    for k in TEXT_META_KEYS:
        v = md.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""

def extract_url_candidates(md: dict) -> list[str]:
    urls = []
    if not md:
        return urls
    for k in URL_META_KEYS:
        v = md.get(k)
        if isinstance(v, str) and (v.startswith("http://") or v.startswith("https://")):
            urls.append(v.strip())
    return urls

async def embed_query(text: str):
    r = await openai_client.embeddings.create(model=settings.EMB_MODEL, input=[text])
    vec = r.data[0].embedding
    emb_tokens = getattr(r.usage, "total_tokens", 0)
    return vec, emb_tokens

async def retrieve_context(user_id: str, question: str, length: str, score_threshold: float):
    top_k = LENGTH_TO_TOPK.get(length, 5)
    max_context_chars = LENGTH_TO_CONTEXT.get(length, 6000)

    q_vec, emb_tokens = await embed_query(question)

    # pinecone query is sync -> run in threadpool
    res = await run_in_threadpool(
        pinecone_index.query,
        vector=q_vec,
        top_k=top_k,
        namespace=str(user_id),
        include_metadata=True
    )

    matches = getattr(res, "matches", []) or []
    filtered = [m for m in matches if float(getattr(m, "score", 0.0)) >= score_threshold]

    chunks = []
    sources = []
    missing_text_count = 0
    url_candidates = []

    for m in filtered:
        md = getattr(m, "metadata", {}) or {}
        txt = extract_text_from_metadata(md)
        if not txt:
            missing_text_count += 1
            continue

        url_candidates.extend(extract_url_candidates(md))
        src = md.get("source") or md.get("url") or md.get("source_url") or "unknown"
        score = float(getattr(m, "score", 0.0))
        sources.append(f"{src} (score={score:.3f})")
        chunks.append(txt)

    base_url = guess_base_url(url_candidates)

    context = "\n\n---\n\n".join(chunks)
    if len(context) > max_context_chars:
        context = context[:max_context_chars] + "\n\n[Context truncated to save tokens.]"

    return {
        "context": context,
        "sources": sources,
        "base_url": base_url,
        "emb_tokens": emb_tokens,
        "retrieved_cnt": len(filtered),
        "missing_text_cnt": missing_text_count,
    }

async def answer_with_llm(system_prompt: str, context: str, history: list[dict], question: str, max_out: int):
    # keep small for cost (same as you)
    last_msgs = history[-6:]

    messages = [{"role": "system", "content": system_prompt}]
    for m in last_msgs:
        messages.append({"role": m["role"], "content": m["content"]})

    messages.append({"role": "user", "content": f"Context:\n{context}\n\nUser question: {question}"})

    r = await openai_client.chat.completions.create(
        model=settings.CHAT_MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=max_out,
    )

    ans = (r.choices[0].message.content or "").strip()
    in_tok = getattr(r.usage, "prompt_tokens", 0)
    out_tok = getattr(r.usage, "completion_tokens", 0)
    return ans, in_tok, out_tok
