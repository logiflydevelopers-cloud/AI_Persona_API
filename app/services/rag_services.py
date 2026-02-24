from urllib.parse import urlparse
from starlette.concurrency import run_in_threadpool
from openai import AsyncOpenAI

try:
    from pinecone.grpc import PineconeGRPC as Pinecone
except Exception:
    from pinecone import Pinecone

from app.core.settings import settings
from app.services.prompt_service import LENGTH_TO_CONTEXT, LENGTH_TO_TOPK


TEXT_META_KEYS = ["text", "content", "chunk", "page_content", "body"]
URL_META_KEYS = ["url", "source_url", "page_url", "source"]


openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

pc = Pinecone(api_key=settings.PINECONE_API_KEY)
pinecone_index = pc.Index(settings.PINECONE_INDEX_NAME)


# =========================
# Helpers
# =========================

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


# =========================
# Embedding
# =========================

async def embed_query(text: str):
    response = await openai_client.embeddings.create(
        model=settings.EMB_MODEL,
        input=[text]
    )
    return response.data[0].embedding


# =========================
# Retrieval
# =========================

async def retrieve_context(
    userId: str,
    question: str,
    length: str,
    score_threshold: float
):
    top_k = LENGTH_TO_TOPK.get(length, 5)
    max_context_chars = LENGTH_TO_CONTEXT.get(length, 6000)

    q_vec = await embed_query(question)

    # Pinecone query is sync → run in threadpool
    res = await run_in_threadpool(
        pinecone_index.query,
        vector=q_vec,
        top_k=top_k,
        namespace=str(userId),
        include_metadata=True,
        include_values=False
    )

    matches = getattr(res, "matches", []) or []

    # Filter by similarity threshold
    filtered = [
        m for m in matches
        if float(getattr(m, "score", 0.0)) >= score_threshold
    ]

    # Sort by highest similarity first
    filtered.sort(
        key=lambda m: float(getattr(m, "score", 0.0)),
        reverse=True
    )

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

        src = (
            md.get("source")
            or md.get("url")
            or md.get("source_url")
            or "unknown"
        )

        score = float(getattr(m, "score", 0.0))
        sources.append(f"{src} (score={score:.3f})")
        chunks.append(txt)

    base_url = guess_base_url(url_candidates)

    context = "\n\n---\n\n".join(chunks)

    # Safe truncation (avoid cutting mid-line)
    if len(context) > max_context_chars:
        context = context[:max_context_chars]
        context = context.rsplit("\n", 1)[0]
        context += "\n\n[Context truncated.]"

    return {
        "context": context,
        "sources": sources,
        "base_url": base_url,
        "retrieved_cnt": len(filtered),
        "missing_text_cnt": missing_text_count,
    }


# =========================
# LLM Answer
# =========================

async def answer_with_llm(
    system_prompt: str,
    context: str,
    history: list[dict],
    question: str,
    max_out: int
):
    last_msgs = history[-6:]  # limit memory window

    messages = [{"role": "system", "content": system_prompt}]

    for m in last_msgs:
        messages.append({
            "role": m.get("role", "user"),
            "content": m.get("content", "")
        })

    messages.append({
        "role": "user",
        "content": f"Context:\n{context}\n\nUser question: {question}"
    })

    response = await openai_client.chat.completions.create(
        model=settings.CHAT_MODEL,
        messages=messages,
        temperature=0.2,   # low temp for grounded RAG
        max_tokens=max_out,
    )

    answer = (response.choices[0].message.content or "").strip()

    return answer