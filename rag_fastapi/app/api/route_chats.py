from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException

from app.api.schemas import ChatRequest, ChatResponse, Usage
from app.core.settings import settings
from app.db.mongo import get_db
from app.services.prompt_service import (
    build_system_prompt,
    LENGTH_SETTINGS,
    fallback_not_found,
    is_greeting,
    greeting_reply,
)
from app.services.rag_services import retrieve_context, answer_with_llm, calc_cost

router = APIRouter()


def now():
    return datetime.now(timezone.utc)


DEFAULT_ROLE = "Help Desk Specialist"
DEFAULT_TONE = "Friendly"
DEFAULT_LENGTH = "Short"

MAX_MESSAGES_STORE = 300    # stored per user
HISTORY_FOR_LLM = 50        # used for LLM history


@router.get("/health")
async def health():
    return {"ok": True}


async def ensure_user_doc(col, user_id: str):
    await col.update_one(
        {"_id": user_id},
        {
            "$setOnInsert": {
                "_id": user_id,
                "created_at": now(),
                "messages": [],
                "settings": {"role": DEFAULT_ROLE, "tone": DEFAULT_TONE, "length": DEFAULT_LENGTH},
                "usage": {"emb_tokens": 0, "chat_in_tokens": 0, "chat_out_tokens": 0, "total_cost_usd": 0.0},
            },
            "$set": {"updated_at": now()},
        },
        upsert=True,
    )


async def reset_user(col, user_id: str):
    await col.update_one(
        {"_id": user_id},
        {
            "$set": {
                "messages": [],
                "usage": {"emb_tokens": 0, "chat_in_tokens": 0, "chat_out_tokens": 0, "total_cost_usd": 0.0},
                "updated_at": now(),
            }
        },
    )


async def push_msg(col, user_id: str, role: str, content: str, base_url=None):
    msg = {"role": role, "content": content, "base_url": base_url, "created_at": now()}
    await col.update_one(
        {"_id": user_id},
        {
            "$set": {"updated_at": now()},
            "$push": {"messages": {"$each": [msg], "$slice": -MAX_MESSAGES_STORE}},
        },
    )


async def load_doc(col, user_id: str):
    return await col.find_one({"_id": user_id}) or {}


async def load_history(col, user_id: str):
    doc = await col.find_one(
        {"_id": user_id},
        {"messages": {"$slice": -HISTORY_FOR_LLM}, "settings": 1, "usage": 1},
    ) or {}

    msgs = doc.get("messages", [])
    history = [{"role": m.get("role"), "content": m.get("content")} for m in msgs]
    return history, doc


@router.post("/v1/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    db = get_db()
    col = db["messages"]  # ONE doc per user

    user_id = req.user_id
    message = (req.message or "").strip()

    await ensure_user_doc(col, user_id)

    # reset chat if needed
    if req.reset:
        await reset_user(col, user_id)

    # ===== CASE 1: SETTINGS-ONLY payload (no message) =====
    if not message:
        if not (req.role and req.tone and req.length):
            raise HTTPException(
                status_code=400,
                detail="Settings payload must include role, tone, length (or send question for chat).",
            )

        await col.update_one(
            {"_id": user_id},
            {"$set": {"settings": {"role": req.role, "tone": req.tone, "length": req.length}, "updated_at": now()}},
        )

        doc = await load_doc(col, user_id)
        usage = doc.get("usage", {"emb_tokens": 0, "chat_in_tokens": 0, "chat_out_tokens": 0, "total_cost_usd": 0.0})

        return ChatResponse(
            mode="settings",
            answer="Settings saved successfully.",
            usage=Usage(**usage),
            effective_settings=doc.get("settings", {"role": DEFAULT_ROLE, "tone": DEFAULT_TONE, "length": DEFAULT_LENGTH}),
            debug={},
        )

    # ===== CASE 2: CHAT payload =====
    history, doc = await load_history(col, user_id)

    stored = doc.get("settings") or {"role": DEFAULT_ROLE, "tone": DEFAULT_TONE, "length": DEFAULT_LENGTH}
    role = req.role or stored.get("role", DEFAULT_ROLE)
    tone = req.tone or stored.get("tone", DEFAULT_TONE)
    length = req.length or stored.get("length", DEFAULT_LENGTH)

    # if chat request included settings, persist them (optional)
    if req.role or req.tone or req.length:
        await col.update_one(
            {"_id": user_id},
            {"$set": {"settings": {"role": role, "tone": tone, "length": length}, "updated_at": now()}},
        )

    usage = doc.get("usage", {"emb_tokens": 0, "chat_in_tokens": 0, "chat_out_tokens": 0, "total_cost_usd": 0.0})

    # store user msg
    await push_msg(col, user_id, "user", message, base_url=None)

    # ✅ GREETING BYPASS (NO Pinecone / NO LLM)
    if is_greeting(message):
        answer = greeting_reply(role, tone, length)
        await push_msg(col, user_id, "assistant", answer, base_url=None)

        return ChatResponse(
            mode="chat",
            answer=answer,
            base_url=None,
            sources=[],
            usage=Usage(**usage),
            effective_settings={"role": role, "tone": tone, "length": length},
            debug={"small_talk": "greeting"},
        )

    # reload history after storing user message
    history, doc = await load_history(col, user_id)
    usage = doc.get("usage", usage)

    # RAG retrieve (namespace = user_id)
    r = await retrieve_context(
        user_id=user_id,
        question=message,
        length=length,
        score_threshold=settings.DEFAULT_SCORE_THRESHOLD,
    )

    usage["emb_tokens"] += int(r.get("emb_tokens") or 0)
    usage["total_cost_usd"] += float(calc_cost(emb_tokens=r.get("emb_tokens") or 0))

    # no context => skip LLM
    if not (r.get("context") or "").strip():
        answer = fallback_not_found(length)

        await push_msg(col, user_id, "assistant", answer, base_url=r.get("base_url"))
        await col.update_one({"_id": user_id}, {"$set": {"usage": usage, "updated_at": now()}})

        return ChatResponse(
            mode="chat",
            answer=answer,
            base_url=r.get("base_url"),
            sources=(r.get("sources") or [])[:10],
            usage=Usage(**usage),
            effective_settings={"role": role, "tone": tone, "length": length},
            debug={"retrieved_cnt": r.get("retrieved_cnt"), "missing_text_cnt": r.get("missing_text_cnt")},
        )

    # LLM answer
    system_prompt = build_system_prompt(role, tone, length)
    max_out = LENGTH_SETTINGS[length]["max_out"]

    ans, in_tok, out_tok = await answer_with_llm(
        system_prompt=system_prompt,
        context=r["context"],
        history=history,
        question=message,
        max_out=max_out,
    )

    usage["chat_in_tokens"] += int(in_tok or 0)
    usage["chat_out_tokens"] += int(out_tok or 0)
    usage["total_cost_usd"] += float(calc_cost(chat_in=in_tok or 0, chat_out=out_tok or 0))

    await push_msg(col, user_id, "assistant", ans, base_url=r.get("base_url"))
    await col.update_one({"_id": user_id}, {"$set": {"usage": usage, "updated_at": now()}})

    return ChatResponse(
        mode="chat",
        answer=ans,
        base_url=r.get("base_url"),
        sources=(r.get("sources") or [])[:10],
        usage=Usage(**usage),
        effective_settings={"role": role, "tone": tone, "length": length},
        debug={"retrieved_cnt": r.get("retrieved_cnt"), "missing_text_cnt": r.get("missing_text_cnt")},
    )
