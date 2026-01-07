from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException
import re
from typing import Optional

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

MAX_MESSAGES_STORE = 300
HISTORY_FOR_LLM = 50


# ======================================================
# STRICT EMAIL VALIDATION
# ======================================================
EMAIL_REGEX = re.compile(
    r"""
    ^
    (?![._-])
    (?!.*[._-]{2})
    [a-zA-Z0-9._-]{1,64}
    (?<![._-])
    @
    (?!-)
    (?:[a-zA-Z0-9-]{1,63}\.)+
    [a-zA-Z]{2,63}
    $
    """,
    re.VERBOSE | re.IGNORECASE,
)


def extract_email_from_text(text: str) -> Optional[str]:
    if not text or "@" not in text:
        return None

    text = text.strip()

    if EMAIL_REGEX.match(text):
        return text.lower()

    for token in text.split():
        token = token.strip(".,;:()[]<>\"'")
        if EMAIL_REGEX.match(token):
            return token.lower()

    return None


@router.get("/health")
async def health():
    return {"ok": True}


# ======================================================
# DB HELPERS
# ======================================================
async def ensure_user_doc(col, user_id: str):
    await col.update_one(
        {"_id": user_id},
        {
            "$setOnInsert": {
                "_id": user_id,
                "created_at": now(),
                "messages": [],
                "settings": {
                    "role": DEFAULT_ROLE,
                    "tone": DEFAULT_TONE,
                    "length": DEFAULT_LENGTH,
                },
                "usage": {
                    "emb_tokens": 0,
                    "chat_in_tokens": 0,
                    "chat_out_tokens": 0,
                    "total_tokens": 0,
                    "total_cost_usd": 0.0,
                },
                "email_prompted": False,
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
                "usage": {
                    "emb_tokens": 0,
                    "chat_in_tokens": 0,
                    "chat_out_tokens": 0,
                    "total_tokens": 0,
                    "total_cost_usd": 0.0,
                },
                "email_prompted": False,
                "updated_at": now(),
            }
        },
    )


async def push_msg(col, user_id: str, role: str, content: str, base_url=None):
    msg = {
        "role": role,
        "content": content,
        "base_url": base_url,
        "created_at": now(),
    }
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
    doc = (
        await col.find_one(
            {"_id": user_id},
            {"messages": {"$slice": -HISTORY_FOR_LLM}, "settings": 1, "usage": 1, "mail": 1, "email_prompted": 1},
        )
        or {}
    )

    history = [{"role": m["role"], "content": m["content"]} for m in doc.get("messages", [])]
    return history, doc


# ======================================================
# CHAT API
# ======================================================
@router.post("/v1/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    db = get_db()
    col = db["messages"]

    user_id = req.user_id
    message = (req.message or "").strip()

    await ensure_user_doc(col, user_id)

    if req.reset:
        await reset_user(col, user_id)

    # ===== SETTINGS ONLY =====
    if not message:
        if not (req.role and req.tone and req.length):
            raise HTTPException(status_code=400, detail="Settings payload incomplete")

        await col.update_one(
            {"_id": user_id},
            {"$set": {"settings": {"role": req.role, "tone": req.tone, "length": req.length}, "updated_at": now()}},
        )

        doc = await load_doc(col, user_id)
        return ChatResponse(
            mode="settings",
            answer="Settings saved successfully.",
            usage=Usage(**doc["usage"]),
            effective_settings=doc["settings"],
            debug={},
        )

    # ===== CHAT =====
    history, doc = await load_history(col, user_id)
    usage = doc["usage"]

    # Store user message
    await push_msg(col, user_id, "user", message)

    # Silent email capture
    email = extract_email_from_text(message)
    if email:
        await col.update_one(
            {"_id": user_id, "mail": {"$exists": False}},
            {"$set": {"mail": email, "updated_at": now()}},
        )

    # ======================================================
    # GREETING + EMAIL ASK (ONCE)
    # ======================================================
    if is_greeting(message):
        answer = greeting_reply(DEFAULT_ROLE, DEFAULT_TONE, DEFAULT_LENGTH)

        if not doc.get("mail") and not doc.get("email_prompted"):
            answer += (
                "\n\nIf you'd like updates or a summary later, "
                "you can share your email anytime — totally optional 😊"
            )
            await col.update_one(
                {"_id": user_id},
                {"$set": {"email_prompted": True, "updated_at": now()}},
            )

        await push_msg(col, user_id, "assistant", answer)

        return ChatResponse(
            mode="chat",
            answer=answer,
            usage=Usage(**usage),
            effective_settings=doc["settings"],
            debug={"small_talk": "greeting"},
        )

    # ======================================================
    # RAG FLOW
    # ======================================================
    r = await retrieve_context(
        user_id=user_id,
        question=message,
        length=doc["settings"]["length"],
        score_threshold=settings.DEFAULT_SCORE_THRESHOLD,
    )

    usage["emb_tokens"] += int(r.get("emb_tokens") or 0)
    usage["total_cost_usd"] += float(calc_cost(emb_tokens=r.get("emb_tokens") or 0))

    if not (r.get("context") or "").strip():
        usage["total_tokens"] = usage["emb_tokens"] + usage["chat_in_tokens"] + usage["chat_out_tokens"]
        await col.update_one({"_id": user_id}, {"$set": {"usage": usage, "updated_at": now()}})
        answer = fallback_not_found(doc["settings"]["length"])
        await push_msg(col, user_id, "assistant", answer)
        return ChatResponse(mode="chat", answer=answer, usage=Usage(**usage))

    ans, in_tok, out_tok = await answer_with_llm(
        system_prompt=build_system_prompt(**doc["settings"]),
        context=r["context"],
        history=history,
        question=message,
        max_out=LENGTH_SETTINGS[doc["settings"]["length"]]["max_out"],
    )

    usage["chat_in_tokens"] += int(in_tok or 0)
    usage["chat_out_tokens"] += int(out_tok or 0)
    usage["total_tokens"] = usage["emb_tokens"] + usage["chat_in_tokens"] + usage["chat_out_tokens"]
    usage["total_cost_usd"] += float(calc_cost(chat_in=in_tok or 0, chat_out=out_tok or 0))

    await push_msg(col, user_id, "assistant", ans)
    await col.update_one({"_id": user_id}, {"$set": {"usage": usage, "updated_at": now()}})

    return ChatResponse(
        mode="chat",
        answer=ans,
        usage=Usage(**usage),
        effective_settings=doc["settings"],
        debug={},
    )
