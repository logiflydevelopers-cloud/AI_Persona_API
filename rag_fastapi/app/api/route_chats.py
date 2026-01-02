from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException

from app.api.schemas import ChatRequest, ChatResponse, Usage
from app.core.settings import settings
from app.db.mongo import get_db
from app.services.prompt_service import build_system_prompt, LENGTH_SETTINGS
from app.services.rag_services import retrieve_context, answer_with_llm, calc_cost

router = APIRouter()

def now():
    return datetime.now(timezone.utc)

DEFAULT_ROLE = "Help Desk Specialist"
DEFAULT_TONE = "Friendly"
DEFAULT_LENGTH = "Short"

MAX_MESSAGES_STORE = 300   # stored per user
HISTORY_FOR_LLM = 50       # used for LLM history

@router.get("/health")
async def health():
    return {"ok": True}

async def ensure_user_doc(col, user_id: str):
    await col.update_one(
        {"_id": user_id},
        {"$setOnInsert": {
            "_id": user_id,
            "created_at": now(),
            "messages": [],
            "settings": {"role": DEFAULT_ROLE, "tone": DEFAULT_TONE, "length": DEFAULT_LENGTH},
            "usage": {"emb_tokens": 0, "chat_in_tokens": 0, "chat_out_tokens": 0, "total_cost_usd": 0.0},
        }, "$set": {"updated_at": now()}},
        upsert=True
    )

async def reset_user(col, user_id: str):
    await col.update_one(
        {"_id": user_id},
        {"$set": {
            "messages": [],
            "usage": {"emb_tokens": 0, "chat_in_tokens": 0, "chat_out_tokens": 0, "total_cost_usd": 0.0},
            "updated_at": now(),
        }}
    )

async def push_msg(col, user_id: str, role: str, content: str, base_url=None):
    msg = {"role": role, "content": content, "base_url": base_url, "created_at": now()}
    await col.update_one(
        {"_id": user_id},
        {"$set": {"updated_at": now()},
         "$push": {"messages": {"$each": [msg], "$slice": -MAX_MESSAGES_STORE}}}
    )

async def load_doc(col, user_id: str):
    return await col.find_one({"_id": user_id}) or {}

async def load_history(col, user_id: str):
    doc = await col.find_one({"_id": user_id}, {"messages": {"$slice": -HISTORY_FOR_LLM}, "settings": 1, "usage": 1})
    doc = doc or {}
    msgs = doc.get("messages", [])
    history = [{"role": m.get("role"), "content": m.get("content")} for m in msgs]
    return history, doc

@router.post("/v1/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    db = get_db()
    col = db["messages"]  # ONE doc per user

    await ensure_user_doc(col, req.user_id)

    # reset chat if needed
    if req.reset:
        await reset_user(col, req.user_id)

    # ===== CASE 1: SETTINGS-ONLY payload (no message) =====
    if not req.message or not req.message.strip():
        if not (req.role and req.tone and req.length):
            raise HTTPException(
                status_code=400,
                detail="Settings payload must include role, tone, length (or send message for chat)."
            )

        await col.update_one(
            {"_id": req.user_id},
            {"$set": {
                "settings": {"role": req.role, "tone": req.tone, "length": req.length},
                "updated_at": now()
            }}
        )

        doc = await load_doc(col, req.user_id)
        usage = doc.get("usage", {"emb_tokens": 0, "chat_in_tokens": 0, "chat_out_tokens": 0, "total_cost_usd": 0.0})

        return ChatResponse(
            mode="settings",
            answer="Settings saved successfully.",
            usage=Usage(**usage),
            effective_settings=doc.get("settings", {"role": DEFAULT_ROLE, "tone": DEFAULT_TONE, "length": DEFAULT_LENGTH}),
            debug={}
        )

    # ===== CASE 2: CHAT payload =====
    history, doc = await load_history(col, req.user_id)

    # decide effective settings:
    # - if chat payload includes role/tone/length → use & store
    # - else use stored settings
    stored = doc.get("settings") or {"role": DEFAULT_ROLE, "tone": DEFAULT_TONE, "length": DEFAULT_LENGTH}

    role = req.role or stored.get("role", DEFAULT_ROLE)
    tone = req.tone or stored.get("tone", DEFAULT_TONE)
    length = req.length or stored.get("length", DEFAULT_LENGTH)

    # if chat request provided settings, persist them (optional but useful)
    if req.role or req.tone or req.length:
        await col.update_one(
            {"_id": req.user_id},
            {"$set": {"settings": {"role": role, "tone": tone, "length": length}, "updated_at": now()}}
        )

    # store user msg
    await push_msg(col, req.user_id, "user", req.message.strip(), base_url=None)

    # reload last history after push (optional)
    history, doc = await load_history(col, req.user_id)

    # RAG retrieve (namespace=user_id)
    r = await retrieve_context(
        user_id=req.user_id,
        question=req.message.strip(),
        length=length,
        score_threshold=settings.DEFAULT_SCORE_THRESHOLD
    )

    usage = doc.get("usage", {"emb_tokens": 0, "chat_in_tokens": 0, "chat_out_tokens": 0, "total_cost_usd": 0.0})
    usage["emb_tokens"] += int(r["emb_tokens"])
    usage["total_cost_usd"] += float(calc_cost(emb_tokens=r["emb_tokens"]))

    # no context => skip LLM
    if not (r["context"] or "").strip():
        answer = (
            "I don't have enough information in your saved data to answer that.\n\n"
            "Tip: ensure Pinecone metadata contains chunk text in keys like "
            "['text','content','chunk','page_content','body']."
        )

        await push_msg(col, req.user_id, "assistant", answer, base_url=r["base_url"])
        await col.update_one({"_id": req.user_id}, {"$set": {"usage": usage, "updated_at": now()}})

        return ChatResponse(
            mode="chat",
            answer=answer,
            base_url=r["base_url"],
            sources=r["sources"][:10],
            usage=Usage(**usage),
            effective_settings={"role": role, "tone": tone, "length": length},
            debug={"retrieved_cnt": r["retrieved_cnt"], "missing_text_cnt": r["missing_text_cnt"]}
        )

    # LLM answer
    system_prompt = build_system_prompt(role, tone, length)
    max_out = LENGTH_SETTINGS[length]["max_out"]

    ans, in_tok, out_tok = await answer_with_llm(
        system_prompt=system_prompt,
        context=r["context"],
        history=history,
        question=req.message.strip(),
        max_out=max_out
    )

    usage["chat_in_tokens"] += int(in_tok)
    usage["chat_out_tokens"] += int(out_tok)
    usage["total_cost_usd"] += float(calc_cost(chat_in=in_tok, chat_out=out_tok))

    await push_msg(col, req.user_id, "assistant", ans, base_url=r["base_url"])
    await col.update_one({"_id": req.user_id}, {"$set": {"usage": usage, "updated_at": now()}})

    return ChatResponse(
        mode="chat",
        answer=ans,
        base_url=r["base_url"],
        sources=r["sources"][:10],
        usage=Usage(**usage),
        effective_settings={"role": role, "tone": tone, "length": length},
        debug={"retrieved_cnt": r["retrieved_cnt"], "missing_text_cnt": r["missing_text_cnt"]}
    )
