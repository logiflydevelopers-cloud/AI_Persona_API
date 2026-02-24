from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException

from app.api.schemas import ChatRequest, ChatResponse, Usage
from app.core.settings import settings
from app.db.mongo import (
    get_chats_collection,
    get_chatsettings_collection,
)
from app.services.prompt_service import (
    build_system_prompt,
    LENGTH_SETTINGS,
    fallback_not_found,
    is_greeting,
    greeting_reply,
)
from app.services.rag_services import retrieve_context, answer_with_llm
from app.services.token_counter import count_rag_tokens

router = APIRouter()


def now():
    return datetime.now(timezone.utc)


HISTORY_FOR_LLM = 50

DEFAULT_SETTINGS = {
    "role": "Help Desk Specialist",
    "tone": "Professional",
    "length": "Short",
}


@router.get("/health")
async def health():
    return {"ok": True}


@router.post("/v1/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    chats_col = get_chats_collection()
    settings_col = get_chatsettings_collection()

    userId = req.userId
    leadId = req.leadId

    is_settings_mode = req.settings is not None
    is_chat_mode = bool(req.message)

    if not is_settings_mode and not is_chat_mode:
        raise HTTPException(
            status_code=400,
            detail="Invalid payload: provide message or settings",
        )

    # =====================================================
    # SETTINGS MODE
    # =====================================================
    if is_settings_mode:
        clean_settings = {
            "role": req.settings.role,
            "tone": req.settings.tone,
            "length": req.settings.length,
        }

        await settings_col.update_one(
            {"userId": userId},
            {
                "$set": {
                    **clean_settings,
                    "updated_at": now(),
                },
                "$setOnInsert": {
                    "userId": userId,
                    "created_at": now(),
                },
            },
            upsert=True,
        )

        effective_settings = {
            **DEFAULT_SETTINGS,
            **clean_settings,
        }

        return ChatResponse(
            answer="Settings saved successfully.",
            effective_settings=effective_settings,
        )

    # =====================================================
    # CHAT MODE
    # =====================================================

    message = (req.message or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message is required")

    if not leadId:
        raise HTTPException(status_code=400, detail="leadId is required")

    # Load settings
    settings_doc = await settings_col.find_one({"userId": userId}) or {}

    effective_settings = {
        "role": settings_doc.get("role") or DEFAULT_SETTINGS["role"],
        "tone": settings_doc.get("tone") or DEFAULT_SETTINGS["tone"],
        "length": settings_doc.get("length") or DEFAULT_SETTINGS["length"],
    }

    # Store user message
    await chats_col.update_one(
        {"userId": userId, "leadId": leadId},
        {
            "$setOnInsert": {
                "userId": userId,
                "leadId": leadId,
                "created_at": now(),
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_tokens": 0,
            },
            "$push": {
                "messages": {
                    "role": "user",
                    "content": message,
                    "timestamp": now(),
                }
            },
            "$set": {"updated_at": now()},
        },
        upsert=True,
    )

    # Greeting shortcut (no LLM call → no token count)
    if is_greeting(message):
        answer = greeting_reply(
            effective_settings["tone"],
            effective_settings["length"],
        )

        await chats_col.update_one(
            {"userId": userId, "leadId": leadId},
            {
                "$push": {
                    "messages": {
                        "role": "assistant",
                        "content": answer,
                        "timestamp": now(),
                    }
                },
                "$set": {"updated_at": now()},
            },
        )

        return ChatResponse(
            answer=answer,
            effective_settings=effective_settings,
            debug={"small_talk": True},
        )

    # Load history
    chat_doc = await chats_col.find_one(
        {"userId": userId, "leadId": leadId}
    )
    history = (chat_doc.get("messages") or [])[-HISTORY_FOR_LLM:]

    # RAG retrieval
    r = await retrieve_context(
        userId=userId,
        question=message,
        length=effective_settings["length"],
        score_threshold=settings.DEFAULT_SCORE_THRESHOLD,
    )

    if not (r.get("context") or "").strip():
        answer = fallback_not_found(effective_settings["length"])
        usage_data = None
    else:
        system_prompt, max_out = build_system_prompt(
            role=effective_settings["role"],
            tone=effective_settings["tone"],
            length=effective_settings["length"],
        )

        answer = await answer_with_llm(
            system_prompt=system_prompt,
            context=r["context"],
            history=history,
            question=message,
            max_out=max_out,
        )

        # 🔥 Token counting
        token_data = count_rag_tokens(
            model=settings.CHAT_MODEL,
            system_prompt=system_prompt,
            context=r["context"],
            history=history,
            question=message,
            answer=answer,
        )

        usage_data = Usage(**token_data)

        # Store tokens inside chat doc
        await chats_col.update_one(
            {"userId": userId, "leadId": leadId},
            {
                "$inc": {
                    "total_input_tokens": token_data["input_tokens"],
                    "total_output_tokens": token_data["output_tokens"],
                    "total_tokens": token_data["total_tokens"],
                }
            },
        )

    # Store assistant reply
    update_payload = {
        "role": "assistant",
        "content": answer,
        "timestamp": now(),
    }

    if usage_data:
        update_payload["usage"] = usage_data.model_dump()

    await chats_col.update_one(
        {"userId": userId, "leadId": leadId},
        {
            "$push": {"messages": update_payload},
            "$set": {"updated_at": now()},
        },
    )

    return ChatResponse(
        answer=answer,
        effective_settings=effective_settings,
        usage=usage_data if usage_data else Usage(),
    )