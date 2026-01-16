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

router = APIRouter()


def now():
    return datetime.now(timezone.utc)


HISTORY_FOR_LLM = 50


@router.get("/health")
async def health():
    return {"ok": True}


@router.post("/v1/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    chats_col = get_chats_collection()
    settings_col = get_chatsettings_collection()

    user_id = req.user_id
    lead_id = req.lead_id

    # =====================================================
    # SETTINGS MODE
    # =====================================================
    if req.mode == "settings":
        if not req.settings:
            raise HTTPException(400, "Settings payload missing")

        await settings_col.update_one(
            {"userId": user_id, "leadId": lead_id},
            {
                "$set": {
                    **req.settings,
                    "updatedAt": now(),
                },
                "$setOnInsert": {
                    "userId": user_id,
                    "leadId": lead_id,
                    "createdAt": now(),
                },
            },
            upsert=True,
        )

        settings_doc = await settings_col.find_one(
            {"userId": user_id, "leadId": lead_id},
            {"_id": 0},
        )

        return ChatResponse(
            mode="settings",
            answer="Settings saved successfully.",
            effective_settings=settings_doc or {},
        )

    # =====================================================
    # CHAT MODE
    # =====================================================
    if not req.session_id:
        raise HTTPException(400, "sessionId is required")

    message = (req.message or "").strip()
    if not message:
        raise HTTPException(400, "Message is required")

    session_id = req.session_id

    # Store user message
    await chats_col.update_one(
        {"userId": user_id, "leadId": lead_id, "sessionId": session_id},
        {
            "$setOnInsert": {
                "userId": user_id,
                "leadId": lead_id,
                "sessionId": session_id,
                "messages": [],
                "createdAt": now(),
            },
            "$push": {
                "messages": {
                    "role": "user",
                    "content": message,
                    "timestamp": now(),
                }
            },
            "$set": {"updatedAt": now()},
        },
        upsert=True,
    )

    # Load settings (lead → org fallback)
    settings_doc = await settings_col.find_one(
        {"userId": user_id, "leadId": lead_id}
    ) or await settings_col.find_one(
        {"userId": user_id, "leadId": None}
    )

    if not settings_doc:
        raise HTTPException(400, "Chat settings not found")

    # Greeting shortcut
    if is_greeting(message):
        answer = greeting_reply(
            settings_doc.get("role"),
            settings_doc.get("tone"),
            settings_doc.get("length"),
        )

        await chats_col.update_one(
            {"userId": user_id, "leadId": lead_id, "sessionId": session_id},
            {
                "$push": {
                    "messages": {
                        "role": "assistant",
                        "content": answer,
                        "timestamp": now(),
                    }
                },
                "$set": {"updatedAt": now()},
            },
        )

        return ChatResponse(
            mode="chat",
            answer=answer,
            effective_settings=settings_doc,
            debug={"small_talk": "greeting"},
        )

    # Load history
    chat_doc = await chats_col.find_one(
        {"userId": user_id, "leadId": lead_id, "sessionId": session_id}
    )
    history = (chat_doc.get("messages") or [])[-HISTORY_FOR_LLM:]

    # RAG
    r = await retrieve_context(
        user_id=user_id,
        question=message,
        length=settings_doc["length"],
        score_threshold=settings.DEFAULT_SCORE_THRESHOLD,
    )

    if not (r.get("context") or "").strip():
        answer = fallback_not_found(settings_doc["length"])
    else:
        answer, _, _ = await answer_with_llm(
            system_prompt=build_system_prompt(**settings_doc),
            context=r["context"],
            history=history,
            question=message,
            max_out=LENGTH_SETTINGS[settings_doc["length"]]["max_out"],
        )

    # Store assistant message
    await chats_col.update_one(
        {"userId": user_id, "leadId": lead_id, "sessionId": session_id},
        {
            "$push": {
                "messages": {
                    "role": "assistant",
                    "content": answer,
                    "timestamp": now(),
                }
            },
            "$set": {"updatedAt": now()},
        },
    )

    return ChatResponse(
        mode="chat",
        answer=answer,
        effective_settings=settings_doc,
    )
