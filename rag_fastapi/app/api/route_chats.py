from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException

from app.api.schemas import ChatRequest, ChatResponse
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

# =========================
# DEFAULT SETTINGS (FALLBACK)
# =========================
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

    user_id = req.user_id
    lead_id = req.lead_id

    is_settings_mode = req.settings is not None
    is_chat_mode = bool(req.message)

    if not is_settings_mode and not is_chat_mode:
        raise HTTPException(
            status_code=400,
            detail="Invalid payload: provide message or settings",
        )

    # =====================================================
    # SETTINGS MODE (USER-LEVEL ONLY)
    # =====================================================
    if is_settings_mode:
        await settings_col.update_one(
            {"userId": user_id},
            {
                "$set": {
                    **req.settings,
                    "updatedAt": now(),
                },
                "$setOnInsert": {
                    "userId": user_id,
                    "createdAt": now(),
                },
            },
            upsert=True,
        )

        settings_doc = await settings_col.find_one(
            {"userId": user_id},
            {"_id": 0},
        ) or {}

        effective_settings = {
            **DEFAULT_SETTINGS,
            **settings_doc,
        }

        return ChatResponse(
            answer="Settings saved successfully.",
            effective_settings=effective_settings,
        )

    # =====================================================
    # CHAT MODE (LEAD REQUIRED)
    # =====================================================
    message = (req.message or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message is required")

    if not lead_id:
        raise HTTPException(status_code=400, detail="leadId is required for chat")

    # Load settings (lead → org → defaults)
    settings_doc = (
        await settings_col.find_one(
            {"userId": user_id, "leadId": lead_id},
            {"_id": 0},
        )
        or await settings_col.find_one(
            {"userId": user_id},
            {"_id": 0},
        )
        or {}
    )

    effective_settings = {
        **DEFAULT_SETTINGS,
        **settings_doc,
    }

    # Store user message
    await chats_col.update_one(
        {"userId": user_id, "leadId": lead_id},
        {
            "$setOnInsert": {
                "userId": user_id,
                "leadId": lead_id,
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

    # Greeting shortcut
    if is_greeting(message):
        answer = greeting_reply(
            effective_settings["role"],
            effective_settings["tone"],
            effective_settings["length"],
        )

        await chats_col.update_one(
            {"userId": user_id, "leadId": lead_id},
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
            answer=answer,
            effective_settings=effective_settings,
            debug={"small_talk": "greeting"},
        )

    # Load chat history
    chat_doc = await chats_col.find_one(
        {"userId": user_id, "leadId": lead_id}
    )
    history = (chat_doc.get("messages") or [])[-HISTORY_FOR_LLM:]

    # RAG
    r = await retrieve_context(
        user_id=user_id,
        question=message,
        length=effective_settings["length"],
        score_threshold=settings.DEFAULT_SCORE_THRESHOLD,
    )

    if not (r.get("context") or "").strip():
        answer = fallback_not_found(effective_settings["length"])
    else:
        answer, _, _ = await answer_with_llm(
            system_prompt=build_system_prompt(**effective_settings),
            context=r["context"],
            history=history,
            question=message,
            max_out=LENGTH_SETTINGS[effective_settings["length"]]["max_out"],
        )

    # Store assistant reply
    await chats_col.update_one(
        {"userId": user_id, "leadId": lead_id},
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
        answer=answer,
        effective_settings=effective_settings,
    )
