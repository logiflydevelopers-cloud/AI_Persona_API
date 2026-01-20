from motor.motor_asyncio import AsyncIOMotorClient
from app.core.settings import settings
from typing import Optional

_client: Optional[AsyncIOMotorClient] = None


# =========================
# Mongo Client
# =========================
def get_client() -> AsyncIOMotorClient:
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(settings.MONGO_URI)
    return _client


# =========================
# Database
# =========================
def get_db():
    return get_client()[settings.MONGO_DB]


# =========================
# Collections (EXPLICIT)
# =========================
def get_chats_collection():
    """
    Stores chat history

    One document = userId + leadId
    One chat per lead
    """
    return get_db()["chats"]


def get_chatsettings_collection():
    """
    Stores chat settings

    One document = userId
    Settings are org / main-user level
    """
    return get_db()["chatsettings"]
