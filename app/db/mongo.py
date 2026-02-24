from motor.motor_asyncio import AsyncIOMotorClient
from typing import Optional
from app.core.settings import settings


_client: Optional[AsyncIOMotorClient] = None


# =========================
# CONNECTION MANAGEMENT
# =========================

def get_client() -> AsyncIOMotorClient:
    """
    Lazy singleton Mongo client.
    """
    global _client

    if _client is None:
        _client = AsyncIOMotorClient(
            settings.MONGO_URI,
            maxPoolSize=25,
            minPoolSize=5,
            serverSelectionTimeoutMS=5000,
            retryWrites=True,
        )

    return _client


def close_client():
    """
    Call on FastAPI shutdown.
    """
    global _client

    if _client is not None:
        _client.close()
        _client = None


# =========================
# DATABASE
# =========================

def get_db():
    return get_client()[settings.MONGO_DB]


# =========================
# COLLECTIONS
# =========================

def get_chats_collection():
    """
    One document per (user_id, lead_id)
    Contains:
      - messages[]
      - total_input_tokens
      - total_output_tokens
      - total_tokens
    """
    return get_db()["chats"]


def get_chatsettings_collection():
    """
    One document per user_id
    Stores org-level assistant settings.
    """
    return get_db()["chatsettings"]


# =========================
# INDEX SETUP (CALL ON STARTUP)
# =========================

async def init_indexes():
    db = get_db()

    chats = db["chats"]
    settings_col = db["chatsettings"]

    # -----------------------------------------------------
    # Chats Collection Indexes
    # -----------------------------------------------------

    # Unique chat per user + lead
    await chats.create_index(
        [("user_id", 1), ("lead_id", 1)],
        unique=True
    )

    # Fast lookup by user
    await chats.create_index(
        [("user_id", 1)]
    )

    # Recent chats sorting
    await chats.create_index(
        [("updated_at", -1)]
    )

    # Analytics: token aggregation queries
    await chats.create_index(
        [("user_id", 1), ("updated_at", -1)]
    )

    # -----------------------------------------------------
    # Chat Settings Collection Index
    # -----------------------------------------------------

    await settings_col.create_index(
        [("user_id", 1)],
        unique=True
    )

    # -----------------------------------------------------
    # OPTIONAL: TTL (Auto delete old chats)
    # Uncomment if you want automatic cleanup
    # -----------------------------------------------------
    #
    # await chats.create_index(
    #     [("updated_at", 1)],
    #     expireAfterSeconds=60 * 60 * 24 * 30  # 30 days
    # )