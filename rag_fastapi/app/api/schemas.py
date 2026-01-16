from pydantic import BaseModel, Field, ConfigDict
from typing import Literal, Optional, Dict, Any, List

# =========================
# ENUMS (Settings Only)
# =========================
Role = Literal[
    "Help Desk Specialist",
    "Client Service Representative",
    "Technical Support Agent"
]
Tone = Literal["Friendly", "Professional", "Casual"]
Length = Literal["Minimal", "Short", "Long", "Chatty"]

Mode = Literal["chat", "settings"]


# =========================
# REQUEST
# =========================
class ChatRequest(BaseModel):
    """
    Single API payload

    CHAT MODE:
    {
      "mode": "chat",
      "userId": "Logifly",
      "leadId": "lead.123",
      "sessionId": "session_001",
      "message": "Hi"
    }

    SETTINGS MODE:
    {
      "mode": "settings",
      "userId": "Logifly",
      "leadId": "lead.123",
      "settings": {
        "role": "Help Desk Specialist",
        "tone": "Professional",
        "length": "Short"
      }
    }
    """

    model_config = ConfigDict(populate_by_name=True)

    # Mode switch
    mode: Mode = "chat"

    # Ownership (ALWAYS REQUIRED)
    user_id: str = Field(..., min_length=1, alias="userId")
    lead_id: str = Field(..., min_length=1, alias="leadId")

    # Chat only
    session_id: Optional[str] = Field(None, alias="sessionId")
    message: Optional[str] = Field(None, min_length=1)

    # Settings only
    settings: Optional[Dict[str, Any]] = None


# =========================
# USAGE
# =========================
class Usage(BaseModel):
    emb_tokens: int = 0
    chat_in_tokens: int = 0
    chat_out_tokens: int = 0
    total_cost_usd: float = 0.0


# =========================
# RESPONSE
# =========================
class ChatResponse(BaseModel):
    """
    mode:
    - chat → assistant reply
    - settings → confirmation
    """

    mode: Mode
    answer: str

    base_url: Optional[str] = None
    sources: List[str] = []

    usage: Usage = Usage()

    # Read-only snapshot of applied settings
    effective_settings: Dict[str, Any] = {}

    debug: Dict[str, Any] = {}
