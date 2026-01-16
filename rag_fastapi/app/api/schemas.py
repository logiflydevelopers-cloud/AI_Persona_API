from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Dict, Any, List, Literal


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


# =========================
# REQUEST
# =========================
class ChatRequest(BaseModel):
    """
    SINGLE API – AUTO-DETECTED INTENT

    SETTINGS PAYLOAD:
    {
      "userId": "Logifly",
      "leadId": "lead.123",
      "settings": {
        "role": "Help Desk Specialist",
        "tone": "Professional",
        "length": "Short"
      }
    }

    CHAT PAYLOAD:
    {
      "userId": "Logifly",
      "leadId": "lead.123",
      "message": "Hi"
    }
    """

    model_config = ConfigDict(populate_by_name=True)

    # Ownership (ALWAYS REQUIRED)
    user_id: str = Field(..., min_length=1, alias="userId")
    lead_id: str = Field(..., min_length=1, alias="leadId")

    # Chat (auto-detected)
    message: Optional[str] = Field(None, min_length=1)

    # Settings (auto-detected)
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
    Unified response for both chat & settings
    """

    answer: str

    base_url: Optional[str] = None
    sources: List[str] = []

    usage: Usage = Usage()

    # Read-only snapshot of applied settings
    effective_settings: Dict[str, Any] = {}

    debug: Dict[str, Any] = {}

