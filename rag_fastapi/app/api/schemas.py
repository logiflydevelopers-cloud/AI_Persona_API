from pydantic import BaseModel, Field, ConfigDict, model_validator
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

    SETTINGS PAYLOAD (MAIN USER ONLY):
    {
      "userId": "Logifly",
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
      "question": "Hi"
    }
    """

    model_config = ConfigDict(populate_by_name=True)

    # ALWAYS REQUIRED
    user_id: str = Field(..., min_length=1, alias="userId")

    # CHAT ONLY
    lead_id: Optional[str] = Field(None, min_length=1, alias="leadId")
    message: Optional[str] = Field(None, min_length=1, alias="question")

    # SETTINGS ONLY
    settings: Optional[Dict[str, Any]] = None

    # =========================
    # VALIDATION
    # =========================
    @model_validator(mode="after")
    def validate_intent(self):
        # SETTINGS FLOW
        if self.settings is not None:
            if self.lead_id is not None:
                raise ValueError("leadId is not allowed when updating settings")

            if self.message is not None:
                raise ValueError("message is not allowed when updating settings")

            return self

        # CHAT FLOW
        if self.message is not None:
            if not self.lead_id:
                raise ValueError("leadId is required for chat messages")

            return self

        raise ValueError("Payload must contain either settings or message")


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
