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
# SETTINGS MODEL (Strongly Typed)
# =========================

class ChatSettings(BaseModel):
    role: Role
    tone: Tone
    length: Length


# =========================
# REQUEST
# =========================

class ChatRequest(BaseModel):
    """
    SINGLE API – AUTO-DETECTED INTENT
    """

    model_config = ConfigDict(populate_by_name=True)

    # ALWAYS REQUIRED
    user_id: str = Field(..., min_length=1, alias=["userId"])

    # CHAT ONLY
    lead_id: Optional[str] = Field(None, min_length=1, alias=["leadId"])
    message: Optional[str] = Field(None, min_length=1, alias="question")

    # SETTINGS ONLY
    settings: Optional[ChatSettings] = None

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
                raise ValueError("question is not allowed when updating settings")

            return self

        # CHAT FLOW
        if self.message is not None:
            if not self.lead_id:
                raise ValueError("leadId is required for chat messages")

            return self

        raise ValueError("Payload must contain either settings or question")


# =========================
# USAGE (Token Tracking Only)
# =========================

class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


# =========================
# RESPONSE
# =========================

class ChatResponse(BaseModel):
    """
    Unified response for both chat & settings
    """

    answer: str

    base_url: Optional[str] = None
    sources: List[str] = Field(default_factory=list)

    usage: Usage = Field(default_factory=Usage)

    # Snapshot of applied settings
    effective_settings: Dict[str, Any] = Field(default_factory=dict)

    # Debug info (optional)
    debug: Dict[str, Any] = Field(default_factory=dict)