from pydantic import BaseModel, Field, ConfigDict
from typing import Literal, Optional, Dict, Any, List

Role = Literal[
    "Help Desk Specialist",
    "Client Service Representative",
    "Technical Support Agent"
]
Tone = Literal["Friendly", "Professional", "Casual"]
Length = Literal["Minimal", "Short", "Long", "Chatty"]


class ChatRequest(BaseModel):
    """
    Incoming payload supports:
    - userId (camelCase) from frontend/node
    - question (camelCase) from frontend/node

    But inside Python we use:
    - user_id
    - message
    """
    model_config = ConfigDict(populate_by_name=True)

    user_id: str = Field(..., min_length=1, alias="userId")

    # Chat payload
    message: Optional[str] = Field(None, alias="question")

    # Settings payload
    role: Optional[Role] = None
    tone: Optional[Tone] = None
    length: Optional[Length] = None

    # optional: clear chat history for that user
    reset: bool = False


class Usage(BaseModel):
    emb_tokens: int = 0
    chat_in_tokens: int = 0
    chat_out_tokens: int = 0
    total_cost_usd: float = 0.0


class ChatResponse(BaseModel):
    mode: str  # "settings" | "chat"
    answer: str
    base_url: Optional[str] = None
    sources: List[str] = []
    usage: Usage
    effective_settings: Dict[str, str] = {}
    debug: Dict[str, Any] = {}
