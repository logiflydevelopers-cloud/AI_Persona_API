from pydantic import BaseModel, Field
from typing import Literal, Optional, Dict, Any, List

Role = Literal["Help Desk Specialist", "Client Service Representative", "Technical Support Agent"]
Tone = Literal["Friendly", "Professional", "Casual"]
Length = Literal["Minimal", "Short", "Long", "Chatty"]

class ChatRequest(BaseModel):
    userId: str = Field(..., min_length=1)

    # payload-2 (chat)
    question: Optional[str] = None

    # payload-1 (settings)
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


