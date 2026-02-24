# token_counter.py

from __future__ import annotations
import tiktoken
from typing import List, Dict, Optional


# =========================
# ENCODING
# =========================

def get_encoding(model: str):
    """
    Returns the correct tokenizer for a model.
    Falls back to cl100k_base if model is unknown.
    """
    try:
        return tiktoken.encoding_for_model(model)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


# =========================
# BASIC TOKEN COUNTERS
# =========================

def count_text_tokens(text: str, model: str) -> int:
    """
    Count tokens for plain text.
    """
    if not text:
        return 0

    enc = get_encoding(model)
    return len(enc.encode(text))


def count_chat_tokens(messages: List[Dict[str, str]], model: str) -> int:
    """
    Rough estimation of chat-format tokens.
    Includes role + content + small structural overhead.
    """
    enc = get_encoding(model)
    tokens = 0

    for msg in messages:
        tokens += 4  # base overhead per message
        tokens += len(enc.encode(msg.get("role", "")))
        tokens += len(enc.encode(msg.get("content", "")))

    tokens += 2  # assistant reply primer
    return tokens


# =========================
# RAG TOKEN COUNTER
# =========================

def count_rag_tokens(
    model: str,
    system_prompt: str,
    context: str,
    history: List[Dict[str, str]],
    question: str,
    answer: Optional[str] = None,
):
    """
    Counts:
    - prompt tokens (system + history + context + question)
    - completion tokens (if answer provided)
    """

    messages = [{"role": "system", "content": system_prompt}]

    # limit history to last 6 for consistency
    for m in history[-6:]:
        messages.append({
            "role": m.get("role", ""),
            "content": m.get("content", "")
        })

    messages.append({
        "role": "user",
        "content": f"Context:\n{context}\n\nUser question: {question}"
    })

    input_tokens = count_chat_tokens(messages, model)

    output_tokens = 0
    if answer:
        output_tokens = count_text_tokens(answer, model)

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }