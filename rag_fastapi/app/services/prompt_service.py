from __future__ import annotations

# =========================
# USER STYLE OPTIONS
# =========================
ROLE_PROMPTS = {
    "Help Desk Specialist": (
        "You handle common product/app issues, troubleshooting steps, FAQs, and quick resolutions."
    ),
    "Client Service Representative": (
        "You handle customer requests, billing/account questions, policy guidance, and polite resolutions."
    ),
    "Technical Support Agent": (
        "You handle deeper technical debugging with precise steps, logs, edge cases, and accurate guidance."
    ),
}

TONE_GUIDE = {
    "Friendly": "Be warm, approachable, empathetic, and helpful.",
    "Professional": "Be clear, respectful, structured, and business-like.",
    "Casual": "Be relaxed, simple, and conversational (but still accurate).",
}

LENGTH_SETTINGS = {
    "Minimal": {
        "max_out": 120,
        "style": "Answer in 1–2 short sentences. No extra explanation.",
        "format": "No bullets unless absolutely necessary.",
    },
    "Short": {
        "max_out": 220,
        "style": "Answer in a short paragraph. Use bullets only if needed.",
        "format": "Prefer 1 short paragraph. Use bullets only when listing items.",
    },
    "Long": {
        "max_out": 520,
        "style": "Give a detailed answer with steps if relevant.",
        "format": "Use numbered steps for procedures. Use short sections with headings if needed.",
    },
    "Chatty": {
        "max_out": 850,
        "style": "Be detailed and conversational. Add helpful tips.",
        "format": "Use a friendly flow. Add 1–2 practical tips when helpful.",
    },
}

# Retrieval controls (you already use these)
LENGTH_TO_CONTEXT = {"Minimal": 2500, "Short": 4000, "Long": 6500, "Chatty": 8500}
LENGTH_TO_TOPK = {"Minimal": 4, "Short": 5, "Long": 6, "Chatty": 7}


def _normalize_tones(tone: str) -> list[str]:
    """
    Accepts:
      - "Friendly"
      - "Friendly, Professional"
      - "Friendly | Professional"
    Returns list of normalized tone keys present in TONE_GUIDE.
    """
    if not tone:
        return ["Friendly"]
    parts = [p.strip() for p in tone.replace("|", ",").split(",") if p.strip()]
    out = []
    for p in parts:
        # match keys case-insensitively
        for k in TONE_GUIDE.keys():
            if p.lower() == k.lower():
                out.append(k)
                break
    return out or ["Friendly"]


def fallback_not_found(length: str) -> str:
    """
    Polite reply when Pinecone context has no answer.
    Use it in API code when context is empty to avoid LLM call.
    """
    length = (length or "Short").strip()
    if length == "Minimal":
        return "Sorry — I couldn’t find this information in your saved website data."
    if length == "Short":
        return (
            "I’m sorry, I couldn’t find that information in the data I have saved for this website. "
            "If you share a relevant page link/title (or add that page to Pinecone), I can help right away."
        )
    if length == "Long":
        return (
            "I’m sorry — I couldn’t locate an answer to that in the website data currently saved in my knowledge base. "
            "This usually means the page wasn’t crawled/added or the content isn’t present in the stored chunks. "
            "If you share the page URL or section name, I can guide you on what to add so I can answer accurately."
        )
    # Chatty default
    return (
        "Hmm — I couldn’t find a clear answer to that in the website data I currently have saved, so I don’t want to guess. "
        "If you share the URL or the page name you’re referring to, I’ll help you add it and then answer properly."
    )


def build_system_prompt(
    role: str | None = None,
    tone: str | None = None,
    length: str | None = None,
    **_: dict,   # 👈 absorbs userId and any future extras safely
) -> str:
    role = (role or "Help Desk Specialist").strip()
    if role not in ROLE_PROMPTS:
        role = "Help Desk Specialist"

    length = (length or "Short").strip()
    if length not in LENGTH_SETTINGS:
        length = "Short"

    tone_list = _normalize_tones(tone)
    tone_text = " + ".join(tone_list)
    tone_rules = " ".join([TONE_GUIDE[t] for t in tone_list if t in TONE_GUIDE])

    style = LENGTH_SETTINGS[length]["style"]
    fmt = LENGTH_SETTINGS[length].get("format", "")

    return (
        f"You are a {role}. {ROLE_PROMPTS[role]}\n"
        f"Tone: {tone_text}. {tone_rules}\n"
        f"Response style: {style}\n"
        f"Formatting: {fmt}\n\n"
        "STRICT RULES:\n"
        "- Use ONLY the provided context to answer.\n"
        "- If the context does not contain the answer, say politely that you cannot find it in the saved website data.\n"
        "- Do NOT invent details, prices, terms, contacts, or policy statements.\n"
        "- If the question asks for something missing, suggest what page/link the user should share or add.\n\n"
        "LINK RULES:\n"
        "- Only include links that appear in the context/metadata.\n"
        "- If a link in context is relative like /privacy-policy, keep it exactly as-is (do NOT guess the domain).\n"
        "- Always format links as markdown: [Title](url)\n\n"
        "OUTPUT RULES:\n"
        "- Be concise and structured.\n"
        "- Use steps only when the user asks 'how to' or when troubleshooting.\n"
    )


import re

_GREETINGS = {
    "hi", "hii", "hiii", "hello", "hey", "heyy", "hola",
    "good morning", "good afternoon", "good evening",
    "namaste", "yo"
}

def _norm(text: str) -> str:
    t = (text or "").strip().lower()
    t = re.sub(r"[^\w\s]", "", t)         # remove punctuation
    t = re.sub(r"\s+", " ", t).strip()    # normalize spaces
    return t

def is_greeting(text: str) -> bool:
    t = _norm(text)
    if not t:
        return False

    # exact match or startswith greeting phrase
    if t in _GREETINGS:
        return True

    for g in _GREETINGS:
        if t.startswith(g + " "):
            return True

    return False

def greeting_reply(role: str, tone: str, length: str) -> str:
    # choose base line by tone
    tone_key = (tone or "Friendly").strip()
    if tone_key == "Professional":
        base = "Hello! How may I assist you today?"
    elif tone_key == "Casual":
        base = "Hey! What can I help you with?"
    else:
        base = "Hi! 😊 How can I help you today?"

    length_key = (length or "Short").strip()

    if length_key == "Minimal":
        return base

    if length_key == "Short":
        return base + " Ask me anything about the website data."

    if length_key == "Long":
        return (
            base
            + "\n\nYou can ask things like:\n"
            + "1) Pricing / plans / features\n"
            + "2) Policies (refund, shipping, privacy)\n"
            + "3) Setup / troubleshooting steps"
        )

    # Chatty
    return (
        base
        + "\n\nTell me what you want to know — for example *pricing*, *policies*, *features*, or *how something works* — and I’ll guide you."
    )




