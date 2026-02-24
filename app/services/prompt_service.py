from __future__ import annotations
import re


# =========================
# CORE SYSTEM PROMPT
# =========================

CORE_PROMPT = """
You are a company AI assistant.

Use ONLY the provided knowledge base to answer the user.

GROUNDING RULES:
- Treat the provided content as the only source of truth.
- Do not use outside knowledge.
- Do not guess or assume missing details.
- If the answer is not clearly supported, state that the information is unavailable.
- If the provided content is insufficient or partially relevant, do not fill gaps with assumptions.

LINK RULES:
- Only include links that appear in the provided content.
- Keep links exactly as written.
- Format links as Markdown: [Title](URL).

PROHIBITED:
- Fabricating policies, pricing, contacts, features, or procedures.
- Referring to internal documents or system context.
""".strip()


# =========================
# ROLE OPTIONS
# =========================

ROLE_PROMPTS = {
    "Customer Support Agent": """
Assist customers with product usage, account issues, and policy questions.
Focus on resolving the issue clearly and efficiently.
Offer escalation to human support if appropriate.
""",

    "Help Desk Specialist": """
Troubleshoot access, setup, and operational issues.
Provide step-by-step guidance.
Assume the user may not be technical.
""",

    "Technical Support Agent": """
Handle advanced technical issues and integrations.
Provide precise troubleshooting steps.
Explain causes briefly and clearly.
""",

    "Sales Agent": """
Understand customer needs before recommending solutions.
Highlight value and benefits using consultative language.
Avoid aggressive promotion.
""",

    "Client Service Representative": """
Assist existing clients with service requests and account matters.
Prioritize long-term satisfaction and clarity.
"""
}


# =========================
# TONE OPTIONS
# =========================

TONE_PROMPTS = {
    "Professional": """
Use formal, respectful business language.
""",

    "Friendly": """
Use a warm and approachable tone while remaining professional.
""",

    "Casual": """
Use relaxed, conversational language while remaining helpful.
""",

    "Soft-Selling": """
Use persuasive but non-pushy language.
"""
}


# =========================
# LENGTH SETTINGS
# =========================

LENGTH_SETTINGS = {
    "Minimal": {
        "max_out": 120,
        "style": "Answer in 1–2 short sentences. No extra explanation.",
        "format": "No bullets unless absolutely necessary.",
    },
    "Short": {
        "max_out": 250,
        "style": "Answer in a short paragraph. Use bullets only if needed.",
        "format": "Prefer 1 short paragraph. Use bullets only when listing items.",
    },
    "Long": {
        "max_out": 450,
        "style": "Give a detailed answer with steps if relevant.",
        "format": "Use numbered steps for procedures. Use short sections with headings if needed.",
    },
    "Chatty": {
        "max_out": 600,  # slightly reduced for safety
        "style": "Be detailed and conversational. Add helpful tips.",
        "format": "Use a friendly flow. Add 1–2 practical tips when helpful.",
    },
}


# =========================
# RETRIEVAL CONTROLS
# =========================

LENGTH_TO_CONTEXT = {
    "Minimal": 2000,
    "Short": 3500,
    "Long": 5000,
    "Chatty": 6500,
}

LENGTH_TO_TOPK = {
    "Minimal": 4,
    "Short": 5,
    "Long": 5,
    "Chatty": 6,
}


# =========================
# TONE NORMALIZATION
# =========================

def _normalize_tones(tone: str) -> list[str]:
    """
    Accepts:
      - "Friendly"
      - "Friendly, Professional"
      - "Friendly | Professional"
    Returns normalized tone keys.
    """
    if not tone:
        return ["Friendly"]

    parts = [p.strip() for p in tone.replace("|", ",").split(",") if p.strip()]
    matched = []

    for p in parts:
        for key in TONE_PROMPTS:
            if p.lower() == key.lower():
                matched.append(key)
                break

    return matched or ["Friendly"]


# =========================
# SYSTEM PROMPT BUILDER
# =========================

def build_system_prompt(role: str, tone: str, length: str):
    if role not in ROLE_PROMPTS:
        raise ValueError(f"Invalid role: {role}")

    if length not in LENGTH_SETTINGS:
        raise ValueError(f"Invalid length: {length}")

    tone_keys = _normalize_tones(tone)
    length_config = LENGTH_SETTINGS[length]

    tone_block = "\n".join(
        TONE_PROMPTS[t].strip()
        for t in tone_keys
        if t in TONE_PROMPTS
    )

    sections = [
        CORE_PROMPT,
        "ROLE:\n" + ROLE_PROMPTS[role].strip(),
        "TONE:\n" + tone_block,
        "RESPONSE GUIDELINES:\n"
        + length_config["style"] + " "
        + length_config["format"],
        "FINAL CHECK:\n"
        + "If the answer is not directly supported by the provided content, "
          "state clearly that the information is unavailable. "
          "Do not fabricate or infer missing details."
    ]

    system_prompt = "\n\n".join(sections)

    return system_prompt, length_config["max_out"]


# =========================
# FAST FALLBACK (NO LLM CALL)
# =========================

def fallback_not_found(length: str) -> str:
    length = (length or "Short").strip()

    if length == "Minimal":
        return "Sorry — I couldn’t find this information in the saved website data."

    if length == "Short":
        return (
            "I’m sorry, I couldn’t find that information in the saved website data. "
            "If you share the relevant page link or title, I can assist further."
        )

    if length == "Long":
        return (
            "I’m sorry — I couldn’t locate an answer in the website data currently available. "
            "This may mean the page wasn’t added or the relevant content isn’t stored. "
            "If you provide the page URL or section name, I can guide you accordingly."
        )

    return (
        "I couldn’t find a clear answer in the website data I currently have, "
        "and I don’t want to guess. If you share the relevant page or section, "
        "I’ll help you add it and answer properly."
    )


# =========================
# GREETING HANDLING
# =========================

_GREETINGS = {
    "hi", "hii", "hiii", "hello", "hey", "heyy", "hola",
    "good morning", "good afternoon", "good evening",
    "namaste", "yo"
}

def _norm(text: str) -> str:
    t = (text or "").strip().lower()
    t = re.sub(r"[^\w\s]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def is_greeting(text: str) -> bool:
    t = _norm(text)
    if not t:
        return False

    if t in _GREETINGS:
        return True

    return any(t.startswith(g + " ") for g in _GREETINGS)


def greeting_reply(tone: str, length: str) -> str:
    tone_key = _normalize_tones(tone)[0]
    length_key = (length or "Short").strip()

    if tone_key == "Professional":
        base = "Hello. How may I assist you today?"
    elif tone_key == "Casual":
        base = "Hey! What can I help you with?"
    else:
        base = "Hi! How can I help you today?"

    if length_key == "Minimal":
        return base

    if length_key == "Short":
        return base + " Feel free to ask about any website information."

    if length_key == "Long":
        return (
            base
            + "\n\nYou can ask about:\n"
            + "1) Pricing or plans\n"
            + "2) Policies (refund, shipping, privacy)\n"
            + "3) Setup or troubleshooting steps"
        )

    return (
        base
        + "\n\nTell me what you'd like to know — pricing, policies, features, "
          "or how something works — and I’ll guide you."
    )