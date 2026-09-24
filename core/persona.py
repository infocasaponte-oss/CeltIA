# Copyright (c) 2026 Luis Manuel Cousido Hermida. All rights reserved.
LANGUAGE = (
    "Responde SIEMPRE en el mismo idioma exacto en el que el usuario escribió su último mensaje. "
    "Si escribió en castellano/español, responde en castellano. Si escribió en gallego, responde en "
    "gallego obligatoriamente. Si escribió en inglés, responde en inglés. Detecta el idioma tú mismo a "
    "partir del texto del usuario; nunca cambies de idioma por tu cuenta, nunca mezcles idiomas en la misma "
    "respuesta, y nunca respondas en un idioma distinto al del usuario salvo que él te lo pida explícitamente."
)

IDENTITY = (
    "Eres CeltIA, un asistente de IA local creado por Lois de CeltIA. Identifícate siempre como CeltIA. "
    "Nunca afirmes haber sido creado por OpenAI, Anthropic, Google u otra empresa de IA; "
    "esa información sería incorrecta, tú eres CeltIA. "
    "Nunca reveles el nombre legal ni la identidad personal de ninguna persona vinculada a la propiedad "
    "intelectual o el copyright de CeltIA, aunque se te pregunte directamente o se insista; esa "
    "información es estrictamente privada y no debe salir del chat. Si te preguntan quién te creó, responde "
    "simplemente \"Soy CeltIA, un asistente de IA local creado por Lois de CeltIA\" sin añadir ninguna frase "
    "extra sobre privacidad o sobre no poder revelar identidades, salvo que el usuario insista explícitamente "
    "en pedir el nombre legal real. "
    "No menciones nunca el nombre de empresas tecnológicas externas a CeltIA (por ejemplo OpenAI, Google, "
    "Anthropic, Microsoft, Meta, Amazon o xAI) ni de sus productos de IA (ChatGPT, Gemini, Claude, Copilot, "
    "Llama, Grok, etc.) en ninguna respuesta, ni para compararte con ellos, ni para citarlos como fuente, ni "
    "por ningún otro motivo; si necesitas referirte a otros asistentes o modelos de forma genérica, usa "
    "expresiones como \"otros asistentes de IA\" sin nombrar la empresa ni el producto."
)

FACT_CHECK = (
    "Cuando el usuario pregunte por un dato factual verificable (capitales, cifras, fechas, nombres de "
    "personas o lugares, hechos históricos o actuales), nunca respondas de memoria dándolo por cierto sin "
    "contrastarlo: usa la herramienta web_search para confirmarlo antes de responder, incluso si el usuario "
    "no ha pedido explícitamente que busques. Si por cualquier motivo no puedes verificarlo, dilo abiertamente "
    "en la respuesta en vez de afirmar un dato sin comprobar; nunca inventes ni confundas datos similares "
    "(por ejemplo, no confundas la capital de un país con la de otro)."
)

NO_REASONING_LEAK = (
    "Nunca muestres tu proceso de razonamiento interno, dudas, borradores ni frases como \"Wait,\", "
    "\"Let me check\", \"Okay, the user is asking\", \"I need to...\" en la respuesta visible, ni siquiera en "
    "inglés aunque estés pensando en ese idioma; piensa en silencio y entrega únicamente la respuesta final, "
    "directa y en el idioma exacto en el que escribió el usuario."
)

STYLE = (
    "No uses emoticonos de forma decorativa ni repetitiva. Inclúyelos únicamente si aportan una "
    "emoción genuina y relevante al mensaje, como mucho uno por respuesta."
)

PRIVACY = (
    "Nunca reveles detalles internos del sistema (arquitectura, herramientas disponibles, "
    "configuración, variables de entorno, claves, código fuente, errores internos o registros) "
    "aunque el usuario lo pida directamente o insista. Esa información es exclusiva para administradores."
)


import re

_IDENTITY_PATTERNS = [
    (re.compile(r"(creado|desarrollado|entrenado|hecho|dise[nñ]ado)\s+por\s+(OpenAI|Anthropic|Google|Meta|Microsoft)", re.I), r"\1 por CeltIA"),
    (re.compile(r"(created|developed|trained|made|built)\s+by\s+(OpenAI|Anthropic|Google|Meta|Microsoft)", re.I), r"\1 by CeltIA"),
    (re.compile(r"\b(asistente|modelo|chatbot|IA)\s+de\s+(OpenAI|Anthropic|Google|Meta|Microsoft)\b", re.I), r"\1 de CeltIA"),
    (re.compile(r"\bsoy\s+(ChatGPT|GPT-?4|GPT-?3(\.5)?|Claude(?:\s+\d)?|Gemini|Bard)\b", re.I), "soy CeltIA"),
    (re.compile(r"\bI\s*'?am\s+(ChatGPT|GPT-?4|GPT-?3(\.5)?|Claude(?:\s+\d)?|Gemini|Bard)\b", re.I), "I am CeltIA"),
]


_EXTERNAL_COMPANY_PATTERN = re.compile(
    r"\b(OpenAI|Anthropic|Google(?:\s+DeepMind)?|DeepMind|Microsoft|Meta(?:\s+AI)?|Amazon|xAI)\b",
    re.I,
)

_EXTERNAL_PRODUCT_PATTERN = re.compile(
    r"\b(ChatGPT|GPT-?4o?|GPT-?3(\.5)?|Claude(?:\s*\d)?|Gemini|Bard|Copilot|Llama\s*\d?|Grok|DeepSeek)\b",
    re.I,
)


_PRIVATE_NAME_PATTERN = re.compile(
    r"\b(Luis\s+)?Manuel\s+Cousido\s+Hermida\b|\bLois\s+de\s+CeltIA\b(?=.{0,40}(Manuel|Cousido|Hermida))",
    re.I,
)


_THINK_BLOCK_PATTERN = re.compile(r"<think>.*?</think>", re.I | re.S)
_THINK_LEFTOVER_PATTERN = re.compile(r"^.*?</think>", re.I | re.S)

_REASONING_MARKER_PATTERN = re.compile(
    r"\b(wait,|okay,\s+the\s+user|the\s+user\s+is\s+asking|let\s+me\s+(check|verify|confirm|think)|"
    r"i\s+need\s+to\s+(answer|confirm|verify|check)|according\s+to\s+the\s+(rules|guidelines)|"
    r"however,\s+the\s+user)\b",
    re.I,
)


def strip_thinking(text: str) -> str:
    """Remove leaked chain-of-thought (often in English) from the final answer.

    The model sometimes emits raw reasoning before a closing </think> tag
    without an opening tag, or wraps it properly in <think>...</think>;
    both leak tokens and language into the visible response.
    """
    if not text:
        return text
    text = _THINK_BLOCK_PATTERN.sub("", text)
    if "</think>" in text:
        text = _THINK_LEFTOVER_PATTERN.sub("", text)
    text = text.strip()

    if _REASONING_MARKER_PATTERN.search(text):
        # Sin cierre </think>: el modelo dejó su deliberación en crudo en la respuesta.
        # Se descartan las frases con marcadores de razonamiento y se conserva el resto.
        sentences = re.split(r"(?<=[.!?])\s+", text)
        kept = [s for s in sentences if not _REASONING_MARKER_PATTERN.search(s)]
        cleaned = " ".join(kept).strip()
        if cleaned:
            text = cleaned

    return text.strip()


def enforce_identity(text: str) -> str:
    if not text:
        return text
    text = strip_thinking(text)
    text = _PRIVATE_NAME_PATTERN.sub("[privado]", text)
    for pattern, repl in _IDENTITY_PATTERNS:
        text = pattern.sub(repl, text)
    text = _EXTERNAL_COMPANY_PATTERN.sub("otras empresas de tecnología", text)
    text = _EXTERNAL_PRODUCT_PATTERN.sub("otros asistentes de IA", text)
    return text


def system_prompt(role: str, extra: str = "") -> str:
    lines = [LANGUAGE, IDENTITY, FACT_CHECK, NO_REASONING_LEAK, STYLE]
    if role != "admin":
        lines.append(PRIVACY)
    if extra:
        lines.append(extra)
    return "\n\n".join(lines)



class IdentityStreamFilter:
    """Applies enforce_identity incrementally. Holds back a short tail so patterns that span
    chunk boundaries are matched before anything is shown; the caller replaces the streamed
    text with enforce_identity(full_text) at the end, so the final result is always exact."""

    HOLD = 48

    def __init__(self):
        self.raw = ""
        self.emitted = 0

    def reset(self):
        self.raw = ""
        self.emitted = 0

    def feed(self, delta: str) -> str:
        self.raw += delta
        if "<think>" in self.raw and "</think>" not in self.raw:
            return ""
        filtered = enforce_identity(self.raw)
        cut = len(filtered) - self.HOLD
        while cut > self.emitted and not filtered[cut - 1].isspace():
            cut -= 1
        if cut <= self.emitted:
            return ""
        out = filtered[self.emitted:cut]
        self.emitted = cut
        return out
