# Copyright (c) 2026 Luis Manuel Cousido Hermida. All rights reserved.
import re
from dataclasses import dataclass

from core.config import settings


@dataclass(frozen=True)
class Route:
    mode: str
    difficulty: int
    thinking: bool
    max_tokens: int


CODE = re.compile(r"\b(python|javascript|typescript|rust|go|java|sql|bug|debug|exception|function|api|docker|git)\b", re.I)
AGENT = re.compile(
    r"\b(search|busca|buscar|búscame|investiga|encuentra|noticias|web|internet|execute|ejecuta|"
    r"calculate|calcula|tool|herramienta|automatiza|mcp|actualidad|actual(es)?|reciente(s)?)\b|"
    r"\b(capital|población|habitantes|cu[aá]ntos?|cu[aá]ndo\s+(fue|naci[oó]|muri[oó]|ocurri[oó])|"
    r"qui[eé]n\s+es|d[oó]nde\s+(queda|est[aá]|se\s+encuentra)|en\s+qu[eé]\s+año|fecha\s+(de|exacta))\b",
    re.I,
)
COMPLEX = re.compile(r"\b(plan|strategy|estrategia|analiza|analysis|compare|compara|design|diseña|implementa|arquitectura)\b", re.I)


def difficulty(text: str) -> int:
    n = 1
    if len(text) > 700: n += 1
    if COMPLEX.search(text): n += 1
    if CODE.search(text): n += 1
    if AGENT.search(text): n += 1
    return min(n, 5)


def route(text: str) -> Route:
    d = difficulty(text)
    if len(text) >= settings.router_long_context_chars:
        return Route("long", max(d, 4), True, 2048)
    if AGENT.search(text):
        return Route("agent", max(d, 4), True, 2048)
    if CODE.search(text):
        return Route("code", max(d, 3), True, 2048)
    if d >= 4:
        return Route("think", d, True, 2048)
    if d == 3:
        return Route("think", d, True, 1024)
    return Route("fast", d, False, 512)
