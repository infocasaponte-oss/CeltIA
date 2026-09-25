from __future__ import annotations

ROUTES = ("fast", "think", "code", "agent", "long")

ROUTE_SYSTEM_GUIDANCE = """Trusted CeltIA routing contract:
- fast: short, direct tasks that need little deliberation and no code execution or external tools.
- think: tasks that primarily need multi-step reasoning, analysis, comparison, planning, trade-off evaluation, or careful derivation.
- code: tasks whose primary output is code, debugging, a patch, a query, tests, or implementation guidance tied directly to source code.
- agent: tasks that require current/external information, web lookup, tools, execution, or interaction with external systems.
- long: tasks whose effective input/context is large enough to require long-context handling. Use trusted size metadata such as input_chars and long_context_chars; do not infer long only from words like 'long document'.
Do not use any legacy-router decision or benchmark label; those are not supplied to the scorer."""
