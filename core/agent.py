# Copyright (c) 2026 Luis Manuel Cousido Hermida. All rights reserved.
import json
import logging

from core.verification import Verifier

from . import persona
from .config import settings

logger = logging.getLogger(__name__)

AGENT_INSTRUCTIONS = """Tienes capacidad de usar herramientas, incluida web_search para buscar en internet.
Si el usuario te pide buscar, investigar, encontrar información, noticias, o cualquier dato actual/verificable,
DEBES llamar a web_search antes de responder. Nunca digas que no tienes acceso a internet o a información en
tiempo real: sí lo tienes, a través de la herramienta web_search, y negarte a usarla cuando corresponde es un error.
Usa el resto de herramientas solo cuando sea necesario. Nunca afirmes que una herramienta se ejecutó sin su resultado.
Después de cada resultado, verifica la salida y solo entonces entrega una respuesta final breve.
Si un resultado de la herramienta indica un error, corrige el plan antes de finalizar."""

DEFAULT_PLAN = {
    "requires_tool": True,
    "verification_required": True,
    "steps": ["plan", "tool", "verify", "answer"],
    "reason": "Use tools when needed and verify results before finalizing the answer.",
}


MAX_TOOL_RESULT_CHARS = 4000  # keep tool output from eating the context window that the answer needs


def tool_content(result) -> str:
    text = json.dumps(result, ensure_ascii=False)
    if len(text) > MAX_TOOL_RESULT_CHARS:
        text = text[:MAX_TOOL_RESULT_CHARS] + '... [resultado recortado]'
    return text


class Agent:
    def __init__(self, llm, registry, policy=None, verifier=None, planner=None):
        self.llm = llm
        self.registry = registry
        self.policy = policy
        self.verifier = verifier or Verifier()
        self.planner = planner

    async def run(self, messages, allowed_tools: set[str] | None = None, on_tool_call=None, role: str = "user",
                   registry=None, extra_instructions: str = "", on_event=None, on_token=None):
        """`registry`, when given, overrides self.registry for this call only (e.g. Creador
        Studio binds a per-project CreatorTools registry). Its own tool handlers are
        responsible for validating their arguments, so the agent's generic ToolPolicy is
        skipped in that case rather than rejecting tool names it doesn't recognize."""
        if not messages:
            return {"answer": "No hay mensaje para procesar.", "steps": 0, "tool_calls": 0,
                    "verification": {"ok": False, "checked": False}}

        def emit(event):
            if on_event is not None:
                on_event(event)

        reg = registry or self.registry
        policy = self.policy if registry is None else None

        user_text = messages[-1].get("content", "")
        plan = self.planner.plan(user_text) if self.planner is not None else DEFAULT_PLAN

        instructions = AGENT_INSTRUCTIONS + (f"\n\n{extra_instructions}" if extra_instructions else "")
        system_content = persona.system_prompt(role, extra=instructions)
        if plan.get("requires_tool"):
            system_content += f"\n\nPlan sugerido: {', '.join(plan.get('steps', []))}. {plan.get('reason', '')}"

        working = [{"role": "system", "content": system_content}] + messages
        calls = 0
        verification = {"ok": False, "checked": False}

        emit({"type": "status", "text": "Planificando: " + ", ".join(plan.get("steps", []))})
        for step in range(settings.max_agent_steps):
            emit({"type": "status", "text": f"Pensando (paso {step + 1})…"})
            try:
                out = await self.llm.chat(working, thinking=True, max_tokens=2048, tools=reg.schemas(only=allowed_tools), on_token=on_token)
                msg = out["choices"][0]["message"]
            except (KeyError, IndexError, TypeError):
                logger.exception("Malformed response from LLM backend during agent run")
                return {"answer": "El modelo devolvió una respuesta no válida.", "steps": step + 1,
                        "tool_calls": calls, "verification": verification}

            tool_calls = msg.get("tool_calls") or []

            if not tool_calls:
                answer = msg.get("content", "") or "No answer produced."
                return {"answer": answer, "steps": step + 1, "tool_calls": calls, "verification": verification}

            working.append(msg)
            for call in tool_calls:
                if calls >= settings.max_tool_calls:
                    return {"answer": "Tool-call limit reached.", "steps": step + 1, "tool_calls": calls, "verification": verification}

                fn = call.get("function", {})
                name = fn.get("name", "")
                raw_args = fn.get("arguments", "{}")
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except Exception:
                    args = {}

                emit({"type": "tool_start", "name": name, "args": args})
                if policy is not None:
                    try:
                        args = policy.validate(name, args)
                    except Exception as exc:
                        result = {"ok": False, "error": str(exc)}
                        working.append({"role": "tool", "tool_call_id": call.get("id", f"call-{calls}"), "content": tool_content(result)})
                        calls += 1
                        verification = {"ok": False, "checked": True, "error": str(exc)}
                        if on_tool_call is not None:
                            on_tool_call(name, args, result)
                        emit({"type": "tool_result", "name": name, "ok": False, "error": str(exc)})
                        continue

                try:
                    result = await reg.call(name, args, only=allowed_tools)
                except Exception as exc:
                    result = {"ok": False, "error": str(exc)}

                if name == "python_exec":
                    verification = self.verifier.verify_code(str(args.get("code", "")))
                    result = {**result, "verification": verification}
                elif name == "calculator":
                    result = {**result, "verification": {"ok": True, "checked": True, "value": result.get("result")}}
                    verification = {"ok": True, "checked": True, "value": result.get("result")}

                calls += 1
                working.append({"role": "tool", "tool_call_id": call.get("id", f"call-{calls}"), "content": tool_content(result)})
                if on_tool_call is not None:
                    on_tool_call(name, args, result)
                emit({"type": "tool_result", "name": name, "ok": bool(result.get("ok", True)) if isinstance(result, dict) else True})

            emit({"type": "status", "text": "Redactando la respuesta con los resultados…"})
            try:
                final_out = await self.llm.chat(working, thinking=True, max_tokens=2048, tools=reg.schemas(only=allowed_tools), on_token=on_token)
                final_msg = final_out["choices"][0]["message"]
            except (KeyError, IndexError, TypeError):
                logger.exception("Malformed response from LLM backend during agent run")
                return {"answer": "El modelo devolvió una respuesta no válida tras usar las herramientas.",
                        "steps": step + 1, "tool_calls": calls, "verification": verification}
            if final_msg.get("content"):
                return {"answer": final_msg["content"], "steps": step + 1, "tool_calls": calls, "verification": verification}

        return {"answer": "Agent step limit reached before completion.", "steps": settings.max_agent_steps, "tool_calls": calls, "verification": verification}
