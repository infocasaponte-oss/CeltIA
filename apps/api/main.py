# Copyright (c) 2026 Luis Manuel Cousido Hermida. All rights reserved.
import asyncio
import json
import logging
import re
import time
import uuid
from pathlib import Path
from typing import Annotated, Literal

import httpx
from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from core import files as files_mod
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

from core import billing, diagnostics, oauth, persona
from core.agent import Agent, tool_content
from core.apikeys import generate_client_key, generate_key, hash_key, hash_password, verify_password
from core.config import settings
import secrets
from core.gateway import AttemptLimiter, Gateway, GatewayRejection
from core.creator import filesystem as creator_fs
from core.creator import git_tools as creator_git
from core.creator.projects import project_manager
from core.creator.sandbox import sandbox_configured, sandbox_for
from core.creator.tools import CREATOR_TOOL_NAMES, build_creator_registry
from core.inference import VLLMClient
from core.decision_runtime import CeltIADecisionRuntime
from core.decision_shadow import evaluate_route_shadow
from core.memory import Memory
from core.planner import Planner
from core.policy import ToolPolicy
from core.router import route
from core.tools import PUBLIC_SAFE_TOOLS, Tool, builtins

logger = logging.getLogger(__name__)

app = FastAPI(title="CeltIA", version="1.0.0")
app.state.metrics = {"requests": 0, "agent_runs": 0, "tool_calls": 0, "stream_requests": 0, "decision_requests": 0}
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "PATCH", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)

@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    return response

web_dir = Path(__file__).resolve().parent.parent / "web"
if web_dir.exists():
    app.mount("/ui", StaticFiles(directory=str(web_dir), html=True), name="web")

memory = Memory(settings.sqlite_path)
registry = builtins(policy=ToolPolicy())
llm = VLLMClient(settings.vllm_base_url, settings.model_serve_name)
decision_runtime = CeltIADecisionRuntime(
    llm,
    abstain_below=settings.decision_abstain_below,
    temperature=settings.decision_temperature,
    reject_suspected_ood=settings.decision_reject_ood,
    ood_entropy_threshold=settings.decision_ood_entropy_threshold,
    ood_margin_threshold=settings.decision_ood_margin_threshold,
    max_questions=settings.decision_max_questions,
    max_output_tokens=settings.decision_max_output_tokens,
    max_total_output_tokens=settings.decision_max_total_output_tokens,
    max_total_prompt_chars=settings.decision_max_total_prompt_chars,
)
agent = Agent(llm, registry, policy=ToolPolicy(), planner=Planner())
gateway = Gateway(settings.gateway_max_concurrency, settings.gateway_queue_wait_seconds,
                  month_usage=lambda key_id: memory.month_tokens(key_id),
                  month_usage_client=lambda cid: memory.month_tokens_client(cid))
login_limiter = AttemptLimiter(max_attempts=8, window_seconds=15 * 60)         # per email
login_ip_limiter = AttemptLimiter(max_attempts=40, window_seconds=15 * 60)      # per IP (proxies share one IP)
register_limiter = AttemptLimiter(max_attempts=10, window_seconds=60 * 60)
AUTO_SEARCH_TOOLS = {"web_search"}
CODE_TOOLS = {"install_package"}
ROLES = {"admin", "premium", "user"}

def role_tools(role: str):
    if role == "admin":
        return None
    return PUBLIC_SAFE_TOOLS  # install_package runs pip on the host: admin only

def role_has_history(role: str) -> bool:
    return role in {"admin", "premium"}

async def system_audit():
    vllm_status = "unreachable"
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{settings.vllm_base_url}/models")
            vllm_status = "ok" if resp.status_code == 200 else f"http {resp.status_code}"
    except Exception as exc:
        vllm_status = f"error: {exc}"

    try:
        memory.db.execute("SELECT 1")
        db_status = "ok"
    except Exception as exc:
        db_status = f"error: {exc}"

    keys = memory.list_api_keys()
    return {
        "ok": vllm_status == "ok" and db_status == "ok",
        "vllm_backend": vllm_status,
        "database": db_status,
        "metrics": dict(app.state.metrics),
        "api_keys": {
            "total": len(keys),
            "active": sum(1 for k in keys if k["active"]),
            "revoked": sum(1 for k in keys if not k["active"]),
            "admin": sum(1 for k in keys if k["role"] == "admin"),
            "premium": sum(1 for k in keys if k["role"] == "premium"),
            "user": sum(1 for k in keys if k["role"] == "user"),
        },
        "recent_errors": diagnostics.recent_errors(10),
    }

registry.add(Tool(
    "system_audit",
    "Audit CeltIA's own backend: model connectivity, database health, live metrics, API key counts, and recent internal errors. Admin/trusted use only.",
    {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
    system_audit,
))

class Message(BaseModel):
    role:str
    content:str

class ChatRequest(BaseModel):
    model:str="CeltIA V4"
    messages:list[Message]
    session_id:str|None=None
    mode:str|None=None
    temperature:float|None=None
    max_tokens:int|None=None

DecisionOptionInput = Annotated[str, Field(min_length=1, max_length=1000)]

class DecisionQuestionInput(BaseModel):
    id: str = Field(min_length=1, max_length=128)
    prompt: str = Field(min_length=1, max_length=8000)
    type: Literal["boolean", "choice", "score"]
    options: list[DecisionOptionInput] = Field(default_factory=list, max_length=64)
    minimum: int | None = None
    maximum: int | None = None

    @model_validator(mode="after")
    def validate_type_fields(self):
        if self.type == "boolean":
            if self.options or self.minimum is not None or self.maximum is not None:
                raise ValueError("boolean questions do not accept options or score bounds")
            return self

        if self.type == "choice":
            if self.minimum is not None or self.maximum is not None:
                raise ValueError("choice questions do not accept score bounds")
            if len(self.options) < 2:
                raise ValueError("choice questions require at least two options")
            if len(set(self.options)) != len(self.options):
                raise ValueError("choice options must be unique")
            return self

        if self.options:
            raise ValueError("score questions do not accept choice options")
        if self.minimum is None or self.maximum is None or self.minimum > self.maximum:
            raise ValueError("score questions require a valid minimum/maximum")
        if self.maximum - self.minimum + 1 > 101:
            raise ValueError("score questions support at most 101 candidate values")
        return self

class DecisionApiRequest(BaseModel):
    context: object
    questions: list[DecisionQuestionInput] = Field(min_length=1, max_length=32)

class ApiKeyCreateRequest(BaseModel):
    name: str
    role: str = "user"

class ApiKeyUpdateRequest(BaseModel):
    role: str | None = None
    active: bool | None = None

class BillingSignupRequest(BaseModel):
    email: str

async def require_api_key(authorization: str | None = Header(default=None)):
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "missing API key")
    raw_key = authorization.split(" ", 1)[1].strip()
    record = memory.find_api_key(hash_key(raw_key))
    if not record or not record["active"]:
        raise HTTPException(401, "invalid or revoked API key")
    balance = record.get("token_balance")
    if balance is not None and balance <= 0:
        raise HTTPException(402, "Sin saldo de tokens. Recarga tu cuenta para continuar.")
    return record

def _admin_token_ok(candidate: str | None) -> bool:
    return bool(settings.admin_token and candidate and secrets.compare_digest(candidate.encode(), settings.admin_token.encode()))

async def require_admin(x_admin_token: str | None = Header(default=None), authorization: str | None = Header(default=None)):
    """Admin access: either a logged-in account with role=admin (Bearer key) or the legacy X-Admin-Token."""
    if authorization and authorization.lower().startswith("bearer "):
        record = memory.find_api_key(hash_key(authorization.split(" ", 1)[1].strip()))
        if record and record["active"]:
            if record["role"] == "admin":
                return True
            if not _admin_token_ok(x_admin_token):
                raise HTTPException(403, "admin role required")  # valid session, but not an admin
    if _admin_token_ok(x_admin_token):
        return True
    raise HTTPException(401, "invalid or expired session")

class CookieConsent(BaseModel):
    necessary: bool = True
    analytics: bool = False

PLANS = {
    "free": {"label": "Free", "role": "user", "token_grant": None},
    "pro": {"label": "Pro", "role": "premium", "token_grant": None},
    "payg": {"label": "Pay-as-you-go", "role": "premium", "token_grant": None},
}

class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str
    gdpr_accepted: bool
    cookies: CookieConsent
    plan: str = "free"

class LoginRequest(BaseModel):
    email: str
    password: str

@app.get("/billing/plans")
async def billing_plans():
    return {
        "plans": [
            {"id": "free", "label": "Free", "description": f"{settings.free_plan_token_grant} tokens de prueba, sin tarjeta.",
             "requires_payment": False},
            {"id": "pro", "label": "Pro", "description": f"Cuota fija mensual, incluye {settings.pro_plan_token_grant} tokens/mes.",
             "requires_payment": billing.is_configured()},
            {"id": "payg", "label": "Pay-as-you-go", "description": "Sin cuota fija, se factura mensualmente según los tokens que consumas.",
             "requires_payment": billing.metered_configured()},
        ]
    }

@app.post("/auth/register")
async def auth_register(req: RegisterRequest, request: Request):
    ip_key = f"ip:{request.client.host if request.client else 'unknown'}"
    register_limiter.check(ip_key)
    register_limiter.hit(ip_key)
    if len(req.password) < 8:
        raise HTTPException(400, "La contraseña debe tener al menos 8 caracteres")
    if not req.gdpr_accepted:
        raise HTTPException(400, "Debes aceptar la política de privacidad (RGPD) para crear una cuenta")
    if req.plan not in PLANS:
        raise HTTPException(400, f"plan must be one of {sorted(PLANS)}")
    if memory.find_by_email(req.email):
        raise HTTPException(409, "Ya existe una cuenta con ese correo")

    raw, key_hash, prefix = generate_key()
    key_id = memory.create_account(
        name=req.name, email=req.email, key_hash=key_hash, prefix=prefix, role="user",
        password_hash=hash_password(req.password), gdpr_accepted=True, cookie_consent=req.cookies.model_dump(),
        plan=req.plan,
    )
    result = {"id": key_id, "api_key": raw, "warning": "Guarda esta clave ahora, no volverá a mostrarse."}

    if req.plan == "free":
        memory.add_token_credit(key_id, settings.free_plan_token_grant)
    elif req.plan == "pro" and billing.is_configured():
        try:
            result["checkout_url"] = await billing.create_checkout_session(req.email, key_id=key_id, plan="pro")
        except Exception:
            logger.exception("failed to create Pro checkout session on registration")
    elif req.plan == "payg" and billing.metered_configured():
        try:
            result["checkout_url"] = await billing.create_checkout_session(req.email, key_id=key_id, plan="payg")
        except Exception:
            logger.exception("failed to create Pay-as-you-go checkout session on registration")
    return result

@app.post("/auth/login")
async def auth_login(req: LoginRequest, request: Request):
    ip = request.client.host if request.client else "unknown"
    email_key = f"email:{req.email.strip().lower()}"
    login_limiter.check(email_key)
    login_ip_limiter.check(f"ip:{ip}")
    creds = memory.get_account_credentials(req.email)
    if not creds or not creds["active"] or not creds["password_hash"] or not verify_password(req.password, creds["password_hash"]):
        login_limiter.hit(email_key)
        login_ip_limiter.hit(f"ip:{ip}")
        raise HTTPException(401, "Correo o contraseña incorrectos")
    login_limiter.clear(email_key)
    raw, key_hash, prefix = generate_key()
    memory.rotate_api_key(creds["id"], key_hash, prefix)
    return {"id": creds["id"], "api_key": raw}

def _oauth_redirect_uri(provider: str) -> str:
    return f"{settings.public_base_url}/auth/{provider}/callback"

@app.get("/auth/{provider}/start")
async def auth_oauth_start(provider: str, gdpr_accepted: bool = False, analytics: bool = False):
    if provider not in ("google", "github"):
        raise HTTPException(404, "unknown provider")
    if not gdpr_accepted:
        raise HTTPException(400, "Debes aceptar la política de privacidad (RGPD) antes de continuar")
    if provider == "google" and not oauth.google_configured():
        raise HTTPException(503, "El inicio de sesión con Google no está configurado todavía")
    if provider == "github" and not oauth.github_configured():
        raise HTTPException(503, "El inicio de sesión con GitHub no está configurado todavía")
    state = uuid.uuid4().hex
    memory.save_oauth_state(state, provider, gdpr_accepted, {"necessary": True, "analytics": analytics})
    redirect_uri = _oauth_redirect_uri(provider)
    url = oauth.google_authorize_url(state, redirect_uri) if provider == "google" else oauth.github_authorize_url(state, redirect_uri)
    return RedirectResponse(url=url)

@app.get("/auth/{provider}/callback")
async def auth_oauth_callback(provider: str, code: str, state: str):
    if provider not in ("google", "github"):
        raise HTTPException(404, "unknown provider")
    pending = memory.pop_oauth_state(state)
    if not pending or pending["provider"] != provider:
        raise HTTPException(400, "Estado OAuth inválido o expirado")
    redirect_uri = _oauth_redirect_uri(provider)
    try:
        identity = await (oauth.google_fetch_identity(code, redirect_uri) if provider == "google"
                           else oauth.github_fetch_identity(code, redirect_uri))
    except Exception:
        logger.exception("oauth callback failed")
        raise HTTPException(502, "No se pudo completar el inicio de sesión")
    key_id = memory.find_by_oauth(provider, identity["subject"])
    if key_id is None and identity.get("email"):
        key_id = memory.find_by_email(identity["email"])
    if key_id is None:
        raw, key_hash, prefix = generate_key()
        key_id = memory.create_account(
            name=identity.get("name") or identity.get("email") or provider, email=identity.get("email"),
            key_hash=key_hash, prefix=prefix, role="user", oauth_provider=provider,
            oauth_subject=identity["subject"], gdpr_accepted=pending["gdpr_accepted"],
            cookie_consent=pending["cookie_consent"],
        )
    else:
        raw, key_hash, prefix = generate_key()
        memory.rotate_api_key(key_id, key_hash, prefix)
        memory.record_consent(key_id, pending["gdpr_accepted"], pending["cookie_consent"])
    return RedirectResponse(url=f"/ui/#key={raw}")

@app.get("/")
async def root():
    return RedirectResponse(url="/ui/")

async def _retention_loop():
    while True:
        try:
            result = memory.purge_older_than(settings.data_retention_days)
            if result["messages_deleted"] or result["activity_deleted"]:
                logger.info("RGPD retention purge: %s", result)
        except Exception:
            logger.exception("retention purge failed")
        await asyncio.sleep(24 * 60 * 60)

@app.on_event("startup")
async def _start_retention_loop():
    asyncio.create_task(_retention_loop())

@app.exception_handler(GatewayRejection)
async def gateway_rejection_handler(request: Request, exc: GatewayRejection):
    return JSONResponse(status_code=429, content={"error": {"type": "rate_limit_exceeded", "message": exc.reason}},
                        headers={"Retry-After": str(exc.retry_after)})

@app.get("/health")
async def health(): return {"status":"ok","model":settings.model_serve_name}

@app.post("/v1/decide")
async def decide(req: DecisionApiRequest, key: dict = Depends(require_api_key)):
    """Structured decision endpoint. Results are advisory; this endpoint executes no tools."""
    if not req.questions or len(req.questions) > settings.decision_max_questions:
        raise HTTPException(
            400,
            f"questions must contain between 1 and {settings.decision_max_questions} items",
        )
    payload = [q.model_dump() for q in req.questions]
    started_at = time.monotonic()
    try:
        async with gateway.slot(key):
            results, usage = await decision_runtime.decide_with_usage(req.context, payload)
    except (ValueError, TypeError) as exc:
        raise HTTPException(400, str(exc))
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))
    app.state.metrics["decision_requests"] += 1
    if key.get("id") is not None:
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))
        total_tokens = int(usage.get("total_tokens", prompt_tokens + completion_tokens))
        memory.record_usage(
            key["id"],
            prompt_tokens,
            completion_tokens,
            model=settings.model_serve_name,
            latency_ms=int((time.monotonic() - started_at) * 1000),
            client_key_id=key.get("client_key_id"),
        )
        gateway.record_tokens(key["id"], total_tokens, key.get("client_key_id"))
        if key.get("token_balance") is not None:
            memory.decrement_token_balance(key["id"], total_tokens)
        customer_id = key.get("stripe_customer_id")
        if customer_id:
            asyncio.create_task(billing.report_usage(customer_id, total_tokens))
    return {"object": "decision.list", "usage": usage, "data": [
        {"id": r.id, "probabilities": r.probabilities, "decision": r.decision,
         "confidence": r.confidence, "abstained": r.abstained,
         "abstention_reason": r.abstention_reason,
         "normalized_entropy": r.normalized_entropy,
         "margin": r.margin,
         "expected_score": r.expected_score}
        for r in results
    ]}


@app.get("/metrics")
async def metrics():
    return {"status":"ok","metrics":app.state.metrics}

@app.get("/admin/decision-shadow")
async def decision_shadow_report(days: int = 30, _: bool = Depends(require_admin)):
    return memory.decision_shadow_summary(days=min(max(days, 1), 365))

@app.get("/v1/models")
async def models():
    return {"object":"list","data":[{"id":"CeltIA V4","object":"model","owned_by":"local"}]}

def _log_tool_activity(key_id, name, args, result):
    if key_id is None or not isinstance(result, dict):
        return
    if name == "web_search":
        urls = [item.get("url") for item in result.get("results", []) if isinstance(item, dict) and item.get("url")]
        memory.record_activity(key_id, "web_search", json.dumps({"query": args.get("query"), "urls": urls}, ensure_ascii=False))
    elif name == "install_package":
        memory.record_activity(key_id, "dependency_install", json.dumps(
            {"package": args.get("package"), "ok": result.get("ok")}, ensure_ascii=False))

async def _build_response(req: ChatRequest, sid: str, key: dict, on_event=None, stream_tokens=False):
    def emit(event):
        if on_event is not None:
            on_event(event)

    on_token = None
    if stream_tokens:
        id_filter = persona.IdentityStreamFilter()

        def on_token(delta):
            if delta is None:
                id_filter.reset()
                emit({"type": "token_reset"})
                return
            piece = id_filter.feed(delta)
            if piece:
                emit({"type": "token", "text": piece})

    started_at = time.monotonic()
    cap = gateway.limits_for(key.get("role", "user"))["max_tokens_per_request"]
    if req.max_tokens:
        req.max_tokens = min(req.max_tokens, cap)
    text = "\n".join(m.content for m in req.messages if m.role == "user")[-20000:]
    r = route(text)
    if settings.decision_shadow_routing:
        shadow = await evaluate_route_shadow(decision_runtime, text, r.mode)
        if shadow:
            logger.info("CDE shadow route: %s", shadow)
            memory.record_decision_shadow(
                key.get("id"),
                shadow["heuristic"],
                shadow["cde"],
                shadow["confidence"],
                shadow["abstained"],
                shadow.get("abstention_reason"),
                shadow.get("normalized_entropy"),
                shadow.get("margin"),
            )
            emit({"type": "decision_shadow", **shadow})
    auto_agent = r.mode == "agent"
    if not auto_agent and req.mode in {"fast", "think", "code", "long"}:
        forced_tokens = 512 if req.mode == "fast" else 1024 if req.mode == "think" else 2048
        r = r.__class__(req.mode, r.difficulty, req.mode in {"think", "code", "long"}, req.max_tokens or forced_tokens)
    role = key.get("role", "user")
    keep_history = role_has_history(role)
    msgs = (memory.history(sid, api_key_id=key.get("id")) if keep_history else []) + [m.model_dump() for m in req.messages]
    allowed_tools = role_tools(role)
    prompt_tokens = completion_tokens = 0
    emit({"type": "status", "text": f"Entendido. Ruta elegida: {r.mode}"})
    if r.mode == "agent":
        app.state.metrics["agent_runs"] += 1
        result = await agent.run(msgs, allowed_tools=allowed_tools, role=role,
                                  on_tool_call=lambda n, a, res: _log_tool_activity(key.get("id"), n, a, res),
                                  on_event=on_event, on_token=on_token)
        answer = result["answer"]
        verification = result.get("verification", {})
        meta = {"route": r.mode, "difficulty": r.difficulty, "steps": result["steps"], "tool_calls": result["tool_calls"],
                "verification_checked": verification.get("checked", False), "verified": bool(verification.get("ok"))}
        completion_tokens = max(1, len(answer) // 4)
    else:
        mode_tools = AUTO_SEARCH_TOOLS | (CODE_TOOLS if r.mode == "code" else set())
        effective_tools = mode_tools if allowed_tools is None else (mode_tools & allowed_tools)
        hint_lines = []
        if "web_search" in effective_tools:
            hint_lines.append(
                "You have a working internet search tool named web_search. If the user asks you to search, "
                "investigate, find, or look up anything (in any language: busca, buscar, investiga, encuentra), "
                "or if answering accurately requires current or verifiable information, you MUST call web_search "
                "before answering. Never claim you lack real-time or internet access — you have it via this tool."
            )
        if "install_package" in effective_tools:
            hint_lines.append(
                "If you are writing code that needs a third-party library, call install_package to install it first."
            )
        hint_lines.append("Only use the tools listed for this request; do not invent or call tools that were not provided. Otherwise answer directly.")
        search_hint = {"role": "system", "content": persona.system_prompt(role, extra=" ".join(hint_lines))}
        emit({"type": "status", "text": "Consultando al modelo…"})
        search_result = await llm.chat(
            [search_hint] + msgs, thinking=r.thinking, max_tokens=req.max_tokens or r.max_tokens,
            temperature=req.temperature, tools=registry.schemas(only=effective_tools), on_token=on_token,
        )
        search_msg = search_result["choices"][0]["message"]
        tool_calls = search_msg.get("tool_calls") or []
        web_search_used = False
        if tool_calls:
            working = [search_hint] + msgs + [search_msg]
            for call in tool_calls[: settings.max_tool_calls]:
                fn = call.get("function", {})
                name = fn.get("name", "")
                raw_args = fn.get("arguments", "{}")
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except Exception:
                    args = {}
                emit({"type": "tool_start", "name": name, "args": args})
                try:
                    tool_result = await registry.call(name, args, only=effective_tools)
                except Exception as exc:
                    tool_result = {"ok": False, "error": str(exc)}
                emit({"type": "tool_result", "name": name,
                      "ok": bool(tool_result.get("ok", True)) if isinstance(tool_result, dict) else True})
                _log_tool_activity(key.get("id"), name, args, tool_result)
                if name == "web_search":
                    web_search_used = True
                working.append({"role": "tool", "tool_call_id": call.get("id", "call-0"),
                                 "content": tool_content(tool_result)})
            emit({"type": "status", "text": "Redactando la respuesta con los resultados…"})
            result = await llm.chat(working, thinking=r.thinking, max_tokens=req.max_tokens or r.max_tokens, temperature=req.temperature, on_token=on_token)
        else:
            result = search_result
        answer = result["choices"][0]["message"].get("content", "")
        meta = {"route": r.mode, "difficulty": r.difficulty, "thinking": r.thinking, "verified": False, "web_search_used": web_search_used}
        if result.get("meta", {}).get("offline_fallback"):
            meta["offline_fallback"] = True
            meta["model_status"] = "offline-demo-mode"
        usage = result.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", max(1, len(answer) // 4))
    answer = persona.enforce_identity(answer)
    if keep_history:
        memory.add(sid, "user", text, api_key_id=key.get("id"))
        memory.add(sid, "assistant", answer, api_key_id=key.get("id"))
        memory.prune_conversations(key["id"], keep=10)
    app.state.metrics["requests"] += 1
    if key.get("id") is not None:
        memory.record_usage(key["id"], prompt_tokens, completion_tokens, model=settings.model_serve_name,
                            latency_ms=int((time.monotonic() - started_at) * 1000),
                            client_key_id=key.get("client_key_id"))
        gateway.record_tokens(key["id"], prompt_tokens + completion_tokens, key.get("client_key_id"))
        if key.get("token_balance") is not None:
            memory.decrement_token_balance(key["id"], prompt_tokens + completion_tokens)
        customer_id = key.get("stripe_customer_id")
        if customer_id:
            asyncio.create_task(billing.report_usage(customer_id, prompt_tokens + completion_tokens))
    return {"id": "celtia-" + uuid.uuid4().hex, "object": "chat.completion", "model": settings.model_serve_name, "choices": [{"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens},
            "metadata": meta, "session_id": sid}

@app.post("/v1/chat/completions")
async def chat(req:ChatRequest, key=Depends(require_api_key)):
    if not req.messages: raise HTTPException(400, "messages cannot be empty")
    sid = req.session_id or str(uuid.uuid4())
    async with gateway.slot(key):
        return await _build_response(req, sid, key)

@app.post("/v1/chat/completions/stream")
async def chat_stream(req:ChatRequest, key=Depends(require_api_key)):
    if not req.messages: raise HTTPException(400, "messages cannot be empty")
    sid = req.session_id or str(uuid.uuid4())
    app.state.metrics["stream_requests"] += 1
    slot = gateway.slot(key)
    await slot.__aenter__()  # 429 is raised here, before the SSE response starts

    async def generate():
      try:
        queue: asyncio.Queue = asyncio.Queue()
        task = asyncio.create_task(_build_response(req, sid, key, on_event=queue.put_nowait, stream_tokens=True))

        def sse(obj):
            return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

        while not (task.done() and queue.empty()):
            try:
                yield sse(await asyncio.wait_for(queue.get(), timeout=0.2))
            except asyncio.TimeoutError:
                continue
        try:
            output = task.result()
        except HTTPException as exc:
            yield sse({"type": "error", "detail": exc.detail, "status": exc.status_code})
        except Exception as exc:
            logger.exception("stream failed")
            yield sse({"type": "error", "detail": str(exc), "status": 500})
        else:
            yield sse({"type": "chunk", "data": output})
        yield "data: [DONE]\n\n"
      finally:
        await slot.__aexit__(None, None, None)

    return StreamingResponse(generate(), media_type="text/event-stream")

@app.post("/admin/api-keys")
async def create_api_key(req: ApiKeyCreateRequest, _=Depends(require_admin)):
    if not req.name.strip():
        raise HTTPException(400, "name cannot be empty")
    if req.role not in ROLES:
        raise HTTPException(400, f"role must be one of {sorted(ROLES)}")
    raw_key, key_hash, prefix = generate_key()
    key_id = memory.create_api_key(req.name.strip(), key_hash, prefix, role=req.role)
    return {"id": key_id, "name": req.name.strip(), "role": req.role, "api_key": raw_key, "prefix": prefix,
            "warning": "Store this key now, it will not be shown again."}

@app.get("/admin/api-keys")
async def list_api_keys(_=Depends(require_admin)):
    return {"keys": memory.list_api_keys()}

@app.patch("/admin/api-keys/{key_id}")
async def update_api_key(key_id: int, req: ApiKeyUpdateRequest, _=Depends(require_admin)):
    target = memory.get_api_key(key_id)
    if not target:
        raise HTTPException(404, "API key not found")
    demoting = req.role is not None and req.role != "admin"
    deactivating = req.active is False
    if target["role"] == "admin" and target["active"] and (demoting or deactivating):
        active_admins = [k for k in memory.list_api_keys() if k["role"] == "admin" and k["active"]]
        if len(active_admins) <= 1:
            raise HTTPException(400, "No puedes quitar el rol ni desactivar al único administrador activo")
    if req.role is not None:
        if req.role not in ROLES:
            raise HTTPException(400, f"role must be one of {sorted(ROLES)}")
        if not memory.set_api_key_role(key_id, req.role):
            raise HTTPException(404, "API key not found")
    if req.active is not None and not memory.set_api_key_active(key_id, req.active):
        raise HTTPException(404, "API key not found")
    return memory.get_api_key(key_id)

class AdminPasswordRequest(BaseModel):
    new_password: str

@app.post("/admin/users/{user_id}/password")
async def admin_set_password(user_id: int, req: AdminPasswordRequest, _=Depends(require_admin)):
    if len(req.new_password) < 8:
        raise HTTPException(400, "La contraseña debe tener al menos 8 caracteres")
    if not memory.get_api_key(user_id):
        raise HTTPException(404, "Usuario no encontrado")
    memory.set_password_hash(user_id, hash_password(req.new_password))
    return {"ok": True}

@app.get("/v1/me")
async def me(key=Depends(require_api_key)):
    record = memory.get_api_key(key["id"])
    if not record:
        raise HTTPException(404, "API key not found")
    usage = memory.sum_usage(key["id"])
    subscription_status = await billing.get_subscription_status(record.get("stripe_subscription_id"))
    return {**record, "usage": usage, "subscription_status": subscription_status,
            "stripe_configured": billing.is_configured(), "history_enabled": role_has_history(record["role"])}

@app.post("/v1/files/extract")
async def extract_file(file: UploadFile = File(...), key=Depends(require_api_key)):
    data = await file.read(files_mod.MAX_UPLOAD_BYTES + 1)
    if len(data) > files_mod.MAX_UPLOAD_BYTES:
        raise HTTPException(413, "El archivo supera el máximo de 10 MB")
    try:
        return files_mod.extract_text(file.filename or "archivo", data)
    except files_mod.UnsupportedFile as exc:
        raise HTTPException(415, str(exc))

class ClientKeyCreateRequest(BaseModel):
    name: str
    requests_per_minute: int | None = None
    tokens_per_minute: int | None = None
    monthly_tokens: int | None = None

class ClientKeyLimitsRequest(BaseModel):
    requests_per_minute: int | None = None
    tokens_per_minute: int | None = None
    monthly_tokens: int | None = None

def _clean_limit(value):
    if value is None:
        return None
    if value <= 0:
        raise HTTPException(400, "Los límites deben ser números positivos (vacío = sin límite propio)")
    return value

@app.get("/v1/me/keys")
async def list_my_keys(key=Depends(require_api_key)):
    return {"keys": memory.list_client_keys(key["id"])}

@app.post("/v1/me/keys")
async def create_my_key(req: ClientKeyCreateRequest, key=Depends(require_api_key)):
    name = req.name.strip()
    if not name or len(name) > 60:
        raise HTTPException(400, "El nombre debe tener entre 1 y 60 caracteres")
    if len([k for k in memory.list_client_keys(key["id"]) if not k["revoked"]]) >= 10:
        raise HTTPException(400, "Máximo 10 claves activas; revoca alguna antes de crear otra")
    raw, key_hash, prefix = generate_client_key()
    key_id = memory.create_client_key(key["id"], name, key_hash, prefix, _clean_limit(req.requests_per_minute),
                                      _clean_limit(req.tokens_per_minute), _clean_limit(req.monthly_tokens))
    return {"id": key_id, "name": name, "prefix": prefix, "api_key": raw,
            "warning": "Guarda esta clave ahora, no volverá a mostrarse."}

@app.post("/v1/me/keys/{key_id}/limits")
async def set_my_key_limits(key_id: int, req: ClientKeyLimitsRequest, key=Depends(require_api_key)):
    if not memory.set_client_key_limits(key["id"], key_id, _clean_limit(req.requests_per_minute),
                                        _clean_limit(req.tokens_per_minute), _clean_limit(req.monthly_tokens)):
        raise HTTPException(404, "Clave no encontrada o revocada")
    return {"ok": True}

@app.delete("/v1/me/keys/{key_id}")
async def revoke_my_key(key_id: int, key=Depends(require_api_key)):
    if not memory.revoke_client_key(key["id"], key_id):
        raise HTTPException(404, "Clave no encontrada o ya revocada")
    return {"ok": True}

@app.get("/v1/me/usage")
async def my_usage(days: int = 30, key=Depends(require_api_key)):
    days = max(1, min(days, 365))
    return {**memory.usage_summary(key["id"], days), "limits": gateway.limits_for(key.get("role", "user")),
            "month_tokens": memory.month_tokens(key["id"]), "tokens_last_minute": gateway.tokens_last_minute(key["id"]),
            "token_balance": key.get("token_balance")}

class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str

@app.post("/v1/me/password")
async def change_my_password(req: ChangePasswordRequest, key=Depends(require_api_key)):
    stored = memory.get_password_hash(key["id"])
    if not stored or not verify_password(req.current_password, stored):
        raise HTTPException(401, "La contraseña actual no es correcta")
    if len(req.new_password) < 8:
        raise HTTPException(400, "La nueva contraseña debe tener al menos 8 caracteres")
    memory.set_password_hash(key["id"], hash_password(req.new_password))
    return {"ok": True}

class CreditRequest(BaseModel):
    amount: int

@app.post("/admin/api-keys/{key_id}/credit")
async def credit_api_key(key_id: int, req: CreditRequest, _=Depends(require_admin)):
    if req.amount <= 0:
        raise HTTPException(400, "amount must be positive")
    if not memory.add_token_credit(key_id, req.amount):
        raise HTTPException(404, "API key not found")
    return memory.get_api_key(key_id)

@app.get("/v1/conversations")
async def my_conversations(key=Depends(require_api_key)):
    if not role_has_history(key.get("role", "user")):
        raise HTTPException(403, "Your plan does not include conversation history")
    return {"conversations": memory.list_conversations(key["id"])}

@app.get("/v1/conversations/{session_id}")
async def my_conversation(session_id: str, key=Depends(require_api_key)):
    if not role_has_history(key.get("role", "user")):
        raise HTTPException(403, "Your plan does not include conversation history")
    return {"session_id": session_id, "messages": memory.get_conversation(key["id"], session_id)}

@app.delete("/v1/conversations/{session_id}")
async def delete_my_conversation(session_id: str, key=Depends(require_api_key)):
    if not role_has_history(key.get("role", "user")):
        raise HTTPException(403, "Your plan does not include conversation history")
    memory.delete_conversation(key["id"], session_id)
    return {"deleted": True}

@app.get("/v1/me/export")
async def export_my_data(key=Depends(require_api_key)):
    """RGPD art. 20: exporta todos los datos personales asociados a la clave del solicitante."""
    return memory.export_user_data(key["id"])

@app.delete("/v1/me")
async def delete_my_data(key=Depends(require_api_key)):
    """RGPD art. 17: borra historial/actividad y desactiva la cuenta (derecho al olvido)."""
    memory.delete_user_data(key["id"])
    return {"deleted": True}

@app.delete("/admin/api-keys/{key_id}")
async def revoke_api_key(key_id: int, _=Depends(require_admin)):
    if not memory.revoke_api_key(key_id):
        raise HTTPException(404, "API key not found")
    return {"revoked": key_id}

@app.get("/admin/api-keys/{key_id}/billing")
async def api_key_billing(key_id: int, _=Depends(require_admin)):
    record = memory.get_api_key(key_id)
    if not record:
        raise HTTPException(404, "API key not found")
    usage = memory.sum_usage(key_id)
    subscription_status = await billing.get_subscription_status(record.get("stripe_subscription_id"))
    return {
        "id": key_id,
        "email": record.get("email"),
        "usage": usage,
        "stripe_configured": billing.is_configured(),
        "stripe_customer_id": record.get("stripe_customer_id"),
        "subscription_status": subscription_status,
        "token_balance": record.get("token_balance"),
        "token_balance_max": record.get("token_balance_max"),
    }

@app.get("/admin/activity")
async def list_activity(api_key_id: int | None = None, kind: str | None = None, limit: int = 100, _=Depends(require_admin)):
    return {"activity": memory.list_activity(api_key_id=api_key_id, kind=kind, limit=min(limit, 500))}

@app.post("/billing/signup")
async def billing_signup(req: BillingSignupRequest):
    if not billing.is_configured():
        raise HTTPException(503, "Billing is not configured yet")
    if not req.email.strip():
        raise HTTPException(400, "email cannot be empty")
    checkout_url = await billing.create_checkout_session(req.email.strip())
    return {"checkout_url": checkout_url}

@app.get("/billing/complete")
async def billing_complete(session_id: str):
    if not billing.is_configured():
        raise HTTPException(503, "Billing is not configured yet")
    session = await billing.retrieve_session(session_id)
    if session.payment_status not in {"paid", "no_payment_required"}:
        raise HTTPException(402, "Payment was not completed")
    customer_id = session.customer.id if session.customer else None
    subscription_id = session.subscription.id if session.subscription else None
    email = session.customer_details.email if session.customer_details else None
    ref = getattr(session, "client_reference_id", None)
    referenced_id, referenced_plan = None, "pro"
    if ref:
        parts = ref.split(":", 1)
        referenced_id = int(parts[0])
        referenced_plan = parts[1] if len(parts) > 1 else "pro"

    if referenced_id and memory.get_api_key(referenced_id):
        # El pago confirma la cuenta que ya se creó en /auth/register: se sube a premium
        # y, si el plan lo incluye, se le acreditan tokens fijos, sin duplicar cuentas.
        key_id = referenced_id
        memory.link_stripe_customer(key_id, customer_id, subscription_id)
        memory.set_api_key_role(key_id, "premium")
        memory.set_plan(key_id, referenced_plan)
        if referenced_plan == "pro":
            memory.add_token_credit(key_id, settings.pro_plan_token_grant)
        # payg: sin tope de saldo (token_balance queda NULL = ilimitado), se factura por uso real vía Stripe metered billing
        return RedirectResponse(url="/ui/#billing=success")

    existing_id = memory.find_api_key_by_customer(customer_id) if customer_id else None
    if existing_id:
        raw_key = None
        key_id = existing_id
    else:
        raw_key, key_hash, prefix = generate_key()
        key_id = memory.create_api_key(
            email or "stripe-customer", key_hash, prefix,
            email=email, role="premium",
            stripe_customer_id=customer_id, stripe_subscription_id=subscription_id,
        )
        memory.add_token_credit(key_id, settings.pro_plan_token_grant)
    body = (
        f"<h1>CeltIA</h1><p>Subscription active.</p>"
        f"<p>Your API key: <code>{raw_key}</code></p><p>Store it now, it will not be shown again.</p>"
        if raw_key else "<h1>CeltIA</h1><p>This subscription already has an API key issued.</p>"
    )
    return HTMLResponse(body)

@app.post("/webhooks/stripe")
async def stripe_webhook(request: Request, stripe_signature: str | None = Header(default=None)):
    payload = await request.body()
    try:
        event = billing.verify_webhook(payload, stripe_signature or "")
    except Exception:
        logger.exception("Stripe webhook signature verification failed")
        raise HTTPException(400, "invalid webhook signature")

    obj = event["data"]["object"]
    event_type = event["type"]
    if event_type in {"customer.subscription.deleted", "invoice.payment_failed"}:
        customer_id = obj.get("customer")
        key_id = memory.find_api_key_by_customer(customer_id) if customer_id else None
        if key_id:
            memory.set_api_key_active(key_id, False)
    elif event_type == "invoice.paid":
        customer_id = obj.get("customer")
        key_id = memory.find_api_key_by_customer(customer_id) if customer_id else None
        if key_id:
            memory.set_api_key_active(key_id, True)
    return {"received": True}


# ---------------------------------------------------------------------------
# Creador Studio: proyectos reales, herramientas del agente existente ligadas
# a un workspace, sandbox Docker por proyecto y vista previa en vivo.
# ---------------------------------------------------------------------------

def require_creator_access(key=Depends(require_api_key)):
    if not role_has_history(key.get("role", "user")):
        raise HTTPException(403, "Tu plan no incluye Creador Studio. Pasa a premium para usarlo.")
    return key

CREATOR_AUTO_REPAIR_INSTRUCTIONS = (
    "Estás operando dentro de Creador Studio sobre un proyecto real, no solo generando texto. "
    "Usa listFiles/readFile antes de editar para entender el proyecto existente. Usa createFile o "
    "updateFile para escribir ficheros completos, y applyPatch para cambios pequeños y localizados. "
    "Tras crear o modificar ficheros de un proyecto con dependencias, instala paquetes con "
    "installPackage y comprueba que compila con build. Si build, runTests o runCommand devuelven un "
    "error, NUNCA le pidas al usuario que te copie el error: localiza el fichero y la línea "
    "implicados con el propio mensaje de error, aplica un patch que lo corrija, y vuelve a ejecutar "
    "build antes de dar tu respuesta final. Solo cuando el proyecto tenga un servidor de desarrollo "
    "arrancable, usa startPreview para exponer la vista previa en vivo. "
    "Regla estricta: si runCommand, build o runTests devuelven ok=false, tu SIGUIENTE acción en esa misma "
    "respuesta debe ser una llamada a la herramienta applyPatch o updateFile con la corrección — nunca "
    "termines el turno con solo una explicación en texto de lo que habría que arreglar. Después de aplicar "
    "el patch, vuelve a llamar a runCommand/build para confirmar que el error desapareció antes de responder."
)

class CreateProjectRequest(BaseModel):
    name: str
    template: str = "blank"

@app.get("/creator/sandbox-status")
async def creator_sandbox_status(_=Depends(require_creator_access)):
    return {"configured": sandbox_configured()}

@app.post("/creator/projects")
async def create_project(req: CreateProjectRequest, key=Depends(require_creator_access)):
    meta = project_manager.create(req.name, key["id"], req.template)
    creator_git.ensure_repo(project_manager.workspace_path(meta["id"]))
    return meta

@app.get("/creator/projects")
async def list_projects(key=Depends(require_creator_access)):
    return {"projects": project_manager.list(key["id"])}

@app.delete("/creator/projects/{project_id}")
async def delete_project(project_id: str, key=Depends(require_creator_access)):
    try:
        meta = project_manager.assert_owner(project_id, key["id"])
    except PermissionError:
        raise HTTPException(404, "project not found")
    if meta and sandbox_configured():
        try:
            sandbox_for(project_id, meta["container_name"], project_manager.host_workspace_path(project_id)).stop_and_remove()
        except Exception:
            logger.exception("failed to remove sandbox container for project %s", project_id)
    if not project_manager.delete(project_id, key["id"]):
        raise HTTPException(404, "project not found")
    return {"deleted": True}

def _project_root(project_id: str, key):
    project_manager.assert_owner(project_id, key["id"])
    return project_manager.workspace_path(project_id)

@app.get("/creator/projects/{project_id}/files")
async def creator_list_files(project_id: str, path: str = "", key=Depends(require_creator_access)):
    try:
        root = _project_root(project_id, key)
        return {"entries": creator_fs.list_files(root, path)}
    except PermissionError as exc:
        raise HTTPException(403, str(exc))

@app.get("/creator/projects/{project_id}/file")
async def creator_read_file(project_id: str, path: str, key=Depends(require_creator_access)):
    try:
        root = _project_root(project_id, key)
        return {"path": path, "content": creator_fs.read_file(root, path)}
    except PermissionError as exc:
        raise HTTPException(403, str(exc))
    except FileNotFoundError:
        raise HTTPException(404, "file not found")

class WriteFileRequest(BaseModel):
    path: str
    content: str

@app.put("/creator/projects/{project_id}/file")
async def creator_write_file(project_id: str, req: WriteFileRequest, key=Depends(require_creator_access)):
    try:
        root = _project_root(project_id, key)
        creator_fs.write_file(root, req.path, req.content)
        return {"ok": True}
    except PermissionError as exc:
        raise HTTPException(403, str(exc))

@app.delete("/creator/projects/{project_id}/file")
async def creator_delete_file(project_id: str, path: str, key=Depends(require_creator_access)):
    try:
        root = _project_root(project_id, key)
        creator_fs.delete_file(root, path)
        return {"ok": True}
    except PermissionError as exc:
        raise HTTPException(403, str(exc))

class MoveFileRequest(BaseModel):
    from_path: str
    to_path: str

@app.post("/creator/projects/{project_id}/move")
async def creator_move_file(project_id: str, req: MoveFileRequest, key=Depends(require_creator_access)):
    try:
        root = _project_root(project_id, key)
        creator_fs.move_file(root, req.from_path, req.to_path)
        return {"ok": True}
    except PermissionError as exc:
        raise HTTPException(403, str(exc))

class PreviewRequest(BaseModel):
    command: str = "npm run dev -- --host 0.0.0.0 --port 3000"

@app.post("/creator/projects/{project_id}/preview/start")
async def creator_start_preview(project_id: str, req: PreviewRequest, key=Depends(require_creator_access)):
    if not sandbox_configured():
        raise HTTPException(503, "El sandbox Docker no está configurado (revisa CREATOR_HOST_PROJECTS_DIR).")
    meta = project_manager.assert_owner(project_id, key["id"])
    sandbox = sandbox_for(project_id, meta["container_name"], project_manager.host_workspace_path(project_id))
    return sandbox.start_preview(req.command)

@app.get("/creator/projects/{project_id}/logs")
async def creator_get_logs(project_id: str, key=Depends(require_creator_access)):
    meta = project_manager.assert_owner(project_id, key["id"])
    sandbox = sandbox_for(project_id, meta["container_name"], project_manager.host_workspace_path(project_id))
    return {"logs": sandbox.get_runtime_logs()}

@app.get("/creator/projects/{project_id}/git/status")
async def creator_git_status(project_id: str, key=Depends(require_creator_access)):
    root = _project_root(project_id, key)
    return creator_git.status(root)

@app.get("/creator/projects/{project_id}/git/diff")
async def creator_git_diff(project_id: str, key=Depends(require_creator_access)):
    root = _project_root(project_id, key)
    return creator_git.diff(root)

class CommitRequest(BaseModel):
    message: str

@app.post("/creator/projects/{project_id}/git/commit")
async def creator_git_commit(project_id: str, req: CommitRequest, key=Depends(require_creator_access)):
    root = _project_root(project_id, key)
    return creator_git.commit(root, req.message)

class PushRequest(BaseModel):
    remote_url: str
    branch: str = "main"

@app.post("/creator/projects/{project_id}/git/push")
async def creator_git_push(project_id: str, req: PushRequest, key=Depends(require_creator_access)):
    root = _project_root(project_id, key)
    return creator_git.add_remote_and_push(root, req.remote_url, req.branch)

class CreatorChatRequest(BaseModel):
    message: str

@app.post("/creator/projects/{project_id}/chat")
async def creator_chat(project_id: str, req: CreatorChatRequest, key=Depends(require_creator_access)):
    project_manager.assert_owner(project_id, key["id"])
    try:
        project_registry = build_creator_registry(project_id)
    except ValueError:
        raise HTTPException(404, "project not found")

    activity = []
    def on_tool_call(name, args, result):
        activity.append({"tool": name, "args": args, "ok": bool(result.get("ok", True)) if isinstance(result, dict) else True})

    result = await agent.run(
        [{"role": "user", "content": req.message}],
        allowed_tools=CREATOR_TOOL_NAMES,
        on_tool_call=on_tool_call,
        role=key.get("role", "user"),
        registry=project_registry,
        extra_instructions=CREATOR_AUTO_REPAIR_INSTRUCTIONS,
    )
    answer = persona.enforce_identity(result.get("answer", ""))
    if _looks_like_leaked_reasoning(answer):
        answer = _summarize_activity(activity)
    memory.record_activity(key["id"], "creator_chat", json.dumps({"project_id": project_id, "message": req.message[:200]}))
    return {"answer": answer, "steps": result.get("steps", 0), "tool_calls": result.get("tool_calls", 0), "activity": activity}


_LEAKED_REASONING_HINTS = re.compile(
    r"\bwait,|\bthe user\b|\bi need to\b|\blet me\b|\bi'm not sure\b|\bhmm,|\bso the project\b|"
    r"\bel usuario\b|\bseg[uú]n la instrucci[oó]n\b|\bparece ser\b|\bpero el error\b|\blas herramientas disponibles\b|"
    r"\bcreo que\b|\bentonces\b.{0,20}\bdebo\b",
    re.I,
)


def _looks_like_leaked_reasoning(text: str) -> bool:
    if not text:
        return True
    return bool(_LEAKED_REASONING_HINTS.search(text)) and len(text) > 200


def _summarize_activity(activity: list[dict]) -> str:
    if not activity:
        return "Tarea completada."
    done = [a["tool"] for a in activity if a.get("ok")]
    failed = [a["tool"] for a in activity if not a.get("ok")]
    parts = []
    if done:
        parts.append("Herramientas ejecutadas correctamente: " + ", ".join(done) + ".")
    if failed:
        parts.append("Fallaron: " + ", ".join(failed) + ".")
    return " ".join(parts) or "Tarea completada."

