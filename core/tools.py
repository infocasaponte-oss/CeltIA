# Copyright (c) 2026 Luis Manuel Cousido Hermida. All rights reserved.
import ast, operator, os, subprocess, sys, tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

import httpx

from core import webfetch
from core.config import settings
from core.policy import ToolPolicy


@dataclass
class Tool:
    name: str
    description: str
    schema: dict[str, Any]
    handler: Callable[..., Any]

    def openai(self):
        return {"type":"function","function":{
            "name":self.name,"description":self.description,"parameters":self.schema}}


PUBLIC_SAFE_TOOLS = {"calculator", "current_time", "web_search", "fetch_url"}


_DEFAULT_POLICY = object()


class Registry:
    def __init__(self, policy=_DEFAULT_POLICY):
        self.tools = {}
        # policy=None means "no generic validation" (each tool validates its own args),
        # distinct from the omitted-argument default, which falls back to ToolPolicy().
        self.policy = ToolPolicy() if policy is _DEFAULT_POLICY else policy

    def add(self, t): self.tools[t.name] = t

    def schemas(self, only: set[str] | None = None):
        return [t.openai() for t in self.tools.values() if only is None or t.name in only]

    async def call(self, name, args, only: set[str] | None = None):
        if name not in self.tools: raise ValueError("tool not allowed")
        if only is not None and name not in only: raise ValueError("tool not allowed for this API key")
        if self.policy is not None:
            validated = self.policy.validate(name, args)
        else:
            validated = args
        result = self.tools[name].handler(**validated)
        if hasattr(result, "__await__"): result = await result
        return result

OPS={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,
     ast.Div:operator.truediv,ast.Pow:operator.pow,ast.Mod:operator.mod,
     ast.USub:operator.neg}

def calc_node(n):
    if isinstance(n,ast.Constant) and isinstance(n.value,(int,float)): return n.value
    if isinstance(n,ast.UnaryOp) and type(n.op) in OPS: return OPS[type(n.op)](calc_node(n.operand))
    if isinstance(n,ast.BinOp) and type(n.op) in OPS: return OPS[type(n.op)](calc_node(n.left),calc_node(n.right))
    raise ValueError("unsupported expression")

def calculator(expression: str):
    return {"expression":expression,"result":calc_node(ast.parse(expression,mode="eval").body)}

def python_exec(code: str, timeout: int = 8):
    if len(code)>50000: raise ValueError("code too large")
    with tempfile.TemporaryDirectory() as td:
        path=os.path.join(td,"main.py")
        open(path,"w",encoding="utf8").write(code)
        env={"PATH":os.environ.get("PATH",""),"HOME":td,"PYTHONUNBUFFERED":"1"}
        try:
            p=subprocess.run([sys.executable,"-I",path],cwd=td,env=env,capture_output=True,text=True,timeout=timeout)
            return {"ok":p.returncode==0,"returncode":p.returncode,"stdout":p.stdout[-20000:],"stderr":p.stderr[-20000:]}
        except subprocess.TimeoutExpired:
            return {"ok":False,"timeout":True,"stdout":"","stderr":"timeout"}

def read_file(path: str):
    p=os.path.realpath(path)
    root=os.path.realpath("/workspace")
    if not (p==root or p.startswith(root+os.sep)): raise PermissionError("restricted to /workspace")
    if not os.path.isfile(p): raise FileNotFoundError(path)
    return {"path":p,"content":open(p,encoding="utf8",errors="replace").read()[:100000]}


def list_dir(path: str = "/workspace"):
    p = os.path.realpath(path)
    root = os.path.realpath("/workspace")
    if not (p == root or p.startswith(root + os.sep)):
        raise PermissionError("restricted to /workspace")
    if not os.path.isdir(p):
        raise FileNotFoundError(path)
    entries = []
    for name in sorted(os.listdir(p)):
        full = os.path.join(p, name)
        entries.append({"name": name, "type": "dir" if os.path.isdir(full) else "file"})
    return {"path": p, "entries": entries}


def grep_text(path: str, pattern: str):
    p = os.path.realpath(path)
    root = os.path.realpath("/workspace")
    if not (p == root or p.startswith(root + os.sep)):
        raise PermissionError("restricted to /workspace")
    if not os.path.exists(p):
        raise FileNotFoundError(path)
    matches = []
    for dirpath, _, files in os.walk(p):
        for filename in files:
            file_path = os.path.join(dirpath, filename)
            try:
                with open(file_path, "r", encoding="utf-8", errors="replace") as handle:
                    for lineno, line in enumerate(handle, 1):
                        if pattern.lower() in line.lower():
                            matches.append({"file": file_path, "line": lineno, "text": line.strip()})
            except OSError:
                continue
    return {"path": p, "pattern": pattern, "matches": matches[:50]}


def current_time():
    now = datetime.now(timezone.utc)
    return {"iso": now.isoformat(), "utc": now.astimezone(timezone.utc).isoformat()}


async def web_search(query: str, count: int = 3):
    if not settings.brave_search_api_key:
        return {"ok": False, "error": "BRAVE_SEARCH_API_KEY is not configured"}
    count = max(1, min(int(count), 5))
    headers = {"Accept": "application/json", "X-Subscription-Token": settings.brave_search_api_key}
    params = {"q": query, "count": count}
    try:
        async with httpx.AsyncClient(timeout=12) as client:
            r = await client.get("https://api.search.brave.com/res/v1/web/search", headers=headers, params=params)
            r.raise_for_status()
            data = r.json()
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    results = [
        {"title": item.get("title"), "url": item.get("url"), "snippet": (item.get("description") or "")[:300]}
        for item in data.get("web", {}).get("results", [])[:count]
    ]
    return {"ok": True, "query": query, "results": results}


def install_package(package: str):
    try:
        p = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", "--no-input", package],
            capture_output=True, text=True, timeout=60,
        )
        return {"ok": p.returncode == 0, "package": package, "stdout": p.stdout[-4000:], "stderr": p.stderr[-4000:]}
    except subprocess.TimeoutExpired:
        return {"ok": False, "package": package, "error": "timeout"}


def builtins(policy=None):
    r = Registry(policy=policy)
    r.add(Tool("calculator","Evaluate safe arithmetic.",{"type":"object","properties":{"expression":{"type":"string"}},"required":["expression"],"additionalProperties":False},calculator))
    r.add(Tool("python_exec","Run Python in a restricted subprocess.",{"type":"object","properties":{"code":{"type":"string"},"timeout":{"type":"integer","minimum":1,"maximum":15}},"required":["code"],"additionalProperties":False},python_exec))
    r.add(Tool("read_file","Read text under /workspace.",{"type":"object","properties":{"path":{"type":"string"}},"required":["path"],"additionalProperties":False},read_file))
    r.add(Tool("list_dir","List files and directories under /workspace.",{"type":"object","properties":{"path":{"type":"string"}},"required":[],"additionalProperties":False},list_dir))
    r.add(Tool("grep_text","Search text within /workspace files.",{"type":"object","properties":{"path":{"type":"string"},"pattern":{"type":"string"}},"required":["pattern"],"additionalProperties":False},grep_text))
    r.add(Tool("current_time","Return the current UTC time.",{"type":"object","properties":{},"required":[],"additionalProperties":False},current_time))
    r.add(Tool("web_search","Search the web for current information using Brave Search.",{"type":"object","properties":{"query":{"type":"string"},"count":{"type":"integer","minimum":1,"maximum":10}},"required":["query"],"additionalProperties":False},web_search))
    r.add(Tool("fetch_url","Read the text content of a public web page (http/https). Use it to open a link the user mentions or a URL returned by web_search.",{"type":"object","properties":{"url":{"type":"string"}},"required":["url"],"additionalProperties":False},webfetch.fetch_url))
    r.add(Tool("install_package","Install a Python package needed to write or run code, via pip.",{"type":"object","properties":{"package":{"type":"string"}},"required":["package"],"additionalProperties":False},install_package))
    return r
