# Copyright (c) 2026 Luis Manuel Cousido Hermida. All rights reserved.
import ast
import re
from dataclasses import dataclass

PACKAGE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,213}(==[A-Za-z0-9_.\-]+)?$")


@dataclass(frozen=True)
class SandboxConfig:
    max_timeout_seconds: int = 10
    max_ram_mb: int = 512
    max_cpu_cores: int = 1
    allow_network: bool = False
    allow_filesystem: bool = False
    max_code_chars: int = 50000


class ToolPolicy:
    """Minimal allow-list and validation policy for safe local tool use."""

    def __init__(self, config: SandboxConfig | None = None):
        self.config = config or SandboxConfig()
        self.allowed_tools = {"calculator", "python_exec", "read_file", "list_dir", "grep_text", "current_time", "web_search", "install_package", "system_audit"}

    def is_allowed(self, name: str) -> bool:
        return name in self.allowed_tools

    def validate_python_code(self, code: str) -> str:
        if not code or not code.strip():
            raise ValueError("code cannot be empty")
        if len(code) > self.config.max_code_chars:
            raise ValueError("code too large")
        try:
            tree = ast.parse(code, mode="exec")
        except SyntaxError as exc:
            raise ValueError(f"invalid Python: {exc.msg}") from exc
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                raise ValueError("import statements are disabled in the sandbox")
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute):
                    if func.attr in {"system", "popen", "run", "exec", "eval", "compile"}:
                        raise ValueError("dangerous Python calls are blocked")
                if isinstance(func, ast.Name) and func.id in {"__import__", "open", "exec", "eval"}:
                    raise ValueError("dangerous Python calls are blocked")
        return code

    def validate(self, name: str, args: dict) -> dict:
        if not self.is_allowed(name):
            raise ValueError(f"tool not allowed: {name}")
        if name == "calculator":
            expression = str(args.get("expression", "")).strip()
            if not expression:
                raise ValueError("expression cannot be empty")
            if len(expression) > 1000:
                raise ValueError("expression too long")
            return {"expression": expression}
        if name == "python_exec":
            code = self.validate_python_code(str(args.get("code", "")))
            timeout = int(args.get("timeout", self.config.max_timeout_seconds))
            if timeout < 1 or timeout > self.config.max_timeout_seconds:
                raise ValueError("timeout out of range")
            return {"code": code, "timeout": timeout}
        if name in {"read_file", "list_dir", "grep_text"}:
            path = str(args.get("path", "/workspace")).strip()
            if not path:
                raise ValueError("path cannot be empty")
            return {"path": path}
        if name == "current_time":
            return {}
        if name == "web_search":
            query = str(args.get("query", "")).strip()
            if not query:
                raise ValueError("query cannot be empty")
            if len(query) > 400:
                raise ValueError("query too long")
            count = int(args.get("count", 5))
            if count < 1 or count > 10:
                raise ValueError("count out of range")
            return {"query": query, "count": count}
        if name == "install_package":
            package = str(args.get("package", "")).strip()
            if not package:
                raise ValueError("package cannot be empty")
            if not PACKAGE_NAME_RE.match(package):
                raise ValueError("invalid package name")
            return {"package": package}
        return dict(args)
