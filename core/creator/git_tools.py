# Copyright (c) 2026 Luis Manuel Cousido Hermida. All rights reserved.
"""Local Git operations for a project's workspace.

These run `git` directly on the host (not inside the sandbox container) because
git itself doesn't execute arbitrary project code -- only the project's own
scripts (npm install, dev servers, etc.) go through the Docker sandbox.
"""
import subprocess

GIT_TIMEOUT = 30


def _run(root, args):
    try:
        p = subprocess.run(
            ["git", *args], cwd=str(root), capture_output=True, text=True, timeout=GIT_TIMEOUT
        )
        return {"ok": p.returncode == 0, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}
    except subprocess.TimeoutExpired:
        return {"ok": False, "stdout": "", "stderr": "git command timed out"}
    except FileNotFoundError:
        return {"ok": False, "stdout": "", "stderr": "git is not installed in the API container"}


def ensure_repo(root) -> dict:
    if (root / ".git").exists():
        return {"ok": True, "already_initialized": True}
    _run(root, ["init"])
    _run(root, ["config", "user.email", "celtia@local"])
    _run(root, ["config", "user.name", "CeltIA Creador"])
    return {"ok": True, "already_initialized": False}


def status(root) -> dict:
    return _run(root, ["status", "--porcelain=v1", "-b"])


def diff(root) -> dict:
    return _run(root, ["diff"])


def commit(root, message: str) -> dict:
    ensure_repo(root)
    add_result = _run(root, ["add", "-A"])
    if not add_result["ok"]:
        return add_result
    return _run(root, ["commit", "-m", message, "--allow-empty-message"])


def log(root, limit: int = 20) -> dict:
    return _run(root, ["log", f"-{limit}", "--pretty=format:%h|%ad|%s", "--date=iso"])


def add_remote_and_push(root, remote_url: str, branch: str = "main") -> dict:
    _run(root, ["branch", "-M", branch])
    existing = _run(root, ["remote"])
    if "origin" in (existing.get("stdout") or "").split():
        _run(root, ["remote", "set-url", "origin", remote_url])
    else:
        _run(root, ["remote", "add", "origin", remote_url])
    return _run(root, ["push", "-u", "origin", branch])
