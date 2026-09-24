# Copyright (c) 2026 Luis Manuel Cousido Hermida. All rights reserved.
import ast
import subprocess
import sys
import tempfile


class Verifier:
    """Minimal verification layer for code and answer quality."""

    def verify_code(self, code: str) -> dict:
        if not code.strip():
            return {"ok": False, "error": "empty code"}
        try:
            compile(code, "<mini_council>", "exec")
        except SyntaxError as exc:
            return {"ok": False, "error": f"syntax error: {exc.msg} at line {exc.lineno}"}

        with tempfile.TemporaryDirectory() as td:
            path = f"{td}/verification_check.py"
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(code)
            try:
                proc = subprocess.run(
                    [sys.executable, "-I", path],
                    cwd=td,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
            except subprocess.TimeoutExpired:
                return {"ok": False, "error": "execution timeout"}

            if proc.returncode != 0:
                return {
                    "ok": False,
                    "error": proc.stderr.strip() or proc.stdout.strip() or "execution failed",
                    "stdout": proc.stdout[:2000],
                    "stderr": proc.stderr[:2000],
                }

            return {"ok": True, "stdout": proc.stdout[:2000], "stderr": proc.stderr[:2000]}

    def verify_text(self, text: str) -> dict:
        return {"ok": bool(text and text.strip()), "length": len(text.strip())}
