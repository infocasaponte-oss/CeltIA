# Copyright (c) 2026 Luis Manuel Cousido Hermida. All rights reserved.
"""Filesystem operations scoped to a single project's workspace directory.

Every function takes a `root` (the project's workspace, an absolute Path) and a
relative `path`, and refuses to escape `root` via `..` or symlinks.
"""
import os
import shutil
from pathlib import Path

MAX_FILE_BYTES = 2_000_000


def _resolve(root: Path, rel_path: str) -> Path:
    rel_path = (rel_path or "").lstrip("/\\")
    candidate = (root / rel_path).resolve()
    root_resolved = root.resolve()
    if candidate != root_resolved and root_resolved not in candidate.parents:
        raise PermissionError(f"path escapes project workspace: {rel_path}")
    return candidate


def list_files(root: Path, rel_path: str = "") -> list[dict]:
    base = _resolve(root, rel_path)
    if not base.exists():
        return []
    nodes = []
    for entry in sorted(base.iterdir(), key=lambda e: (e.is_file(), e.name.lower())):
        if entry.name in {".git", "node_modules", "__pycache__"}:
            continue
        rel = str(entry.relative_to(root.resolve())).replace("\\", "/")
        nodes.append({"path": rel, "name": entry.name, "type": "dir" if entry.is_dir() else "file"})
    return nodes


def list_files_recursive(root: Path, rel_path: str = "", max_entries: int = 2000) -> list[dict]:
    base = _resolve(root, rel_path)
    nodes = []
    skip_dirs = {".git", "node_modules", "__pycache__", "dist", "build"}
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for filename in filenames:
            full = Path(dirpath) / filename
            rel = str(full.relative_to(root.resolve())).replace("\\", "/")
            nodes.append({"path": rel, "name": filename, "type": "file"})
            if len(nodes) >= max_entries:
                return nodes
    return nodes


def read_file(root: Path, rel_path: str) -> str:
    target = _resolve(root, rel_path)
    if not target.is_file():
        raise FileNotFoundError(rel_path)
    data = target.read_bytes()[:MAX_FILE_BYTES]
    return data.decode("utf-8", errors="replace")


def write_file(root: Path, rel_path: str, content: str) -> None:
    target = _resolve(root, rel_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def delete_file(root: Path, rel_path: str) -> None:
    target = _resolve(root, rel_path)
    if target.is_dir():
        shutil.rmtree(target, ignore_errors=True)
    elif target.exists():
        target.unlink()


def move_file(root: Path, from_path: str, to_path: str) -> None:
    src = _resolve(root, from_path)
    dst = _resolve(root, to_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))


def search_project(root: Path, query: str, max_results: int = 50) -> list[dict]:
    query_lower = query.lower()
    matches = []
    for node in list_files_recursive(root):
        full = root.resolve() / node["path"]
        try:
            with open(full, "r", encoding="utf-8", errors="replace") as handle:
                for lineno, line in enumerate(handle, 1):
                    if query_lower in line.lower():
                        matches.append({"path": node["path"], "line": lineno, "text": line.strip()[:200]})
                        if len(matches) >= max_results:
                            return matches
        except OSError:
            continue
    return matches


def apply_patch(root: Path, rel_path: str, patch: str) -> None:
    """Minimal unified-diff-free patch: applies a simple search/replace patch format:

        <<<<<<< SEARCH
        old text
        =======
        new text
        >>>>>>> REPLACE

    Chosen over full unified diffs because it's far more reliable for a small local
    model to emit correctly, at the cost of only supporting one hunk per call.
    """
    target = _resolve(root, rel_path)
    if not target.is_file():
        raise FileNotFoundError(rel_path)
    original = target.read_text(encoding="utf-8")

    if "<<<<<<< SEARCH" in patch and ">>>>>>> REPLACE" in patch:
        search_block = patch.split("<<<<<<< SEARCH", 1)[1].split("=======", 1)[0].strip("\n")
        replace_block = patch.split("=======", 1)[1].split(">>>>>>> REPLACE", 1)[0].strip("\n")
        if search_block not in original:
            raise ValueError("patch search block not found in file")
        updated = original.replace(search_block, replace_block, 1)
    else:
        updated = patch

    target.write_text(updated, encoding="utf-8")
