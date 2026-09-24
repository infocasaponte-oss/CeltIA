# Copyright (c) 2026 Luis Manuel Cousido Hermida. All rights reserved.
"""Builds a per-project Registry implementing the CreatorTools contract, so the
existing Agent (core/agent.py) can operate on a real project workspace instead
of just returning code in a chat bubble. No new agent, no new model: this only
gives the current agent a bound set of tools scoped to one project's sandbox.
"""
from core.creator import filesystem as fs
from core.creator import git_tools
from core.creator.projects import project_manager
from core.creator.sandbox import sandbox_for
from core.tools import Registry, Tool


def build_creator_registry(project_id: str) -> Registry:
    meta = project_manager.get(project_id)
    if not meta:
        raise ValueError("unknown project")
    root = project_manager.workspace_path(project_id)
    sandbox = sandbox_for(project_id, meta["container_name"], project_manager.host_workspace_path(project_id))

    # Sin ToolPolicy genérica: cada función de esta registry ya valida sus propios
    # argumentos (rutas restringidas al workspace, ejecución solo dentro del sandbox).
    registry = Registry(policy=None)

    def list_files(path: str = ""):
        return {"path": path, "entries": fs.list_files(root, path)}

    def read_file(path: str):
        return {"path": path, "content": fs.read_file(root, path)}

    def create_file(path: str, content: str = ""):
        fs.write_file(root, path, content)
        return {"ok": True, "path": path}

    def update_file(path: str, content: str):
        fs.write_file(root, path, content)
        return {"ok": True, "path": path}

    def apply_patch(path: str, patch: str):
        fs.apply_patch(root, path, patch)
        return {"ok": True, "path": path}

    def delete_file(path: str):
        fs.delete_file(root, path)
        return {"ok": True, "path": path}

    def move_file(from_path: str, to_path: str):
        fs.move_file(root, from_path, to_path)
        return {"ok": True, "from": from_path, "to": to_path}

    def search_project(query: str):
        return {"query": query, "results": fs.search_project(root, query)}

    def run_command(command: str):
        return sandbox.exec_run(command)

    def install_package(name: str, manager: str = "npm"):
        cmd = f"{manager} install {name}" if manager != "pip" else f"pip install {name}"
        return sandbox.exec_run(cmd)

    def build(command: str = "npm run build"):
        return sandbox.exec_run(command)

    def run_tests(command: str = "npm test"):
        return sandbox.exec_run(command)

    def get_diagnostics():
        # v1: usa el propio compilador/linter del proyecto vía runCommand; esto es un
        # atajo genérico que intenta tsc si existe, y si no, informa que no hay diagnóstico.
        result = sandbox.exec_run("npx --no-install tsc --noEmit 2>&1 || true")
        return {"diagnostics_raw": result.get("stdout", "") + result.get("stderr", "")}

    def start_preview(command: str = "npm run dev -- --host 0.0.0.0 --port 3000"):
        return sandbox.start_preview(command)

    def restart_preview(command: str = "npm run dev -- --host 0.0.0.0 --port 3000"):
        sandbox.exec_run("pkill -f 'npm run dev' 2>/dev/null || true")
        return sandbox.start_preview(command)

    def get_runtime_logs(lines: int = 200):
        return {"logs": sandbox.get_runtime_logs(lines)}

    def git_status():
        return git_tools.status(root)

    def git_diff():
        return git_tools.diff(root)

    def git_commit(message: str):
        return git_tools.commit(root, message)

    registry.add(Tool("listFiles", "List files and folders inside the project workspace.",
        {"type": "object", "properties": {"path": {"type": "string"}}, "required": [], "additionalProperties": False},
        list_files))
    registry.add(Tool("readFile", "Read the full text content of a file in the project workspace.",
        {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"], "additionalProperties": False},
        read_file))
    registry.add(Tool("createFile", "Create a new file with the given content in the project workspace.",
        {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path"], "additionalProperties": False},
        create_file))
    registry.add(Tool("updateFile", "Overwrite a file's full content in the project workspace.",
        {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"], "additionalProperties": False},
        update_file))
    registry.add(Tool("applyPatch",
        "Apply a small targeted change to a file using a SEARCH/REPLACE block: "
        "'<<<<<<< SEARCH\\nold text\\n=======\\nnew text\\n>>>>>>> REPLACE'. "
        "Prefer this over updateFile when only part of a file changes.",
        {"type": "object", "properties": {"path": {"type": "string"}, "patch": {"type": "string"}}, "required": ["path", "patch"], "additionalProperties": False},
        apply_patch))
    registry.add(Tool("deleteFile", "Delete a file or folder from the project workspace.",
        {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"], "additionalProperties": False},
        delete_file))
    registry.add(Tool("moveFile", "Move or rename a file within the project workspace.",
        {"type": "object", "properties": {"from_path": {"type": "string"}, "to_path": {"type": "string"}}, "required": ["from_path", "to_path"], "additionalProperties": False},
        move_file))
    registry.add(Tool("searchProject", "Search for a text string across all files in the project workspace.",
        {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"], "additionalProperties": False},
        search_project))
    registry.add(Tool("runCommand", "Run a shell command inside the project's isolated sandbox container (never on the host).",
        {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"], "additionalProperties": False},
        run_command))
    registry.add(Tool("installPackage", "Install a package (npm by default, or pip) inside the project's sandbox container.",
        {"type": "object", "properties": {"name": {"type": "string"}, "manager": {"type": "string", "enum": ["npm", "pip"]}}, "required": ["name"], "additionalProperties": False},
        install_package))
    registry.add(Tool("build", "Run the project's build command inside the sandbox.",
        {"type": "object", "properties": {"command": {"type": "string"}}, "required": [], "additionalProperties": False},
        build))
    registry.add(Tool("runTests", "Run the project's test command inside the sandbox.",
        {"type": "object", "properties": {"command": {"type": "string"}}, "required": [], "additionalProperties": False},
        run_tests))
    registry.add(Tool("getDiagnostics", "Get compiler/type diagnostics for the project (best-effort, tries tsc).",
        {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        get_diagnostics))
    registry.add(Tool("startPreview", "Start the project's dev server inside the sandbox and expose a live preview URL.",
        {"type": "object", "properties": {"command": {"type": "string"}}, "required": [], "additionalProperties": False},
        start_preview))
    registry.add(Tool("restartPreview", "Restart the project's dev server inside the sandbox.",
        {"type": "object", "properties": {"command": {"type": "string"}}, "required": [], "additionalProperties": False},
        restart_preview))
    registry.add(Tool("getRuntimeLogs", "Get the recent stdout/stderr logs of the running dev server.",
        {"type": "object", "properties": {"lines": {"type": "integer"}}, "required": [], "additionalProperties": False},
        get_runtime_logs))
    registry.add(Tool("gitStatus", "Get the git status of the project workspace.",
        {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        git_status))
    registry.add(Tool("gitDiff", "Get the unstaged git diff of the project workspace.",
        {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        git_diff))
    registry.add(Tool("gitCommit", "Stage all changes and commit them with the given message.",
        {"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"], "additionalProperties": False},
        git_commit))

    return registry


CREATOR_TOOL_NAMES = {
    "listFiles", "readFile", "createFile", "updateFile", "applyPatch", "deleteFile", "moveFile",
    "searchProject", "runCommand", "installPackage", "build", "runTests", "getDiagnostics",
    "startPreview", "restartPreview", "getRuntimeLogs", "gitStatus", "gitDiff", "gitCommit",
}
