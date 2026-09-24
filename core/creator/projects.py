# Copyright (c) 2026 Luis Manuel Cousido Hermida. All rights reserved.
"""Project Manager for Creador Studio.

Every project is isolated on disk under CREATOR_PROJECTS_DIR/{project_id}/:
  workspace/   -- the actual project files (bind-mounted into the sandbox container)
  .creator/    -- project.json, agent-state.json, permissions.json, snapshots/
"""
import json
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from core.config import settings

_SLUG_PATTERN = re.compile(r"[^a-z0-9-]+")
_PROJECT_ID_PATTERN = re.compile(r"[0-9a-f]{12}")


def _slugify(name: str) -> str:
    slug = _SLUG_PATTERN.sub("-", name.lower()).strip("-")
    return slug or "proyecto"


class ProjectManager:
    def __init__(self, root: str | None = None):
        self.root = Path(root or settings.creator_projects_dir)
        self.root.mkdir(parents=True, exist_ok=True)

    def _project_dir(self, project_id: str) -> Path:
        if not _PROJECT_ID_PATTERN.fullmatch(project_id or ""):
            raise PermissionError("invalid project id")
        return self.root / project_id

    def workspace_path(self, project_id: str) -> Path:
        return self._project_dir(project_id) / "workspace"

    def host_workspace_path(self, project_id: str) -> str:
        """Absolute HOST path used to bind-mount this project's workspace into the sandbox
        container. Requires CREATOR_HOST_PROJECTS_DIR to be set (Docker-outside-of-Docker
        cannot resolve the API container's own internal path)."""
        if not settings.creator_host_projects_dir:
            raise RuntimeError(
                "CREATOR_HOST_PROJECTS_DIR is not configured; set it in .env.local to the "
                "absolute host path of the data/projects folder."
            )
        base = settings.creator_host_projects_dir.rstrip("/\\")
        return f"{base}/{project_id}/workspace"

    def create(self, name: str, api_key_id: int, template: str = "blank") -> dict:
        project_id = uuid.uuid4().hex[:12]
        project_dir = self._project_dir(project_id)
        (project_dir / "workspace").mkdir(parents=True, exist_ok=True)
        (project_dir / ".creator" / "snapshots").mkdir(parents=True, exist_ok=True)

        meta = {
            "id": project_id,
            "name": name,
            "slug": _slugify(name),
            "owner_api_key_id": api_key_id,
            "template": template,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "container_name": f"celtia-creator-{project_id}",
        }
        (project_dir / ".creator" / "project.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        (project_dir / ".creator" / "agent-state.json").write_text("{}", encoding="utf-8")
        (project_dir / ".creator" / "permissions.json").write_text(
            json.dumps({"github_token": None}, indent=2), encoding="utf-8"
        )
        return meta

    def get(self, project_id: str) -> dict | None:
        meta_path = self._project_dir(project_id) / ".creator" / "project.json"
        if not meta_path.exists():
            return None
        return json.loads(meta_path.read_text(encoding="utf-8"))

    def list(self, api_key_id: int) -> list[dict]:
        results = []
        if not self.root.exists():
            return results
        for entry in self.root.iterdir():
            meta_path = entry / ".creator" / "project.json"
            if not meta_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if meta.get("owner_api_key_id") == api_key_id:
                results.append(meta)
        return sorted(results, key=lambda m: m.get("created_at", ""), reverse=True)

    def delete(self, project_id: str, api_key_id: int) -> bool:
        meta = self.get(project_id)
        if not meta or meta.get("owner_api_key_id") != api_key_id:
            return False
        shutil.rmtree(self._project_dir(project_id), ignore_errors=True)
        return True

    def assert_owner(self, project_id: str, api_key_id: int) -> dict:
        meta = self.get(project_id)
        if not meta or meta.get("owner_api_key_id") != api_key_id:
            raise PermissionError("project not found or not owned by this account")
        return meta

    def read_permissions(self, project_id: str) -> dict:
        path = self._project_dir(project_id) / ".creator" / "permissions.json"
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def write_permissions(self, project_id: str, permissions: dict) -> None:
        path = self._project_dir(project_id) / ".creator" / "permissions.json"
        path.write_text(json.dumps(permissions, indent=2), encoding="utf-8")


project_manager = ProjectManager()
