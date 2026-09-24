# Copyright (c) 2026 Luis Manuel Cousido Hermida. All rights reserved.
"""Docker-outside-of-Docker sandbox: one container per project.

The agent (and the user) never execute commands directly on the host. Every
`runCommand`/`installPackage`/dev-server invocation happens inside a dedicated,
resource-limited container for that project, with the project's workspace bind
-mounted in and no access to the host's environment variables or secrets.

Convention: any dev server the agent starts inside the sandbox must listen on
0.0.0.0:3000 (container-internal). That single fixed port is published to a
free host port chosen when the container is first created, which keeps the
Docker-outside-of-Docker networking simple (no reverse proxy needed: the
browser opens http://localhost:<published-port> directly for the live preview).
"""
import logging
import socket
import threading

import docker
from docker.errors import DockerException, NotFound

from core.config import settings

logger = logging.getLogger(__name__)

PREVIEW_CONTAINER_PORT = 3000

_client_lock = threading.Lock()
_client = None


def _docker_client():
    global _client
    with _client_lock:
        if _client is None:
            _client = docker.from_env()
        return _client


def sandbox_configured() -> bool:
    try:
        _docker_client().ping()
        return bool(settings.creator_host_projects_dir)
    except DockerException:
        return False


def _free_host_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class ProjectSandbox:
    def __init__(self, project_id: str, container_name: str, host_workspace_path: str):
        self.project_id = project_id
        self.container_name = container_name
        self.host_workspace_path = host_workspace_path

    def _container(self):
        try:
            return _docker_client().containers.get(self.container_name)
        except NotFound:
            return None

    def ensure_running(self):
        container = self._container()
        if container is not None:
            container.reload()
            if container.status != "running":
                container.start()
            return container

        container = _docker_client().containers.run(
            settings.creator_sandbox_image,
            name=self.container_name,
            command="sleep infinity",
            working_dir="/workspace",
            volumes={self.host_workspace_path: {"bind": "/workspace", "mode": "rw"}},
            mem_limit=settings.creator_container_mem_limit,
            nano_cpus=int(settings.creator_container_cpu_quota) * 10_000,
            network_mode="bridge",
            ports={f"{PREVIEW_CONTAINER_PORT}/tcp": None},
            environment={},
            detach=True,
            tty=False,
            security_opt=["no-new-privileges"],
        )
        return container

    def published_preview_port(self) -> int | None:
        container = self._container()
        if container is None:
            return None
        container.reload()
        bindings = container.attrs.get("NetworkSettings", {}).get("Ports", {}) or {}
        entries = bindings.get(f"{PREVIEW_CONTAINER_PORT}/tcp")
        if not entries:
            return None
        return int(entries[0]["HostPort"])

    def exec_run(self, command: str, timeout: int | None = None) -> dict:
        container = self.ensure_running()
        try:
            exit_code, output = container.exec_run(
                ["/bin/sh", "-c", command],
                workdir="/workspace",
                demux=True,
            )
            stdout, stderr = output if isinstance(output, tuple) else (output, b"")
            return {
                "ok": exit_code == 0,
                "exit_code": exit_code,
                "stdout": (stdout or b"").decode("utf-8", errors="replace")[-8000:],
                "stderr": (stderr or b"").decode("utf-8", errors="replace")[-8000:],
            }
        except DockerException as exc:
            logger.exception("sandbox exec_run failed for project %s", self.project_id)
            return {"ok": False, "exit_code": -1, "stdout": "", "stderr": str(exc)}

    def start_preview(self, command: str) -> dict:
        self.ensure_running()
        self.exec_run("pkill -f '.creator-preview' 2>/dev/null || true")
        self.exec_run(
            f"( {command} ) > /workspace/.creator-preview.log 2>&1 & echo started > /tmp/.creator-preview-marker"
        )
        port = self.published_preview_port()
        return {"ok": True, "port": port, "url": f"http://localhost:{port}" if port else None}

    def get_runtime_logs(self, lines: int = 200) -> str:
        result = self.exec_run(f"tail -n {lines} /workspace/.creator-preview.log 2>/dev/null || true")
        return result.get("stdout", "")

    def stop_and_remove(self):
        container = self._container()
        if container is None:
            return
        try:
            container.remove(force=True)
        except DockerException:
            logger.exception("failed to remove sandbox container %s", self.container_name)


def sandbox_for(project_id: str, container_name: str, host_workspace_path: str) -> ProjectSandbox:
    return ProjectSandbox(project_id, container_name, host_workspace_path)
