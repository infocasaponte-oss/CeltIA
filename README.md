# CeltIA

Local-first AI assistant and agent built around an OpenBMB MiniCPM model with a premium, Spanish-first local experience.

## Included
- Fast / Think / Code / Agent / Long routing
- OpenAI-compatible FastAPI endpoint
- vLLM backend integration
- planner + ReAct-style tool loop
- calculator, Python subprocess sandbox, restricted file reads
- SQLite session memory and document memory
- result verification and tool validation
- MCP server scaffold
- tests and smoke checks
- training and LoRA notes

## Quick start

Requirements: Python 3.11+, Docker Compose, NVIDIA GPU, NVIDIA Container Toolkit.

For a real local model run on Windows with an RTX 3060 Ti, use a CUDA-capable NVIDIA GPU, the latest GeForce Game Ready driver, WSL2 + Docker Desktop with GPU support, and the NVIDIA Container Toolkit enabled inside WSL.

```powershell
cd D:\mini-council
copy .env.example .env
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
```

### GPU setup for RTX 3060 Ti (Windows + WSL2)

1. Install the latest NVIDIA Game Ready Driver for GeForce RTX 3060 Ti from the official NVIDIA driver page.
2. Enable WSL2 and install Ubuntu from Microsoft Store if needed:
   ```powershell
   wsl --install
   wsl --status
   ```
3. In Docker Desktop, enable:
   - Use the WSL 2 based engine
   - Enable NVIDIA Container Toolkit support (if offered by your Docker Desktop build)
4. Inside the WSL distro, install the NVIDIA Container Toolkit:
   ```bash
   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   ```
5. Verify GPU access from WSL:
   ```bash
   nvidia-smi
   docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
   ```
6. Start the full stack with the GPU override:
   ```powershell
   cd D:\mini-council
   docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
   ```

API: http://localhost:18080
Model server: http://localhost:18000

```bash
curl http://localhost:18080/health
curl http://localhost:18080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"CeltIA V4","messages":[{"role":"user","content":"Haz una suma simple para probar la herramienta."}]}'
```

## Runtime notes

For an 8 GB GPU keep `MODEL_CONTEXT=32768` initially. Move to 128K only after measuring KV-cache memory.

The runtime works without LoRA. Add adapters only after the baseline has been benchmarked.

## Tool security

The built-in Python executor is a restricted subprocess suitable for local trusted use, not hostile multi-tenant workloads. For public exposure replace it with a dedicated Docker/Firecracker sandbox and enforce authentication, quotas and network isolation.

## MCP

Install `requirements-mcp.txt` and run:

```bash
python mcp_server.py
```

The server uses the official MCP Python SDK. Put it behind authentication/TLS before exposing it remotely.

## Training guidance

Recommended order: Agent, Code, Reasoning, then investigate Long/Vision adapters.

Do not train on private chain-of-thought from closed models. Use observable answers, concise plans, tool calls, tool results, verification outcomes and properly licensed public/open datasets.
