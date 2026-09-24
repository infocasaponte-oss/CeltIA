# Rexistro de cambios — CeltIA V4

Autor: Luis Manuel Cousido Hermida

## 2026-09 — Versión inicial no repositorio
- **Arranque:** `start_ui.ps1` crea a `.venv`, instala dependencias, lanza API (8080) e web (4173).
- **Modelo local:** Ollama (`celtia-qwen3`, contexto 8192) en vez de vLLM/Docker.
- **Gateway** (`core/gateway`): rate limit por minuto, cola de inferencia, límites de tokens por minuto e mes,
  límites propios por chave.
- **API keys para clientes externos** (`sk_live_…`): crear, revogar, límites e consumo por chave; só se garda o hash.
- **Streaming en vivo:** pasos do axente e tokens reais, con filtro de identidade.
- **Adxuntos:** PDF, DOCX, XLSX, texto e código nas barras de chat (`core/files.py`).
- **Panel de consumo** (peticións, tokens, latencia, cota mensual) e **rexistro de usuarios** para administradores
  (rol, estado, contrasinal).
- **Seguridade:** escape de nomes no panel admin, protección do último administrador, 403/401 diferenciados.
