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

## 2026-09 — Revisión de seguridade e licenza
- **Licenza propietaria** (`LICENSE`) e `SECURITY.md` cos riscos coñecidos.
- **Corrixido:** un usuario podía parar o sandbox de proxectos alleos (IDOR en `DELETE /creator/projects/{id}`).
- **Corrixido:** un cliente podía ler o historial doutra sesión elixindo o seu `session_id`.
- **Corrixido:** XSS almacenado no historial, na táboa de facturación e no selector de proxectos (escape de HTML).
- **Corrixido:** enlace de contas OAuth por email non verificado (Google/GitHub).
- **Engadido:** bloqueo por forza bruta no login (8 fallos/email, 40/IP en 15 min) e no rexistro.
- **Engadido:** contrasinal mínimo de 8 caracteres, comparación en tempo constante do token admin,
  validación do id de proxecto e cabeceiras de seguridade.
- **Restrinxido:** `install_package` (pip no host) só para administradores.
