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

## 2026-09 — Chat profesional
- **Contexto de conversación:** o cliente envía os últimos turnos en cada petición; o modelo xa non "esquece" o anterior.
- **Lectura de enlaces:** nova ferramenta `fetch_url` (protexida contra SSRF) e lectura automática dos enlaces
  que o usuario pega ou menciona ("o enlace que me deches").
- **Ferramentas encadeadas:** ata 3 rondas (buscar → ler páxina → responder).
- **Rede de seguridade:** se o modelo devolve baleiro, reintenta unha vez; se non, mensaxe clara.
- **Data actual** no prompt e detección de idioma (español/galego/inglés) para evitar respostas mesturadas.
- **Estilo profesional** no prompt (resposta primeiro, fontes con enlace, informes xa redactados).
- **Interface:** Markdown seguro (listas, táboas, código, enlaces), botón Copiar e mensaxes de erro claras.
- **Data e hora actuais sempre presentes:** o prompt leva data, hora e zona horaria (a do navegador), con datas
  relativas precalculadas e cálculo verificado para "dentro de N días/semanas/meses" e "hai/hace N ...".
- **A conversa non se perde ao recargar:** gárdase no navegador por conta (`localStorage`), tamén nos plans sen historial.

## 2026-09 — Stripe (test), imaxes con Grok e preparación de celtiaia.com
- **Plans mensuais Basic/Pro/Ultra** (5/18/48 €; 500k/2M/6M tokens) con produtos propios en Stripe (test).
- **Webhook de Stripe** (`checkout.session.completed`, `invoice.paid`, `invoice.payment_failed`,
  `customer.subscription.deleted`): activación e renovación mensual de tokens, idempotentes.
- **Corrixido:** `/billing/complete` acreditaba tokens en cada recarga da URL de éxito.
- **Corrixido:** `.env.local` tiña `STRIPE_SECRET_KEY=` e `STRIPE_WEBHOOK_SECRET=` baleiros que pisaban o `.env`.
- **Imaxes con Grok Imagine:** `POST /v1/images/generations` (premium/admin) e comando `/imagen …` no chat.
- **Seguridade:** documentación pública da API desactivada; IP real detrás do túnel (`CF-Connecting-IP`) para o
  límite de intentos; contrasinais admin rotados.
