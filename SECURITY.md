# Seguridad

Informar de vulnerabilidades de forma privada a infocasaponte@gmail.com (no abrir issues públicos).

## Riesgos conocidos y pendientes
- **`python_exec` no es un sandbox real.** Ejecuta código en el host con el mismo usuario que la API; la
  lista de bloqueo por AST se puede evadir. Solo está disponible para administradores, pero una inyección de
  prompt (p. ej. desde resultados de `web_search`) podría inducir al agente a ejecutar código. No exponer
  cuentas admin a entradas no fiables hasta aislarlo (Docker/contenedor sin red) o desactivarlo.
- **`install_package` (pip en el host) es solo para administradores**, por el mismo motivo.
- Los límites y bloqueos de intentos son en memoria: se reinician con el servidor y no se comparten entre
  varias instancias.
- La API escucha en `0.0.0.0`. No exponerla a Internet sin HTTPS (p. ej. túnel de Cloudflare) ni sin revisar CORS.
