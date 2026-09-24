# Despliegue en celtiaia.com (Cloudflare Tunnel)

La API + interfaz corren en local (`start_ui.ps1`, puerto 8080) y se publican por el túnel `celtiav2`.

## Cambios en Cloudflare (dashboard)
1. **Zero Trust → Networks → Tunnels → `celtiav2` → Public Hostname → Add**: hostname `celtiaia.com`,
   servicio `HTTP` → `127.0.0.1:8080`. (Deja las rutas existentes de `app`, `bff` y `native1`.)
2. **Workers & Pages → `celtia-llms` → Custom domains**: quitar `celtiaia.com` (el proyecto no se borra;
   sigue en `celtia-llms.pages.dev`).
3. **DNS → `celtiaia.com`** (CNAME): cambiar el destino de `celtia-llms.pages.dev` a
   `acf924f6-bb77-4898-a2d3-ed332f46122c.cfargotunnel.com` (proxied). (Al añadir el hostname del paso 1 Cloudflare
   puede ofrecer hacerlo solo.)
4. Comprobar: `https://celtiaia.com/health` devuelve `{"status":"ok",...}`.

**Volver atrás:** re-añadir `celtiaia.com` como custom domain de `celtia-llms` y poner el CNAME de nuevo a
`celtia-llms.pages.dev`.

## Stripe (modo test)
- Webhook: `https://celtiaia.com/webhooks/stripe` (`we_1UJFroAgk5iaw71TkDgYwqxR`), eventos
  `checkout.session.completed`, `invoice.paid`, `invoice.payment_failed`, `customer.subscription.deleted`.
- Secreto en `.env` → `STRIPE_WEBHOOK_SECRET`. Precios mensuales `STRIPE_PRICE_BASIC/PRO/ULTRA` (5 / 18 / 48 EUR).
- Modo live: cuando exista la clave `sk_live_`, repetir la creación de productos/precios y del webhook en modo live.
