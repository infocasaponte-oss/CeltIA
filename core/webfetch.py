# Copyright (c) 2026 Luis Manuel Cousido Hermida. All rights reserved.
"""fetch_url tool: read a public web page as plain text, with SSRF protection.

Only http/https on standard ports, every hop (including redirects) must resolve to a public IP,
bounded size/time, and only textual content types are returned.
"""
import asyncio
import ipaddress
import re
import socket
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

import httpx

MAX_BYTES = 600_000
MAX_CHARS = 6000
MAX_REDIRECTS = 3
ALLOWED_PORTS = {80, 443, 8080, 8443}
TEXT_TYPES = ("text/html", "application/xhtml+xml", "text/plain", "application/json", "text/xml", "application/xml")
USER_AGENT = "CeltIA/4 (+read-only page fetch)"


class BlockedURL(ValueError):
    pass


async def _assert_public(url: str) -> None:
    parts = urlparse(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise BlockedURL("Solo se permiten URLs http/https")
    if parts.username or parts.password:
        raise BlockedURL("No se permiten credenciales en la URL")
    port = parts.port or (443 if parts.scheme == "https" else 80)
    if port not in ALLOWED_PORTS:
        raise BlockedURL("Puerto no permitido")
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(parts.hostname, port, type=socket.SOCK_STREAM)
    except socket.gaierror:
        raise BlockedURL("No se pudo resolver el dominio")
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        ip = getattr(ip, "ipv4_mapped", None) or ip
        if not ip.is_global:
            raise BlockedURL("La URL apunta a una dirección privada o interna")


_NOISE = re.compile(r"cookie|consent|gdpr|banner|modal|popup|newsletter|breadcrumb|footer|menu|navbar|sidebar")


class _TextExtractor(HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "template", "head", "iframe"}
    BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article", "table", "ul", "ol"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title = ""
        self._skip = 0
        self._in_title = False
        self._noise_tag = None  # tag name of a cookie/consent/nav container being skipped
        self._noise_depth = 0

    def handle_starttag(self, tag, attrs):
        if self._noise_tag:
            if tag == self._noise_tag:
                self._noise_depth += 1
            return
        marker = " ".join(v or "" for k, v in attrs if k in ("id", "class", "role", "aria-label")).lower()
        if tag not in ("html", "body", "main") and _NOISE.search(marker):
            self._noise_tag, self._noise_depth = tag, 1
            return
        if tag == "title":
            self._in_title = True
        elif tag in self.SKIP:
            self._skip += 1
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if self._noise_tag:
            if tag == self._noise_tag:
                self._noise_depth -= 1
                if self._noise_depth == 0:
                    self._noise_tag = None
            return
        if tag == "title":
            self._in_title = False
        elif tag in self.SKIP:
            self._skip = max(0, self._skip - 1)
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._noise_tag:
            return
        if self._in_title:
            self.title += data
        elif not self._skip:
            self.parts.append(data)


def _clean(text: str) -> str:
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


async def fetch_url(url: str) -> dict:
    current = url.strip()
    try:
        async with httpx.AsyncClient(timeout=10, follow_redirects=False, headers={"User-Agent": USER_AGENT}) as client:
            for _ in range(MAX_REDIRECTS + 1):
                await _assert_public(current)
                async with client.stream("GET", current) as resp:
                    if resp.is_redirect and resp.headers.get("location"):
                        current = urljoin(current, resp.headers["location"])
                        continue
                    if resp.status_code >= 400:
                        return {"ok": False, "url": current, "error": f"La página respondió con HTTP {resp.status_code}"}
                    ctype = resp.headers.get("content-type", "").split(";")[0].strip().lower()
                    if ctype and not ctype.startswith(TEXT_TYPES):
                        return {"ok": False, "url": current, "error": f"Tipo de contenido no compatible: {ctype}"}
                    body = bytearray()
                    async for chunk in resp.aiter_bytes():
                        body.extend(chunk)
                        if len(body) > MAX_BYTES:
                            break
                    html = bytes(body).decode(resp.encoding or "utf-8", errors="replace")
                    if ctype in ("text/html", "application/xhtml+xml", ""):
                        parser = _TextExtractor()
                        parser.feed(html)
                        text, title = _clean("".join(parser.parts)), _clean(parser.title)
                    else:
                        text, title = _clean(html), ""
                    if not text:
                        return {"ok": False, "url": current, "error": "La página no contiene texto legible (¿requiere JavaScript?)"}
                    return {"ok": True, "url": current, "title": title, "text": text[:MAX_CHARS],
                            "truncated": len(text) > MAX_CHARS}
            return {"ok": False, "url": current, "error": "Demasiadas redirecciones"}
    except BlockedURL as exc:
        return {"ok": False, "url": current, "error": str(exc)}
    except Exception as exc:
        return {"ok": False, "url": current, "error": f"No se pudo leer la página: {type(exc).__name__}"}
