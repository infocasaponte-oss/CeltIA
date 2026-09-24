# Copyright (c) 2026 Luis Manuel Cousido Hermida. All rights reserved.
import httpx

from core.config import settings

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"

GITHUB_AUTH_URL = "https://github.com/login/oauth/authorize"
GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"
GITHUB_USER_URL = "https://api.github.com/user"
GITHUB_EMAILS_URL = "https://api.github.com/user/emails"


def google_configured() -> bool:
    return bool(settings.google_client_id and settings.google_client_secret)


def github_configured() -> bool:
    return bool(settings.github_client_id and settings.github_client_secret)


def google_authorize_url(state: str, redirect_uri: str) -> str:
    params = {
        "client_id": settings.google_client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": "openid email profile",
        "state": state,
        "prompt": "select_account",
    }
    return f"{GOOGLE_AUTH_URL}?{httpx.QueryParams(params)}"


def github_authorize_url(state: str, redirect_uri: str) -> str:
    params = {
        "client_id": settings.github_client_id,
        "redirect_uri": redirect_uri,
        "scope": "read:user user:email",
        "state": state,
    }
    return f"{GITHUB_AUTH_URL}?{httpx.QueryParams(params)}"


async def google_fetch_identity(code: str, redirect_uri: str) -> dict:
    async with httpx.AsyncClient(timeout=15) as client:
        token_res = await client.post(GOOGLE_TOKEN_URL, data={
            "client_id": settings.google_client_id,
            "client_secret": settings.google_client_secret,
            "code": code,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        })
        token_res.raise_for_status()
        access_token = token_res.json()["access_token"]
        info_res = await client.get(GOOGLE_USERINFO_URL, headers={"Authorization": f"Bearer {access_token}"})
        info_res.raise_for_status()
        info = info_res.json()
        return {"subject": info["sub"], "email": info.get("email"), "name": info.get("name") or info.get("email")}


async def github_fetch_identity(code: str, redirect_uri: str) -> dict:
    async with httpx.AsyncClient(timeout=15) as client:
        token_res = await client.post(
            GITHUB_TOKEN_URL,
            data={
                "client_id": settings.github_client_id,
                "client_secret": settings.github_client_secret,
                "code": code,
                "redirect_uri": redirect_uri,
            },
            headers={"Accept": "application/json"},
        )
        token_res.raise_for_status()
        access_token = token_res.json()["access_token"]
        headers = {"Authorization": f"Bearer {access_token}", "Accept": "application/json"}
        user_res = await client.get(GITHUB_USER_URL, headers=headers)
        user_res.raise_for_status()
        user = user_res.json()
        email = user.get("email")
        if not email:
            emails_res = await client.get(GITHUB_EMAILS_URL, headers=headers)
            if emails_res.status_code == 200:
                for entry in emails_res.json():
                    if entry.get("primary"):
                        email = entry.get("email")
                        break
        return {"subject": str(user["id"]), "email": email, "name": user.get("name") or user.get("login")}
