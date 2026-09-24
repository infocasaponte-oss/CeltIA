# Copyright (c) 2026 Luis Manuel Cousido Hermida. All rights reserved.
import asyncio
import logging

import stripe

from core.config import settings

logger = logging.getLogger(__name__)


def is_configured() -> bool:
    return bool(settings.stripe_secret_key and settings.stripe_price_id)


def _client() -> "stripe.StripeClient":
    return stripe.StripeClient(settings.stripe_secret_key)


def metered_configured() -> bool:
    return bool(settings.stripe_secret_key and settings.stripe_metered_price_id)


async def create_checkout_session(email: str, key_id: int | None = None, plan: str = "pro") -> str:
    price_id = settings.stripe_metered_price_id if plan == "payg" else settings.stripe_price_id
    if not settings.stripe_secret_key or not price_id:
        raise RuntimeError("Stripe is not configured yet")

    def _create():
        client = _client()
        params = {
            "mode": "subscription",
            "customer_email": email,
            "line_items": [{"price": price_id, "quantity": 1}] if plan != "payg" else [{"price": price_id}],
            "success_url": f"{settings.public_base_url}/billing/complete?session_id={{CHECKOUT_SESSION_ID}}",
            "cancel_url": f"{settings.public_base_url}/ui/",
        }
        if key_id is not None:
            params["client_reference_id"] = f"{key_id}:{plan}"
        session = client.checkout.sessions.create(params)
        return session.url

    return await asyncio.to_thread(_create)


async def retrieve_session(session_id: str) -> dict:
    if not is_configured():
        raise RuntimeError("Stripe is not configured yet")

    def _retrieve():
        client = _client()
        return client.checkout.sessions.retrieve(session_id, {"expand": ["customer", "subscription"]})

    return await asyncio.to_thread(_retrieve)


async def get_subscription_status(subscription_id: str) -> str | None:
    if not is_configured() or not subscription_id:
        return None

    def _retrieve():
        client = _client()
        return client.subscriptions.retrieve(subscription_id)

    try:
        sub = await asyncio.to_thread(_retrieve)
        return sub.status
    except Exception:
        logger.exception("Failed to fetch Stripe subscription status")
        return None


async def report_usage(customer_id: str, quantity: int) -> None:
    if not is_configured() or not customer_id or quantity <= 0:
        return

    def _report():
        client = _client()
        client.billing.meter_events.create({
            "event_name": settings.stripe_meter_event_name,
            "payload": {"stripe_customer_id": customer_id, "value": str(quantity)},
        })

    try:
        await asyncio.to_thread(_report)
    except Exception:
        logger.exception("Failed to report usage to Stripe")


def verify_webhook(payload: bytes, sig_header: str) -> dict:
    if not settings.stripe_webhook_secret:
        raise RuntimeError("Stripe webhook secret is not configured")
    return stripe.Webhook.construct_event(payload, sig_header, settings.stripe_webhook_secret)
