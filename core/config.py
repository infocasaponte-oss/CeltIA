# Copyright (c) 2026 Luis Manuel Cousido Hermida. All rights reserved.
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_serve_name: str = "CeltIA V4"
    vllm_base_url: str = "http://localhost:8000/v1"
    sqlite_path: str = "./mini_council.db"
    max_agent_steps: int = 8
    gateway_max_concurrency: int = 1
    gateway_queue_wait_seconds: float = 30.0
    max_tool_calls: int = 12
    python_tool_timeout_seconds: int = 8
    router_long_context_chars: int = 12000
    model_context: int = 4096
    brave_search_api_key: str = ""
    admin_token: str = ""
    stripe_secret_key: str = ""
    stripe_webhook_secret: str = ""
    stripe_price_id: str = ""  # legacy alias of the PRO price
    stripe_price_basic: str = ""
    stripe_price_pro: str = ""
    stripe_price_ultra: str = ""
    basic_plan_token_grant: int = 500_000
    ultra_plan_token_grant: int = 6_000_000
    stripe_meter_event_name: str = "celtia_tokens"
    public_base_url: str = "http://localhost:8080"
    # Generación de imágenes (xAI / Grok Imagine)
    image_api_url: str = "https://api.x.ai/v1"
    image_api_key: str = ""
    xai_api_key: str = ""
    image_model: str = "grok-imagine-image"
    image_token_cost: int = 1500  # tokens descontados del saldo por imagen generada

    data_retention_days: int = 365
    google_client_id: str = ""
    google_client_secret: str = ""
    github_client_id: str = ""
    github_client_secret: str = ""
    stripe_metered_price_id: str = ""
    free_plan_token_grant: int = 20000
    pro_plan_token_grant: int = 2000000

    # Creador Studio: proyectos aislados + sandbox Docker por proyecto
    creator_projects_dir: str = "/data/projects"
    creator_host_projects_dir: str = ""
    creator_sandbox_image: str = "node:20-slim"
    creator_container_mem_limit: str = "1g"
    creator_container_cpu_quota: int = 100000
    creator_command_timeout_seconds: int = 120
    creator_max_auto_repair_attempts: int = 2
    model_config = SettingsConfigDict(env_file=(".env", ".env.local"), extra="ignore")

settings = Settings()
