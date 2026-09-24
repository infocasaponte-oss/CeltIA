# Copyright (c) 2026 Luis Manuel Cousido Hermida. All rights reserved.
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_serve_name: str = "CeltIA V4"
    vllm_base_url: str = "http://localhost:8000/v1"
    sqlite_path: str = "./mini_council.db"
    max_agent_steps: int = 8
    gateway_max_concurrency: int = 4  # local GPU requests queue inside Ollama; the cloud primary handles parallel calls
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
    public_base_url: str = "http://localhost:8081"
    # Generación de imágenes (xAI / Grok Imagine)
    image_api_url: str = "https://api.x.ai/v1"
    image_api_key: str = ""
    xai_api_key: str = ""
    image_model: str = "grok-imagine-image"
    image_token_cost: int = 1500  # tokens descontados del saldo por imagen generada

    # Modelo principal del chat (Grok / xAI, API compatible con OpenAI). El modelo local (Ollama) queda de reserva.
    llm_primary_enabled: bool = True
    llm_primary_base_url: str = "https://api.x.ai/v1"
    llm_primary_api_key: str = ""  # si está vacío se reutiliza XAI_API_KEY / IMAGE_API_KEY
    llm_primary_model: str = "grok-4.20-0309-non-reasoning"
    llm_primary_reasoning_model: str = "grok-4.20-0309-reasoning"  # solo para respuestas sin herramientas en modo "pensar"
    llm_primary_context: int = 128000
    llm_primary_timeout_seconds: int = 60
    llm_primary_cooldown_seconds: int = 60  # tras un fallo de disponibilidad se usa el modelo local durante este tiempo

    data_retention_days: int = 365
    google_client_id: str = ""
    google_client_secret: str = ""
    github_client_id: str = ""
    github_client_secret: str = ""
    stripe_metered_price_id: str = ""
    free_plan_token_grant: int = 20000
    pro_plan_token_grant: int = 2000000

    decision_abstain_below: float = 0.55
    decision_temperature: float = 1.0
    decision_shadow_routing: bool = False
    decision_reject_ood: bool = True
    decision_ood_entropy_threshold: float = 0.90
    decision_ood_margin_threshold: float = 0.10
    decision_max_questions: int = 32
    decision_max_output_tokens: int = 1024
    decision_max_total_output_tokens: int = 8192
    decision_max_total_prompt_chars: int = 250000
    decision_call_timeout_seconds: float = 30.0
    decision_request_timeout_seconds: float = 120.0

    # Creador Studio: proyectos aislados + sandbox Docker por proyecto
    creator_projects_dir: str = "/data/projects"
    creator_host_projects_dir: str = ""
    creator_sandbox_image: str = "node:20-slim"
    creator_container_mem_limit: str = "1g"
    creator_container_cpu_quota: int = 100000
    creator_command_timeout_seconds: int = 120
    creator_max_auto_repair_attempts: int = 2
    @property
    def primary_llm_key(self) -> str:
        return self.llm_primary_api_key or self.xai_api_key or self.image_api_key

    model_config = SettingsConfigDict(env_file=(".env", ".env.local"), extra="ignore")

settings = Settings()
