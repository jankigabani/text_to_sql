"""Central configuration. Every tunable the experiments touch lives here."""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    # ── Postgres ──────────────────────────────────────────────────────────────
    pghost: str = "localhost"
    pgport: int = 5432
    pgdatabase: str = "adventureworks"
    pguser: str = "postgres"
    pgpassword: str = ""

    pg_readonly_user: str = "t2s_readonly"
    pg_readonly_password: str = ""

    # ── LLM ───────────────────────────────────────────────────────────────────
    llm_provider: Literal["ollama", "gemini", "openai", "anthropic", "mock"] = "ollama"
    llm_model: str = "qwen2.5-coder:7b"
    llm_temperature: float = 0.0
    llm_timeout_s: int = 120
    ollama_base_url: str = "http://127.0.0.1:11434"
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    # Gemini (Google AI Studio). Free-tier models work; set the key in .env only.
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash-lite"
    # Minimum seconds between Gemini calls. 0 for the UI; the benchmark sets ~4.5 so the
    # free tier's per-minute limit is never hit.
    gemini_min_interval_s: float = 0.0
    # Retries on 429/5xx. 3 keeps the UI responsive; the benchmark raises it.
    gemini_retries: int = 3

    # Ollama's default context window is too small for a whole-schema prompt.
    ollama_num_ctx: int = 16384

    # Model used when a request does not name one, as a provider:model spec,
    # e.g. "ollama:qwen2.5-coder:7b" or "gemini:gemini-3.5-flash-lite". Empty = LLM_PROVIDER:LLM_MODEL.
    default_model: str = ""

    # ── v2 engine (UI / /api) ──────────────────────────────────────────────────
    # The schema-shrink prompt is skipped in favour of a keyword pre-filter above this size.
    prune_max_prompt_tokens: int = 12000
    prune_prefilter_tables: int = 30
    prune_max_tables: int = 8
    connection_cache_size: int = 6

    # ── Embeddings ────────────────────────────────────────────────────────────
    embed_provider: Literal["ollama", "hashing"] = "ollama"
    embed_model: str = "nomic-embed-text"

    # ── Retrieval ─────────────────────────────────────────────────────────────
    retrieve_top_tables: int = 8
    retrieve_top_columns: int = 40
    retrieve_sample_values: int = 4
    index_dir: str = "data/index"

    # ── Verification gate ─────────────────────────────────────────────────────
    gate_max_repairs: int = 2
    gate_plan_cost_ceiling: float = 5_000_000
    gate_row_limit: int = 200
    gate_statement_timeout_ms: int = 15_000
    gate_confidence_threshold: float = 0.55
    gate_disabled_layers: str = ""

    # ── Service ───────────────────────────────────────────────────────────────
    log_level: str = "INFO"
    run_log_enabled: bool = True
    run_log_schema: str = "t2s"

    include_schemas: str = "public,sales,person,production,purchasing,humanresources"

    @field_validator("gate_disabled_layers", "include_schemas", mode="before")
    @classmethod
    def _strip(cls, v: object) -> str:
        return str(v or "").strip()

    # ── Derived ───────────────────────────────────────────────────────────────
    @property
    def dsn(self) -> str:
        return (
            f"host={self.pghost} port={self.pgport} dbname={self.pgdatabase} "
            f"user={self.pguser} password={self.pgpassword}"
        )

    @property
    def readonly_dsn(self) -> str:
        """Stage 5 connects with this. Falls back to the owner DSN if no role exists yet."""
        if not self.pg_readonly_password:
            return self.dsn
        return (
            f"host={self.pghost} port={self.pgport} dbname={self.pgdatabase} "
            f"user={self.pg_readonly_user} password={self.pg_readonly_password}"
        )

    @property
    def disabled_layers(self) -> set[str]:
        return {p.strip().lower() for p in self.gate_disabled_layers.split(",") if p.strip()}

    @property
    def schema_allowlist(self) -> list[str]:
        return [p.strip().lower() for p in self.include_schemas.split(",") if p.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()


def reload_settings() -> Settings:
    """Used by the ablation runner, which mutates the environment between arms."""
    get_settings.cache_clear()
    return get_settings()
