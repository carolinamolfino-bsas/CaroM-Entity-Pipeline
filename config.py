"""Configuración del modelo (lógica reutilizada del Módulo 1).

- El proveedor se elige con la variable LLM_PROVIDER (openai | anthropic).
- Los parámetros se validan con Pydantic (temperatura 0-2, max_tokens > 0, ...).
- La API key se guarda como SecretStr para que no aparezca en logs.
"""

from __future__ import annotations

import inspect
import logging
import os
from enum import StrEnum

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, ConfigDict, Field, SecretStr

logger = logging.getLogger("pipeline")


class Provider(StrEnum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


DEFAULT_MODELS = {Provider.OPENAI: "gpt-4o-mini", Provider.ANTHROPIC: "claude-haiku-4-5"}
_KEY_ENV = {Provider.OPENAI: "OPENAI_API_KEY", Provider.ANTHROPIC: "ANTHROPIC_API_KEY"}
_MODEL_ENV = {Provider.OPENAI: "OPENAI_MODEL", Provider.ANTHROPIC: "ANTHROPIC_MODEL"}


class MissingAPIKeyError(ValueError):
    """La API key del proveedor seleccionado no está configurada."""


class LLMSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Provider = Provider.OPENAI
    model: str = Field(min_length=1)
    api_key: SecretStr | None = None
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, gt=0, le=32_000)
    timeout: float = Field(default=60.0, gt=0)
    network_retries: int = Field(default=2, ge=0, le=10)  # reintentos del SDK (red/429/5xx)
    output_retries: int = Field(default=3, ge=1, le=10)   # intentos de .with_retry()

    @classmethod
    def from_env(cls, provider: str | None = None) -> "LLMSettings":
        prov = Provider((provider or os.getenv("LLM_PROVIDER", "openai")).lower())
        key = os.getenv(_KEY_ENV[prov])
        data: dict = {
            "provider": prov,
            "model": os.getenv(_MODEL_ENV[prov]) or DEFAULT_MODELS[prov],
            "api_key": SecretStr(key) if key else None,
        }
        for env, field in (
            ("LLM_TEMPERATURE", "temperature"),
            ("LLM_MAX_TOKENS", "max_tokens"),
            ("LLM_TIMEOUT", "timeout"),
            ("LLM_NETWORK_RETRIES", "network_retries"),
            ("LLM_OUTPUT_RETRIES", "output_retries"),
        ):
            if (value := os.getenv(env)) not in (None, ""):
                data[field] = value  # Pydantic convierte y valida
        return cls(**data)


def _anthropic_accepts_temperature() -> bool:
    """El SDK anthropic >= 2.0 (API 2026-10-21) rechaza temperature/top_p."""
    from anthropic.resources.messages import AsyncMessages

    return "temperature" in inspect.signature(AsyncMessages.create).parameters


def build_chat_model(settings: LLMSettings) -> BaseChatModel:
    """Crea ChatOpenAI o ChatAnthropic según la configuración."""
    if settings.api_key is None:
        raise MissingAPIKeyError(
            f"Falta {_KEY_ENV[settings.provider]} para '{settings.provider}'. "
            "Configurala en el archivo .env"
        )

    if settings.provider is Provider.OPENAI:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.model,
            api_key=settings.api_key,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            timeout=settings.timeout,
            max_retries=settings.network_retries,
        )

    from langchain_anthropic import ChatAnthropic

    kwargs: dict = {}
    if _anthropic_accepts_temperature():
        kwargs["temperature"] = min(settings.temperature, 1.0)  # Anthropic: rango 0-1
    else:
        logger.info("SDK anthropic >= 2.0: no se envía temperature (la API la rechaza).")
    return ChatAnthropic(
        model=settings.model,
        api_key=settings.api_key,
        max_tokens=settings.max_tokens,
        timeout=settings.timeout,
        max_retries=settings.network_retries,
        **kwargs,
    )
