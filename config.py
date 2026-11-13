"""Configuración del pipeline, reutilizando el cliente del Módulo 1.

`llm_client` (repo CaroM-LLM-Client, instalado como dependencia) aporta:
  - ManagerSettings.from_env(): proveedor, API key (SecretStr) y ModelConfig
    validados con Pydantic (temperatura 0-2, max_tokens > 0, timeout, reintentos).
  - MissingAPIKeyError y la detección de parámetros que Anthropic ya no acepta.

Acá sólo se agrega lo propio del pipeline: reintentos de salida, modelo de
respaldo y el tope de max_tokens para el escalado ante respuestas truncadas.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from langchain_core.language_models.base import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from llm_client import (
    API_KEY_ENV,
    ManagerSettings,
    MissingAPIKeyError,
    Provider,
    anthropic_supports_sampling,
)
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger("pipeline")

__all__ = [
    "MissingAPIKeyError",
    "PipelineSettings",
    "Provider",
    "build_chat_model",
]


class PipelineSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    llm: ManagerSettings
    output_retries: int = Field(default=3, ge=1, le=10)
    max_tokens_cap: int = Field(default=4096, gt=0, le=32_000)
    fallback: ManagerSettings | None = None

    @classmethod
    def from_env(cls, provider: str | None = None) -> PipelineSettings:
        llm = ManagerSettings.from_env(provider)
        data: dict = {"llm": llm}
        for env, field in (
            ("LLM_OUTPUT_RETRIES", "output_retries"),
            ("LLM_MAX_TOKENS_CAP", "max_tokens_cap"),
        ):
            if (value := os.getenv(env)) not in (None, ""):
                data[field] = value
        data["fallback"] = _fallback_from_env(llm.provider)
        return cls(**data)


def _fallback_from_env(primary: Provider) -> ManagerSettings | None:
    """LLM_FALLBACK_PROVIDER: auto (default) | none | openai | anthropic.

    'auto' usa el otro proveedor sólo si su API key está configurada.
    """
    choice = (os.getenv("LLM_FALLBACK_PROVIDER") or "auto").strip().lower()
    if choice in ("none", "no", "off", "false"):
        return None
    if choice == "auto":
        other = Provider.ANTHROPIC if primary is Provider.OPENAI else Provider.OPENAI
        if not os.getenv(API_KEY_ENV[other]):
            return None
        return ManagerSettings.from_env(other)
    fallback = ManagerSettings.from_env(choice)
    if fallback.provider is primary:
        logger.warning("LLM_FALLBACK_PROVIDER igual al principal (%s): se ignora.", primary)
        return None
    return fallback


class ChatOpenAICompatible(ChatOpenAI):
    """ChatOpenAI para servidores compatibles (Ollama, LM Studio, vLLM...).

    LangChain renombra max_tokens a max_completion_tokens (exclusivo de la API
    oficial de OpenAI). Los servidores compatibles documentan max_tokens, así
    que acá se vuelve al nombre clásico.
    """

    def _get_request_payload(
        self, input_: LanguageModelInput, *, stop: list[str] | None = None, **kwargs: Any
    ) -> dict:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        if "max_completion_tokens" in payload:
            payload["max_tokens"] = payload.pop("max_completion_tokens")
        return payload


def build_chat_model(settings: ManagerSettings) -> BaseChatModel:
    """Crea ChatOpenAI o ChatAnthropic a partir de la configuración del Módulo 1."""
    api_key = settings.require_api_key()  # MissingAPIKeyError con mensaje claro
    cfg = settings.model

    if settings.provider is Provider.OPENAI:
        # OPENAI_BASE_URL (Módulo 1) -> servidor compatible, ej. Ollama local.
        clase = ChatOpenAICompatible if settings.base_url else ChatOpenAI
        if settings.base_url:
            logger.info("Usando servidor compatible con OpenAI: %s", settings.base_url)
        return clase(
            model=cfg.model,
            base_url=settings.base_url,
            api_key=api_key,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,  # type: ignore[call-arg]
            timeout=cfg.timeout,
            max_retries=cfg.max_retries,  # reintentos del SDK ante red / 429 / 5xx
        )

    from langchain_anthropic import ChatAnthropic

    kwargs: dict = {}
    if anthropic_supports_sampling():
        kwargs["temperature"] = min(cfg.temperature, 1.0)  # Anthropic: rango 0-1
    elif "temperature" in cfg.model_fields_set:
        logger.warning(
            "AVISO: LLM_TEMPERATURE ignorada para Anthropic "
            "(la API 2026-10-21 ya no acepta temperature)."
        )
    return ChatAnthropic(  # type: ignore[call-arg]
        model=cfg.model,
        api_key=api_key,
        max_tokens=cfg.max_tokens,
        timeout=cfg.timeout,
        max_retries=cfg.max_retries,
        **kwargs,
    )
