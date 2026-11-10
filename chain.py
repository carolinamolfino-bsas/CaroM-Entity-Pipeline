"""Pipeline LCEL de extracción de entidades técnicas.

    prompt | model.with_structured_output(ExtraccionTecnica, include_raw=True)
           | validar_salida            # finish_reason + errores de parseo/validación
    .with_retry(...)                   # reintenta ante JSON mal formado o incompleto

`process_text()` es la función asíncrona pública: ejecuta la cadena con
`.ainvoke()` y devuelve un `ExtraccionTecnica` validado, o lanza un
`ExtraccionError` controlado si se agotan los reintentos.
"""

from __future__ import annotations

import logging
import time
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import ValidationError

from config import LLMSettings, build_chat_model
from prompts import build_prompt
from schemas import ExtraccionTecnica

logger = logging.getLogger("pipeline")

# finish_reason / stop_reason que indican que la respuesta quedó cortada.
_TRUNCATION_REASONS = {"length", "max_tokens"}


# --------------------------------------------------------------------------- errores

class RespuestaIncompletaError(Exception):
    """El LLM cortó la respuesta (finish_reason=length/max_tokens) o no devolvió el objeto."""


class ExtraccionError(Exception):
    """Error controlado que devuelve process_text() tras agotar los reintentos."""

    def __init__(self, message: str, *, tipo: str, intentos: int, causa: BaseException | None):
        super().__init__(message)
        self.tipo = tipo
        self.intentos = intentos
        self.causa = causa


# Errores ante los que tiene sentido reintentar la cadena completa.
# (Los errores de red / 429 / 5xx ya los reintenta el SDK con `max_retries`.)
RETRYABLE_OUTPUT_ERRORS: tuple[type[BaseException], ...] = (
    ValidationError,           # el JSON no cumple el schema Pydantic
    OutputParserException,     # JSON mal formado
    RespuestaIncompletaError,  # cortado por tokens o sin tool call
)


# --------------------------------------------------------------------------- validación

def _finish_reason(raw: AIMessage | Any) -> str | None:
    meta = getattr(raw, "response_metadata", None) or {}
    # OpenAI -> finish_reason ; Anthropic -> stop_reason
    return meta.get("finish_reason") or meta.get("stop_reason")


def validar_salida(result: dict[str, Any]) -> ExtraccionTecnica:
    """Paso de validación de la cadena.

    Recibe la salida de `with_structured_output(..., include_raw=True)`:
    {"raw": AIMessage, "parsed": ExtraccionTecnica | None, "parsing_error": Exception | None}

    1. Revisa finish_reason ANTES de usar el objeto: si el modelo cortó por
       tokens, el JSON está incompleto aunque parezca parseable.
    2. Propaga errores de parseo/validación para que .with_retry() reintente.
    """
    raw, parsed, error = result.get("raw"), result.get("parsed"), result.get("parsing_error")
    reason = _finish_reason(raw)
    usage = getattr(raw, "usage_metadata", None) or {}
    logger.debug("finish_reason=%s usage=%s", reason, usage)

    if reason in _TRUNCATION_REASONS:
        logger.warning(
            "Respuesta truncada (finish_reason=%s, output_tokens=%s). "
            "Se descarta el objeto; considerá subir LLM_MAX_TOKENS.",
            reason, usage.get("output_tokens"),
        )
        raise RespuestaIncompletaError(f"Respuesta truncada por el modelo (finish_reason={reason})")

    if error is not None:
        detalle = error.errors() if isinstance(error, ValidationError) else error
        logger.warning("Validación fallida (%s): %s", type(error).__name__, detalle)
        raise error

    if parsed is None:
        logger.warning("El modelo no devolvió el objeto estructurado (sin tool call).")
        raise RespuestaIncompletaError("El modelo no llamó a la herramienta de salida")

    logger.info(
        "Validación OK: %d tecnologías, criticidad=%s",
        len(parsed.tecnologias), parsed.nivel_de_criticidad,
    )
    return parsed


# --------------------------------------------------------------------------- observabilidad

class LogIntentos(BaseCallbackHandler):
    """Cuenta y registra cada llamada al LLM (1 por intento de .with_retry())."""

    run_inline = True  # ejecutar en el mismo hilo/loop: el contador es exacto

    def __init__(self) -> None:
        self.intentos = 0

    def on_chat_model_start(self, serialized: dict, messages: list, *, run_id: UUID, **kw: Any) -> None:
        self.intentos += 1
        if self.intentos == 1:
            logger.info("Intento 1: llamando al modelo...")
        else:
            logger.warning("Reintento -> intento %d: llamando al modelo...", self.intentos)


# --------------------------------------------------------------------------- cadena

def build_chain(
    model: BaseChatModel, *, max_attempts: int = 3, backoff: bool = True
) -> Runnable:
    """Ensambla la cadena LCEL con reintentos ante salida inválida.

    `backoff=False` desactiva la espera entre reintentos (útil en tests).
    """
    structured_kwargs: dict = {"include_raw": True}
    try:
        from langchain_openai import ChatOpenAI

        if isinstance(model, ChatOpenAI):
            # function_calling (no-strict): las restricciones (min_length, enum...)
            # las valida Pydantic del lado del cliente y disparan el reintento.
            structured_kwargs["method"] = "function_calling"
    except ImportError:  # pragma: no cover
        pass

    prompt = build_prompt()
    structured_model = model.with_structured_output(ExtraccionTecnica, **structured_kwargs)

    chain = (
        prompt
        | structured_model
        | RunnableLambda(validar_salida, name="validar_salida")
    )
    return chain.with_retry(
        retry_if_exception_type=RETRYABLE_OUTPUT_ERRORS,
        stop_after_attempt=max_attempts,
        wait_exponential_jitter=backoff,
    ).with_config(run_name="extraccion_tecnica")


_default_chain: Runnable | None = None


def get_default_chain() -> Runnable:
    """Cadena perezosa construida desde el .env (se crea una sola vez)."""
    global _default_chain
    if _default_chain is None:
        settings = LLMSettings.from_env()
        logger.info("Proveedor=%s modelo=%s", settings.provider, settings.model)
        _default_chain = build_chain(
            build_chat_model(settings), max_attempts=settings.output_retries
        )
    return _default_chain


async def process_text(text: str, chain: Runnable | None = None) -> ExtraccionTecnica:
    """Extrae entidades técnicas de `text` y devuelve un objeto validado.

    Lanza ExtraccionError (nunca una excepción cruda del SDK/Pydantic) si no
    se pudo obtener un objeto válido.
    """
    if not text or not text.strip():
        raise ExtraccionError("El texto de entrada está vacío", tipo="entrada_invalida",
                              intentos=0, causa=None)

    chain = chain or get_default_chain()
    contador = LogIntentos()
    inicio = time.perf_counter()
    logger.info("Procesando texto (%d caracteres)", len(text))

    try:
        resultado = await chain.ainvoke({"texto": text}, config={"callbacks": [contador]})
    except RETRYABLE_OUTPUT_ERRORS as exc:
        logger.error("Reintentos agotados (%d intentos): %s", contador.intentos, exc)
        raise ExtraccionError(
            "El modelo no devolvió un objeto válido tras los reintentos",
            tipo="salida_invalida", intentos=contador.intentos, causa=exc,
        ) from exc
    except Exception as exc:  # red, auth, rate limit agotado, etc.
        logger.error("Error del proveedor: %s: %s", type(exc).__name__, exc)
        raise ExtraccionError(
            f"Error del proveedor: {type(exc).__name__}",
            tipo="proveedor", intentos=contador.intentos, causa=exc,
        ) from exc

    logger.info(
        "OK en %d intento(s), %.0f ms", contador.intentos, (time.perf_counter() - inicio) * 1000
    )
    return resultado
