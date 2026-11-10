"""Pipeline LCEL de extracción de entidades técnicas.

Cada intento ejecuta:

    prompt | model.with_structured_output(ExtraccionTecnica, include_raw=True)
           | validar_salida            # finish_reason + errores de parseo/validación

Resiliencia, de adentro hacia afuera:

1. Autocorrección: si un intento falla la validación, el siguiente incluye un
   mensaje con la respuesta anterior y los errores concretos, para que el
   modelo sepa QUÉ corregir (en vez de repetir el mismo pedido a ciegas).
2. Escalado de tokens: si la respuesta se cortó por max_tokens, el siguiente
   intento duplica el límite (hasta LLM_MAX_TOKENS_CAP).
3. `.with_retry()`: reintenta ante JSON mal formado, inválido o truncado.
4. `.with_fallbacks()`: si el modelo principal agota sus intentos (o falla el
   proveedor), se prueba el modelo de respaldo con su propio `.with_retry()`.

`process_text()` es la función asíncrona pública: ejecuta la cadena con
`.ainvoke()` y devuelve un `ExtraccionTecnica` validado, o lanza un
`ExtraccionError` controlado.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from pydantic import ValidationError

from config import PipelineSettings, build_chat_model
from prompts import CORRECCION_TEMPLATE, build_prompt
from schemas import ExtraccionTecnica

logger = logging.getLogger("pipeline")

# finish_reason (OpenAI) / stop_reason (Anthropic) que indican respuesta cortada.
_TRUNCATION_REASONS = {"length", "max_tokens"}
_DEFAULT_MAX_TOKENS = 1024
ESTADO_KEY = "estado_extraccion"


# --------------------------------------------------------------------------- errores

class RespuestaIncompletaError(Exception):
    """El LLM no devolvió el objeto estructurado (sin tool call)."""


class RespuestaTruncadaError(RespuestaIncompletaError):
    """El LLM cortó la respuesta por límite de tokens (finish_reason=length/max_tokens)."""


class ExtraccionError(Exception):
    """Error controlado que devuelve process_text() tras agotar los reintentos."""

    def __init__(self, message: str, *, tipo: str, intentos: int, causa: BaseException | None):
        super().__init__(message)
        self.tipo = tipo
        self.intentos = intentos
        self.causa = causa


# Errores ante los que tiene sentido reintentar con el mismo modelo.
# (Los errores de red / 429 / 5xx ya los reintenta el SDK con `max_retries`.)
RETRYABLE_OUTPUT_ERRORS: tuple[type[BaseException], ...] = (
    ValidationError,           # el JSON no cumple el schema Pydantic
    OutputParserException,     # JSON mal formado
    RespuestaIncompletaError,  # cortado por tokens o sin tool call
)


# --------------------------------------------------------------------------- validación

def _finish_reason(raw: Any) -> str | None:
    meta = getattr(raw, "response_metadata", None) or {}
    return meta.get("finish_reason") or meta.get("stop_reason")


def validar_salida(result: dict[str, Any]) -> ExtraccionTecnica:
    """Valida la salida de `with_structured_output(..., include_raw=True)`.

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
            "Respuesta truncada (finish_reason=%s, output_tokens=%s). Se descarta el objeto.",
            reason, usage.get("output_tokens"),
        )
        raise RespuestaTruncadaError(f"Respuesta truncada por el modelo (finish_reason={reason})")

    if error is not None:
        detalle = error.errors() if isinstance(error, ValidationError) else error
        logger.warning("Validación fallida (%s): %s", type(error).__name__, detalle)
        raise error

    if parsed is None:
        logger.warning("El modelo no devolvió el objeto estructurado (sin tool call).")
        raise RespuestaIncompletaError("El modelo no llamó a la herramienta de salida")

    if not isinstance(parsed, ExtraccionTecnica):  # defensa: nunca debería pasar
        raise RespuestaIncompletaError(f"Tipo inesperado: {type(parsed).__name__}")

    logger.info(
        "Validación OK: %d tecnologías, criticidad=%s",
        len(parsed.tecnologias), parsed.nivel_de_criticidad,
    )
    return parsed


# --------------------------------------------------------------------------- autocorrección

@dataclass
class EstadoIntentos:
    """Estado compartido entre los intentos de UNA llamada a process_text()."""

    correcciones: list[BaseMessage] = field(default_factory=list)
    max_tokens: int | None = None


def _estado(config: RunnableConfig | None, nombre: str) -> EstadoIntentos:
    registro = ((config or {}).get("configurable") or {}).get(ESTADO_KEY)
    if registro is None:
        # Sin registro (ej. alguien llamó a chain.ainvoke directo): funciona
        # igual, pero sin memoria entre intentos.
        return EstadoIntentos()
    estado: EstadoIntentos = registro.setdefault(nombre, EstadoIntentos())
    return estado


def _describir_salida(raw: Any) -> str:
    tool_calls = getattr(raw, "tool_calls", None) or []
    if tool_calls:
        return json.dumps(tool_calls[0].get("args", {}), ensure_ascii=False)
    contenido = getattr(raw, "content", "") or ""
    return str(contenido)[:1000] or "(respuesta vacía)"


def _describir_error(exc: BaseException) -> str:
    if isinstance(exc, ValidationError):
        lineas = []
        for err in exc.errors():
            campo = ".".join(str(p) for p in err.get("loc", ())) or "(objeto)"
            lineas.append(f"- {campo}: {err.get('msg')}")
        return "\n".join(lineas)
    if isinstance(exc, RespuestaIncompletaError):
        return "- No llamaste a la herramienta 'ExtraccionTecnica'."
    return f"- {exc}"


def _max_tokens_de(model: BaseChatModel) -> int | None:
    if "max_tokens" not in type(model).model_fields:
        return None
    valor = getattr(model, "max_tokens", None)
    return int(valor) if valor else _DEFAULT_MAX_TOKENS


# --------------------------------------------------------------------------- observabilidad

class LogIntentos(BaseCallbackHandler):
    """Cuenta y registra cada llamada al LLM (1 por intento) y el modelo usado."""

    run_inline = True  # mismo hilo/loop: el contador es exacto

    def __init__(self) -> None:
        self.intentos = 0
        self.modelos: list[str] = []

    def on_chat_model_start(
        self, serialized: dict[str, Any], messages: list[list[BaseMessage]], *,
        run_id: UUID, **kwargs: Any,
    ) -> None:
        self.intentos += 1
        params = kwargs.get("invocation_params") or {}
        modelo = str(params.get("model") or params.get("model_name") or params.get("_type", "?"))
        if self.modelos and modelo != self.modelos[-1]:
            logger.warning("Cambio al modelo de respaldo: %s", modelo)
        self.modelos.append(modelo)
        if self.intentos == 1:
            logger.info("Intento 1 (%s): llamando al modelo...", modelo)
        else:
            logger.warning("Reintento -> intento %d (%s): llamando al modelo...",
                           self.intentos, modelo)


# --------------------------------------------------------------------------- cadena

def _structured_kwargs(model: BaseChatModel) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"include_raw": True}
    try:
        from langchain_openai import ChatOpenAI

        if isinstance(model, ChatOpenAI):
            # function_calling (no-strict): las restricciones (min_length, enum...)
            # las valida Pydantic del lado del cliente y disparan el reintento.
            kwargs["method"] = "function_calling"
    except ImportError:  # pragma: no cover
        pass
    return kwargs


def build_attempt(model: BaseChatModel, *, nombre: str, max_tokens_cap: int = 4096) -> Runnable:
    """Un intento de extracción con autocorrección y escalado de max_tokens."""
    prompt = build_prompt()
    kwargs = _structured_kwargs(model)
    base_max_tokens = _max_tokens_de(model)
    cadena_base = prompt | model.with_structured_output(ExtraccionTecnica, **kwargs)

    def _preparar(inputs: dict, config: RunnableConfig) -> tuple[Runnable, dict, EstadoIntentos]:
        estado = _estado(config, nombre)
        cadena = cadena_base
        if estado.max_tokens and base_max_tokens:
            modelo = model.model_copy(update={"max_tokens": estado.max_tokens})
            cadena = prompt | modelo.with_structured_output(ExtraccionTecnica, **kwargs)
        return cadena, {**inputs, "correcciones": list(estado.correcciones)}, estado

    def _procesar(result: dict, estado: EstadoIntentos) -> ExtraccionTecnica:
        try:
            return validar_salida(result)
        except RespuestaTruncadaError:
            if base_max_tokens:
                actual = estado.max_tokens or base_max_tokens
                nuevo = min(actual * 2, max_tokens_cap)
                if nuevo > actual:
                    estado.max_tokens = nuevo
                    logger.warning("Próximo intento con max_tokens=%d (antes %d).", nuevo, actual)
                else:
                    logger.warning("max_tokens ya está en el tope (%d).", max_tokens_cap)
            raise
        except RETRYABLE_OUTPUT_ERRORS as exc:
            mensaje = CORRECCION_TEMPLATE.format(
                salida_previa=_describir_salida(result.get("raw")),
                errores=_describir_error(exc),
            )
            # Se conserva sólo la última corrección: el prompt no crece sin límite.
            estado.correcciones = [HumanMessage(content=mensaje)]
            logger.info("Autocorrección preparada para el próximo intento.")
            raise

    def invocar(inputs: dict, config: RunnableConfig) -> ExtraccionTecnica:
        cadena, entrada, estado = _preparar(inputs, config)
        return _procesar(cadena.invoke(entrada, config), estado)

    async def ainvocar(inputs: dict, config: RunnableConfig) -> ExtraccionTecnica:
        cadena, entrada, estado = _preparar(inputs, config)
        return _procesar(await cadena.ainvoke(entrada, config), estado)

    return RunnableLambda(invocar, afunc=ainvocar, name=f"intento_{nombre}")


def build_chain(
    model: BaseChatModel,
    *,
    max_attempts: int = 3,
    backoff: bool = True,
    fallback_model: BaseChatModel | None = None,
    max_tokens_cap: int = 4096,
) -> Runnable:
    """Ensambla la cadena: intento -> .with_retry() -> .with_fallbacks() (opcional).

    `backoff=False` desactiva la espera entre reintentos (útil en tests).
    """

    def con_reintentos(m: BaseChatModel, nombre: str) -> Runnable:
        return build_attempt(m, nombre=nombre, max_tokens_cap=max_tokens_cap).with_retry(
            retry_if_exception_type=RETRYABLE_OUTPUT_ERRORS,
            stop_after_attempt=max_attempts,
            wait_exponential_jitter=backoff,
        )

    chain = con_reintentos(model, "principal")
    if fallback_model is not None:
        chain = chain.with_fallbacks([con_reintentos(fallback_model, "respaldo")])
    return chain.with_config(run_name="extraccion_tecnica")


def build_chain_from_settings(settings: PipelineSettings, *, backoff: bool = True) -> Runnable:
    primary = build_chat_model(settings.llm)
    fallback = build_chat_model(settings.fallback) if settings.fallback else None
    logger.info(
        "Proveedor=%s modelo=%s | respaldo=%s",
        settings.llm.provider, settings.llm.model.model,
        f"{settings.fallback.provider}/{settings.fallback.model.model}"
        if settings.fallback else "ninguno",
    )
    return build_chain(
        primary,
        max_attempts=settings.output_retries,
        backoff=backoff,
        fallback_model=fallback,
        max_tokens_cap=settings.max_tokens_cap,
    )


_default_chain: Runnable | None = None


def get_default_chain() -> Runnable:
    """Cadena perezosa construida desde el .env (se crea una sola vez)."""
    global _default_chain
    if _default_chain is None:
        _default_chain = build_chain_from_settings(PipelineSettings.from_env())
    return _default_chain


# --------------------------------------------------------------------------- API pública

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
    config: RunnableConfig = {
        "callbacks": [contador],
        "configurable": {ESTADO_KEY: {}},  # estado nuevo por texto
    }

    try:
        resultado = await chain.ainvoke({"texto": text}, config=config)
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


async def process_many(
    texts: Sequence[str], chain: Runnable | None = None, *, max_concurrency: int = 5
) -> list[ExtraccionTecnica | ExtraccionError]:
    """Procesa varios textos en paralelo, con un máximo de llamadas simultáneas.

    Devuelve, en el mismo orden, el objeto validado o el ExtraccionError de
    cada texto: un texto que falla no interrumpe a los demás.
    """
    chain = chain or get_default_chain()
    semaforo = asyncio.Semaphore(max_concurrency)

    async def uno(t: str) -> ExtraccionTecnica | ExtraccionError:
        async with semaforo:
            try:
                return await process_text(t, chain=chain)
            except ExtraccionError as exc:
                return exc

    return list(await asyncio.gather(*(uno(t) for t in texts)))
