"""Tests offline: un chat model falso simula respuestas buenas, malas y truncadas.

Ejecutar:  python -m pytest -q
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ValidationError

from chain import ExtraccionError, build_chain, process_text
from config import LLMSettings, MissingAPIKeyError, build_chat_model
from prompts import build_prompt
from schemas import ExtraccionTecnica, NivelCriticidad

VALIDO = {
    "tecnologias": ["FastAPI", "Redis", "PostgreSQL"],
    "nivel_de_criticidad": "alta",
    "resumen_tecnico": "API con caché en Redis y persistencia en PostgreSQL; "
                       "cuello de botella en conexiones concurrentes.",
}


def tool_msg(args: dict, finish: str = "tool_calls") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": "ExtraccionTecnica", "args": args, "id": "call_1"}],
        response_metadata={"finish_reason": finish},
    )


class FakeToolModel(BaseChatModel):
    """Devuelve las respuestas de `responses` en orden, una por llamada."""

    responses: list[AIMessage]
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "fake-tool-model"

    def bind_tools(self, tools: Any, **kwargs: Any):
        return self

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kw) -> ChatResult:
        msg = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return ChatResult(generations=[ChatGeneration(message=msg)])


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------- schema

def test_schema_valido_y_normalizacion():
    obj = ExtraccionTecnica(**{**VALIDO, "nivel_de_criticidad": " ALTA ",
                               "tecnologias": ["Redis", "redis", " FastAPI "]})
    assert obj.nivel_de_criticidad is NivelCriticidad.ALTA
    assert obj.tecnologias == ["Redis", "FastAPI"]


@pytest.mark.parametrize("cambio", [
    {"tecnologias": []},                      # lista vacía
    {"tecnologias": ["Redis", "  "]},         # elemento vacío
    {"nivel_de_criticidad": "crítica"},       # fuera del enum
    {"resumen_tecnico": "corto"},             # demasiado corto
    {"campo_extra": 1},                       # campo no permitido
])
def test_schema_rechaza(cambio):
    with pytest.raises(ValidationError):
        ExtraccionTecnica(**{**VALIDO, **cambio})


def test_prompt_template_variables():
    prompt = build_prompt("FORMATO-X")
    assert prompt.input_variables == ["texto"]
    msgs = prompt.format_messages(texto="log de prueba")
    assert "FORMATO-X" in msgs[0].content and "log de prueba" in msgs[1].content


# ---------------------------------------------------------------- cadena

def test_camino_feliz():
    model = FakeToolModel(responses=[tool_msg(VALIDO)])
    res = run(process_text("texto", chain=build_chain(model, backoff=False)))
    assert isinstance(res, ExtraccionTecnica) and model.calls == 1
    assert res.model_dump(mode="json") == VALIDO


def test_reintenta_ante_json_invalido_y_se_recupera(caplog):
    malo = {**VALIDO, "tecnologias": []}  # viola min_length
    model = FakeToolModel(responses=[tool_msg(malo), tool_msg(VALIDO)])
    with caplog.at_level(logging.INFO, logger="pipeline"):
        res = run(process_text("texto", chain=build_chain(model, backoff=False)))
    assert res.tecnologias and model.calls == 2
    assert "Validación fallida" in caplog.text and "intento 2" in caplog.text


def test_reintenta_ante_respuesta_truncada():
    model = FakeToolModel(responses=[tool_msg(VALIDO, finish="length"), tool_msg(VALIDO)])
    run(process_text("texto", chain=build_chain(model, backoff=False)))
    assert model.calls == 2  # el objeto truncado se descartó aunque fuera parseable


def test_truncado_estilo_anthropic():
    truncado = AIMessage(content="", tool_calls=[{"name": "ExtraccionTecnica",
                         "args": VALIDO, "id": "x"}], response_metadata={"stop_reason": "max_tokens"})
    model = FakeToolModel(responses=[truncado])
    with pytest.raises(ExtraccionError) as e:
        run(process_text("texto", chain=build_chain(model, max_attempts=2, backoff=False)))
    assert e.value.tipo == "salida_invalida" and e.value.intentos == 2


def test_sin_tool_call_se_reintenta():
    texto_plano = AIMessage(content="No puedo", response_metadata={"finish_reason": "stop"})
    model = FakeToolModel(responses=[texto_plano, tool_msg(VALIDO)])
    run(process_text("texto", chain=build_chain(model, backoff=False)))
    assert model.calls == 2


def test_agota_reintentos_error_controlado():
    model = FakeToolModel(responses=[tool_msg({"tecnologias": "no es lista"})])
    with pytest.raises(ExtraccionError) as e:
        run(process_text("texto", chain=build_chain(model, max_attempts=3, backoff=False)))
    assert e.value.intentos == 3 and model.calls == 3


def test_texto_vacio():
    with pytest.raises(ExtraccionError):
        run(process_text("   ", chain=build_chain(FakeToolModel(responses=[tool_msg(VALIDO)]))))


def test_procesamiento_concurrente():
    model = FakeToolModel(responses=[tool_msg(VALIDO)])
    chain = build_chain(model, backoff=False)

    async def varios():
        return await asyncio.gather(*(process_text(f"t{i}", chain=chain) for i in range(5)))

    assert len(run(varios())) == 5


# ---------------------------------------------------------------- configuración

def test_settings_validan_rango(monkeypatch):
    monkeypatch.setenv("LLM_TEMPERATURE", "3")
    with pytest.raises(ValidationError):
        LLMSettings.from_env("openai")


def test_falta_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(MissingAPIKeyError):
        build_chat_model(LLMSettings.from_env("openai"))


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_cadena_real_se_construye(monkeypatch, provider):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    model = build_chat_model(LLMSettings.from_env(provider))
    assert build_chain(model) is not None
