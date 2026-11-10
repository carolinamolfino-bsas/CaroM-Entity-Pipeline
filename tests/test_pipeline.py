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
from pydantic import Field, ValidationError

from chain import ExtraccionError, build_chain, process_many, process_text
from config import MissingAPIKeyError, PipelineSettings, build_chat_model
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
    """Devuelve las respuestas de `responses` en orden, una por llamada.

    `registro` es un dict compartido incluso entre copias (model_copy), así se
    puede contar llamadas y ver los mensajes / max_tokens de cada intento.
    """

    responses: list[Any]  # AIMessage o Exception a lanzar
    max_tokens: int | None = 1024
    model_name: str = "fake-principal"
    registro: dict = Field(default_factory=lambda: {"calls": 0, "max_tokens": [], "mensajes": []})

    @property
    def calls(self) -> int:
        return self.registro["calls"]

    @property
    def _llm_type(self) -> str:
        return "fake-tool-model"

    @property
    def _identifying_params(self) -> dict:
        return {"model": self.model_name}

    def bind_tools(self, tools: Any, **kwargs: Any):
        return self

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kw) -> ChatResult:
        i = self.registro["calls"]
        self.registro["calls"] += 1
        self.registro["max_tokens"].append(self.max_tokens)
        self.registro["mensajes"].append(messages)
        msg = self.responses[min(i, len(self.responses) - 1)]
        if isinstance(msg, Exception):
            raise msg
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


# ---------------------------------------------------------------- v2: autocorrección, tokens, fallback

def test_autocorreccion_incluye_errores_en_el_reintento():
    malo = {**VALIDO, "tecnologias": [], "nivel_de_criticidad": "crítica"}
    model = FakeToolModel(responses=[tool_msg(malo), tool_msg(VALIDO)])
    run(process_text("texto", chain=build_chain(model, backoff=False)))
    primer_intento, segundo_intento = model.registro["mensajes"]
    assert len(segundo_intento) == len(primer_intento) + 1
    correccion = segundo_intento[-1].content
    assert "tecnologias" in correccion and "nivel_de_criticidad" in correccion
    assert '"crítica"' in correccion  # incluye la respuesta anterior


def test_sin_tool_call_genera_correccion_especifica():
    texto_plano = AIMessage(content="No puedo", response_metadata={"finish_reason": "stop"})
    model = FakeToolModel(responses=[texto_plano, tool_msg(VALIDO)])
    run(process_text("texto", chain=build_chain(model, backoff=False)))
    assert "No llamaste a la herramienta" in model.registro["mensajes"][1][-1].content


def test_estado_no_se_comparte_entre_textos():
    malo = {**VALIDO, "tecnologias": []}
    model = FakeToolModel(responses=[tool_msg(malo), tool_msg(VALIDO), tool_msg(VALIDO)])
    chain = build_chain(model, backoff=False)
    run(process_text("uno", chain=chain))
    run(process_text("dos", chain=chain))
    # el tercer llamado (otro texto) no arrastra la corrección del primero
    assert len(model.registro["mensajes"][2]) == len(model.registro["mensajes"][0])


def test_truncado_escala_max_tokens():
    model = FakeToolModel(
        max_tokens=500,
        responses=[tool_msg(VALIDO, finish="length"), tool_msg(VALIDO, finish="length"),
                   tool_msg(VALIDO)],
    )
    run(process_text("texto", chain=build_chain(model, backoff=False, max_tokens_cap=1500)))
    assert model.registro["max_tokens"] == [500, 1000, 1500]


def test_escalado_respeta_tope():
    model = FakeToolModel(max_tokens=1000, responses=[tool_msg(VALIDO, finish="length")])
    with pytest.raises(ExtraccionError):
        run(process_text("t", chain=build_chain(model, backoff=False, max_tokens_cap=1200)))
    assert model.registro["max_tokens"] == [1000, 1200, 1200]


def test_fallback_a_otro_modelo_si_el_principal_falla(caplog):
    principal = FakeToolModel(responses=[RuntimeError("401 invalid api key")])
    respaldo = FakeToolModel(model_name="fake-respaldo", responses=[tool_msg(VALIDO)])
    chain = build_chain(principal, backoff=False, fallback_model=respaldo)
    with caplog.at_level(logging.INFO, logger="pipeline"):
        res = run(process_text("texto", chain=chain))
    assert res.tecnologias == VALIDO["tecnologias"]
    assert principal.calls == 1 and respaldo.calls == 1  # el 401 no se reintenta
    assert "Cambio al modelo de respaldo: fake-respaldo" in caplog.text


def test_fallback_tras_agotar_reintentos_de_salida():
    principal = FakeToolModel(responses=[tool_msg({"tecnologias": []})])
    respaldo = FakeToolModel(model_name="fake-respaldo", responses=[tool_msg(VALIDO)])
    chain = build_chain(principal, max_attempts=2, backoff=False, fallback_model=respaldo)
    assert run(process_text("texto", chain=chain)).tecnologias
    assert principal.calls == 2 and respaldo.calls == 1


def test_process_many_no_corta_por_un_error():
    model = FakeToolModel(responses=[tool_msg(VALIDO)])
    chain = build_chain(model, backoff=False)
    resultados = run(process_many(["a", "   ", "c"], chain=chain, max_concurrency=2))
    assert isinstance(resultados[0], ExtraccionTecnica)
    assert isinstance(resultados[1], ExtraccionError)
    assert isinstance(resultados[2], ExtraccionTecnica)


# ---------------------------------------------------------------- configuración

def test_settings_validan_rango(monkeypatch):
    monkeypatch.setenv("LLM_TEMPERATURE", "3")
    with pytest.raises(ValidationError):
        PipelineSettings.from_env("openai")


def test_falta_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(MissingAPIKeyError):
        build_chat_model(PipelineSettings.from_env("openai").llm)


def test_fallback_auto_solo_si_hay_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("LLM_FALLBACK_PROVIDER", raising=False)
    assert PipelineSettings.from_env("openai").fallback is None
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    fb = PipelineSettings.from_env("openai").fallback
    assert fb is not None and fb.provider == "anthropic"
    monkeypatch.setenv("LLM_FALLBACK_PROVIDER", "none")
    assert PipelineSettings.from_env("openai").fallback is None


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_cadena_real_se_construye(monkeypatch, provider):
    from chain import build_chain_from_settings

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    assert build_chain_from_settings(PipelineSettings.from_env(provider)) is not None


# ---------------------------------------------------------------- evaluación

def test_dataset_es_valido():
    from evaluar import cargar_dataset

    casos = cargar_dataset()
    assert len(casos) >= 10
    assert len({c.id for c in casos}) == len(casos)
    assert {c.criticidad for c in casos} >= {NivelCriticidad.BAJA, NivelCriticidad.MEDIA,
                                             NivelCriticidad.ALTA, None}


def test_metricas_de_evaluacion():
    from evaluar import CasoEval, evaluar_caso, metricas

    caso = CasoEval(id="x", texto="t", criticidad="alta",
                    tecnologias=["PostgreSQL", "Redis", "Node.js"])
    salida = ExtraccionTecnica(tecnologias=["postgresql", "NodeJS", "Docker"],
                               nivel_de_criticidad="media",
                               resumen_tecnico="Resumen técnico de prueba suficientemente largo.")
    r = evaluar_caso(caso, salida)
    assert r.encontradas == ["PostgreSQL", "Node.js"] and r.faltantes == ["Redis"]
    assert r.extra == ["Docker"] and r.criticidad_correcta is False

    error = evaluar_caso(CasoEval(id="y", texto="t", criticidad="baja"), ExtraccionError(
        "fallo", tipo="proveedor", intentos=1, causa=None))
    m = metricas([r, error])
    assert m["tasa_exito"] == 0.5 and m["exactitud_criticidad"] == 0.0
    assert m["recall_tecnologias"] == pytest.approx(2 / 3)
