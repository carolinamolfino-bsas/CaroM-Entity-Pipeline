# Pipeline de Extracción de Entidades Técnicas

Pipeline en **LangChain (LCEL)** que recibe un texto sin procesar (un log de error, un incidente o una descripción de arquitectura) y devuelve un **objeto validado con Pydantic**:

```
ChatPromptTemplate  ->  model.with_structured_output(ExtraccionTecnica)  ->  validar_salida
                                         .with_retry()  (JSON mal formado / incompleto / truncado)
```

Funciona con **OpenAI** (`ChatOpenAI`) y **Anthropic** (`ChatAnthropic`). La configuración reutiliza la lógica del Módulo 1: proveedor elegido por `.env`, parámetros validados con Pydantic y API key protegida.

## Ejemplo de salida

Entrada:

```
ERROR [api-gateway] FastAPI worker 7: psycopg.OperationalError: connection to server
at 'db-prod' (PostgreSQL 16) failed: remaining connection slots are reserved.
Redis cache hit ratio cayó a 12%. /checkout devuelve 503 al 40% de los usuarios.
```

Salida (`ExtraccionTecnica` validado, serializado a JSON):

```json
{
  "tecnologias": ["FastAPI", "Redis", "PostgreSQL"],
  "nivel_de_criticidad": "alta",
  "resumen_tecnico": "API con caché en Redis y persistencia en PostgreSQL; cuello de botella en conexiones concurrentes."
}
```

> El texto exacto del resumen varía según el modelo.

## Estructura

```
technical-entity-pipeline/
├── schemas.py        # Contrato Pydantic: ExtraccionTecnica + enum NivelCriticidad
├── prompts.py        # ChatPromptTemplate modular (texto + instrucciones de formato)
├── config.py         # Settings validados + creación de ChatOpenAI / ChatAnthropic
├── chain.py          # Cadena LCEL, validación de finish_reason, .with_retry(), process_text()
├── main.py           # Mini-script de prueba asíncrono (3 textos en paralelo)
├── tests/test_pipeline.py   # 19 tests offline con un modelo simulado
├── .env.example
└── requirements.txt
```

## Instalación

```bash
git clone <URL-del-repo>
cd technical-entity-pipeline

python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env               # Windows: copy .env.example .env
# editar .env y completar la API key del proveedor que vas a usar
```

## Variables de entorno

| Variable | Default | Descripción |
|---|---|---|
| `LLM_PROVIDER` | `openai` | `openai` o `anthropic` |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | — | Obligatoria la del proveedor elegido |
| `OPENAI_MODEL` | `gpt-4o-mini` | Modelo de OpenAI |
| `ANTHROPIC_MODEL` | `claude-haiku-4-5` | Modelo de Anthropic |
| `LLM_TEMPERATURE` | `0` | 0 a 2. Para extracción conviene que sea bajo |
| `LLM_MAX_TOKENS` | `1024` | Límite de salida. Si es muy bajo, la respuesta se trunca y se reintenta |
| `LLM_TIMEOUT` | `60` | Segundos por request |
| `LLM_NETWORK_RETRIES` | `2` | Reintentos del SDK ante red, 429 y 5xx |
| `LLM_OUTPUT_RETRIES` | `3` | Intentos de `.with_retry()` ante salida inválida |
| `LOG_LEVEL` | `INFO` | `DEBUG` muestra también `finish_reason` y tokens |

## Ejecutar el script de prueba

```bash
python main.py                         # proveedor de LLM_PROVIDER
python main.py --provider anthropic
python main.py --texto "Kafka consumer lag de 2M mensajes en el cluster de pagos"
```

Procesa en paralelo (`asyncio.gather`) tres textos: un log de error, una descripción de arquitectura y un **texto ambiguo** de prueba de estrés. Imprime el JSON de cada uno, o un error controlado si después de los reintentos no se obtuvo un objeto válido. En los logs se ve el proceso:

```
INFO    pipeline: Procesando texto (412 caracteres)
INFO    pipeline: Intento 1: llamando al modelo...
WARNING pipeline: Validación fallida (ValidationError): [{'type': 'too_short', 'loc': ('tecnologias',), ...}]
WARNING pipeline: Reintento -> intento 2: llamando al modelo...
INFO    pipeline: Validación OK: 3 tecnologías, criticidad=alta
INFO    pipeline: OK en 2 intento(s), 1840 ms
```

## Ejecutar los tests

```bash
python -m pytest -q
```

No requieren API keys. Usan un chat model simulado para cubrir el camino feliz, la recuperación tras un JSON inválido, la respuesta truncada (`finish_reason=length` en OpenAI y `stop_reason=max_tokens` en Anthropic), la respuesta sin tool call, el agotamiento de reintentos y la concurrencia.

## Uso desde código

```python
import asyncio
from dotenv import load_dotenv
from chain import process_text, ExtraccionError

load_dotenv()

async def demo():
    try:
        r = await process_text("Nginx devuelve 502; el upstream Node.js se reinicia por OOM")
        print(r.tecnologias, r.nivel_de_criticidad, r.resumen_tecnico)
    except ExtraccionError as e:
        print("No se pudo extraer:", e.tipo, e)

asyncio.run(demo())
```

## Decisiones de diseño

**Contrato (`schemas.py`)**
- `tecnologias` no puede estar vacía (`min_length=1`). Se limpian espacios, se rechazan elementos vacíos y se eliminan duplicados sin distinguir mayúsculas (`Redis` y `redis` cuentan como uno).
- `nivel_de_criticidad` es un enum (`baja`/`media`/`alta`). Acepta `"Alta"` o `" ALTA "`, pero rechaza valores fuera del enum, como `"crítica"`.
- `resumen_tecnico` tiene entre 20 y 400 caracteres, y `extra="forbid"` rechaza campos inventados.
- Las `description` de cada campo viajan al LLM como parte del schema y le explican qué poner.

**Prompt (`prompts.py`)**: `ChatPromptTemplate` sin f-strings. Las instrucciones de formato son una variable (`{instrucciones_formato}`) fijada con `.partial()`, así se pueden cambiar sin tocar el resto. La única entrada libre es `{texto}`.

**Cadena (`chain.py`)**
- `with_structured_output(..., include_raw=True)` conserva el `AIMessage` crudo además del objeto parseado. Así se puede **revisar el `finish_reason` antes de usar el objeto**: si el modelo cortó por tokens, se descarta aunque parezca válido.
- `validar_salida` convierte cualquier problema (truncado, JSON mal formado, falla de Pydantic, falta de tool call) en una excepción. `.with_retry()` reintenta la cadena completa solo ante esos errores, con backoff exponencial y jitter.
- Una API key inválida **no** se reintenta (reintentar no sirve). Los errores de red y 429 los reintenta el SDK (`LLM_NETWORK_RETRIES`).
- En OpenAI se usa `method="function_calling"`: las restricciones las valida Pydantic del lado del cliente, que es lo que dispara el reintento.
- `process_text()` siempre devuelve un `ExtraccionTecnica` válido o lanza un `ExtraccionError` controlado (`tipo`: `salida_invalida`, `proveedor` o `entrada_invalida`, junto con la cantidad de `intentos`). Nunca deja escapar una excepción cruda del SDK.
- Un callback (`LogIntentos`) registra cada llamada al modelo, así en los logs se ve cuándo hubo un reintento.

**Nota sobre Anthropic**: desde la versión de API `2026-10-21` (SDK `anthropic` >= 2.0), la API rechaza `temperature`. `config.py` lo detecta y no la envía en ese caso. En SDK 1.x sí la envía, limitada a 1.0.
