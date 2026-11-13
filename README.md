# CaroM - Pre-entrega 2: Pipeline de Extracción de Entidades Técnicas
[![Python Tests](https://github.com/carolinamolfino-bsas/CaroM-Entity-Pipeline/actions/workflows/tests.yml/badge.svg)](https://github.com/carolinamolfino-bsas/CaroM-Entity-Pipeline/actions/workflows/tests.yml)

Pipeline en **LangChain (LCEL)** que recibe un texto sin procesar (un log de error, un incidente o una descripción de arquitectura) y devuelve un **objeto validado con Pydantic**.

```
                 ┌──────────────── .with_retry() (hasta LLM_OUTPUT_RETRIES) ────────────────┐
texto ──► ChatPromptTemplate ──► model.with_structured_output(ExtraccionTecnica) ──► validar_salida ──► objeto
                 ▲ correcciones                                                         │
                 └──── autocorrección + escalado de max_tokens si el intento falla ◄────┘
          .with_fallbacks() ──► misma cadena con el otro proveedor si el principal no logra un objeto válido
```

Funciona con **OpenAI** (`ChatOpenAI`) y **Anthropic** (`ChatAnthropic`). La configuración **reutiliza como librería el cliente del Módulo 1** ([CaroM-LLM-Client](https://github.com/carolinamolfino-bsas/CaroM-LLM-Client)).

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

> Ejemplo tomado de la consigna; el texto exacto del resumen varía según el modelo.

## Estructura

```
CaroM-Entity-Pipeline/
├── schemas.py        # Contrato Pydantic: ExtraccionTecnica + enum NivelCriticidad
├── prompts.py        # ChatPromptTemplate modular + template de autocorrección
├── config.py         # Settings (reutiliza llm_client del Módulo 1) + ChatOpenAI / ChatAnthropic
├── chain.py          # Cadena LCEL, autocorrección, .with_retry(), .with_fallbacks(), process_text()
├── main.py           # Mini-script de prueba asíncrono con verificación de resultados
├── evaluar.py        # Mini evaluación contra un dataset con respuestas esperadas
├── evals/dataset.jsonl          # 10 casos: alta, media, baja y un texto ambiguo
├── tests/                       # 32 tests offline + servidor HTTP falso compatible con OpenAI
├── .github/workflows/tests.yml  # CI: ruff + mypy + pytest en cada push
├── pyproject.toml    # Configuración de pytest, ruff y mypy
├── requirements.txt / requirements-dev.txt
└── .env.example
```

## Instalación

```bash
git clone https://github.com/carolinamolfino-bsas/CaroM-Entity-Pipeline.git
cd CaroM-Entity-Pipeline

python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

cp .env.example .env               # Windows: copy .env.example .env
# editar .env y completar la API key del proveedor que vas a usar
```

`requirements.txt` instala el cliente del Módulo 1 directamente desde GitHub (`carom-llm-client @ git+https://...@v1.2.0`), así que hace falta tener `git` instalado.

## Variables de entorno

| Variable | Default | Descripción |
|---|---|---|
| `LLM_PROVIDER` | `openai` | Proveedor principal: `openai` o `anthropic` |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | — | Obligatoria la del proveedor principal |
| `OPENAI_MODEL` | `gpt-4o-mini` | Modelo de OpenAI |
| `ANTHROPIC_MODEL` | `claude-haiku-4-5` | Modelo de Anthropic |
| `OPENAI_BASE_URL` | — | Servidor compatible con OpenAI (ej. Ollama: `http://localhost:11434/v1`). Vacío = OpenAI oficial |
| `LLM_TEMPERATURE` | `0.7` (el `.env.example` usa `0`) | 0 a 2. Para extracción conviene que sea bajo |
| `LLM_MAX_TOKENS` | `512` (el `.env.example` usa `1024`) | Límite inicial de salida |
| `LLM_MAX_TOKENS_CAP` | `4096` | Tope al duplicar `max_tokens` cuando una respuesta se trunca |
| `LLM_TIMEOUT` | `60` | Segundos por request |
| `LLM_MAX_RETRIES` | `3` | Reintentos del SDK ante red, 429 y 5xx |
| `LLM_OUTPUT_RETRIES` | `3` | Intentos de `.with_retry()` ante salida inválida |
| `LLM_FALLBACK_PROVIDER` | `auto` | `auto` (el otro proveedor si su key está cargada), `none`, `openai` o `anthropic` |
| `LLM_MAX_CONCURRENCY` | `5` | Textos procesados en simultáneo |
| `LOG_LEVEL` | `INFO` | `DEBUG` muestra también `finish_reason` y tokens |

Las variables compartidas con el Módulo 1 se validan con el mismo `ManagerSettings` de `llm_client`.

## Ejecutar el script de prueba

```bash
python main.py                         # proveedor de LLM_PROVIDER
python main.py --provider anthropic
python main.py --texto "Kafka consumer lag de 2M mensajes en el cluster de pagos"
```

Procesa en paralelo tres textos y **verifica** cada resultado:

| Caso | Comportamiento esperado |
|---|---|
| Log de error | criticidad `alta` |
| Descripción de arquitectura | criticidad `baja` o `media` |
| Texto ambiguo (estrés) | no se rompe: el modelo se recupera (objeto válido) o el validador lo rechaza (error controlado) |

Termina con `Resumen: N/3 casos con el comportamiento esperado` y código de salida 0 si todos cumplen. En los logs se ve el proceso:

```
INFO    pipeline: Intento 1 (gpt-4o-mini): llamando al modelo...
WARNING pipeline: Validación fallida (ValidationError): [{'type': 'too_short', 'loc': ('tecnologias',), ...}]
INFO    pipeline: Autocorrección preparada para el próximo intento.
WARNING pipeline: Reintento -> intento 2 (gpt-4o-mini): llamando al modelo...
INFO    pipeline: Validación OK: 3 tecnologías, criticidad=alta
INFO    pipeline: OK en 2 intento(s), 1840 ms
```

## Probar gratis, sin API key (Ollama)

[Ollama](https://ollama.com) corre modelos de lenguaje en tu propia computadora y expone una API compatible con la de OpenAI. Con `OPENAI_BASE_URL` el pipeline usa `ChatOpenAI` contra Ollama, sin key ni costo. **Ollama y el modelo no son parte del repo:** se instalan sólo en la máquina donde querés probar.

1. Instalá Ollama desde https://ollama.com/download (Windows, macOS o Linux).
2. Descargá un modelo que soporte *tools* (`with_structured_output` las usa). Se hace una sola vez, son unos 4,7 GB y conviene tener 8 GB de RAM libres:
   ```bash
   ollama pull qwen2.5:7b
   ```
3. En tu `.env`:
   ```
   LLM_PROVIDER=openai
   OPENAI_BASE_URL=http://localhost:11434/v1
   OPENAI_API_KEY=ollama
   OPENAI_MODEL=qwen2.5:7b
   LLM_FALLBACK_PROVIDER=none
   ```
4. Corré `python main.py` y, si querés, `python evaluar.py`.

Qué tener en cuenta:
- Ollama no admite `tool_choice`, así que el modelo podría contestar con texto en vez de llamar a la herramienta. En ese caso actúa la autocorrección ("No llamaste a la herramienta…") y se reintenta.
- Un modelo local chico es más lento y menos preciso que GPT o Claude. Sirve para ver el pipeline funcionando con respuestas reales; las métricas de `evaluar.py` van a ser más bajas.
- Con un servidor compatible se envía `max_tokens` en lugar de `max_completion_tokens`, que es exclusivo de OpenAI (clase `ChatOpenAICompatible` en `config.py`).

## Evaluación

```bash
python evaluar.py                          # imprime una tabla por caso y las métricas
python evaluar.py --umbral 0.8 --salida evals/resultados.json
```

Corre el pipeline sobre `evals/dataset.jsonl` y mide:

- **Tasa de éxito:** porcentaje de textos con un objeto válido.
- **Exactitud de criticidad:** porcentaje de casos en que acertó `baja`, `media` o `alta`.
- **Recall de tecnologías:** porcentaje de las tecnologías esperadas que encontró (sin distinguir mayúsculas ni signos, así `Node.js` y `NodeJS` cuentan como la misma).

Devuelve código 1 si la exactitud queda debajo del umbral. Requiere una API key real: cada corrida hace 10 llamadas como mínimo.

## Tests y controles de calidad

```bash
python -m pytest -q   # 32 tests, sin API keys
ruff check .          # estilo y errores comunes
mypy                  # chequeo de tipos
```

Los tests usan un chat model simulado y cubren:

- el caso que sale bien;
- la autocorrección: el reintento incluye la respuesta anterior y los errores concretos;
- el escalado de `max_tokens` ante una respuesta truncada (y que respete el tope);
- el respaldo con otro modelo, tanto ante un error del proveedor como al agotar los reintentos;
- que el estado de un texto no contamine al siguiente;
- `process_many()` con un texto que falla en el medio;
- las métricas de evaluación;
- una corrida completa por HTTP contra un servidor local que imita a Ollama (`ChatOpenAI` + LCEL + autocorrección).

## Uso desde código

```python
import asyncio
from dotenv import load_dotenv
from chain import process_text, process_many, ExtraccionError

load_dotenv()

async def demo():
    try:
        r = await process_text("Nginx devuelve 502; el upstream Node.js se reinicia por OOM")
        print(r.tecnologias, r.nivel_de_criticidad, r.resumen_tecnico)
    except ExtraccionError as e:
        print("No se pudo extraer:", e.tipo, e)

    # Muchos textos: cada uno devuelve su objeto o su ExtraccionError, en orden
    resultados = await process_many(["texto 1", "texto 2", "texto 3"], max_concurrency=3)

asyncio.run(demo())
```

## Decisiones de diseño

**Contrato (`schemas.py`)**
- `tecnologias` no puede estar vacía (`min_length=1`). Se limpian espacios, se rechazan elementos vacíos y se eliminan duplicados sin distinguir mayúsculas.
- `nivel_de_criticidad` es un enum (`baja`/`media`/`alta`). Acepta `"Alta"` o `" ALTA "`, pero rechaza valores fuera del enum, como `"crítica"`.
- `resumen_tecnico` tiene entre 20 y 400 caracteres, y `extra="forbid"` rechaza campos inventados.

**Prompt (`prompts.py`)**: `ChatPromptTemplate` sin f-strings. Las instrucciones de formato son una variable fijada con `.partial()`. Hay un `MessagesPlaceholder` opcional para las correcciones, y el mensaje de corrección también es un template (`PromptTemplate`).

**Resiliencia (`chain.py`)**, de adentro hacia afuera:
1. **Revisión del `finish_reason`.** `with_structured_output(..., include_raw=True)` conserva el mensaje crudo. Si el modelo cortó por tokens, el objeto se descarta aunque parezca válido.
2. **Autocorrección.** Si la salida no pasa la validación, el reintento agrega un mensaje con la respuesta anterior y la lista de errores (`- tecnologias: List should have at least 1 item`). El modelo sabe qué corregir, en vez de recibir el mismo pedido a ciegas. Sólo se conserva la última corrección, para que el prompt no crezca sin límite.
3. **Escalado de tokens.** Si la respuesta se truncó, el siguiente intento duplica `max_tokens`, hasta `LLM_MAX_TOKENS_CAP`.
4. **`.with_retry()`.** Reintenta sólo ante salida inválida (JSON mal formado, falla de Pydantic, truncado o sin tool call), con backoff exponencial. Una API key inválida no se reintenta.
5. **`.with_fallbacks()`.** Si el modelo principal agota sus intentos o falla el proveedor, se prueba el otro proveedor con su propio `.with_retry()`. Los logs muestran `Cambio al modelo de respaldo`.

El estado entre intentos (correcciones y `max_tokens`) es propio de cada llamada a `process_text()`: un texto nunca hereda las correcciones de otro.

**Reutilización del Módulo 1 (`config.py`).** Proveedor, API key (`SecretStr`), validación de parámetros, `MissingAPIKeyError` y la detección de parámetros que Anthropic ya no acepta vienen de `llm_client`. Este repo sólo agrega lo propio del pipeline.

**Errores controlados.** `process_text()` siempre devuelve un `ExtraccionTecnica` válido o lanza un `ExtraccionError` (`tipo`: `salida_invalida`, `proveedor` o `entrada_invalida`, junto con la cantidad de `intentos`). Nunca deja escapar una excepción cruda del SDK.

**Versiones acotadas.** Las dependencias aceptan mejoras menores pero no una versión mayor nueva (`langchain-core>=1.7,<2`).

**Nota sobre Anthropic:** desde la versión de API `2026-10-21` (SDK `anthropic` >= 2.0), la API rechaza `temperature`. El pipeline no la envía y avisa en el log si la configuraste.

## Validación

**Alcance:** los 32 tests usan un modelo simulado y no consumen créditos. Se verificó contra la API real de Anthropic con una key inválida: devuelve un 401 controlado, activa el respaldo y no reintenta.

**Limitación:** las respuestas reales de los modelos no fueron verificadas por no disponer de créditos de API.
