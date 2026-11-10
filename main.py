"""Mini-script de prueba asíncrono.

Procesa en paralelo (asyncio.gather) tres textos:
  1. Log de error claro          -> se espera criticidad alta
  2. Descripción de arquitectura -> se espera criticidad media/baja
  3. Texto ambiguo (estrés)      -> el modelo se recupera o el validador lo rechaza

Uso:
    python main.py                        # proveedor de LLM_PROVIDER (.env)
    python main.py --provider anthropic
    python main.py --texto "mi propio log..."
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

from dotenv import load_dotenv
from pydantic import ValidationError

from chain import ExtraccionError, build_chain, process_text
from config import LLMSettings, MissingAPIKeyError, build_chat_model

EJEMPLOS: dict[str, str] = {
    "log de error": (
        "2026-11-09 03:12:44 ERROR [api-gateway] FastAPI worker 7: "
        "psycopg.OperationalError: connection to server at 'db-prod' (PostgreSQL 16) "
        "failed: remaining connection slots are reserved. Redis cache hit ratio cayó a 12%. "
        "El endpoint /checkout devuelve 503 al 40% de los usuarios desde hace 20 minutos."
    ),
    "arquitectura": (
        "El servicio de notificaciones está escrito en Go, consume eventos de Kafka y "
        "persiste el estado en DynamoDB. Corre en Kubernetes (EKS) con autoscaling por HPA. "
        "Estamos evaluando migrar las plantillas de email de Handlebars a MJML."
    ),
    "ambiguo (estrés)": (
        "desde ayer anda raro, a veces tarda y a veces no, el de infra dice que es la base "
        "pero no sé, capaz es el cache o el balanceador. ojo que el viernes hay release."
    ),
}


async def correr(nombre: str, texto: str, chain) -> bool:
    try:
        resultado = await process_text(texto, chain=chain)
    except ExtraccionError as exc:
        print(f"\n### {nombre}\nERROR CONTROLADO [{exc.tipo}] tras {exc.intentos} intento(s): "
              f"{exc}\n  causa: {type(exc.causa).__name__}: {str(exc.causa)[:300]}")
        return False
    print(f"\n### {nombre}\n{resultado.model_dump_json(indent=2)}")
    return True


async def main() -> int:
    parser = argparse.ArgumentParser(description="Pipeline de extracción de entidades técnicas")
    parser.add_argument("--provider", choices=["openai", "anthropic"], default=None)
    parser.add_argument("--texto", help="Procesar sólo este texto")
    args = parser.parse_args()

    try:
        settings = LLMSettings.from_env(args.provider)
        chain = build_chain(build_chat_model(settings), max_attempts=settings.output_retries)
    except (ValidationError, MissingAPIKeyError) as exc:
        print(f"Configuración inválida: {exc}")
        return 2

    print(f"Proveedor: {settings.provider} · modelo: {settings.model} · "
          f"intentos máx.: {settings.output_retries}")

    casos = {"texto propio": args.texto} if args.texto else EJEMPLOS
    resultados = await asyncio.gather(*(correr(n, t, chain) for n, t in casos.items()))

    print(f"\nResumen: {sum(resultados)}/{len(resultados)} textos procesados correctamente.")
    return 0 if all(resultados) else 1


if __name__ == "__main__":
    load_dotenv()
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Silenciar logs HTTP de bajo nivel para que se vean los del pipeline.
    for noisy in ("httpx", "httpx2", "openai", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    sys.exit(asyncio.run(main()))
