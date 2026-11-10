"""Mini-script de prueba asíncrono.

Procesa en paralelo (process_many) tres textos y verifica el comportamiento:
  1. Log de error claro          -> se espera criticidad "alta"
  2. Descripción de arquitectura -> se espera criticidad "baja" o "media"
  3. Texto ambiguo (estrés)      -> se espera que NO se rompa: objeto válido
                                    o error controlado, nunca un traceback

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
from dataclasses import dataclass

from dotenv import load_dotenv
from pydantic import ValidationError

from chain import ExtraccionError, build_chain_from_settings, process_many
from config import MissingAPIKeyError, PipelineSettings
from schemas import ExtraccionTecnica


@dataclass
class Caso:
    nombre: str
    texto: str
    criticidad_aceptada: set[str] | None  # None = prueba de estrés: sólo que no se rompa


EJEMPLOS = [
    Caso(
        "log de error",
        "2026-11-09 03:12:44 ERROR [api-gateway] FastAPI worker 7: "
        "psycopg.OperationalError: connection to server at 'db-prod' (PostgreSQL 16) "
        "failed: remaining connection slots are reserved. Redis cache hit ratio cayó a 12%. "
        "El endpoint /checkout devuelve 503 al 40% de los usuarios desde hace 20 minutos.",
        {"alta"},
    ),
    Caso(
        "arquitectura",
        "El servicio de notificaciones está escrito en Go, consume eventos de Kafka y "
        "persiste el estado en DynamoDB. Corre en Kubernetes (EKS) con autoscaling por HPA. "
        "Estamos evaluando migrar las plantillas de email de Handlebars a MJML.",
        {"baja", "media"},
    ),
    Caso(
        "ambiguo (estrés)",
        "desde ayer anda raro, a veces tarda y a veces no, el de infra dice que es la base "
        "pero no sé, capaz es el cache o el balanceador. ojo que el viernes hay release.",
        None,
    ),
]


def verificar(caso: Caso, salida: ExtraccionTecnica | ExtraccionError) -> bool:
    print(f"\n### {caso.nombre}")
    if isinstance(salida, ExtraccionError):
        print(f"ERROR CONTROLADO [{salida.tipo}] tras {salida.intentos} intento(s): {salida}")
        print(f"  causa: {type(salida.causa).__name__}: {str(salida.causa)[:300]}")
        if caso.criticidad_aceptada is None and salida.tipo == "salida_invalida":
            print("✔ Esperado: el validador rechazó el texto ambiguo sin romper el programa.")
            return True
        print("✘ No se obtuvo un objeto válido.")
        return False

    print(salida.model_dump_json(indent=2))
    if caso.criticidad_aceptada is None:
        print("✔ Esperado: el modelo se recuperó y devolvió un objeto válido.")
        return True
    if salida.nivel_de_criticidad.value in caso.criticidad_aceptada:
        print(f"✔ Criticidad '{salida.nivel_de_criticidad}' dentro de lo esperado.")
        return True
    print(f"✘ Criticidad '{salida.nivel_de_criticidad}', se esperaba "
          f"{' o '.join(sorted(caso.criticidad_aceptada))}.")
    return False


async def main() -> int:
    parser = argparse.ArgumentParser(description="Pipeline de extracción de entidades técnicas")
    parser.add_argument("--provider", choices=["openai", "anthropic"], default=None)
    parser.add_argument("--texto", help="Procesar sólo este texto")
    args = parser.parse_args()

    try:
        settings = PipelineSettings.from_env(args.provider)
        chain = build_chain_from_settings(settings)
    except (ValidationError, MissingAPIKeyError) as exc:
        print(f"Configuración inválida: {exc}")
        return 2

    respaldo = (f"{settings.fallback.provider}/{settings.fallback.model.model}"
                if settings.fallback else "ninguno")
    print(f"Proveedor: {settings.llm.provider} · modelo: {settings.llm.model.model} · "
          f"intentos máx.: {settings.output_retries} · respaldo: {respaldo}")

    casos = [Caso("texto propio", args.texto, None)] if args.texto else EJEMPLOS
    salidas = await process_many([c.texto for c in casos], chain=chain,
                                 max_concurrency=settings.llm.max_concurrency)
    resultados = [verificar(c, s) for c, s in zip(casos, salidas, strict=True)]

    print(f"\nResumen: {sum(resultados)}/{len(resultados)} casos con el comportamiento esperado.")
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
