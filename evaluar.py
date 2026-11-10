"""Mini evaluación del pipeline contra un dataset con respuestas esperadas.

Mide, sobre evals/dataset.jsonl:
  - tasa de éxito:          % de textos con objeto válido (sin ExtraccionError)
  - exactitud de criticidad: % de casos con criticidad esperada que acertó
  - recall de tecnologías:   % de tecnologías esperadas que el modelo encontró
  - tecnologías extra:       nombres devueltos que no estaban en lo esperado
                             (no siempre es un error: puede ser una variante del nombre)

Los casos con `criticidad: null` (ej. el texto ambiguo) sólo verifican que el
pipeline no se rompa: un objeto válido o un error controlado son aceptables.

Uso:
    python evaluar.py                     # proveedor del .env
    python evaluar.py --umbral 0.8        # exit code 1 si la exactitud queda debajo
    python evaluar.py --salida evals/resultados.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

from schemas import ExtraccionTecnica, NivelCriticidad

DATASET = Path(__file__).parent / "evals" / "dataset.jsonl"


class CasoEval(BaseModel):
    id: str
    texto: str
    criticidad: NivelCriticidad | None = None
    tecnologias: list[str] = Field(default_factory=list)


@dataclass
class ResultadoCaso:
    id: str
    ok: bool
    criticidad_esperada: str | None
    criticidad_obtenida: str | None = None
    criticidad_correcta: bool | None = None
    encontradas: list[str] = field(default_factory=list)
    faltantes: list[str] = field(default_factory=list)
    extra: list[str] = field(default_factory=list)
    error: str | None = None


def _norm(nombre: str) -> str:
    return "".join(ch for ch in nombre.casefold() if ch.isalnum())


def evaluar_caso(caso: CasoEval, salida: ExtraccionTecnica | Exception) -> ResultadoCaso:
    esperada = caso.criticidad.value if caso.criticidad else None
    if isinstance(salida, Exception):
        return ResultadoCaso(id=caso.id, ok=False, criticidad_esperada=esperada,
                             faltantes=list(caso.tecnologias), error=str(salida))

    obtenidas = {_norm(t): t for t in salida.tecnologias}
    esperadas = {_norm(t): t for t in caso.tecnologias}
    return ResultadoCaso(
        id=caso.id,
        ok=True,
        criticidad_esperada=esperada,
        criticidad_obtenida=salida.nivel_de_criticidad.value,
        criticidad_correcta=None if esperada is None else salida.nivel_de_criticidad.value == esperada,
        encontradas=[esperadas[k] for k in esperadas if k in obtenidas],
        faltantes=[esperadas[k] for k in esperadas if k not in obtenidas],
        extra=[obtenidas[k] for k in obtenidas if k not in esperadas],
    )


def metricas(resultados: list[ResultadoCaso]) -> dict[str, float]:
    total = len(resultados)
    con_criticidad = [r for r in resultados if r.criticidad_esperada is not None]
    esperadas = sum(len(r.encontradas) + len(r.faltantes) for r in resultados)
    return {
        "casos": total,
        "tasa_exito": sum(r.ok for r in resultados) / total if total else 0.0,
        "exactitud_criticidad": (
            sum(bool(r.criticidad_correcta) for r in con_criticidad) / len(con_criticidad)
            if con_criticidad else 0.0
        ),
        "recall_tecnologias": (
            sum(len(r.encontradas) for r in resultados) / esperadas if esperadas else 0.0
        ),
    }


def cargar_dataset(path: Path = DATASET) -> list[CasoEval]:
    with path.open(encoding="utf-8") as f:
        return [CasoEval.model_validate_json(linea) for linea in f if linea.strip()]


def imprimir(resultados: list[ResultadoCaso], m: dict[str, float]) -> None:
    print(f"\n{'caso':<28} {'crit. esperada':<15} {'obtenida':<10} {'tecnologías':<12} faltantes")
    print("-" * 90)
    for r in resultados:
        if not r.ok:
            print(f"{r.id:<28} {str(r.criticidad_esperada):<15} {'ERROR':<10} {'-':<12} "
                  f"{(r.error or '')[:40]}")
            continue
        marca = {True: "✔", False: "✘", None: "·"}[r.criticidad_correcta]
        tecs = f"{len(r.encontradas)}/{len(r.encontradas) + len(r.faltantes)}"
        print(f"{r.id:<28} {str(r.criticidad_esperada):<15} {marca} {r.criticidad_obtenida:<8} "
              f"{tecs:<12} {', '.join(r.faltantes) or '-'}")
    print("-" * 90)
    print(f"Tasa de éxito:            {m['tasa_exito']:.0%}")
    print(f"Exactitud de criticidad:  {m['exactitud_criticidad']:.0%}")
    print(f"Recall de tecnologías:    {m['recall_tecnologias']:.0%}")


async def main() -> int:
    from chain import build_chain_from_settings, process_many
    from config import MissingAPIKeyError, PipelineSettings

    parser = argparse.ArgumentParser(description="Evaluación del pipeline")
    parser.add_argument("--provider", choices=["openai", "anthropic"], default=None)
    parser.add_argument("--umbral", type=float, default=0.7,
                        help="Exactitud mínima de criticidad (0-1). Default 0.7")
    parser.add_argument("--salida", type=Path, help="Guardar resultados en JSON")
    args = parser.parse_args()

    try:
        settings = PipelineSettings.from_env(args.provider)
        chain = build_chain_from_settings(settings)
    except (ValidationError, MissingAPIKeyError) as exc:
        print(f"Configuración inválida: {exc}")
        return 2

    casos = cargar_dataset()
    salidas = await process_many([c.texto for c in casos], chain=chain,
                                 max_concurrency=settings.llm.max_concurrency)
    resultados = [evaluar_caso(c, s) for c, s in zip(casos, salidas, strict=True)]
    m = metricas(resultados)
    imprimir(resultados, m)

    if args.salida:
        args.salida.write_text(
            json.dumps({"metricas": m, "casos": [asdict(r) for r in resultados]},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nResultados guardados en {args.salida}")

    aprobado = m["exactitud_criticidad"] >= args.umbral
    print(f"\n{'APROBADO' if aprobado else 'DEBAJO DEL UMBRAL'} (umbral {args.umbral:.0%})")
    return 0 if aprobado else 1


if __name__ == "__main__":
    load_dotenv()
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "WARNING"),
                        format="%(levelname)-7s %(name)s: %(message)s")
    sys.exit(asyncio.run(main()))
