"""Contrato de salida del pipeline (Pydantic v2).

Este modelo es la "fuente de verdad": LangChain lo convierte en el schema de la
herramienta que el LLM debe completar, y luego valida la respuesta contra él.
Si el LLM devuelve algo que no cumple estas restricciones, la validación falla
y la cadena reintenta.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class NivelCriticidad(StrEnum):
    BAJA = "baja"
    MEDIA = "media"
    ALTA = "alta"


class ExtraccionTecnica(BaseModel):
    """Entidades técnicas extraídas de un texto (log, incidente o descripción de arquitectura)."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    tecnologias: list[str] = Field(
        min_length=1,
        max_length=30,
        description=(
            "Tecnologías, lenguajes, frameworks, bases de datos, servicios o "
            "herramientas mencionados en el texto, con su nombre oficial "
            "(ej. 'PostgreSQL', no 'postgres'). Sin duplicados."
        ),
    )
    nivel_de_criticidad: NivelCriticidad = Field(
        description=(
            "'alta': caída, pérdida de datos, seguridad o impacto en producción; "
            "'media': degradación de rendimiento o errores parciales; "
            "'baja': informativo, mejora o sin impacto en usuarios."
        ),
    )
    resumen_tecnico: str = Field(
        min_length=20,
        max_length=400,
        description="Resumen técnico de 1 o 2 oraciones, en español.",
    )

    @field_validator("nivel_de_criticidad", mode="before")
    @classmethod
    def _normalizar_criticidad(cls, value: Any) -> Any:
        # Tolera "Alta", " MEDIA " (mayúsculas/espacios), pero no valores fuera del enum.
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("tecnologias")
    @classmethod
    def _limpiar_tecnologias(cls, values: list[str]) -> list[str]:
        limpias: list[str] = []
        vistas: set[str] = set()
        for v in values:
            v = v.strip()
            if not v:
                raise ValueError("La lista de tecnologías contiene un elemento vacío")
            if v.casefold() not in vistas:  # dedup sin distinguir mayúsculas
                vistas.add(v.casefold())
                limpias.append(v)
        return limpias
