"""Prompt template modular (sin f-strings: LangChain gestiona las variables)."""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

SYSTEM_TEMPLATE = """Sos un ingeniero de software senior especializado en análisis de \
incidentes y arquitectura de sistemas.

Tu tarea: extraer información técnica del texto que te pase el usuario.

Reglas:
- Incluí sólo tecnologías mencionadas explícitamente o inequívocamente implicadas \
(ej. "psql" implica PostgreSQL). No inventes tecnologías.
- Usá el nombre oficial de cada tecnología y no repitas ninguna.
- Elegí la criticidad según el impacto descrito, no según el tono del texto.
- Si el texto es ambiguo, elegí la interpretación más conservadora y aclaralo en el resumen.

Formato de salida:
{instrucciones_formato}"""

INSTRUCCIONES_FORMATO_DEFAULT = (
    "Respondé exclusivamente llamando a la herramienta 'ExtraccionTecnica' con "
    "todos sus campos completos. No agregues texto fuera de la herramienta."
)

HUMAN_TEMPLATE = """Texto a analizar:
<texto>
{texto}
</texto>"""


def build_prompt(instrucciones_formato: str = INSTRUCCIONES_FORMATO_DEFAULT) -> ChatPromptTemplate:
    """Devuelve el template con las instrucciones de formato ya fijadas.

    La única variable de entrada que queda libre es `texto`. Las instrucciones
    de formato se pueden reemplazar sin tocar el resto del prompt.
    """
    return ChatPromptTemplate.from_messages(
        [("system", SYSTEM_TEMPLATE), ("human", HUMAN_TEMPLATE)]
    ).partial(instrucciones_formato=instrucciones_formato)
