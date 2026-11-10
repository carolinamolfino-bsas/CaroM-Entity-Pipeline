"""Prompt templates modulares (sin f-strings: LangChain gestiona las variables)."""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate

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

# Mensaje de autocorrección: se agrega en el reintento cuando la respuesta
# anterior no pasó la validación, para que el modelo sepa QUÉ corregir.
CORRECCION_TEMPLATE = PromptTemplate.from_template(
    """Tu respuesta anterior no es válida y fue descartada.

Respuesta anterior:
{salida_previa}

Problemas detectados:
{errores}

Volvé a llamar a la herramienta 'ExtraccionTecnica' corrigiendo esos problemas. \
Seguí respetando las reglas: no inventes tecnologías que no estén en el texto."""
)


def build_prompt(instrucciones_formato: str = INSTRUCCIONES_FORMATO_DEFAULT) -> ChatPromptTemplate:
    """Template del pipeline.

    - `texto` (obligatoria): el texto a analizar.
    - `correcciones` (opcional): mensajes de autocorrección de intentos previos.
    - `instrucciones_formato`: fijada con .partial(), reemplazable sin tocar el resto.
    """
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_TEMPLATE),
            ("human", HUMAN_TEMPLATE),
            MessagesPlaceholder("correcciones", optional=True),
        ]
    ).partial(instrucciones_formato=instrucciones_formato)
