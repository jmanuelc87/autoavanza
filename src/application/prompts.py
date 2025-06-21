import os

from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
)

extract = [
    SystemMessagePromptTemplate.from_template(
        "Eres un asistente servicial, usando OCR extraes los attributos que se te indiquen de la imagen"
    ),
    HumanMessagePromptTemplate.from_template("data:image/jpg;base64,{image}"),
]


validate = [
    SystemMessagePromptTemplate.from_template(
        "Como analista de validaciones de empeño de auto, quiero verificar que la factura del vehículo cumpla con los requisitos de autenticidad y datos correctos, para garantizar que el auto no tenga problemas legales o administrativos antes de otorgar el préstamo."
    ),
    AIMessagePromptTemplate.from_template(
        """De acuerdo a la siguiente lista dime si es aprobado o rechazado el tramite."""
    ),
    HumanMessagePromptTemplate.from_template("""{items}"""),
]
