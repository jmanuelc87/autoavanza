import os

from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
)

extract = [
    SystemMessagePromptTemplate.from_template(
        "Eres un asistente servicial, extrae los campos que se te indiquen del documento {doc_name}."
    ),
    # HumanMessagePromptTemplate.from_template("data:image/jpg;base64,{image}"),
    HumanMessagePromptTemplate.from_template("El documento {doc_name}: {page_content}"),
]


validate = [
    SystemMessagePromptTemplate.from_template(
        "Como analista de validaciones de empeño de auto, quiero verificar que la factura del vehículo cumpla con los requisitos de autenticidad y datos correctos, para garantizar que el auto no tenga problemas legales o administrativos antes de otorgar el préstamo."
    ),
    AIMessagePromptTemplate.from_template(
        """De acuerdo a la siguiente lista, de forma conscisa escribe Aprobado o Rechazado el tramite de acuerdo a las siguientes reglas. La Marca, version, y modelo no rechazan el trámite si no coinciden todas las demas si."""
    ),
    HumanMessagePromptTemplate.from_template("""{items}"""),
]
