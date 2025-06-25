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

normalize = [
    SystemMessagePromptTemplate.from_template(
        "Eres un asistente servicial, normalizas los siguientes valores de acuerdo a las reglas propuestas"
    ),
    HumanMessagePromptTemplate.from_template("{values}"),
    HumanMessagePromptTemplate.from_template(
        """Reglas: 
            - La carroceria debe caer en alguna de estas categorias: SUV, Hatchback, Sedán, Pickup, Van, Minivan, Deportivo, Camioneta, Familiar, u Otro.
            - El color debe caer en alguna de estar categorias: Blanco, Gris/Plata, Negro, Rojo/Burdeos, Azul, Verde, Marrón, Amarillo/Dorado, Naranja, Morado, u Otro.
            - El combustible debe caer en alguna de estas categorias: Gasolina o Diésel.
            - La marca debe caer en alguna de estas categorias: Mercedes Benz, BMW, Audi, Porsche, Lexus, Volvo, Jaguar, Land Rover, Tesla, Ferrari, Aston Martin, Maserati, Cupra, Infiniti, Cadillac, Acura, Alfa Romeo, Mazda, Mini, Jeep, Volkswagen, Kia, Toyota, Honda, Hyundai, SEAT, BYD, Chevrolet, Nissan, Ford, Peugeot, Renault, Fiat, Mitsubishi, Dodge, Chrysler, Buick, RAM, MG, GMC, JAC, Changan, Jetour, Omoda, GWM, Smart, Baic, Chirey, Geely, Jaecoo, GAC, Pontiac, Otra Marca, JMC, can-am, Hummer, Saab, Kenworth, Hino, Mercury u Otro.
            - El modelo debe caer en alguna de estas categorias: {modelo_categorias}, u Otro Modelo."""
    ),
]
