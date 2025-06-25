import joblib
import numpy as np
import pandas as pd

from datetime import datetime


def __agrupar_carroceria(valor):
    if valor in ["Camioneta SUV", "Crossover", "Todoterreno Ligero"]:
        return "SUV"
    elif valor in [
        "Hatchback (5 Puertas)",
        "Hatchback (3 Puertas)",
    ]:
        return "Hatchback"
    elif valor == "Sedán":
        return "Sedán"
    elif valor == "Pickup":
        return "Pickup"
    elif valor in ["Van", "Van de Carga", "Van Pasajeros"]:
        return "Van"
    elif valor in ["Mini van (MPV)"]:
        return "Minivan"
    elif valor in ["Convertible", "Coupé"]:
        return "Deportivo"
    elif valor in [
        "Camioneta Cabina Media",
        "Camioneta Cabina Simple",
        "Camioneta Chasis",
    ]:
        return "Camioneta"
    elif valor == "Station Wagon":
        return "Familiar"
    elif valor in ["Utilitario", "Buggy", "Otro"]:
        return "Otro"
    else:
        return "Otro"


def __agrupar_color(color):
    color = color.lower()
    if color in ["blanco", "perla", "marfil", "crema"]:
        return "Blanco"
    elif color in ["gris", "plata", "plomo", "acero", "antracita"]:
        return "Gris/Plata"
    elif color in ["negro", "carbón"]:
        return "Negro"
    elif color in ["rojo", "guinda", "vino", "tinto", "mora"]:
        return "Rojo/Burdeos"
    elif color in ["azul", "acua", "quasar", "smoke blue"]:
        return "Azul"
    elif color in ["verde", "uva"]:
        return "Verde"
    elif color in ["café", "marrón", "cobre", "moka", "cobrizo"]:
        return "Marrón"
    elif color in [
        "amarillo",
        "dorado",
        "oro",
        "champagne",
        "terracota",
        "arena",
        "ocre",
        "beige",
    ]:
        return "Amarillo/Dorado"
    elif color in ["naranja", "orange"]:
        return "Naranja"
    elif color in ["morado", "violeta"]:
        return "Morado"
    else:
        return "Otro"


__alta_gama = [
    "Mercedes Benz",
    "BMW",
    "Audi",
    "Porsche",
    "Lexus",
    "Volvo",
    "Jaguar",
    "Land Rover",
    "Tesla",
    "Ferrari",
    "Aston Martin",
    "Maserati",
    "Cupra",
    "Infiniti",
    "Cadillac",
    "Acura",
    "Alfa Romeo",
]

__media_alta = [
    "Mazda",
    "Mini",
    "Jeep",
    "Volkswagen",
    "Kia",
    "Toyota",
    "Honda",
    "Hyundai",
    "SEAT",
    "BYD",
]

__media_baja = [
    "Chevrolet",
    "Nissan",
    "Ford",
    "Peugeot",
    "Renault",
    "Fiat",
    "Mitsubishi",
    "Dodge",
    "Chrysler",
    "Buick",
    "RAM",
    "MG",
    "GMC",
]

__baja_gama = [
    "JAC",
    "Changan",
    "Jetour",
    "Omoda",
    "GWM",
    "Smart",
    "Baic",
    "Chirey",
    "Geely",
    "Jaecoo",
    "GAC",
    "Pontiac",
    "Otra Marca",
    "JMC",
    "can-am",
    "Hummer",
    "Saab",
    "Kenworth",
    "Hino",
    "Mercury",
]

# Diccionario de mapeo ordinal
__gama_orden = {"Baja": 1, "Media-Baja": 2, "Media-Alta": 3, "Alta": 4}


# Clasificación textual
def __clasificar_marca_4gamas(marca):
    if marca in __alta_gama:
        return "Alta"
    elif marca in __media_alta:
        return "Media-Alta"
    elif marca in __media_baja:
        return "Media-Baja"
    else:
        return "Baja"


def predict(df):
    df["color_CAT"] = df["color"].apply(__agrupar_color)
    df["combustible_bin"] = df["combustible"].apply(
        lambda x: 1 if x in ["Gasolina", "Diésel"] else 0
    )
    df["marca_gama"] = df["marca"].apply(__clasificar_marca_4gamas)
    df["marca_ordinal"] = df["marca_gama"].map(__gama_orden)
    df["edad_auto"] = datetime.now().year - df["fecha"].astype(int)

    seminuevos_df = pd.read_csv("data/seminuevos.csv")
    top_models = seminuevos_df["modelo"].value_counts().nlargest(300).index

    df["modelo_simplificado"] = df["modelo"].where(
        seminuevos_df["modelo"].isin(top_models), "Otro Modelo"
    )

    with open("models/MLP_pipeline.joblib", "rb") as file:
        model = joblib.load(file)

    result = model.predict(df)

    return 0.7 * np.expm1(result)
