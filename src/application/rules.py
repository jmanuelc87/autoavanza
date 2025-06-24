import os
import datetime

from .models import Factura, TarjetaCirculacion, CredencialVotar

from rapidfuzz import fuzz, utils


def compare_doc_names(
    factura: Factura,
    tarjeta_circulacion: TarjetaCirculacion,
    credencial_votar: CredencialVotar,
    **kwargs,
) -> str:
    """Compara el propietario o nombre en los documentos Factura, Tarjeta de Circulacion y Credencial para Votar

    Args:
        factura (Factura): La factura del vehiculo
        tarjeta_circulacion (TarjetaCirculacion): La Tarjeta de circulacion
        credencial_votar (CredencialVotar): La Credencial para Votar

    Returns:
        str: El resultado de la validacion es aprovado o rechazado
    """
    print(factura, tarjeta_circulacion, credencial_votar)
    wr1 = fuzz.WRatio(
        factura.nombre_cliente,
        tarjeta_circulacion.nombre,
        processor=utils.default_process,
    )
    wr2 = fuzz.WRatio(
        factura.nombre_cliente, credencial_votar.nombre, processor=utils.default_process
    )
    wr3 = fuzz.WRatio(
        tarjeta_circulacion.nombre,
        credencial_votar.nombre,
        processor=utils.default_process,
    )

    if wr1 >= 90.0 and wr2 >= 90.0 and wr3 >= 90.0:
        return "El nombre del propietario es correcto!"
    elif 70.0 <= wr1 < 90.0 and 70.0 <= wr2 < 90.0 and 70.0 <= wr3 < 90.0:
        return "Favor de verificar que el nombre aparezca correctamente en los tres documentos"
    else:
        return "El nombre no aparece correctamente en los documentos"


def compare_vin(
    factura: Factura, tarjeta_circulacion: TarjetaCirculacion, **kwargs
) -> list[str]:
    """Compara el numero de serie, NIV o VIN en la factura y tarjeta de circulacion

    Args:
        factura (Factura): La Factura
        tc (TarjetaCirculacion): La Tarjeta de Circulacion

    Returns:
        str: El resultado de la validacion es aprovado o rechazado
    """

    wr = fuzz.WRatio(factura.no_serie, tarjeta_circulacion.no_niv)

    result = []

    if wr >= 70.0:
        result.append("El NIV del vehiculo es correcto!")
    else:
        result.append("El NIV no coincide")

    wr2 = fuzz.WRatio(factura.no_motor, tarjeta_circulacion.no_motor)

    if wr >= 70.0:
        result.append("El numero de serie del motor es correcto!")
    else:
        result.append("El numero de serie del motor no coincide")

    return result


def compare_vehicle_data(
    factura: Factura, tarjeta_circulacion: TarjetaCirculacion, **kwargs
) -> list[str]:
    """Compara la marca, linea y modelo del vehiculo en la factura y tarjeta de circulacion

    Args:
        factura (Factura): La Factura
        tc (TarjetaCirculacion): La Tarjeta de Circulacion

    Returns:
        str: El resultado de la validacion es aprovado o rechazado
    """
    result = []

    wr = fuzz.WRatio(factura.marca, tarjeta_circulacion.marca)

    if wr < 80.0:
        label = f"La marca del vehiculo es distinto."
        result.append(label)

    wr = fuzz.WRatio(factura.version, tarjeta_circulacion.vehiculo)

    if wr < 80.0:
        label = "La version del vehiculo es distinta."
        result.append(label)

    wr = fuzz.WRatio(str(factura.modelo), str(tarjeta_circulacion.modelo))

    if wr < 80.0:
        label = "El modelo del vehiculo es distinto."
        result.append(label)

    if len(result) == 0:
        result.append("Los datos del vehiculo son correctos!")

    return result


def compare_motor(
    factura: Factura, tarjeta_circulacion: TarjetaCirculacion, **kwargs
) -> str:
    """Compara el motor en la Factura y Tarjeta de Circulacion

    Args:
        factura (Factura): La Factura
        tc (TarjetaCirculacion): La Tarjeta de Circulacion

    Returns:
        str: El resultado de la validacion es approvado o rechazado
    """
    wr = fuzz.WRatio(factura.no_motor, tarjeta_circulacion.no_motor)

    if wr < 80.0:
        return "El numero de motor del vehiculo es distinto. Favor de verificar"

    return "El numero de motor es correcto!"


def check_validity(credencial_votar: CredencialVotar, **kwargs) -> str:
    """Verfica la validez de la credencial para votar

    Args:
        ine (CredencialVotar): La credencial para votar

    Returns:
        str: El resultado de la validacion es approvador o rechazado
    """
    vigencia = credencial_votar.vigencia

    if "-" in vigencia:
        vigencia = vigencia.split("-")[1]

    vigencia = int(vigencia)

    current_year = datetime.datetime.now().year
    if current_year > vigencia:
        return "La Credencial para Votar no esta vigente"
    return "La Credencial para Votar es vigente"


tools = [
    compare_doc_names,
    compare_vin,
    compare_vehicle_data,
    compare_motor,
    # check_validity,
]


def validate_documents(inputs, tools):
    """Validacion de los documentos

    Args:
        inputs (dict): Los datos proporcionados por el llm
        tools (list): Las herramientas que validaran los documentos

    Yields:
        str: el resultado de la validacion
    """
    results = []
    for tool in tools:
        result = tool(**inputs)
        if isinstance(result, list):
            results.extend(result)
        else:
            results.append(result)

    print(results, end="\n\n")

    return {"items": results}
