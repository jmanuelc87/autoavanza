from pydantic import BaseModel, Field


class Factura(BaseModel):
    nombre_cliente: str = Field(description="El nombre del cliente")
    marca: str = Field(description="La marca")
    version: str = Field(description="La version o tipo")
    modelo: str = Field(description="El modelo")
    color: str = Field(description="El color")
    combustible: str = Field(description="El tipo de combustible")
    no_serie: str = Field(description="El numero de serie, NIV o VIN")
    no_motor: str = Field(description="El numero de motor")


class TarjetaCirculacion(BaseModel):
    nombre: str = Field(description="El nombre del cliente")
    rfc: str = Field(description="El rfc del propietario")
    vehiculo: str = Field(description="La clase y tipo de vehiculo")
    marca: str = Field(description="La marca")
    modelo: int = Field(description="El modelo del vehiculo")
    no_motor: str = Field(description="El numero de serie del motor")
    no_niv: str = Field(description="El numero de identificacion vehicular")
    expedicion: str = Field(description="La fecha de expedicion")
    vigencia: str = Field(description="La fecha de vigencia")
    placa: str = Field(description="El numero de placa")
    combustible: str = Field(description="El tipo de combustible")


class CredencialVotar(BaseModel):
    nombre: str = Field(description="El nombre de la persona")
    domicilio: str = Field(description="El domicilio de la persona")
    emision: int = Field(description="La fecha de emision")
    vigencia: str = Field(description="Los años de vigencia")


class ToModel(BaseModel):
    carroceria: str = Field(description="carroceria")
    color: str = Field(description="color")
    combustible: str = Field(description="combustible")
    fecha: str = Field(description="fecha")
    marca: str = Field(description="marca")
    modelo: str = Field(description="modelo")
