import os
import sys
import cv2
import json
import fitz
import dotenv
import functools
import pandas as pd

from PIL import Image

from tempfile import NamedTemporaryFile

from application.inference import predict
from application.prompts import extract, validate, normalize
from application.rules import tools, validate_documents
from application.document import Scanner
from application.models import Factura, TarjetaCirculacion, CredencialVotar, ToModel

from operator import itemgetter

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders.image import UnstructuredImageLoader


dotenv.load_dotenv(dotenv.find_dotenv())


def __load_pdf2img(args):
    mat = fitz.Matrix(3, 3)
    filename: str = args["filename"]
    if filename.endswith(("jpg", "jpeg", "png")):
        return args

    doc = fitz.open(args["filename"])
    temp_file = ""
    for page in doc:
        pix = page.get_pixmap(matrix=mat)
        with NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            pix.save(tmp.name)
            temp_file = tmp.name

    return {"filename": temp_file, "doc_name": args["doc_name"]}


def __load_unstructured_image(args):
    pil_image = args["image"]
    with NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        pil_image.save(tmp.name, format="JPEG")
        tmp_path = tmp.name
    loader = UnstructuredImageLoader(tmp_path)
    try:
        docs = loader.load()
        page = docs.pop()
        return page.page_content or ""
    except TypeError as e:
        print(e)
        return ""
    finally:
        os.remove(tmp_path)


def __scanner(args, scanner: Scanner):
    image = cv2.imread(args["filename"])
    out = scanner.get_document_from(image)
    return {"image": out, "doc_name": args["doc_name"]}


def __image_preprocessing_card(args):
    image = args["image"]
    print(image.shape)
    w, h = image.shape[:2]
    rw, rh = (2 * w) / w, (2 * h) / h
    image = cv2.resize(image, None, fx=rw, fy=rh, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite(args["doc_name"] + ".jpg", thresh)
    return {"image": Image.fromarray(thresh)}


def __image_preprocessing_doc(args):
    image = args["image"]
    print(image.shape)
    w, h = image.shape[:2]
    rw, rh = (2 * w) / w, (2 * h) / h
    image = cv2.resize(image, None, fx=rw, fy=rh, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite(args["doc_name"] + ".jpg", thresh)
    return {"image": Image.fromarray(thresh)}


def __normalize_document(args):
    factura = args["factura"]
    tarjeta = args["tarjeta_circulacion"]

    rows = {
        "carroceria": tarjeta.vehiculo,
        "color": factura.color,
        "combustible": tarjeta.combustible,
        "fecha": factura.modelo,
        "marca": factura.marca,
        "modelo": factura.version,
    }

    seminuevos_df = pd.read_csv("data/seminuevos.csv")
    top_models = seminuevos_df["modelo"].value_counts().nlargest(300).index

    return {
        "values": json.dumps(rows),
        "modelo_categorias": ",".join(top_models.to_list()),
    }


def __joiner(args):
    dictamen_documental = args["dictamen_documental"]
    result = 0

    if "Aprobado" in dictamen_documental.content:
        to_model: ToModel = args["model_call"]
        rows = [
            to_model.carroceria,
            to_model.color,
            to_model.combustible,
            to_model.fecha,
            to_model.marca,
            to_model.modelo,
            args["odometro"],
        ]
        cols = [
            "carroceria_cat",
            "color",
            "combustible",
            "fecha",
            "marca",
            "modelo",
            "odometro_clean",
        ]

        df = pd.DataFrame(data=[rows], columns=cols)
        print(df)
        result = predict(df)

    return {
        "dictamen_documental": dictamen_documental.content,
        "value": int(result),
    }


def get_validation_pipeline():
    __llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0,
        streaming=False,
    )

    scanner = Scanner()

    load_pdf2img = RunnableLambda(__load_pdf2img)
    preprocessing_card = RunnableLambda(__image_preprocessing_card)
    preprocessing_fac = RunnableLambda(__image_preprocessing_doc)
    loader1 = RunnableLambda(functools.partial(__scanner, scanner=scanner))
    loader3 = RunnableLambda(__load_unstructured_image)

    prompt_extraction = ChatPromptTemplate.from_messages(extract)

    llm_with_fac_structured_output = __llm.with_structured_output(Factura)

    extract_fac_chain = (
        {
            "page_content": load_pdf2img | loader1 | preprocessing_fac | loader3,
            "doc_name": itemgetter("doc_name"),
        }
        | prompt_extraction
        | llm_with_fac_structured_output
    )

    llm_with_tc_structured_output = __llm.with_structured_output(TarjetaCirculacion)

    extract_tc_chain = (
        {
            "page_content": load_pdf2img | loader1 | preprocessing_card | loader3,
            "doc_name": itemgetter("doc_name"),
        }
        | prompt_extraction
        | llm_with_tc_structured_output
    )

    llm_with_ine_structured_output = __llm.with_structured_output(CredencialVotar)

    extract_ine_chain = (
        {
            "page_content": load_pdf2img | loader1 | preprocessing_card | loader3,
            "doc_name": itemgetter("doc_name"),
        }
        | prompt_extraction
        | llm_with_ine_structured_output
    )

    validation_chain = RunnableLambda(
        functools.partial(validate_documents, tools=tools)
    )

    extract_chain = {
        "factura": {
            "filename": itemgetter("factura"),
            "doc_name": lambda x: "Factura",
        }
        | extract_fac_chain,
        "tarjeta_circulacion": {
            "filename": itemgetter("tc"),
            "doc_name": lambda x: "Tarjeta de Circulacion",
        }
        | extract_tc_chain,
        "credencial_votar": {
            "filename": itemgetter("ine"),
            "doc_name": lambda x: "Credencial para Votar",
        }
        | extract_ine_chain,
    }

    validation_prompt = ChatPromptTemplate.from_messages(validate)
    normalize_prompt = ChatPromptTemplate.from_messages(normalize)

    normalize_chain = RunnableLambda(__normalize_document)

    return {
        "dictamen_documental": extract_chain
        | validation_chain
        | validation_prompt
        | __llm,
        "model_call": extract_chain
        | normalize_chain
        | normalize_prompt
        | __llm.with_structured_output(ToModel),
        "odometro": itemgetter("odometro"),
    } | RunnableLambda(__joiner)


if __name__ == "__main__":
    bot = get_validation_pipeline()

    result = bot.invoke(
        input={
            "factura": "data/bronze/BASE_AUTOAVANZA/Caso 0/20250612_181153.jpg",
            "tc": "data/bronze/BASE_AUTOAVANZA/Caso 0/20250612_183027.jpg",
            "ine": "data/bronze/BASE_AUTOAVANZA/Caso 0/20250612_181121.jpg",
            "odometro": 45000,
        }
    )

    print(result)
