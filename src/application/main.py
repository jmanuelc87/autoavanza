import os
import sys
import cv2
import fitz
import dotenv
import base64
import functools

from io import BytesIO
from PIL import Image

from tempfile import NamedTemporaryFile

from application.prompts import extract, validate
from application.rules import tools, validate_documents
from application.document import Scanner
from application.models import Factura, TarjetaCirculacion, CredencialVotar

from operator import itemgetter

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
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


def __scanner(args):
    scanner = Scanner()
    image = cv2.imread(args["filename"])
    out = scanner.get_document_from(image)
    return {"image": out, "doc_name": args["doc_name"]}


def __image_preprocessing_card(args):
    image = args["image"]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite(args["doc_name"] + ".jpg", thresh)
    return {"image": Image.fromarray(thresh)}


def __image_preprocessing_doc(args):
    image = args["image"]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite(args["doc_name"] + ".jpg", thresh)
    return {"image": Image.fromarray(thresh)}


def get_validation_chatbot():
    __llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0,
        streaming=False,
    )

    load_pdf2img = RunnableLambda(__load_pdf2img)
    preprocessing_card = RunnableLambda(__image_preprocessing_card)
    preprocessing_fac = RunnableLambda(__image_preprocessing_doc)
    loader1 = RunnableLambda(__scanner)
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

    chain = {
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
    } | validation_chain

    chat_validation = ChatPromptTemplate.from_messages(validate)

    return chain | chat_validation | __llm


if __name__ == "__main__":
    chat = get_validation_chatbot()

    result = chat.invoke(
        input={
            "factura": "data/bronze/BASE_AUTOAVANZA/Caso 0/20250612_181153.jpg",
            "tc": "data/bronze/BASE_AUTOAVANZA/Caso 0/20250612_183027.jpg",
            "ine": "data/bronze/BASE_AUTOAVANZA/Caso 0/20250612_181121.jpg",
        }
    )

    print(result.content)
