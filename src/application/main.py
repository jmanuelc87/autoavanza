import os
import cv2
import dotenv
import base64
import functools

from io import BytesIO
from PIL import Image

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


__llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
)


scanner = Scanner()


def __load_image(inputs):
    """Load image from file and encode it as base64."""
    pil_image = inputs["image"]

    def encode_image(pil_image):
        img_buffer = BytesIO()
        pil_image.save(img_buffer, format="JPEG")
        byte_data = img_buffer.getvalue()
        base64_str = base64.b64encode(byte_data)
        return base64_str

    image_base64 = encode_image(pil_image)
    return image_base64


def __scanner(args):
    image = cv2.imread(args["filename"])
    out = scanner.get_document_from(image)
    h, w = out.shape[:2]
    r_h, r_w = (h * 0.6) / h, (w * 0.6) / w
    out = cv2.resize(out, None, fx=r_w, fy=r_h, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(args["doc_name"] + ".jpg", out)
    return {"image": Image.fromarray(out)}


def __image_preprocessing(args):
    pass


def get_validation_chatbot():
    loader1 = RunnableLambda(__scanner)
    loader2 = RunnableLambda(__load_image)

    prompt_extraction = ChatPromptTemplate.from_messages(extract)

    llm_with_fac_structured_output = __llm.with_structured_output(Factura)

    extract_fac_chain = (
        {
            "image": loader1 | loader2,
            "doc_name": itemgetter("doc_name"),
        }
        | prompt_extraction
        | llm_with_fac_structured_output
    )

    llm_with_tc_structured_output = __llm.with_structured_output(TarjetaCirculacion)

    extract_tc_chain = (
        {
            "image": loader1 | loader2,
            "doc_name": itemgetter("doc_name"),
        }
        | prompt_extraction
        | llm_with_tc_structured_output
    )

    llm_with_ine_structured_output = __llm.with_structured_output(CredencialVotar)

    extract_ine_chain = (
        {
            "image": loader1 | loader2,
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
    llm = get_validation_chatbot()

    result = llm.invoke(
        input={
            "factura": "20250612_181153.jpg",
            "tc": "20250612_183027.jpg",
            "ine": "20250612_181121.jpg",
        }
    )

    print(result.content)
