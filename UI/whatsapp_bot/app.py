from flask import Flask, request
from collections import defaultdict
from application.validation import get_validation_pipeline
import os
import requests

app = Flask(__name__)

# Meta credentials
ACCESS_TOKEN = "EAAZABXF6q620BO6lsZANu8COvoenApKsir8HWIovZCTaqeJA9r8BZCy3ViRRaq0N4nveGyJZAxr1AepWot9MV4Ja0azDP5AwSzmgVDS0nlTIWS3xoSpNRxsk6DZBtxaAiIPG8fUNn2JWsYUZAh2RVtCTPZAfkD2KP2sMoezmqbLkrvCD5bkTOWyzYYGsxcuShw7ouTJOPKUZAyHWsQCMnAyS0OlqbEfQ2TraDGxmDsmlHUaxykIKNzSRksOpK6"
VERIFY_TOKEN = "NMP123"

# User state: tracks what step each user is on
user_state = {}
user_state_files_holder = {}
odometro = 0

# Ordered list of required documents
document_steps = [
    "INE_front",
    "INE_back",
    "LICENSE_front",
    "LICENSE_back",
    "INVOICE_front",
    "INVOICE_back",
    "ODOMETRO",
]

# ===========
# ROUTES
# ===========


@app.route("/webhook", methods=["GET", "POST"])
def webhook():
    if request.method == "GET":
        mode = request.args.get("hub.mode")
        token = request.args.get("hub.verify_token")
        challenge = request.args.get("hub.challenge")

        if mode == "subscribe" and token == VERIFY_TOKEN:
            print("✅ Webhook verified!")
            return challenge, 200
        else:
            return "❌ Verification failed", 403

    if request.method == "POST":
        data = request.get_json()
        print("📩 Webhook triggered!")

        if data.get("entry"):
            for entry in data["entry"]:
                for change in entry["changes"]:
                    value = change["value"]
                    messages = value.get("messages")

                    if messages:
                        for message in messages:
                            phone_number = message["from"]
                            message_type = message["type"]

                            # First-time user setup
                            if phone_number not in user_state:
                                user_state[phone_number] = 0
                                send_message(
                                    phone_number,
                                    "👋 Hola! Vamos a iniciar. Por favor envía la *foto del frente de tu INE*.",
                                )
                                return "ok", 200

                            # If finished
                            step_index = user_state[phone_number]
                            if step_index >= len(document_steps):
                                send_message(
                                    phone_number,
                                    "✅ Ya recibimos los 6 documentos vamos a procesar tu solicitud. ¡Gracias!",
                                )

                                ine = user_state_files_holder[phone_number]["INE_front"]
                                tc = user_state_files_holder[phone_number][
                                    "LICENSE_front"
                                ]
                                factura = user_state_files_holder[phone_number][
                                    "INVOICE_front"
                                ]

                                result = predict_validate(
                                    ine=ine,
                                    tarjeta=tc,
                                    factura=factura,
                                    odometro=odometro,
                                )

                                send_message(
                                    phone_number,
                                    f"✅ Tu resultado es {result['dictamen_documental']} y te prestamos hasta {result['value']} por tu coche!",
                                )

                                return "ok", 200

                            if message_type == "text":
                                current_step = document_steps[step_index]
                                odometro = int(message["text"]["body"])

                                user_state[phone_number] += 1
                                if user_state[phone_number] < len(document_steps):
                                    next_step = document_steps[user_state[phone_number]]
                                    prompt = step_to_prompt(next_step)
                                    send_message(phone_number, prompt)
                                else:
                                    send_message(
                                        phone_number,
                                        "✅ ¡Todos los documentos han sido recibidos correctamente!",
                                    )

                            # Handle image upload
                            elif message_type == "image":
                                current_step = document_steps[step_index]
                                image_id = message["image"]["id"]
                                file_data = get_image(image_id)

                                filename = f"{current_step}.jpg"
                                folder_path = f"received_images/{phone_number}"
                                os.makedirs(folder_path, exist_ok=True)
                                file_path = os.path.join(folder_path, filename)

                                with open(file_path, "wb") as f:
                                    f.write(file_data)
                                    user_state_files_holder[phone_number][
                                        current_step
                                    ] = file_path
                                print(f"✅ Image saved as {file_path}")

                                # Move to next step
                                user_state[phone_number] += 1
                                if user_state[phone_number] < len(document_steps):
                                    next_step = document_steps[user_state[phone_number]]
                                    prompt = step_to_prompt(next_step)
                                    send_message(phone_number, prompt)
                                else:
                                    send_message(
                                        phone_number,
                                        "✅ ¡Todos los documentos han sido recibidos correctamente!",
                                    )

                            else:
                                send_message(
                                    phone_number, "📸 Por favor envía una *imagen*."
                                )
        return "ok", 200


# ===========
# UTILITIES
# ===========


def send_message(to_number, message):
    url = "https://graph.facebook.com/v19.0/659482707251850/messages"
    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }
    data = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "type": "text",
        "text": {"body": message},
    }
    response = requests.post(url, headers=headers, json=data)
    print(f"📤 Sent message: {message} | Status: {response.status_code}")


def get_image(media_id):
    # Step 1: Get image URL
    url = f"https://graph.facebook.com/v19.0/{media_id}"
    headers = {"Authorization": f"Bearer {ACCESS_TOKEN}"}
    response = requests.get(url, headers=headers)
    file_url = response.json().get("url")

    # Step 2: Download the file
    response = requests.get(file_url, headers=headers)
    return response.content


def step_to_prompt(step):
    prompts = {
        "INE_front": "📸 Por favor envía la *foto del frente de tu INE*.",
        "INE_back": "📸 Ahora envía la *foto del reverso de tu INE*.",
        "LICENSE_front": "📸 Envíame la *foto del frente de tu licencia de conducir*.",
        "LICENSE_back": "📸 Ahora envía la *foto del reverso de tu licencia*.",
        "INVOICE_front": "📸 Envíame la *foto del frente de la factura del auto*.",
        "INVOICE_back": "📸 Envíame la *foto del reverso de la factura*.",
        "ODOMETRO": "Finalmente, envia el kilometraje de tu vehiculo.",
    }
    return prompts.get(step, "📸 Por favor envía una imagen.")


def predict_validate(ine: str, tarjeta: str, factura: str, odometro: int):
    llm = get_validation_pipeline()

    return llm.invoke(
        input={
            "factura": factura,
            "tc": tarjeta,
            "ine": ine,
            "odometro": odometro,
        }
    )


# ===========
# START APP
# ===========
if __name__ == "__main__":
    app.run(port=5000)
