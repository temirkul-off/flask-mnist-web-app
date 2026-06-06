from flask import Flask, render_template, request, jsonify
from keras.models import load_model
import numpy as np
from PIL import Image
import io
import base64
import os
import json
import psycopg2
from psycopg2.extras import Json
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

DB_CONFIG = {
    "host": os.getenv("PGHOST"),
    "port": os.getenv("PGPORT"),
    "dbname": os.getenv("PGDATABASE"),
    "user": os.getenv("PGUSER"),
    "password": os.getenv("PGPASSWORD"),
}


def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)

print("DB_HOST =", os.getenv("PGHOST"))
print("DB_NAME =", os.getenv("PGDATABASE"))
print("DB_USER =", os.getenv("PGUSER"))
print("DB_PASS =", "***" if os.getenv("PGPASSWORD") else "NOT FOUND")

MODEL_PATH = "m.keras"
model = load_model(MODEL_PATH)
input_shape = model.input_shape


@app.route("/")
def index():
    return render_template("index.html")


def preprocess_pil_image(img):
    img = img.convert("L")
    print(f"Original size: {img.size}")

    size = (28, 28)
    new = img.resize(size, Image.LANCZOS)

    arr = np.array(new).astype("float32") / 255.0
    flat = arr.reshape(784,)
    x = np.expand_dims(flat, axis=0)

    return x, flat.tolist()


@app.route("/predict", methods=["POST"])
def predict():
    try:
        print("Received request for prediction", flush=True)

        data = request.get_json(silent=True)

        if not data or "image" not in data:
            return jsonify({"error": "No image sent"}), 400

        img_b64 = data["image"]

        if not img_b64.startswith("data:"):
            return jsonify({"error": "Неверный формат изображения"}), 400

        _, encoded = img_b64.split(",", 1)
        image_bytes = base64.b64decode(encoded)

        pil_img = Image.open(io.BytesIO(image_bytes))

        x, image_vector = preprocess_pil_image(pil_img)

        probs = model.predict(x)
        probs = probs[0]
        pred_idx = int(np.argmax(probs))

        top3_idx = probs.argsort()[-3:][::-1]
        top3 = [{"class": int(i), "prob": float(probs[i])} for i in top3_idx]

        response = {
            "pred": pred_idx,
            "probs": [float(p) for p in probs.tolist()],
            "top3": top3,
            "image_vector": image_vector
        }

        print("Response:", response["pred"], response["probs"], response["top3"], flush=True)
        return jsonify(response)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/feedback", methods=["POST"])
def feedback():
    try:
        data = request.get_json(silent=True)

        if not data:
            return jsonify({"error": "No JSON sent"}), 400

        required_fields = ["name", "email", "is_correct",
                           "image_data_url", "pred", "probs", "top3"]
        missing = [field for field in required_fields if field not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

        name = data["name"].strip()
        email = data["email"].strip()
        is_correct = bool(data["is_correct"])
        correct_answer = data.get("correct_answer")
        comment = data.get("comment", "").strip()
        image_data_url = data["image_data_url"]
        pred = int(data["pred"])
        probs = data["probs"]
        top3 = data["top3"]

        if not image_data_url.startswith("data:image/"):
            return jsonify({"error": "Invalid image data"}), 400

        _, encoded = image_data_url.split(",", 1)
        image_bytes = base64.b64decode(encoded)

        pil_img = Image.open(io.BytesIO(image_bytes))
        x, image_vector = preprocess_pil_image(pil_img)

        if not is_correct:
            if correct_answer is None or str(correct_answer).strip() == "":
                return jsonify({"error": "Correct answer is required when the answer is wrong"}), 400
            correct_answer = int(correct_answer)
        else:
            correct_answer = None

        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute(
            """
            INSERT INTO feedback_entries
            (name, email, is_correct, correct_answer, comment,
             image_data_url, image_vector, predicted_class, probs, top3)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                name,
                email,
                is_correct,
                correct_answer,
                comment if comment else None,
                image_data_url,
                Json(image_vector),
                pred,
                Json(probs),
                Json(top3),
            )
        )

        conn.commit()
        cur.close()
        conn.close()

        return jsonify({"ok": True})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run()
