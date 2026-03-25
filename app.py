from flask import Flask, render_template, request, jsonify
from keras.models import load_model
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)

MODEL_PATH = "m.keras"
model = load_model(MODEL_PATH)
input_shape = model.input_shape

@app.route("/")
def index():
    return render_template("index.html")

def preprocess_pil_image(img):

    img = img.convert('L')
    print(f"Original size: {img.size}")

    size = (28, 28)

    new = img.resize(size, Image.LANCZOS)

    arr = np.array(new).reshape(784,).astype('float32') / 255
    x = np.expand_dims(arr, axis=0)

    return x

@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json(force=True)
    img_b64 = data.get("image", "")

    if not img_b64.startswith("data:"):
        return jsonify({"error": "Неверный формат изображения"}), 400

    _, encoded = img_b64.split(",", 1)
    image_bytes = base64.b64decode(encoded)
    pil_img = Image.open(io.BytesIO(image_bytes))

    x = preprocess_pil_image(pil_img)

    probs = model.predict(x)
    probs = probs[0]
    pred_idx = int(np.argmax(probs))

    top3_idx = probs.argsort()[-3:][::-1]
    top3 = [{"class": int(i), "prob": float(probs[i])} for i in top3_idx]

    response = {
        "pred": pred_idx,
        "probs": [float(p) for p in probs.tolist()],
        "top3": top3
    }
    return jsonify(response)

if __name__ == "__main__":
    app.run()