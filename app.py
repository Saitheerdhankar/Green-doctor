from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import pickle
import os
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("model.h5")

# Load labels
with open("labels.pkl", "rb") as f:
    labels = pickle.load(f)
labels = {v: k for k, v in labels.items()}  # reverse mapping

# Advice dictionary
disease_advice = {
    "Apple___Black_rot": "Remove infected fruits, apply fungicide (Captan, Mancozeb).",
    "Apple___healthy": "Your crop is healthy! 🌱 Keep monitoring regularly.",
    "Apple___Cedar_apple_rust": "Prune galls, use resistant varieties, apply fungicides.",
    "Apple___Scab": "Apply fungicide (Captan, Mancozeb), remove fallen leaves."
}

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    advice = None
    if request.method == "POST":
        file = request.files["file"]
        filepath = os.path.join("static", file.filename)
        file.save(filepath)

        # Preprocess image
        img = image.load_img(filepath, target_size=(128, 128))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array)
        predicted_class = labels[np.argmax(prediction)]

        # Fetch advice
        advice = disease_advice.get(predicted_class, "No advice available")

        result = f"Disease: {predicted_class}"

    return render_template("index.html", result=result, advice=advice)

if __name__ == "__main__":
    app.run(debug=True)
