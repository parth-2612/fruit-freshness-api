# app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Setup upload directory
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load pre-trained MobileNetV2 model
model = MobileNetV2(weights='imagenet')

# Allowed fruits and their nutrition info
ALLOWED_FRUITS = {
    "banana", "apple", "strawberry", "mango", "orange", 
    "pineapple", "grape", "watermelon", "pomegranate", 
    "lemon", "peach"
}

NUTRITION = {
    "banana": {"calories": 89, "carbs": "23g", "fiber": "2.6g", "vitaminC": "15%"},
    "apple": {"calories": 52, "carbs": "14g", "fiber": "2.4g", "vitaminC": "7%"},
    "strawberry": {"calories": 32, "carbs": "7.7g", "fiber": "2g", "vitaminC": "97%"},
    "orange": {"calories": 62, "carbs": "15.4g", "fiber": "3.1g", "vitaminC": "116%"},
    "mango": {"calories": 60, "carbs": "15g", "fiber": "1.6g", "vitaminC": "44%"},
    "pineapple": {"calories": 50, "carbs": "13g", "fiber": "1.4g", "vitaminC": "79%"},
    "grape": {"calories": 69, "carbs": "18g", "fiber": "0.9g", "vitaminC": "4%"},
    "lemon": {"calories": 29, "carbs": "9g", "fiber": "2.8g", "vitaminC": "88%"},
    "peach": {"calories": 39, "carbs": "10g", "fiber": "1.5g", "vitaminC": "10%"},
    "watermelon": {"calories": 30, "carbs": "8g", "fiber": "0.4g", "vitaminC": "13%"},
    "pomegranate": {"calories": 83, "carbs": "19g", "fiber": "4g", "vitaminC": "17%"}
}

CONFIDENCE_THRESHOLD = 0.65  # Minimum confidence to accept prediction

def predict_image(img_path):
    """Predict the fruit from the given image."""
    img = Image.open(img_path).convert('RGB').resize((224, 224))
    img_array = np.expand_dims(np.array(img), axis=0)
    img_array = preprocess_input(img_array)

    preds = model.predict(img_array)
    decoded_preds = decode_predictions(preds, top=5)[0]

    for pred in decoded_preds:
        label = pred[1].lower()
        confidence = float(pred[2])
        if label in ALLOWED_FRUITS and confidence >= CONFIDENCE_THRESHOLD:
            return label.title(), confidence

    return "Unknown Fruit", 0.0

@app.route("/upload", methods=["POST"])
def upload_image():
    """Handle image uploads and return prediction."""
    if 'image' not in request.files:
        return jsonify({"error": "No image file in request"}), 400

    image = request.files['image']
    filename = secure_filename(image.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    image.save(filepath)

    label, confidence = predict_image(filepath)

    # Dummy freshness logic based on confidence
    fresh = confidence > 0.7

    nutrition = NUTRITION.get(label.lower(), {})

    return jsonify({
        "fruit": label,
        "confidence": confidence,
        "fresh": fresh,
        "nutrition": nutrition
    })

if __name__ == "__main__":
    app.run(debug=True)
