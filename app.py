# ===== Import Libraries =====
from flask import Flask, request, jsonify
from flask_cors import CORS  # <--- NEW IMPORT
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# ===== Initialize Flask App =====
app = Flask(__name__)
CORS(app)  # <--- CRITICAL FIX: This enables the connection!

# ===== Load Trained Model =====
# Ensure this file is in the same folder as app.py
model = load_model("skin_disease_model.h5")

# ===== Class Labels =====
classes = [
    "Acne",
    "Actinic_Keratosis",
    "Benign_tumors",
    "Bullous",
    "Candidiasis",       # <--- FIXED: Added comma here
    "DrugEruption",
    "Eczema",
    "Infestations_Bites",
    "Lichen",
    "Lupus",
    "Moles",
    "Psoriasis",
    "Rosacea", 
    "Seborrh_Keratoses", 
    "SkinCancer",
    "Sun_Sunlight_Damage", 
    "Tinea", 
    "Unknown_Normal",
    "Vascular_Tumors",
    "Vasculitis",
    "Vitilgo", 
    "Warts"
]

IMG_SIZE = 128

# ===== Prediction Function =====
def predict_image(img_path):
    try:
        img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        class_index = np.argmax(prediction, axis=1)[0]
        
        # Check if index is within bounds of our list
        if class_index >= len(classes):
            return "Unknown Class", 0.0

        confidence = float(np.max(prediction)) * 100
        return classes[class_index], round(confidence, 2)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Error", 0.0

# ===== API Route =====
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"})

    # Create uploads folder
    upload_folder = "uploads"
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)

    filepath = os.path.join(upload_folder, file.filename)
    file.save(filepath)

    label, confidence = predict_image(filepath)

    return jsonify({
        "prediction": label,
        "confidence": confidence
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
