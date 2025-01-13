import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
import subprocess
from PIL import Image
from flask_cors import CORS, cross_origin
from fish_disease_cls.utils import preprocess_image, download_dataset
import base64
from io import BytesIO

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load the trained model once the app starts
model = tf.keras.models.load_model(r'D:\Aminur\fish-disease-detection\fish-disease-detection\fish_disease_cls\models\fish_classifier.h5')

# Load the labels (replace with your actual dataset class names)
labels = ['Bacterial diseases - Aeromoniasis', 'Bacterial gill disease', 'Bacterial Red disease', 'Fungal diseases Saprolegniasis',
          'Healthy Fish', 'Parasitic diseases', 'Viral diseases White tail disease']  # Update this with actual classes


@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('home.html')

@app.route("/index")
def index():
    return render_template("index.html")

@app.route("/train", methods=['GET','POST'])
@cross_origin()
def trainRoute():
    try:
        subprocess.run(["python", "main.py"], check=True)
        return "Training done successfully!"
    except subprocess.CalledProcessError as e:
        return jsonify({"error": "Training failed", "details": str(e)}), 500

@app.route("/result")
def result():
    prediction_result = request.args.get('result', default=None)
    return render_template("result.html", prediction_result=prediction_result)

# API route for health check
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'Healthy'}), 200

# API route to classify the fish image@app.route('/predict', methods=['POST'])
@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    try:
        # Check if an image file is provided
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        # Read the uploaded image file
        image_file = request.files['image']
        image = Image.open(image_file)

        # Preprocess the image for prediction
        image_preprocessed = preprocess_image(image)

        # Make predictions
        predictions = model.predict(image_preprocessed)
        predicted_class = labels[np.argmax(predictions)]

        # Return the prediction
        return jsonify(predicted_class), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Download the dataset (if not already present)
    download_dataset()
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
