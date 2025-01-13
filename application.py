import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from fish_disease_cls.utils import download_dataset, preprocess_image

download_dataset()

# Load the trained model
model = tf.keras.models.load_model(r'E:\Trekathon\fish-disease-detection\fish_disease_cls\models\fish_classifier.h5')

# Load the labels (replace with your actual dataset class names)
labels = ['Bacterial diseases - Aeromoniasis', 'Bacterial gill disease', 'Bacterial Red disease', 'Bacterial Red disease', 'Bacterial Red disease', 'Healthy Fish', 'Healthy Fish', 'Parasitic diseases', 'Parasitic diseases', 'Viral diseases White tail disease']  # Example, replace with actual classes

# Streamlit app layout
st.title('Fish Image Classification')
st.write('Upload an image of a fish to get the prediction.')

# Image uploader
uploaded_image = st.file_uploader("Choose a fish image", type=['jpg', 'png', 'jpeg'])

if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image and make predictions
    image_preprocessed = preprocess_image(image)
    predictions = model.predict(image_preprocessed)
    predicted_class = labels[np.argmax(predictions)]

    # Display the prediction
    st.write(f'Prediction: {predicted_class}')
