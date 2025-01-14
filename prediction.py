from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = load_model('models/fish_classifier.h5')

# Load and preprocess a new image
img = image.load_img(r'E:\Trekathon\fish-disease-detection\fish_disease_cls\artifacts\data_ingestion\Freshwater Fish Disease Aquaculture in south asia\Test\Healthy Fish\Healthy Fish (1).jpeg', target_size=(224, 224))
img_array = np.expand_dims(image.img_to_array(img), axis=0) / 255.0

# Make a prediction
prediction = model.predict(img_array)

# Output the prediction
print("Predicted class:", np.argmax(prediction))
