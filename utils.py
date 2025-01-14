import os
import zipfile
import gdown  # Use gdown to download from Google Drive
from PIL import Image
import numpy as np

# Function to preprocess the image for prediction
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to match the input size of the model
    image = np.array(image) / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to download and unzip the dataset from Google Drive
def download_dataset():
    # Path to store the dataset after downloading and unzipping
    extraction_dir = 'artifacts/data_ingestion/'
    
    # Create the directory if it doesn't exist
    if not os.path.exists(extraction_dir):
        os.makedirs(extraction_dir)

    # Check if the dataset folder already exists to avoid downloading again
    if not os.path.exists(os.path.join(extraction_dir, 'train')):  # Assuming 'train' is inside the unzipped folder
        dataset_url = "https://drive.google.com/uc?export=download&id=1SQZ5_wmTlgj5i04qtyMo5BnXbcfq84D-"  # Replace with your actual file ID
        output_path = "dataset.zip"
        
        # Download the dataset using gdown
        print(f"Downloading dataset from {dataset_url}...")
        gdown.download(dataset_url, output_path, quiet=False)

        # Unzip the dataset into the 'artifacts/data_ingestion' folder
        with zipfile.ZipFile(output_path, 'r') as zip_ref:
            zip_ref.extractall(extraction_dir)
        print(f"Dataset downloaded and extracted to '{extraction_dir}' folder.")
    else:
        print(f"Dataset already exists in {extraction_dir}. Skipping download.")
