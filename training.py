import os
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall
import numpy as np

# Import the function to download the dataset from utils
from utils import download_dataset

# Function to preprocess the image for prediction
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to match the input size of the model
    image = np.array(image) / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to train the model
def train_model(train_dir, val_dir, model_save_path='models/fish_classifier.h5'):
    # Image Preprocessing and Augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    # Load data with flow_from_directory method
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )

    # Load MobileNetV2 model pre-trained on ImageNet without the top layers
    base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Freeze the base model layers to prevent retraining
    base_model.trainable = False

    # Add custom layers on top of the pre-trained model
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(1024, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(train_generator.num_classes, activation='softmax')  # Output layer with number of classes
    ])

    # Compile the model with the Adam optimizer and categorical crossentropy loss
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy', Precision(), Recall()])

    # Train the model
    model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=20,
        validation_data=val_generator,
        validation_steps=val_generator.samples // val_generator.batch_size
    )

    # Save the trained model
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

    return model, val_generator

# Function to evaluate the model and save the scores in a JSON file
def evaluate_model(model, val_generator, scores_path='models/scores.json'):
    # Evaluate the model on the validation data
    scores = model.evaluate(val_generator)

    # Extract the metrics
    accuracy = scores[1]
    precision = scores[2]
    recall = scores[3]

    # Save the evaluation scores in a JSON file
    score_dict = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall
    }

    with open(scores_path, 'w') as f:
        json.dump(score_dict, f)

    print(f"Model evaluation scores saved to {scores_path}")

# Main execution
def main():
    # Set directories
    train_dir = 'artifacts/data_ingestion/Freshwater Fish Disease Aquaculture in south asia/Train'
    test_dir = 'artifacts/data_ingestion/Freshwater Fish Disease Aquaculture in south asia/Test'

    # Download the dataset (only if not already downloaded)
    download_dataset()

    # Train the model and get the trained model and validation generator
    model, val_generator = train_model(train_dir, test_dir)

    # Evaluate the model and save the scores
    evaluate_model(model, val_generator)

if __name__ == '__main__':
    main()
