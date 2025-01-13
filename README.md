# Fish Classifier Project

This is a fish classifier model built using TensorFlow's MobileNetV2 architecture. It uses transfer learning to classify fish images based on a dataset provided through Google Drive.

## Setup Instructions

1. **Clone the repository**:

    ```bash
    git clone https://github.com/Ronit-Bhowmick/Fish-Disease-Classifier.git
    cd fish-classifier
    ```

2. **Create and activate a virtual environment** (optional but recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate  # On macOS/Linux
    venv\Scripts\activate     # On Windows
    ```

3. **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

4. **Install the package**:

    ```bash
    pip install fish_disease_cls
    ```

## Running the Project

- **Dataset Link:**: 

- **To train the model**:

    ```bash
    training.py
    ```

## Files and Directories

- `training.py`: Script to train the model.
- `utils.py`: Utility functions for downloading and preprocessing the dataset.
- `models/`: Contains the saved model and evaluation scores.
- `artifacts/`: Contains the dataset after downloading and extraction.

## Evaluation

After training, the model’s evaluation results (accuracy, precision, recall) will be saved in `models/scores.json`.

