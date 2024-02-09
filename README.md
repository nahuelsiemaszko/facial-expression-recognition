# Facial Expression Recognition

## Overview

This Python project utilizes a Convolutional Neural Network (CNN) to first train and test a model with a provided dataset, and then use it to capture and recognize facial expressions in real-time through the device's camera.

The model is designed to identify the following seven facial expressions:
- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

The project uses TensorFlow for model construction and training, and OpenCV for image capturing and recognition.

## Usage

1. Ensure Python 3 (or later) is installed. If not, download and install it from [python.org](https://www.python.org/downloads/).

2. Using a terminal or command prompt, clone the repository:

    ```bash
    git clone https://github.com/nahuelsiemaszko/facial-expression-recognition.git
    ```

3. Set up a virtual environment:

    ```bash
    python -m venv venv
    source venv/bin/activate  # For Linux/Mac
    # or
    .\venv\Scripts\activate  # For Windows
    ```

4. Install dependencies:

    ```bash
    pip install numpy opencv-python pillow scikit-learn tensorflow
    ```

5. Train the model (optional):

- First, download the [FER-2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013) and save it inside your project folder.

- Then, delete the "model.h5" file and run:

    ```bash
    python main.py
    ```

6. Use the model:

    ```bash
    python use.py
    ```

- Press 'q' to exit when done.

7. Deactivate the virtual environment:

    ```bash
    deactivate
    ```
