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

1. Clone the repository:

    ```bash
    git clone https://github.com/nahuelsiemaszko/portfolio.git
    cd portfolio/facial-expression-recognition
    ```

2. Set up a virtual environment:

    ```bash
    python -m venv venv
    source venv/bin/activate  # For Linux/Mac
    # or
    .\venv\Scripts\activate  # For Windows
    ```

3. Install dependencies:

    ```bash
    pip install numpy opencv-python pillow scikit-learn tensorflow
    ```

4. Download the [FER-2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013) and save it inside your project folder.

5. Train the model or skip this step to use a pre-trained model:

- To train the model, first delete the "model.h5.zip" file and run:

    ```bash
    python main.py
    ```

6. Use the model
   
- To use the model, first, if you skipped step 5, unzip the 'model.h5.zip' file, and then run the following command. If you didn't skip step 5, run the command directly:

    ```bash
    python use.py
    ```

7. Press 'q' to exit when done.

8. Deactivate the virtual environment:

    ```bash
    deactivate
    ```
