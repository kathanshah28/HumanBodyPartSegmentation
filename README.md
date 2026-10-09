# Human Body Part Segmentation

A Flask-based web application for segmenting human body parts from an uploaded image using a PyTorch deep learning model.

## Overview

This project allows a user to upload an image, runs it through a trained Multi-UNet segmentation model, and displays the predicted segmentation mask. The interface is implemented with Flask templates and the model is loaded from a checkpoint that is downloaded automatically from Hugging Face if it is not already present.

## Features

- Upload an image through a simple web form
- Preprocess the image for model inference
- Run human body part segmentation using a PyTorch model
- Save and display the predicted segmentation result
- Lightweight Flask web app suitable for deployment

## Project Structure

```text
HumanBodyPartSegmentation/
├── api/
│   ├── MultiUnetModel.py   # Model definition
│   └── app.py              # Flask application and inference pipeline
├── templates/
│   ├── upload.html          # Upload form page
│   └── result.html          # Prediction result page
├── .gitignore
├── Procfile
├── README.md
├── requirements.txt
├── vercel.json
└── model checkpoint downloaded at runtime
```

## Tech Stack

- Python
- Flask
- PyTorch
- NumPy
- Pillow
- Matplotlib
- Gunicorn

## Installation

1. Clone the repository:

```bash
git clone https://github.com/kathanshah28/HumanBodyPartSegmentation.git
cd HumanBodyPartSegmentation
```

2. Create and activate a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate    # On macOS/Linux
venv\Scripts\activate       # On Windows
```

3. Install the dependencies:

```bash
pip install -r requirements.txt
```

## Running the App

From the project root:

```bash
python api/app.py
```

Then open the app in a browser at:

```text
http://localhost:8080
```

The app will automatically download the model checkpoint from the configured Hugging Face URL if it does not already exist locally.

## How It Works

1. The user uploads an image.
2. The Flask app stores the image in a temporary upload directory.
3. The image is converted to grayscale and resized to 512x512.
4. The Multi-UNet model predicts a segmentation map.
5. The result is saved as an image and displayed in the browser.

