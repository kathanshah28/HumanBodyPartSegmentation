from flask import Flask, request, jsonify, render_template, send_file, url_for
import torch
# import pickle
import numpy as np
from MultiUnetModel import MultiUNet
# import random
from werkzeug.utils import secure_filename
import os
import matplotlib.pyplot as plt
from PIL import Image
import logging
import traceback
import requests
import sys

# Setup logging
logging.basicConfig(
    # filename="app_debug.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),  # <== This sends logs to stdout
    ]
)


# model = MultiUNet(n_classes=24, input_channels=1)
# checkpoint = torch.load(os.path.join(checkpoints_dir,"model_epoch_13.pth"),map_location='cpu')
# model.load_state_dict(checkpoint['model_state_dict'])
# # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# # start_epoch = checkpoint['epoch'] + 1  # Resume from next epoch
# # model.load_state_dict(model_state_dict)
# model.eval()


# Initialize Flask app

# BASE_DIR = os.getcwd()

UPLOAD_FOLDER = '/tmp/uploads'
PREDICTIONS_FOLDER = '/tmp/predictions'
CHECKPOINTS_DIR = '/tmp/checkpoints'

# UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
# PREDICTIONS_FOLDER = os.path.join(BASE_DIR, 'predictions')
# CHECKPOINTS_DIR = os.path.join(BASE_DIR, 'checkpoints')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PREDICTIONS_FOLDER, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)


# app = Flask(__name__, template_folder=os.path.join(BASE_DIR, 'templates'))
app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PREDICTIONS_FOLDER'] = PREDICTIONS_FOLDER

MODEL_URL = "https://huggingface.co/kathan2813/HumanBodySegmentationVitonDataset/resolve/main/model_epoch_13.pth"
MODEL_PATH = os.path.join(CHECKPOINTS_DIR, "model_epoch_13.pth")

if not os.path.exists(MODEL_PATH):
    try:
        logging.info("Downloading model checkpoint...")
        response = requests.get(MODEL_URL, timeout=30)
        response.raise_for_status()
        with open(MODEL_PATH, 'wb') as f:
            f.write(response.content)
        logging.info("Model downloaded successfully.")
    except Exception as e:
        logging.error(f"Failed to download model: {e}")
        raise RuntimeError("Model download failed.")

# model = MultiUNet(n_classes=24, input_channels=1)
# checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
# model.load_state_dict(checkpoint['model_state_dict'])
# model.eval()

# Load model
try:
    model = MultiUNet(n_classes=24, input_channels=1)
    checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
except Exception as e:
    logging.error(f"Model loading error: {e}")
    raise RuntimeError("Model loading failed.")

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        logging.debug("Received POST request for file upload.")
        # Check if the post request has the file part
        if 'file' not in request.files:
            logging.error("No file part in the request.")
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            logging.error("No file selected for upload.")
            return "No selected file"
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            logging.debug(f"File saved at {filepath}.")

            # Process the image and predict
            try:
                input_image = Image.open(filepath).convert('L')
                input_image = input_image.resize((512, 512))
                input_image = np.array(input_image).astype(np.float32) / 255.0
                input_image = np.expand_dims(input_image, axis=(0, 1))  # Add batch and channel dimensions
                input_tensor = torch.from_numpy(input_image)

                # Perform prediction
                with torch.no_grad():
                    output = model(input_tensor)
                    prediction = torch.argmax(output, dim=1).squeeze().cpu().numpy()

                # Save the prediction as an image
                prediction_image_path = os.path.join(app.config['PREDICTIONS_FOLDER'], f"prediction_{filename}")
                # prediction_image = Image.fromarray((prediction * 255).astype(np.uint8))
                # prediction_image.save(prediction_image_path)

                plt.imsave(prediction_image_path, prediction, cmap='viridis')

                logging.debug(f"Prediction image saved at {prediction_image_path}.")


                return render_template('result.html', prediction_image=url_for('view_prediction', filename=os.path.basename(prediction_image_path)))
            except Exception as e:
                logging.error(f"Error during image processing or prediction: {e}")
                return "Error during prediction. Check logs for details."

    return render_template('upload.html')

@app.route("/view-prediction/<path:filename>")
def view_prediction(filename):
    full_path = os.path.join(app.config['PREDICTIONS_FOLDER'], filename)
    return send_file(full_path, mimetype='image/png')

@app.route('/favicon.ico')
def favicon():
    return '', 204  # No Content

@app.errorhandler(Exception)
def handle_exception(e):
    logging.error("Unhandled Exception: %s\n%s", str(e), traceback.format_exc())
    return f"Internal Server Error {f}", 500


if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get("PORT", 8080)))