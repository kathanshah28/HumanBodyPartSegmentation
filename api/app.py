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

# Setup logging
logging.basicConfig(
    filename="app_debug.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# # Load the model
# model_path = "checkpoints_new_arctitecture_512512_adam/final_model.pth"
# with open(model_path, "rb") as f:
#     model_state_dict = pickle.load(f)

# model = MultiUNet(n_classes=24, input_channels=1)
# model.load_state_dict(model_state_dict)
# model.eval()

# # Load the model
checkpoints_dir = "checkpoints_new_arctitecture_512512_adam"
# with open(model_path, "rb") as f:
#     model_state_dict = pickle.load(f)



model = MultiUNet(n_classes=24, input_channels=1)
checkpoint = torch.load(os.path.join(checkpoints_dir,"model_epoch_13.pth"),map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# start_epoch = checkpoint['epoch'] + 1  # Resume from next epoch
# model.load_state_dict(model_state_dict)
model.eval()


# Initialize Flask app

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
PREDICTIONS_FOLDER = os.path.join(BASE_DIR, 'predictions')

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, 'templates'))
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PREDICTIONS_FOLDER'] = PREDICTIONS_FOLDER

# app = Flask(__name__, template_folder='../templates')
# UPLOAD_FOLDER = 'uploads'
# PREDICTIONS_FOLDER = 'predictions'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(PREDICTIONS_FOLDER, exist_ok=True)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['PREDICTIONS_FOLDER'] = PREDICTIONS_FOLDER

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


if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get("PORT", 8080)))