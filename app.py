from flask import Flask
from flask import request
from werkzeug.utils import secure_filename
import lungxpert
import os
from flask_cors import CORS 
import pydicom
import numpy as np
import cv2

app = Flask(__name__, static_folder="output")
cors = CORS(app, resources={r"*": {"origins": "*"}})

@app.route("/")
def welcome():
    return "<p>LungXpert</p>"

@app.route("/upload", methods=["POST"])
def upload_file():
    if request.method == "POST":
        file = request.files["image"]
        fileName = f"output/image/{secure_filename(file.filename)}"

        file.save(fileName)

        is_lung = lungxpert.lungDetector(fileName)
        if is_lung:
            prediction = lungxpert.prediction(fileName)

            return {
                "status": "OK",
                "prediction": prediction
            }
        
        else:
            return {
                "status": "NOT OK",
                "message": "Not Lung"
            }

        if os.path.exists(fileName):
            os.remove(fileName)

@app.route("/convert-dcm", methods=["POST"])
def converter():
    file = request.files['img']

    file.save(f"./output/{secure_filename(file.filename)}")

    dicom = pydicom.read_file(f"./output/{secure_filename(file.filename)}")
    img = dicom.pixel_array
    scaled_img = (np.maximum(img,0) / img.max()) * 255.0
    file = f"/output/{secure_filename(file.filename)}.png"
    cv2.imwrite(file, scaled_img)

    return {
        "status": "success",
        "message": "File converted successfully",
        "file": file
    }