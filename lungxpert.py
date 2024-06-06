import tensorflow
import torch
import torchvision
from torch import nn

from tensorflow.keras.models import load_model
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np

# Import function to make predictions on images and plot them 
from modular.predictions import pred_and_plot_image


def prediction(img_path):
    model_name = "vitmodel.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_file = torchvision.models.vit_b_16().to(device)
    class_names = ['COVID19','NORMAL', 'PNEUMONIA', 'TUBERCULOSIS']
    model_file.heads = nn.Linear(in_features=768, out_features=len(class_names)).to(device)
    model_file.load_state_dict(torch.load(f"model/{model_name}", map_location=torch.device(device)))

    # Predict on custom image
    return pred_and_plot_image(model=model_file,
                        image_path=img_path,
                        class_names=class_names, device=torch.device(device))


def lungDetector(img_path):
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    # Load the model
    model = load_model("model/keras_Model.h5", compile=False)

    # Load the labels
    class_names = open("model/labels.txt", "r").readlines()

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    image = Image.open(img_path).convert("RGB")

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    class_true = "Lung X-ray"

    # Print prediction and confidence score
    print("Class:", class_name[2:])
    print("Confidence Score:", confidence_score)

    if(class_name[2:].strip() == "Lung X-ray"):
        return True
    
    return False
    
