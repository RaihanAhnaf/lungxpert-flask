import torch
import torchvision
from torch import nn
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
