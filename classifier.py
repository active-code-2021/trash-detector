from PIL import Image
import torch
from pathlib import Path
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import device_data_loader
loaded_model = torch.load('model.pth')



transformations = transforms.Compose(
    [transforms.Resize((256, 256)), transforms.ToTensor()])

data_dir = './Garbage classification/Garbage classification'
dataset = ImageFolder(data_dir, transform=transformations)


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
device = get_default_device()
def predict_image(img, model):
    # Convert to a batch of 1
    xb = device_data_loader.to_device(img.unsqueeze(0), device)
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    prob, preds = torch.max(yb, dim=1)
    # Retrieve the class label
    return dataset.classes[preds[0].item()]

def predict_external_image(image_name):
    image = Image.open(Path('./uploads/' + image_name))
    example_image = transformations(image)
    plt.imshow(example_image.permute(1, 2, 0))
    result = predict_image(example_image, loaded_model)
    print(result)
    if(result == 'metal' or result == "plastic"):
        res = "כתום"
    elif result == "glass":
        res = "סגול"
    elif(result == "paper" or result == "cardboard"):
        res = "כחול"
    else:
        res = "ירוק"
    return res
