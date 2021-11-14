import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.models as models
from image_classifier import ImageClassifier

class ResNet(ImageClassifier):
        def __init__(self, dataset):
            super().__init__()
            # Use a pretrained model
            self.network = models.resnet50(pretrained=True)
            # Replace last layer
            num_ftrs = self.network.fc.in_features
            self.network.fc = nn.Linear(num_ftrs, len(dataset.classes))
        
        def forward(self, xb):
            return torch.sigmoid(self.network(xb))

   