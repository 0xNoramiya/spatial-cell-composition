import torch.nn as nn
import torchvision.models as models
from torchvision.models import resnet18, ResNet18_Weights

def get_resnet18_model():
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 35)
    return model