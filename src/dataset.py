import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

class SpotDataset(Dataset):
    def __init__(self, patches, labels, transform=None):
        self.patches = patches
        self.labels = labels
        self.transform = transform or transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        
    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        img = self.patches[idx].astype(np.uint8)
        img = self.transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img, label