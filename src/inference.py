import torch
import pandas as pd
from torchvision import transforms
from PIL import Image
import numpy as np
from src.utils import extract_patch

def predict(model, image, spots_df, device='cuda'):
    model.eval()
    test_patches = []
    spot_ids = spots_df['id'].astype(str).tolist()
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    with torch.no_grad():
        for _, row in spots_df.iterrows():
            x, y = int(row['x']), int(row['y'])
            patch = extract_patch(image, x, y)
            img = transform(patch.astype(np.uint8)).unsqueeze(0).to(device)
            pred = model(img).cpu().numpy().flatten()
            test_patches.append(pred)

    return pd.DataFrame(test_patches, columns=[f'C{i}' for i in range(1, 36)]).assign(ID=spot_ids)