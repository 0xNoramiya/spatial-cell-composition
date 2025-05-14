import numpy as np

def extract_patch(image, x, y, patch_size=224):
    half = patch_size // 2
    x_min = max(x - half, 0)
    x_max = min(x + half, image.shape[1])
    y_min = max(y - half, 0)
    y_max = min(y + half, image.shape[0])
    patch = image[y_min:y_max, x_min:x_max]

    if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
        pad_y = patch_size - patch.shape[0]
        pad_x = patch_size - patch.shape[1]
        patch = np.pad(patch, ((0, pad_y), (0, pad_x), (0, 0)), mode='constant')
    
    return patch