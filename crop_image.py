import numpy as np
from typing import Union, Tuple, List

def crop_image_numpy(image: np.ndarray, bbox: List[int]) -> np.ndarray:
    """
    Crop image in numpy array format using bounding box coordinates.
    
    Args:
        image: Input image as numpy array with shape (H, W, C) or (H, W)
        bbox: Bounding box coordinates [left, top, right, bottom]
    
    Returns:
        Cropped image as numpy array
    """
    left, top, right, bottom = bbox
    
    if len(image.shape) == 3:
        height, width, _ = image.shape
    else:
        height, width = image.shape
    
    left = np.clip(left, 0, width)
    top = np.clip(top, 0, height)
    right = np.clip(right, 0, width)
    bottom = np.clip(bottom, 0, height)
    
    if left >= right or top >= bottom:
        raise ValueError("Invalid bounding box coordinates")
    
    cropped_image = image[top:bottom, left:right]
    
    return cropped_image
