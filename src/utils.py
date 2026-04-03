import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def load_image(path: str, monochromatic: bool=True) -> np.ndarray:
    """
    Loads a PNG image from the disk.
    """
        
    if monochromatic:
        # Loads image as a single 2D matrix (M x N)
        flag = cv2.IMREAD_GRAYSCALE
    else:
        # Loads as a 3D matrix (M x N x 3) in BGR format
        flag = cv2.IMREAD_COLOR
    
    
    img = cv2.imread(path, flag)
    
    if img is None:
        raise FileNotFoundError(f"Could not find image at {path}")
    
    return img

def save_image(img: np.ndarray, folder: str, filename: str) -> None:
    """
    Saves a NumPy array as a PNG image.
    """
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Folder does not exist: {folder}")
    
    full_path = os.path.join(folder, filename)
    
    # checks the file extension to decide the format
    image = cv2.imwrite(full_path, img)
    
    if not image:
        print(f"Failed to save image to {full_path}")
    else:
        print(f"Image saved: {full_path}")

def display_results(original: np.ndarray, processed: np.ndarray, title="Result"):
    """
    Displays images side-by-side using Matplotlib.
    """
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(original, cmap='gray' if len(original.shape) == 2 else None)
    
    plt.subplot(1, 2, 2)
    plt.title(title)
    plt.imshow(processed, cmap='gray' if len(processed.shape) == 2 else None)
    
    plt.show()