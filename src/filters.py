import numpy as np
import cv2
from utils import save_image

def pencil_sketch(img: np.ndarray) -> np.ndarray:
    #Verifing if the image is colored
    if img.ndim == 3:
        # Convert to grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img

    img_gray_desfoque = cv2.GaussianBlur(img_gray, (21,21), 0)

    img_non_normalized = np.where(img_gray_desfoque!=0, (img_gray/img_gray_desfoque)*255, 255)

    final_result_normalized = np.where(img_non_normalized > 255, 255, img_non_normalized)

    return final_result_normalized.astype(np.uint8)


def gamma_correction(img:np.ndarray, gamma: float) -> np.ndarray:
    normalized_image = img / 255.0

    normalized_image = normalized_image**(1/gamma)

    final_result = normalized_image*255

    return final_result.astype(np.uint8)


def threshold_binarization(img: np.ndarray, threshold: int) -> np.ndarray:
    result = np.where(img > threshold, 255, 0)
    return result.astype(np.uint8)

def sepia_filter(img: np.ndarray) -> np.ndarray:
    filter = np.array([[0.393, 0.769, 0.189],
                       [0.349, 0.686, 0.168],
                       [0.272, 0.534, 0.131]])

    if img.ndim != 3:
        raise ValueError("Sepia filter requires a color image")

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    sepia_rgb = np.dot(rgb_img, filter.T)
    sepia_rgb = np.where(sepia_rgb > 255, 255, sepia_rgb)
    sepia_rgb = sepia_rgb.astype(np.uint8)

    return cv2.cvtColor(sepia_rgb, cv2.COLOR_RGB2BGR)

def monochrome_filter(img: np.ndarray) -> np.ndarray:
    if img.ndim != 3:
        raise ValueError("Monochrome filter requires a color image")

    # Weights for the BGR
    weights = np.array([0.1140, 0.5870, 0.2989], dtype=np.float32)

    monochromatic_image = np.dot(img.astype(np.float32), weights)

    final_result_normalized = np.where(monochromatic_image > 255, 255, monochromatic_image)

    return final_result_normalized.astype(np.uint8)

def bit_planes (img: np.ndarray, plane: int) -> np.ndarray:
    result = (img >> plane) & 1

    return result.astype(np.uint8) * 255 

def weighted_average_monochromatic_image(img1: np.ndarray, img2: np.ndarray, weight1: float, weight2: float) -> np.ndarray:
    if img1.shape != img2.shape:
        raise ValueError("Both images must have the same dimensions")
    
    weighted_image = (weight1 * img1) + (weight2 * img2)

    weighted_image = np.where(weighted_image > 255, 255, weighted_image)

    return weighted_image.astype(np.uint8)

def mosaic(img: np.ndarray, mosaic_layout: np.ndarray):
    # Extract dimensions
    h, w = img.shape
    grid_size = mosaic_layout.shape[0] 
    
    block_h = h // grid_size
    block_w = w // grid_size

    # Generate base coordinate maps
    row_indices, col_indices = np.indices((h, w))

    # Map pixels to their assigned block from the layout
    mapped_blocks = mosaic_layout[row_indices // block_h, col_indices // block_w]

    # Calculate origin coordinates for each block (The "Head")
    # Using modulo and integer division based on the dynamic grid_size
    head_row = ((mapped_blocks // grid_size) % block_h) * block_h
    head_col = (mapped_blocks % grid_size) * block_w

    normalized_row_indices = (row_indices // block_h) * block_h
    normalized_col_indices = (col_indices // block_w) * block_w

    head_difference_row = row_indices - normalized_row_indices
    head_difference_col = col_indices - normalized_col_indices

    # Final coordinate synthesis
    final_row = head_row + head_difference_row
    final_col = head_col + head_difference_col

    # Vectorized application
    result_image = img[final_row, final_col]
    
    return result_image.astype(np.uint8)


