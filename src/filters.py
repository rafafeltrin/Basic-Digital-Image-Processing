import numpy as np
import cv2
from ultils import save_image

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
    
    sepia_img = np.dot(img, filter.T)

    sepia_img = np.where(sepia_img > 255, 255, sepia_img)

    return sepia_img.astype(np.uint8)

def monochrome_filter(img: np.ndarray) -> np.ndarray:
    filter = np.array([0.2989, 0.5870, 0.1140])

    monochromatic_image = np.dot(img, filter.T)

    return monochromatic_image.astype(np.uint8)

def bit_planes (img: np.ndarray, plane: int) -> np.ndarray:
    result = (img >> plane) & 1

    return result

def weighted_average_monochromatic_image(img1: np.ndarray, img2: np.ndarray, weight1: float, weight2: float) -> np.ndarray:
    if img1.shape != img2.shape:
        raise ValueError("Both images must have the same dimensions")
    
    weighted_image = (weight1 * img1) + (weight2 * img2)

    weighted_image = np.where(weighted_image > 255, 255, weighted_image)

    return weighted_image.astype(np.uint8)