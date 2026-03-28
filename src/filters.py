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