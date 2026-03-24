import numpy as np
import cv2

def rotation_90(img: np.ndarray) -> np.ndarray:
    return np.transpose(img[::-1,::])

def rotation_180(img: np.ndarray) -> np.ndarray:
    return img[::-1, ::-1]

def rotation_270(img: np.ndarray) -> np.ndarray:
    return np.transpose(img[::, ::-1])
