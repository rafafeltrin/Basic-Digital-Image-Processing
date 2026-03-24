import numpy as np
import cv2

def rotation_90(img: np.ndarray) -> np.ndarray:
    return np.transpose(img[::-1,::])

def rotation_180(img: np.ndarray) -> np.ndarray:
    return img[::-1, ::-1]

def rotation_270(img: np.ndarray) -> np.ndarray:
    return np.transpose(img[::, ::-1])

def image_elargement_replication_2_factor(img: np.ndarray) -> np.ndarray:
    x, y = img.shape
    new_image = np.zeros((x * 2, y * 2), dtype=img.dtype)

    new_image[::2, ::2] = img
    new_image[1::2, 1::2] = img
    new_image[::2, 1::2] = img
    new_image[1::2, ::2] = img

    return new_image

def image_elargement_replication_4_factor(img: np.ndarray) -> np.ndarray:
    new_image_2 = image_elargement_replication_2_factor(img)
    new_image_4 = image_elargement_replication_2_factor(new_image_2)

    return new_image_4
