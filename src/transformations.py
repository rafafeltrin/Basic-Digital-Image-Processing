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


def image_elargement_replication(img: np.ndarray, factor: int) -> np.ndarray:
    row_indices = np.arange(img.shape[0] * factor) // factor
    col_indices = np.arange(img.shape[1] * factor) // factor

    return img[row_indices][:, col_indices]


def bit_representation(img: np.ndarray, original_bit_depth:int, final_bit_depth: int):
    factor = (2 ** original_bit_depth) / (2 ** final_bit_depth)
    quantized_image = img // factor

    step = 255.0 / (2 ** final_bit_depth - 1)

    final_image = quantized_image * step
    
    return final_image.astype(np.uint8)
    