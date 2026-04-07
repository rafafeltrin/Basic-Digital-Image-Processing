import numpy as np
import cv2

"""Transformacoes geometricas e de quantizacao para imagens.
"""

def rotation_90(img: np.ndarray) -> np.ndarray:
    """Rotaciona a imagem em 90 graus.

    :param img: Imagem de entrada.
    :returns: Imagem rotacionada em 90 graus.
    """
    return np.transpose(img[::-1,::])

def rotation_180(img: np.ndarray) -> np.ndarray:
    """Rotaciona a imagem em 180 graus.

    :param img: Imagem de entrada.
    :returns: Imagem rotacionada em 180 graus.
    """
    return img[::-1, ::-1]

def rotation_270(img: np.ndarray) -> np.ndarray:
    """Rotaciona a imagem em 270 graus.

    :param img: Imagem de entrada.
    :returns: Imagem rotacionada em 270 graus.
    """
    return np.transpose(img[::, ::-1])



def image_elargement_replication(img: np.ndarray, factor: int) -> np.ndarray:
    """Amplia a imagem por um fator inteiro via replicacao de pixels.

    :param img: Imagem de entrada.
    :param factor: Fator inteiro de ampliacao.
    :returns: Imagem ampliada conforme o fator informado.
    """
    row_indices = np.arange(img.shape[0] * factor) // factor
    col_indices = np.arange(img.shape[1] * factor) // factor

    return img[row_indices][:, col_indices]


def bit_representation(img: np.ndarray, original_bit_depth:int, final_bit_depth: int):
    """Reduz a profundidade de bits da imagem por quantizacao uniforme.

    A imagem e quantizada para a profundidade final e reescalada para o
    intervalo visual de 8 bits [0, 255].

    :param img: Imagem de entrada.
    :param original_bit_depth: Profundidade de bits original da imagem.
    :param final_bit_depth: Profundidade de bits desejada apos quantizacao.
    :returns: Imagem quantizada em `uint8`.
    """
    factor = (2 ** original_bit_depth) / (2 ** final_bit_depth)
    quantized_image = img // factor

    step = 255.0 / (2 ** final_bit_depth - 1)

    final_image = quantized_image * step
    
    return final_image.astype(np.uint8)
    