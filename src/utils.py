import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

"""Funcoes utilitarias para E/S de imagens e visualizacao.

Este modulo concentra operacoes de carregamento, salvamento e exibicao
de imagens usadas no fluxo principal do projeto.
"""

def load_image(path: str, monochromatic: bool=True) -> np.ndarray:
    """Carrega uma imagem do disco usando OpenCV.

    :param path: Caminho do arquivo de imagem.
    :param monochromatic: Se verdadeiro, carrega em tons de cinza.
    :returns: Imagem carregada como arranjo NumPy.
    :raises FileNotFoundError: Se o arquivo nao puder ser lido.
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
    """Salva uma imagem no disco.

    :param img: Imagem a ser salva.
    :param folder: Pasta de destino.
    :param filename: Nome do arquivo de saida.
    :returns: Nao retorna valor.
    :raises FileNotFoundError: Se a pasta de destino nao existir.
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
    """Exibe imagem original e processada lado a lado.

    :param original: Imagem original.
    :param processed: Imagem apos processamento.
    :param title: Titulo do painel da imagem processada.
    :returns: Nao retorna valor.
    """
    def to_matplotlib_color(img: np.ndarray) -> np.ndarray:
        """Converte imagem BGR para RGB quando necessario.

        :param img: Imagem de entrada.
        :returns: Imagem no formato esperado pelo Matplotlib.
        """
        if img.ndim == 3:
            # OpenCV usa BGR, enquanto Matplotlib espera RGB.
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(to_matplotlib_color(original), cmap='gray' if len(original.shape) == 2 else None)
    
    plt.subplot(1, 2, 2)
    plt.title(title)
    plt.imshow(to_matplotlib_color(processed), cmap='gray' if len(processed.shape) == 2 else None)
    
    plt.show()