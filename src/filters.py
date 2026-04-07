import numpy as np
import cv2
from utils import save_image

"""Colecao de filtros e transformacoes de imagem.

Este modulo reune operacoes pontuais e espaciais aplicadas a imagens
representadas por arranjos NumPy, em geral no formato `uint8`.
"""

def pencil_sketch(img: np.ndarray) -> np.ndarray:
    """Aplica um efeito de desenho a lapis na imagem.

    Se a imagem for colorida, ela e convertida para tons de cinza.
    Em seguida, e realizado desfoque gaussiano e uma divisao normalizada
    para realcar contornos claros sobre fundo branco.

    :param img: Imagem de entrada em tons de cinza ou colorida.
    :returns: Imagem transformada no efeito lapis.
    """
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
    """Aplica correcao gama na imagem.

    A imagem e normalizada para o intervalo [0, 1], transformada pela
    potencia `1/gamma` e depois reescalada para [0, 255].

    :param img: Imagem de entrada.
    :param gamma: Valor do parametro gama (deve ser diferente de zero).
    :returns: Imagem com correcao gama aplicada.
    """
    normalized_image = img / 255.0

    normalized_image = normalized_image**(1/gamma)

    final_result = normalized_image*255

    return final_result.astype(np.uint8)


def threshold_binarization(img: np.ndarray, threshold: int) -> np.ndarray:
    """Binariza a imagem a partir de um limiar.

    Pixels com intensidade maior que `threshold` recebem 255; os demais,
    0.

    :param img: Imagem de entrada.
    :param threshold: Limiar de corte para binarizacao.
    :returns: Imagem binaria em `uint8`.
    """
    result = np.where(img > threshold, 255, 0)
    return result.astype(np.uint8)

def sepia_filter(img: np.ndarray) -> np.ndarray:
    """Aplica filtro sepia em uma imagem colorida.

    A imagem e convertida para RGB, transformada por uma matriz sepia e
    depois reconvertida para BGR.

    :param img: Imagem colorida de entrada no formato BGR.
    :returns: Imagem com efeito sepia em BGR.
    :raises ValueError: Se a imagem nao for colorida.
    """
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
    """Converte uma imagem colorida para tons de cinza ponderados.

    Usa pesos para os canais BGR para obter uma luminancia aproximada.

    :param img: Imagem colorida de entrada em BGR.
    :returns: Imagem monocromatica em `uint8`.
    :raises ValueError: Se a imagem nao for colorida.
    """
    if img.ndim != 3:
        raise ValueError("Monochrome filter requires a color image")

    # Weights for the BGR
    weights = np.array([0.1140, 0.5870, 0.2989], dtype=np.float32)

    monochromatic_image = np.dot(img.astype(np.float32), weights)

    final_result_normalized = np.where(monochromatic_image > 255, 255, monochromatic_image)

    return final_result_normalized.astype(np.uint8)

def bit_planes (img: np.ndarray, plane: int) -> np.ndarray:
    """Extrai um plano de bits da imagem.

    O bit selecionado de cada pixel e isolado e reescalado para o intervalo
    visual [0, 255].

    :param img: Imagem de entrada.
    :param plane: Indice do plano de bits a extrair.
    :returns: Imagem binaria representando o plano de bits.
    """
    result = (img >> plane) & 1

    return result.astype(np.uint8) * 255 

def weighted_average_monochromatic_image(img1: np.ndarray, img2: np.ndarray, weight1: float, weight2: float) -> np.ndarray:
    """Combina duas imagens por media ponderada.

    As imagens devem possuir as mesmas dimensoes.

    :param img1: Primeira imagem de entrada.
    :param img2: Segunda imagem de entrada.
    :param weight1: Peso aplicado a `img1`.
    :param weight2: Peso aplicado a `img2`.
    :returns: Imagem resultante da combinacao ponderada.
    :raises ValueError: Se as imagens tiverem dimensoes diferentes.
    """
    if img1.shape != img2.shape:
        raise ValueError("Both images must have the same dimensions")
    
    weighted_image = (weight1 * img1) + (weight2 * img2)

    weighted_image = np.where(weighted_image > 255, 255, weighted_image)

    return weighted_image.astype(np.uint8)

def mosaic(img: np.ndarray, mosaic_layout: np.ndarray):
    """Reorganiza blocos da imagem conforme um layout de mosaico.

    Divide a imagem em uma grade quadrada e remapeia cada bloco de acordo
    com os indices informados em `mosaic_layout`.

    :param img: Imagem de entrada em tons de cinza.
    :param mosaic_layout: Matriz quadrada com o mapeamento dos blocos.
    :returns: Imagem com os blocos reorganizados.
    """
    h, w = img.shape
    grid_size = mosaic_layout.shape[0] 
    
    block_h = h // grid_size
    block_w = w // grid_size

    row_indices, col_indices = np.indices((h, w))

    mapped_blocks = mosaic_layout[row_indices // block_h, col_indices // block_w]

    head_row = ((mapped_blocks // grid_size) % block_h) * block_h
    head_col = (mapped_blocks % grid_size) * block_w

    normalized_row_indices = (row_indices // block_h) * block_h
    normalized_col_indices = (col_indices // block_w) * block_w

    head_difference_row = row_indices - normalized_row_indices
    head_difference_col = col_indices - normalized_col_indices

    final_row = head_row + head_difference_row
    final_col = head_col + head_difference_col

    result_image = img[final_row, final_col]
    
    return result_image.astype(np.uint8)


def negative_filter(img: np.ndarray) -> np.ndarray:
    """Gera o negativo da imagem.

    :param img: Imagem de entrada.
    :returns: Imagem negativa.
    """
    return 255 - img

def intensity_transformed(img: np.ndarray) -> np.ndarray:
    """Aplica transformacao linear por faixas de intensidade.

    Valores abaixo de 100 sao mapeados para 0, acima de 200 para 255 e
    o intervalo intermediario e escalado linearmente.

    :param img: Imagem de entrada.
    :returns: Imagem transformada em `uint8`.
    """
    img_float = img.astype(np.float32)

    result = np.where(
        img_float < 100,
        0,
        np.where(
            img_float > 200,
            255,
            (img_float - 100) * 2.55
        )
    )

    return result.astype(np.uint8)


def inverted_even_rows(img: np.ndarray) -> np.ndarray:
    """Inverte horizontalmente apenas as linhas pares da imagem.

    :param img: Imagem de entrada.
    :returns: Imagem com linhas pares invertidas.
    """
    inverted_image = img.copy()
    inverted_image[::2, :] = inverted_image[::2, ::-1]
    return inverted_image.astype(np.uint8)


def mirror_top_half_to_bottom_half(img: np.ndarray) -> np.ndarray:
    """Espelha a metade superior para preencher a metade inferior.

    A metade inferior passa a ser o reflexo vertical da metade superior.

    :param img: Imagem de entrada.
    :returns: Imagem com espelhamento vertical entre metades.
    """
    mirrored_image = img.copy()
    half = mirrored_image.shape[0] // 2
    mirrored_image[half:, :] = mirrored_image[:half, :][::-1, :]
    
    return mirrored_image.astype(np.uint8)

def vertical_mirror(img: np.ndarray) -> np.ndarray:
    """Inverte a imagem verticalmente (de cima para baixo).

    :param img: Imagem de entrada.
    :returns: Imagem invertida verticalmente.
    """
    # This flips the image upside down
    return img[::-1, :]


def spatial_convolution(img: np.ndarray, filter: np.ndarray) -> np.ndarray:
    """Aplica convolucao espacial com um kernel fornecido.

    O resultado intermediario e calculado em ponto flutuante para preservar
    valores negativos e acima de 255, depois convertido para modulo,
    limitado ao intervalo visual e convertido para `uint8`.

    :param img: Imagem de entrada.
    :param filter: Kernel de convolucao.
    :returns: Imagem resultante da convolucao.
    """
    convolved = cv2.filter2D(img, ddepth=cv2.CV_32F, kernel=filter)
    
    convolved_abs = np.abs(convolved)

    result = np.clip(convolved_abs, 0, 255)
    
    return result.astype(np.uint8)


def combined_gradient_magnitude(img: np.ndarray, filter_x: np.ndarray, filter_y: np.ndarray) -> np.ndarray:
    """Calcula a magnitude do gradiente a partir de dois filtros direcionais.

    Aplica os kernels em x e y, calcula a magnitude euclidiana
    sqrt(g_x^2 + g_y^2) e limita o resultado ao intervalo [0, 255].

    :param img: Imagem de entrada.
    :param filter_x: Kernel para derivada na direcao x.
    :param filter_y: Kernel para derivada na direcao y.
    :returns: Magnitude do gradiente em `uint8`.
    """
    gx = cv2.filter2D(img, ddepth=cv2.CV_32F, kernel=filter_x)
    gy = cv2.filter2D(img, ddepth=cv2.CV_32F, kernel=filter_y)
    
    magnitude = np.sqrt((gx ** 2) + (gy ** 2))
    
    result = np.clip(magnitude, 0, 255)
    
    return result.astype(np.uint8)