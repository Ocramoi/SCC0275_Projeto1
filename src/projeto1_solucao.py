#!/usr/bin/env python
# coding: utf-8

# ### Instalar pacotes
# 
# Este código precisa de algumas bibliotecas para rodar.<br>
# Abaixo estão os comando para sua instalação:

# In[ ]:

# !pip install bitstring

## Código Auxiliar

import pickle
import numpy as np

from itertools import product
from bitstring import BitArray, Bits, BitStream


class MyImgFormat:
    def __init__(self, mat_rgb_ids, rgb_ids_dict):
        self.im_shape_      = mat_rgb_ids.shape
        self.rgb_ids_dict_  = rgb_ids_dict
        self.num_bits_uint_ = int(np.ceil(np.log2(len(rgb_ids_dict))))
        
        self._mat2bytes(mat_rgb_ids)
        
    def _mat2bytes(self, mat):
        self.pixel_bytes_ = BitStream()
        
        pixel_rgb_ids = mat.reshape(-1)
        for rgb_id in pixel_rgb_ids:
            self.pixel_bytes_.append(Bits(uint=int(rgb_id), length=self.num_bits_uint_))
        
    def unpack(self):
        # unpack IDs
        cp_bits = self.pixel_bytes_.copy()
        num_ids = cp_bits.len // self.num_bits_uint_
        
        mat_ids = np.array([
            cp_bits.read('uint:%d' % (self.num_bits_uint_))\
            for i in range(num_ids)\
        ]).reshape(self.im_shape_)
        
        # unpack RGB
        im_rgb = np.zeros((self.im_shape_[0], self.im_shape_[1], 3), dtype='uint8')
        for i in range(self.im_shape_[0]):
            for j in range(self.im_shape_[1]):
                im_rgb[i, j, :] = self.rgb_ids_dict_[mat_ids[i, j]]
                
        return im_rgb

def uniform_quant(im, n_colors):
    # numero de cores e espaco entre as cores (lagura do bin)
    n_vals_ch  = int(np.cbrt(n_colors))
    bin_size   = 256 // n_vals_ch
    
    # possiveis valores por canal e por pixel (combinacao dos 3 canais)
    ch_vals    = np.uint8((np.arange(n_vals_ch)) * bin_size)
    pixel_vals = list(product(ch_vals, ch_vals, ch_vals))
    
    im_qt_rgb  = im // bin_size
    im_qt_rgb[im_qt_rgb >= n_vals_ch] = n_vals_ch - 1
    im_qt_rgb  = np.uint8((im_qt_rgb) * bin_size)
    
    # criar os dicionarios ID -> pixel e pixel -> ID
    dict_id2pixel = {i: list(pixel_vals[i]) for i in range(len(pixel_vals))}
    dict_pixel2id = {pixel_vals[i]: i for i in range(len(pixel_vals))}
    
    mat_ids = np.zeros((im.shape[0], im.shape[1]), dtype='uint8')
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            mat_ids[i, j] = dict_pixel2id[tuple(im_qt_rgb[i, j])]
    
    return mat_ids, dict_id2pixel


def get_bin_size_kb(obj):
    return len(pickle.dumps(obj)) / 1e3


# *CÓDIGO ELABORADO*

# Questão 1

## Q1 setup
from sklearn.datasets import load_sample_image
import matplotlib.pyplot as plt

# Carrega imagem `china.jpg`
china = load_sample_image('china.jpg')
# plt.imshow(china)
# plt.show()

## 1a
def uniform_quant_2(im, n_colors):
    # numero de cores e espaco entre as cores (lagura do bin)
    n_vals_ch  = int(np.cbrt(n_colors))
    bin_size   = 256 // n_vals_ch

    # possiveis valores por canal e por pixel (combinacao dos 3 canais)
    ch_vals    = np.uint8((np.arange(n_vals_ch)) * bin_size)
    pixel_vals = list(product(ch_vals, ch_vals, ch_vals))

    im_qt_rgb  = im // bin_size
    im_qt_rgb[im_qt_rgb >= n_vals_ch] = n_vals_ch - 1
    im_qt_rgb  = np.uint8((im_qt_rgb) * bin_size)

    # criar os dicionarios ID -> pixel e pixel -> ID
    # DIVIDE O VALOR RGB PELA METADE NO DICT DE DADOS
    dict_id2pixel = {i: list([v//2 for v in pixel_vals[i]]) for i in range(len(pixel_vals))}
    dict_pixel2id = {pixel_vals[i]: i for i in range(len(pixel_vals))}

    mat_ids = np.zeros((im.shape[0], im.shape[1]), dtype='uint8')
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            mat_ids[i, j] = dict_pixel2id[tuple(im_qt_rgb[i, j])]

    return mat_ids, dict_id2pixel

## 1b
# Cria 3 subplots, um para cada imagem da figura f
f, subf = plt.subplots(3, 1)

# Ecibe imagem original
subf[0].set_title("Imagem original")
subf[0].imshow(china)

# Carrega matriz de 'labels' e dicionário de cores
m2, d2 = uniform_quant(china, 2**6)
# Cria matriz cópia com cada label transformada em seu valor no dict
im2 = [[ d2[c] for c in r ] for r in m2]
subf[0].set_title("Imagem usando a função dada")
subf[1].imshow(im2)

# Realiza o processo anterior com a função modificada
m3, d3 = uniform_quant_2(china, 2**6)
im3 = [[ d3[c] for c in r ] for r in m3]
subf[2].imshow(im3)

# Exibe imagens
plt.show()

## 1c
print("Tamanhos das imagens:")
print("\tImagem original:", get_bin_size_kb(china))
print("\tImagem quantizada original:", get_bin_size_kb(im2))
print("\tImagem quantizada modificada:", get_bin_size_kb(im3))

# Questão 2

## Q2 setup
from sklearn.cluster import KMeans

## 2a
def kmeansQuant(img: np.uint8, cluster: int = 64) -> np.uint8:
    """
    Calcula centróides para o número dado de clusters e aproxima cada
    pixel da imagem dada para o centróide de seu cluster.

    Parameters:
    ----------
    img: np.uint8
        Imagem carregada como array numpy de uint8
    cluster: int
        Número de clusters para cálculo

    Returns:
    ----------
    np.uint8
        Imagem recontruída em seu tamanho original com a aproximação de cada
        pixel a seu cluster correspondente
    """
    # Transforma tamanho e normaliza array de pixels carregado
    imgArray = np.reshape(img, (-1, 3)) / 255
    # Treina classificador k-means com parâmetros dados sobre o array gerado
    km = KMeans(n_clusters=cluster, random_state=42, max_iter=10, n_init="auto").fit(imgArray)
    # Prediz 'labels' dentre os centróides calculados em todos os pixels da matriz
    labels = km.predict(imgArray)
    # Transforma cada pixel no centróide previsto pelo algoritmo e
    # retoma o tamanho original da imagem
    return np.reshape(km.cluster_centers_[labels], img.shape)

## 2b
#  Cria 3 subplots para as imagens na figura f
f, subf = plt.subplots(3, 1)

subf[0].set_title("Imagem original")
subf[0].imshow(china)
subf[1].set_title("64 cores (k-means)")
subf[1].imshow(kmeansQuant(china))
subf[2].set_title("64 cores (quantização uniforme)")
subf[2].imshow(im2)

plt.show()

# Questão 3

## Q3 setup
