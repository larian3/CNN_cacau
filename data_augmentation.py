import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator


# define caminhos da imagem original e diretório do output
IMAGE_PATH = "C:/cacau/banco_imagens/treino/saudavel/IMG_4238.jpg"
OUTPUT_PATH = "C:/cacau/banco_imagens/treino/saudavel/"

# carrega a imagem original e converter em array
image = load_img(IMAGE_PATH)
image = img_to_array(image)

# adiciona uma dimensão extra no array
image = np.expand_dims(image, axis=0)

# cria um gerador (generator) com as imagens do data augmentation
# shear_range -> cisalhamento da imagem (distorção na largura da imagem)
# horizontal_flip -> espelhamento da imagem na horizontal
# zoom -> aplica uma porcentagem de zoom aleatória na imagem
# rotation_range -> oircentagem de rotação a ser aplicana na imagem
# height_shift_range -> porcentagem de deslocamento vertical da imagem
imgAug = ImageDataGenerator(horizontal_flip=True, zoom_range=0.2, rotation_range= 7, shear_range = 0.2,
                                         height_shift_range = 0.07,)

imgGen = imgAug.flow(image, save_to_dir=OUTPUT_PATH,
                     save_format='jpg', save_prefix='t27_')

# gera 5 imagens por data augmentation
counter = 0

for (i, newImage) in enumerate(imgGen):
    counter += 1
    # ao gerar 10 imagens, parar o loop
    if counter == 10:
        break