# Importação de bibliotecas
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pyrsgis import raster
import cv2

# Importar amostra
imageDirectory = os.getcwd() + '/banco_imagens/amostra'
os.chdir(imageDirectory)
dirFiles = os.listdir(imageDirectory)
print(dirFiles)

# Importar modelo treinado
model_directory = 'C:/cacau/modelos/'
model = tf.keras.models.load_model(model_directory + 'model_w.h5')

# Função para redimensionar e preparar a imagem para predição
def preprocess_image(image_path):
    ds, tempArr = raster.read(image_path)
    tempArrNewDim = np.moveaxis(tempArr, 0, -1).copy()  # Ajusta dimensões para (X,Y,Bandas)
    resized = cv2.resize(tempArrNewDim, (150, 150))  # Redimensiona a proporção da imagem
    # Normaliza a imagem
    resized = resized.astype('float32') / 255.0
    return resized

# Dicionário para mapear os resultados de predição
class_labels = {
    0: 'saudável',
    1: 'podridão parda',
    2: 'vassoura de bruxa'
}

# Passa por todos os elementos (amostras) da pasta, ajusta suas dimensões e faz a predição
for image_name in dirFiles:
    # Processa a imagem
    image_path = os.path.join(imageDirectory, image_name)
    image = preprocess_image(image_path)
    
    # Expande as dimensões da imagem para adicionar o batch size
    image_batch = np.expand_dims(image, axis=0)
    
    # Faz a predição
    y_pred = model.predict(image_batch, verbose=0)
    y_pred_class = np.argmax(y_pred, axis=1)[0]
    
    # Exibe a imagem e a classe prevista
    plt.imshow(image)
    plt.title(f'Imagem: {image_name} \nClasse: {class_labels[y_pred_class]}')
    plt.show()
    
    # Exibe o nome da imagem e a classe prevista no terminal
    print(f'Imagem: {image_name} - Classe prevista: {class_labels[y_pred_class]}')

# Exibe a normalização da amostra
print(f"Shape das amostras: {image_batch.shape}")
