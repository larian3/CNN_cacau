from CNN_load_data import loadData, loadDataTest
import Functions
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import time


inicio = time.time()

# Configuração do Modelo
batch_size = 50 # Número de exemplos em um batch (lote)
epochs = 100 #conjunto de dados que é passado na rede neural UMA VEZ


#Carrega o BD das imagens (Treino e Validação) 
inicio_aux = time.time()
(X, y) = loadData()
fim = time.time()
Functions.printTime("Load Dataset Treino", inicio_aux, fim)

#Carrega o BD das imgagens (Teste)
inicio_aux = time.time()
(X_test, y_test) = loadDataTest()
fim = time.time()
Functions.printTime("Load Dataset Test", inicio_aux, fim)

#Redimensiona os dados para ficar no formato que o tensorflow trabalha
X = X.astype('float32') 
X_test = X_test.astype('float32')


#Normalizando os valores de 0-255 para 0.0-1.0
X /= 255.0
X_test /= 255.0
# print(X.shape)
# print(X_test.shape)


#Divide os dados em 82,35% para treino e 17,65% para validação **Os dados de testes já foram separados
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1765, train_size=0.8235, stratify=y)


#Transformando os rótulos de decimal para vetores com valores binários
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)


#Criação do Modelo
model = Sequential()
model.add(Convolution2D(32, (3, 3), input_shape=(150, 150, 3), activation="relu")) #relu é a função de ativação da rede neural
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(32, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(128, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) #comprime os dados em um array
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(3, activation="sigmoid"))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


model.summary()

# Executa o treinamento do modelo
inicio_aux = time.time()
#função fit é responsável pelo treinamento
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))
fim = time.time()

Functions.printTime("Training", inicio_aux, fim)

print(type(history))

# Plota o gráfico com o histórico da acurácia 
plt.figure(figsize=(10, 6))  # Define o tamanho da figura

# Plota as curvas de acurácia
plt.plot(history.history['accuracy'], color='blue', linestyle='-', linewidth=2, label='Train')
plt.plot(history.history['val_accuracy'], color='orange', linestyle='--', linewidth=2, label='Validation')

# Adiciona título e rótulos dos eixos
plt.title('Learning Curve', fontsize=16)
plt.ylabel('Accuracy', fontsize=14)
plt.xlabel('Epochs', fontsize=14)

# Adiciona uma grade de fundo
plt.grid(True, linestyle='--', alpha=0.7)

# Adiciona legenda
plt.legend(loc='lower right', fontsize=12)

# Salva a figura em alta resolução
plt.savefig('learning_curve.png', dpi=300)

# Exibe o gráfico
plt.show()

print(len(X_test), len(y_test))

# Mostra a potuação da acurácia
scores = model.evaluate(X_test, y_test, verbose=0)
result_error = str("%.2f"%(1-scores[1]))
result = str("%.2f"%(scores[1]))
print("CNN Score:", result)
print("CNN Error:", result_error)


# Salva o modelo no formato JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)


# Salva os pesos em HDF5
model.save("model_2.h5")


print("Modelo salvo no disco")




# Salva os resultados da acurácia em arquivo CSV
index = []
for i in range(1, epochs+1):
    index.append(f'epoca{i}')
result_train = pd.DataFrame(history.history['accuracy'], index=index)
result_test = pd.DataFrame(history.history['val_accuracy'], index=index)
result_train.to_csv('accuracy_trein.csv', header=False)
result_test.to_csv('accuracy_test.csv', header=False)

fim = time.time()
Functions.printTime("Time Run Model", inicio, fim)