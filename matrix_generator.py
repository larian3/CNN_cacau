import CNN_load_data
import confusion_matrix
import matplotlib.pyplot as plt
from keras.models import model_from_json
#from sklearn.preprocessing import LabelEncoder
#from keras.utils import np_utils
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score

#read file
json_file = open("model.json", "r")
load_model_json = json_file.read()
json_file.close()

#load model
model = model_from_json(load_model_json)

#load weights into model
model.load_weights("model_2.h5")    

#compile model and evaluate
model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

#load dataset
X, y = CNN_load_data.loadDataTest()

#normalize dataset from 0-255 to 0.0-1.0
X = X.astype("float32")
X /= 255.0

yp = np.argmax(model.predict(X), axis=-1)
yp = yp.reshape(len(yp), 1)

print(yp.shape)
print(y.shape)
print("Acertos:", sum(y==yp)/len(y))
print("Erros: ", sum(y!=yp)/len(y))

np.set_printoptions(precision=2)
class_names = ["Healthy","Black Pod Rot", "Witches' Broom"]
confusionMatrix = confusion_matrix(y, yp)


accuracy = accuracy_score(y, yp)
print(f"Accuracy: {accuracy}")

recall = recall_score(y, yp, average= "weighted")
print('Recall:', recall)

precision = precision_score(y, yp, average= "weighted")
print('Precision:', precision)

plt.figure()
confusion_matrix.plot_confusion_matrix(confusionMatrix, classes=class_names, title='Confusion Matrix')
plt.tight_layout()
plt.savefig('matriz_de_confusao.png', dpi=300)
plt.show()