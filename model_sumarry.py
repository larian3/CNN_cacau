import tensorflow as tf

directory = 'C:/cacau/'
model = tf.keras.models.load_model(directory+'model_2.h5')

print(model.summary())