import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Embedding, Dropout, GlobalMaxPool2D, Conv2D, BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt 
from support import get_languages

lang = get_languages()
lang_short = ['en','fr','it','es']

X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
print(set(y_train))
X_train = (X_train/10001) * 255
print(np.shape(X_train))
print(np.shape(y_train))
X_train = X_train.reshape(X_train.shape[0], 4, 20,1)
print(X_train[0])
print(np.shape(X_train))
print(np.shape(y_train))
# y_train = tf.keras.utils.to_categorical(y_train,dtype='float32')


model = Sequential()

model.add(Conv2D(32, (3,3), padding='same', input_shape=X_train.shape[1:]))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Dropout(0.25))
# ------------------------------------------------------------
model.add(Conv2D(64, (3,3), padding='same'))
model.add(Activation('relu'))

model.add(Conv2D(64, (3,3), padding='same'))
model.add(Activation('relu'))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(1024))
model.add(Activation('relu'))
# model.add(Dropout(0.5))
model.add(Dense(4))
model.add(Activation('softmax'))



model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=10, batch_size=100, validation_split=0.2)


model.save('model.h5')
plt.plot([x for x in range(1,11)], history.history['accuracy'])
plt.show()
