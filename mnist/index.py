import tensorflow as tf
from keras.datasets import mnist
from keras.utils import np_utils

(x_train, y_train), (x_test, y_test) = mnist.load_data('mnist\mnist.npz')
# 60000张图片 60000个标签
print(x_train.shape, y_train.shape)
# 测试集 10000 10000
print(x_test.shape, y_test.shape)

# 观察数据
X_train = x_train.reshape(60000, 784)
X_test = x_test.reshape(10000, 784)
print(X_train.shape, type(X_train))
print(X_test.shape, type(y_test))

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# 观察标签

import numpy as np
import matplotlib.pyplot as plt

label, count = np.unique(y_train, return_counts=True)
print(label, count)

# one hot
n_classes = 10
print(y_train.shape)
Y_train = np_utils.to_categorical(y_train, n_classes)
print(Y_train.shape)
Y_test = np_utils.to_categorical(y_test, n_classes)
print(y_train[0])
print(Y_train[0])

# 神经网络定义网络
from keras.models import Sequential
from keras.layers.core import Dense, Activation

model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu'))

model.add(Dense(512))
model.add(Activation('relu'))

model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

hisotry = model.fit(X_train, Y_train, batch_size=128, epochs=5, verbose=2, validation_data=(X_test, Y_test))

###卷积神经网络
