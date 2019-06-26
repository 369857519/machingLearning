import tensorflow as tf
import os
from keras.datasets import mnist
from keras.utils import np_utils
from keras import backend as K

(x_train, y_train), (x_test, y_test) = mnist.load_data('mnist\mnist.npz')

img_rows, img_cols = 28, 28

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

X_train = x_train.astype('float32')
X_test = x_test.astype('float32')

X_train /= 255
X_test /= 255

print(x_train.shape, type(x_train))
print(x_test.shape, type(x_test))

from keras.utils import np_utils

n_classes = 10
Y_train = np_utils.to_categorical(y_train, n_classes)
Y_test = np_utils.to_categorical(y_test, n_classes)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

model = Sequential()
# 卷积
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))

# 二层卷积
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))

# 最大池化
model.add(MaxPooling2D(pool_size=(2, 2)))

# Dropout 25% 的输入神经元
model.add(Dropout(0.25))

# 摊平
model.add(Flatten())

### Classification

# 全连接层
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(n_classes, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')

#训练
history=model.fit(X_train,
                  Y_train,
                  batch_size=128,
                  epochs=5,
                  verbose=2,
                  validation_data=(X_test,Y_test))

import tensorflow.gfile as gfile
import numpy as np
save_dir="./mnist/model"
if gfile.Exists(save_dir):
    gfile.DeleteRecursively(save_dir)
gfile.MakeDirs(save_dir)
model_name = 'keras_mnist.h5'
model_path=os.path.join(save_dir,model_name)
model.save(model_path)

