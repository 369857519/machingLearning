import glob
import numpy as np
from PIL import Image
import random
from flask import Flask, request, jsonify
import matplotlib.pyplot as plt
import tensorflow.gfile as gfile
from captcha.image import ImageCaptcha
from keras import backend as K

NUMBER = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
LOWERCASE = list("abcdefghijklmnopqrstuvwxyz")
UPPERCASE = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
CAPTCHA_CHARSET = NUMBER
CAPTCHA_LENGTH = 4
CAPTCHA_HEIGHT = 60
CAPTCHA_WIDTH = 160
TRAIN_DATASET_SIZE = 5000  # 验证码数据集大小
TEST_DATASET_SIZE = 1000
TRAIN_DATA_DIR = './train-data/'  # 验证码数据集目录
TEST_DATA_DIR = './test-data/'


def gen_random_text(charset=CAPTCHA_CHARSET, length=CAPTCHA_LENGTH):
    random.choice(charset)
    text = [random.choice(charset) for _ in range(length)]
    return ''.join(text);


def create_captcha_dataset(size=100, data_dir='./data/', height=60, width=160, image_format='.png'):
    if gfile.Exists(data_dir):
        gfile.DeleteRecursively(data_dir)
    gfile.MakeDirs(data_dir)
    captcha = ImageCaptcha(width=width, height=height)
    for _ in range(size):
        text = gen_random_text(CAPTCHA_CHARSET, CAPTCHA_LENGTH)
        captcha.write(text, data_dir + text + image_format)
    return None


def visualize(image, text):
    plt.figure()
    for i in range(20):
        plt.subplot(5, 4, i + 1)
        plt.tight_layout()
        plt.imshow(image[i], cmap='Greys')
        plt.title("Label: {}".format(text[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()


def fit_keras_channels(batch, rows=CAPTCHA_HEIGHT, cols=CAPTCHA_WIDTH):
    if K.image_data_format() == 'channels_first':
        batch = batch.reshape(batch.shape[0], 1, rows, cols)
        input_shape = (1, rows, cols)
    else:
        batch = batch.reshape(batch.shape[0], rows, cols, 1)
        input_shape = (rows, cols, 1)
    return batch, input_shape


def text2vec(text, length=CAPTCHA_LENGTH, charset=CAPTCHA_CHARSET):
    text_len = len(text)
    if text_len != length:
        raise ValueError("Error: length of captcha should be {},but got {}".format(length, text_len))
    vec = np.zeros(length * len(charset))
    for i in range(length):
        vec[charset.index(text[i]) + i * len(charset)] = 1
    return vec

def vec2text(vector):
    if not isinstance(vector,np.ndarray):
        vector=np.asarray(vector)
    vector=np.reshape(vector,[CAPTCHA_LENGTH,-1])
    text=''
    for item in vector :
        text+=CAPTCHA_CHARSET[np.argmax(item)]
    return text

def prepareData():
    image = []
    text = []
    count = 0
    for filename in glob.glob(TRAIN_DATA_DIR + '*.png'):
        image.append(np.array(Image.open(filename)))
        filename = filename.lstrip(TRAIN_DATA_DIR).rstrip('.png')
        text.append(filename.replace('\\', ''))
        count += 1
        if count >= 100:
            break
    image = np.array(image, dtype=np.float32)
    # 转灰度
    image = np.dot(image[..., :3], [0.299, 0.587, 0.114])
    image = image / 255
    print(image[10])
    # visualize(image,text)
    image, input_shape = fit_keras_channels(image)
    print(image.shape)
    print(input_shape)
    text=list(text)
    vec=[None]*len(text)
    for i in range(len(vec)):
        vec[i]=text2vec(text[i])
    print(vec[0])


# create_captcha_dataset(TRAIN_DATASET_SIZE,TRAIN_DATA_DIR)
# create_captcha_dataset(TEST_DATASET_SIZE,TEST_DATA_DIR)
prepareData()
