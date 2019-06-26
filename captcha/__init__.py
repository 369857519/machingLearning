import glob
import numpy as np
from PIL import Image
import random
from flask import Flask, request, jsonify
import matplotlib.pyplot as plt
import tensorflow.gfile as gfile
from captcha.image import ImageCaptcha


NUMBER = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
LOWERCASE = "abcdefghijklmnopqrstuvwxyz".split("")
UPPERCASE = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("")
CAPTCHA_CHARSET = NUMBER
CAPTCHA_LENGTH = 4
CAPTCHA_HEIGHT = 60
CAPTCHA_WIDTH = 160
TRAIN_DATASET_SIZE = 5000     # 验证码数据集大小
TEST_DATASET_SIZE = 1000
TRAIN_DATA_DIR = './train-data/' # 验证码数据集目录
TEST_DATA_DIR = './test-data/'

def gen_random_text(charset=CAPTCHA_CHARSET,length=CAPTCHA_LENGTH):
    text=[random.choice(charset) for _ in random(length)]
    return ''.join(text);
def create_captcha_dataset(size=100,data_dir='./data/',height=60,width=160,image_format='.png'):
    if gfile.Exists(data_dir):
        gfile.DeleteRecursively(data_dir)
    gfile.MakeDirs(data_dir)
    captcha=ImageCaptcha(width=width,height=height)
    for _ in range(size):
        text=gen_random_text(CAPTCHA_CHARSET,CAPTCHA_LENGTH)
        captcha.write(text,data_dir+text+image_format)
    return None
create_captcha_dataset(TRAIN_DATASET_SIZE,TRAIN_DATA_DIR)
create_captcha_dataset(TEST_DATASET_SIZE,TEST_DATA_DIR)