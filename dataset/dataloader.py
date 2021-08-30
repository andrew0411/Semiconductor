import numpy as np
import os
from PIL import Image
from sklearn.preprocessing import MinMaxScaler


def data_load(type):
    if type == 'CNN':
        true_path = type + '_True'
        false_path = type + '_False'
    elif type == 'CIS':
        true_path = type + '_True'
        false_path = type + '_False'
    else:
        print('Wrong type')

    img_true = []
    img_false = []

    for i in os.listdir(true_path):
        path = true_path + '/' + i
        img = np.array(Image.open(path))
        img_true.append(img)

    for j in os.listdir(false_path):
        path = false_path + '/' + j
        img = np.array(Image.open(path))
        img_false.append(img)

    return img_true, img_false


def cis_scaling(img_set):
    scaled = []
    scaler = MinMaxScaler(feature_range=(0, 0.8))
    for i in range(len(img_set)):
        scaler.fit(img_set[i])
        temp = scaler.transform(img_set[i])
        scaled.append(temp)

    return scaled


def quarter_data_load():
    img_t = []
    img_f = []

    true_path = '1_4/T'
    false_path = '1_4/F'

    for i in os.listdir(true_path):
        path = true_path + '/' + i
        img = np.array(Image.open(path))
        img_t.append(img)

    for j in os.listdir(false_path):
        path = false_path + '/' + j
        img = np.array(Image.open(path))
        img_f.append(img)

    return img_t, img_f

