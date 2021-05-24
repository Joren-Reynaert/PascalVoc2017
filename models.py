import os
from lxml import etree
import numpy as np
from skimage import io
from skimage.transform import resize
import matplotlib.pyplot as plt
import keras
import cv2
from random import randint
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation, Reshape, Permute
from keras.models import Model, Sequential, load_model
from keras.utils import to_categorical


def build_classification_dataset(list_of_files, image_size, class_list, voc_folder):
    """ build training or validation set

    :param list_of_files: list of filenames to build trainset with
    :return: tuple with x np.ndarray of shape (n_images, image_size, image_size, 3) and  y np.ndarray of shape (n_images, n_classes)
    """

    annotation_folder = os.path.join(voc_folder, "VOC2009/Annotations/")
    annotation_files = os.listdir(annotation_folder)
    filtered_filenames = []
    for a_f in annotation_files:
        tree = etree.parse(os.path.join(annotation_folder, a_f))
        if np.any([tag.text == filt for tag in tree.iterfind(".//name") for filt in class_list]):
            filtered_filenames.append(a_f[:-4])

    temp = []
    train_labels = []
    for f_cf in list_of_files:
        with open(f_cf) as file:
            lines = file.read().splitlines()
            temp.append([line.split()[0] for line in lines if int(line.split()[-1]) == 1])
            label_id = [f_ind for f_ind, filt in enumerate(class_list) if filt in f_cf][0]
            train_labels.append(len(temp[-1]) * [label_id])
    train_filter = [item for l in temp for item in l]

    image_folder = os.path.join(voc_folder, "VOC2009/JPEGImages/")
    image_filenames = [os.path.join(image_folder, file) for f in train_filter for file in os.listdir(image_folder) if
                       f in file]
    x = np.array([resize(io.imread(img_f), (image_size, image_size, 3)) for img_f in image_filenames]).astype(
        'float32')

    # changed y to an array of shape (num_examples, num_classes) with 0
    # if class is not present and 1 if class is present
    y_temp = []
    for tf in train_filter:
        y_temp.append([1 if tf in l else 0 for l in temp])
    y = np.array(y_temp)

    return x, y


def build_autoencoder(in_shape, n_blocks):
    input_img = Input(shape=in_shape)
    x = Conv2D(16, (3,3), activation='relu', padding='same', data_format='channels_last',
               kernel_initializer='normal')(input_img)
    for i in range(n_blocks-1):
        x = MaxPooling2D((2, 2), padding='same', data_format='channels_last')(x)
        x = Conv2D(16, (3, 3), activation='relu', padding='same', data_format='channels_last',
                   kernel_initializer='normal')(x)
    encoded = MaxPooling2D((2, 2), padding='same', data_format='channels_last')(x)  # (16, 16, 8)     128-dimensional

    x = Conv2D(8, (3, 3), activation='relu', padding='same', data_format='channels_last', kernel_initializer='normal')(
        encoded)
    for i in range(n_blocks):
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same', data_format='channels_last',
                   kernel_initializer='normal')(x)

    autoencoder = Model(input_img, x)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    return autoencoder
