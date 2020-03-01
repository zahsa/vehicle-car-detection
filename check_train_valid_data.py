

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# os.chdir("/DATA1/zahra/Python Space/pyZahra")

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

import keras.backend as K
from keras import layers
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input


from keras.layers import Input, Flatten, Dense, Dropout
from keras.models import Model

# Optimizers
from keras.optimizers import Adam  # Adam optimizer https://arxiv.org/abs/1412.6980
from keras.optimizers import SGD  # Stochastic gradient descent optimizer

from keras.applications.resnet50 import decode_predictions  # ResNet-specific routines for extracting predictions
from keras.preprocessing.image import load_img

from keras import models
from keras import layers
from keras import optimizers

import tensorflow as tf
import math
from keras.callbacks import TensorBoard

from keras.callbacks import ModelCheckpoint

from keras.callbacks import CSVLogger
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, EarlyStopping

from functools import partial

import pickle

image_size = 224
vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))



# Freeze the layers except the last 4 layers
for layer in vgg_conv.layers[:-4]:
    layer.trainable = False

for layer in vgg_conv.layers:
    print(layer, layer.trainable)


model = models.Sequential()

model.add(vgg_conv)

# Add new layers
model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(8, activation='softmax'))

model.summary()

train_datagen = ImageDataGenerator(
    preprocessing_function= preprocess_input,
    validation_split=0.4,
    brightness_range=[0.5,1.5],
    # rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

valid_datagen = ImageDataGenerator(
    preprocessing_function= preprocess_input,
    validation_split=0.4
)
# Change the batchsize according to your system RAM
train_batchsize = 100
# val_batchsize = 10


train_dir = '/home/carcolordataset200/'
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(image_size, image_size),
    batch_size=train_batchsize,
    class_mode='categorical',
    shuffle=True,
    seed=2,
    subset='training')

validation_generator1 = valid_datagen.flow_from_directory(
    train_dir,
    target_size=(image_size, image_size),
    batch_size=train_batchsize,
    class_mode='categorical',
    shuffle=True,
    seed=2,
    subset='validation')

validation_generator2 = train_datagen.flow_from_directory(
    train_dir,
    target_size=(image_size, image_size),
    batch_size=train_batchsize,
    class_mode='categorical',
    shuffle=True,
    seed=2,
    subset='validation')

print('----train------')
trainnames = train_generator.filenames
print(trainnames)

print('----valid 1------')
validnames1 = validation_generator1.filenames
print(validnames1)

print('----valid 2------')

validnames2 = validation_generator2.filenames
print(validnames2)
