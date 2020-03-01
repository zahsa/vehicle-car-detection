import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import keras
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

from datetime import datetime

import pickle

logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir,
                                                   # histogram_freq=0,
                                                   # batch_size=32,
                                                   # write_graph=True,
                                                   # write_grads=False,
                                                   # write_images=False,
                                                   # embeddings_freq=0,
                                                   # embeddings_layer_names=None,
                                                   # embeddings_metadata=None,
                                                   # embeddings_data=None,
                                                   # update_freq='epoch'
                                                   )

# checkpoints
# checkpoint_path = "training_bestchkpt/stanfcar.ckpt"
checkpoint_path = '/home/training_bestchkpt/stcarbest256_' +  datetime.now().strftime("%Y%m%d-%H%M%S") + '.ckpt'

# weights.{epoch:02d}-{val_loss:.2f}.hdf5

checkpoint_dir = os.path.dirname(checkpoint_path)

# Create checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 # save_weights_only=True,
                                                 save_best_only=True,
                                                 # verbose=1,
                                                 # Save weights, every 5-epochs.
                                                 # period=5,
                                                 monitor='val_acc',
                                                 verbose=1,
                                                 mode='max')
# cp_callback = tf.keras.callbacks.ModelCheckpoint("./weights.{epoch:02d}.hdf5",
#                                           save_weights_only=True,
#                                           verbose=1)
image_size = 224
vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
# baseModel = VGG16(weights="imagenet", include_top=False,input_tensor=Input(shape=(image_size, image_size, 3)))

# Freeze the layers except the last sl layers
sl = 8
for layer in vgg_conv.layers[:-sl]:
    layer.trainable = False
# print('skipped layers:',sl)

# # freeze all layers
# for layer in vgg_conv.layers:
# 	layer.trainable = False

# import ipdb;ipdb.set_trace()
# for layer in vgg_conv.layers:
#     print(layer, layer.trainable)


model = models.Sequential()
model.add(vgg_conv)
model.summary()
model._ckpt_saved_epoch = None
# Add new layers
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(8, activation='softmax'))
model.summary()

train_datagen = ImageDataGenerator(
    preprocessing_function= preprocess_input,
    validation_split=0.2,
    brightness_range=[0.5,1.5],
    # rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

valid_datagen = ImageDataGenerator(
    preprocessing_function= preprocess_input,
    validation_split=0.2,
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


validation_generator = valid_datagen.flow_from_directory(
    train_dir,
    target_size=(image_size, image_size),
    batch_size=train_batchsize,
    class_mode='categorical',
    shuffle=True,
    seed=2,
    subset='validation')


# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)



history = model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples / train_generator.batch_size,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples / validation_generator.batch_size,
    # callbacks=[tensorboard_callback],
    callbacks=[cp_callback,tensorboard_callback],
    verbose=1)


print("Average validation loss: ", np.average(history.history['loss']))

filename = 'valid_history4.pickle'
with open(filename, 'wb') as f:  # Overwrites any existing file.
    pickle.dump(history, f, pickle.HIGHEST_PROTOCOL)


# model.evaluate(test_dataset)

# Save the model
# model.save('model4_carStanf_cropped_color8.h5')
# model.save_weights('weights_carStanfcolor8.h5')
# model_json=model.to_json()
# with open(“classes.json”,”w”) as json_file:
# json_file.write(model_json)

# model.save_weights(checkpoint_path.format(epoch=0))

%matplotlib inline
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# fnames = validation_generator.filenames
# filename = 'valid_imnames4.pickle'
# with open(filename, 'wb') as f:
#     pickle.dump(fnames, f, pickle.HIGHEST_PROTOCOL)
# ground_truth = validation_generator.classes
# filename = 'valid_imlabels4.pickle'
# with open(filename, 'wb') as f:
#     pickle.dump(ground_truth, f, pickle.HIGHEST_PROTOCOL)

fnames = validation_generator.filenames
# valid_imglabels = validation_generator.labels
ground_truth = validation_generator.classes

best_model = load_model(checkpoint_path)

imsize = 224
true_y = []
pred_y = []
error = 0
for i in range(len(fnames)):
    y = ground_truth[i]
    true_y.append(y)
    title = 'Original label:{}'.format(
        fnames[i])

    img = load_img('{}/{}'.format(train_dir,fnames[i]),target_size=(imsize, imsize))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    pred = best_model.predict(x)
    predlabel = np.argmax(pred, axis=1)
    pred_y.append(predlabel)
    if predlabel != y:
       error = error + 1
    print('pred label', predlabel)
    print('ground truth label',y)
    # plt.figure(figsize=[7,7])
    # plt.axis('off')
    # plt.title(title)
    # plt.imshow(img)
    # plt.show()

print('acc rate',(len(pred_y)-error)/len(pred_y))
print('error rate',error/len(pred_y))




import sklearn.metrics as metrics
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(ground_truth, pred_y))
