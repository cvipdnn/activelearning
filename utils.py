import numpy as np 
import tensorflow as tf
from tensorflow.keras import Input, Model
from keras.utils import np_utils
from tensorflow.keras.callbacks import *
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from matplotlib import pyplot

from tensorflow.keras.optimizers import Adam
import random
from keras.utils.np_utils import to_categorical  
import cv2
import h5py
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Model, layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import Input
from  tensorflow.keras.metrics import *
from keras.optimizers import SGD
import random
import pandas as pd
#   Reference https://towardsdatascience.com/active-learning-on-mnist-saving-on-labeling-f3971994c7ba


IMG_SIZEX = 28
IMG_SIZEY = 28

def get_model():

    inputs = Input((IMG_SIZEX, IMG_SIZEY))
  
    x = Flatten()(inputs)
    x = Dense(128, activation = 'relu')(x)
    output = Dense(10, activation = None)(x)

    model = Model(inputs, output)

    optimizer = Adam(lr=0.001)

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(optimizer = optimizer,
            loss = loss_fn, 
             metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]) 

    return model 


def build_callbacks():

    #scheduler = tf.keras.callbacks.LearningRateScheduler(lr_scheduler) 
    #callbacks = [scheduler]

    rlr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.667, patience = 6, verbose = 0,  min_delta = 0.0001, mode = 'min')
    callbacks = [rlr]

    checkpoint = ModelCheckpoint(filepath='params.h5', mointor="val_loss", verbose=0, save_weights_only=False, save_best_only=True, mode="min")
    callbacks.append(checkpoint)

    earlystop = EarlyStopping(monitor='val_loss', patience=12, mode='min', min_delta=0.00001)
    callbacks.append(earlystop)


    return callbacks


