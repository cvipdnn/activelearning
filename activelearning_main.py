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
from activelearning_sample import activelearning


#   Reference https://towardsdatascience.com/active-learning-on-mnist-saving-on-labeling-f3971994c7ba


IMG_SIZEX = 28
IMG_SIZEY = 28


def read_data():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    return x_train,  y_train, x_test, y_test 


if __name__ == '__main__':
    
    x_train,y_train, x_test, y_test = read_data()


    epochs = 156
            
   
    size = x_train.shape[0]
     
    
    my_list = list(range(0,size)) 
                              

    random.shuffle(my_list)

    x_train = x_train[my_list,:,:]
    y_train = y_train[my_list]

    
    start_num_samples=32
    end_num_samples = 16*128+32
    step_size = 16

    # start_num_samples is the number of the first sample, end_num_samples is the number of max samples to try, each time addes step_size samples. 
    #radom sample
    activelearning(x_train, y_train, x_test, y_test, start_num_samples, end_num_samples , step_size , epochs, 3, "rs")
    #least confidence
    activelearning(x_train, y_train, x_test, y_test, start_num_samples, end_num_samples , step_size, epochs , 3, "lc")
    #margin sampling
    activelearning(x_train, y_train, x_test, y_test, start_num_samples, end_num_samples , step_size, epochs , 3, "ms")
    #entropy
    activelearning(x_train, y_train, x_test, y_test, start_num_samples, end_num_samples , step_size, epochs , 3, "en")
    
    
    
    

    



  
    






