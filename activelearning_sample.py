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
from utils import get_model,build_callbacks 

#Reference https://github.com/dhaalves/CEAL_keras/blob/41c14306b328a1ccc87b14b6307c9781813df9a5/CEAL_keras.py#L71

def least_confidence(y_pred_prob):
    
    max_prob = np.max(y_pred_prob, axis=1)

    return np.argsort(max_prob)
    

def margin_sampling(y_pred_prob):

    margim_sampling = np.diff(-np.sort(y_pred_prob)[:, ::-1][:, :2])
    margim_sampling = margim_sampling[:,0]
    return np.argsort(margim_sampling)
  


def entropy_sampling(y_pred_prob):
    entropy = -np.nansum(np.multiply(y_pred_prob, np.log(y_pred_prob)), axis=1)
    
    #sort by -entropy
    return np.argsort(-entropy)
    

def random_sampling(y_pred_prob):

    size = y_pred_prob.shape[0]
    
    my_list = list(range(0,size)) 
                              
    random.shuffle(my_list)

    return np.array(my_list)


def activelearning(x_train, y_train, x_test, y_test, start_num_samples, end_num_samples , step_size, epochs, n_average, criteria ):

    print("===================================")

    submission = pd.DataFrame(columns=['#','accuracy', 'best_accuracy'])
    

    cur_x = x_train[0:start_num_samples,:,:]
    cur_y = y_train[0:start_num_samples]
    
    #iterative retrain after applying the network to trained model 
    back_x = x_train[start_num_samples:,:,:]
    back_y = y_train[start_num_samples:]

    n_iteration = (end_num_samples-start_num_samples)//step_size

    for j in range(n_iteration):
        
        avg_acc = 0 

        best_acc = 0 

        for k in range(n_average):
            model = get_model()
        
            callbacks = build_callbacks()

            model.fit(cur_x, cur_y,  callbacks=[callbacks], epochs=epochs, validation_data=(x_test, y_test), verbose = 0)

            model =tf.keras.models.load_model("params.h5")

            score, acc = model.evaluate(x_test, y_test, verbose=0)

            avg_acc += acc 

            if best_acc < acc : 
                best_acc = acc 
                bestmodel = tf.keras.models.load_model("params.h5")

        submission = submission.append(pd.DataFrame({'#': cur_x.shape[0], 'accuracy': [avg_acc/n_average] , 'best_accuracy':[best_acc] }))
            
        # use the remaining trainning datasets 
        predictions = bestmodel.predict (back_x)
        
        prob = tf.nn.softmax(predictions)

        if criteria == "en":
            index = entropy_sampling(prob)
        elif criteria == "rs":
           index = random_sampling(prob)
        elif criteria == "ms":
            index = margin_sampling(prob)
        elif criteria == "lc":
            index = least_confidence(prob)
        else:
            raise ValueError("Unknow criteria value ")


        # pick up step_size worst samples to train 
        cur_x = np.concatenate((cur_x, back_x[index[0:step_size], :,:]) )
        cur_y = np.concatenate((cur_y, back_y[index[0:step_size]]) )

        back_x  = back_x[index[step_size:],:,:]
        back_y  = back_y[index[step_size:]]
        


    # autolabel : use the remaining data 
    #     back_x, back_y 
    
    size = back_x.shape[0]
    n_iteration = size//step_size

    
     # reduce the size to save time for quick testing, add in the end , we can add during the sampling steps above 
    n_iteration = min(n_iteration, 32)
    for j in range(n_iteration):

        predictions = bestmodel.predict (back_x)
        
        prob = tf.nn.softmax(predictions)
        

        y_autolabeled = np.argmax(prob, axis=1)


        pmax = np.amax(prob, axis=1)
        pidx = np.argsort(pmax)
        back_x = back_x[pidx]
        y_autolabeled = y_autolabeled[pidx]

        cur_x = np.concatenate([cur_x, back_x[-step_size:]])
        cur_y = np.concatenate([cur_y, y_autolabeled[-step_size:]])
        back_x = back_x[:-step_size]

        avg_acc = 0 

        best_acc = 0 

        for k in range(n_average):
            model = get_model()
        
            callbacks = build_callbacks()

            model.fit(cur_x, cur_y,  callbacks=[callbacks], epochs=epochs, validation_data=(x_test, y_test), verbose = 0)

            model =tf.keras.models.load_model("params.h5")

            score, acc = model.evaluate(x_test, y_test, verbose=0)

            avg_acc += acc 

            if best_acc < acc : 
                best_acc = acc 
                bestmodel = tf.keras.models.load_model("params.h5")
    
    
    submission = submission.append(pd.DataFrame({'#': cur_x.shape[0], 'accuracy': [avg_acc/n_average] , 'best_accuracy':[best_acc] }))


    submission.to_csv('method_' + criteria + ".csv", index=False)
