# -*- coding: utf-8 -*-

#Suppress Tensorflow information
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#Import libraries
import numpy as np
import pandas as pd
#import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow import test
from keras.optimizers import Nadam
import datetime
import re

#Local libraries
from data_creator import create_totally_random
from models.model_archt import get_model

#Check gpu support
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return len([x.name for x in local_device_protos if x.device_type == 'GPU'])

def gpu_select(implement = True):
    
    #Suppress Tensorflow information
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    #Ensure GPU functioning
    if implement == True:
        if test.gpu_device_name():
            print('Default GPU Device: {}'.format(test.gpu_device_name()))
            if get_available_gpus() > 1: #Add multi-gpu support if available
                model = multi_gpu(model)
        else:
            print("Please install GPU version of TF")
            quit()

def learn(n_cases, time_sec, n_channels, multiclass):
    
    #Generate random data
    X, y = create_totally_random(n_cases, time_sec, n_channels, multiclass)
    
    #Get model
    model = get_model(X, y, time_sec, conv_node = 32, 
                      dense_node = 96, activation = 'relu', 
                      last_activation = 'sigmoid', initialize = False, 
                      print_model = True)
                      
    #Compile model
    model.compile(loss = 'binary_crossentropy', metrics=['accuracy'],                                   
                  optimizer = Nadam(lr=.001))
                  
    #Train model
    out = model.fit(X, y, batch_size = 200, epochs = 200, validation_split = .10)
    
    return(model, out)
    
def run(parameters, force_gpu = False):
    
    #Force Use GPU?
    gpu_select(force_gpu)
    
    #Specify output
    output = np.zeros((0, 9))
    
    for n_case in parameters["n_cases"]:
        for win_size in parameters["win_sizes"]:
            for n_channel in parameters["n_channels"]:
                for out_class in parameters["multiclass"]:
    
                    #Train model
                    model, out = learn(n_case, win_size, n_channel, out_class)
                    
                    #Append new data to output
                    to_append = np.asarray([str(parameters["randomized"][0]),
                                            n_case, win_size, n_channel, out_class, 
                                            out.history['loss'][199], 
                                            out.history['acc'][199],
                                            out.history['val_loss'][199], 
                                            out.history['val_acc'][199]])
                    to_append.reshape(1,9)
                    output = np.vstack([output, to_append])
                    df = pd.DataFrame(output)
                    df.columns = ["randomized", "n_cases", "win_sizes",
                                  "n_channels", "multiclass", "loss", "accuracy",
                                  "val_loss", "val_accuracy"]
                    
                    #Save to file
                    current_date = re.sub('[^A-Za-z0-9]+', '', str(datetime.datetime.now()))
                    outfile = "".join(["output/", str(parameters["randomized"][0]),
                                       "_exp.csv"])#, current_date, ".csv"])
                    df.to_csv(outfile)

if __name__ == "__main__":

    #Set parameters
    p = {"n_cases":    [1000, 10000, 100000, 1000000, 10000000, 100000000],
         "win_sizes":  [20, 30, 45, 60, 90, 120, 150],
         "n_channels": [1, 2, 6, 8, 15],
         "multiclass": [0, 2, 4, 8, 10, 20],
         "randomized": ["non_sequence_uniform"]}

    total_params = sum(len(item) for item in p.values())
    print("Total items:", total_params)

    #Run experiment
    run(p, force_gpu = True)
