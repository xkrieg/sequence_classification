# -*- coding: utf-8 -*-

#Import libraries
import numpy as np
import pandas as pd

#Create emotion channels
def make_stack(X, window_size):

    X = np.dstack((X[:,window_size*0:window_size*1],  #Anger
                   X[:,window_size*1:window_size*2],  #Contempt
                   X[:,window_size*2:window_size*3],  #Disgust
                   X[:,window_size*3:window_size*4],  #Fear
                   X[:,window_size*4:window_size*5],  #Happiness
                   X[:,window_size*5:window_size*6],  #Neutral
                   X[:,window_size*6:window_size*7],  #Sadness
                   X[:,window_size*7:window_size*8])) #Surprise
    X = X.reshape(X.shape[0], window_size, 8)
    return X

def create_totally_random(n_cases, window_size, n_channels, multilabel = 0):
    
    X = np.random.rand(n_cases, window_size, n_channels)
    if multilabel == 0 or multilabel == 1:
        y = np.random.randint(2, size = (n_cases, ))
        y = y.reshape(n_cases, 1)
    else:
        y = np.random.randint(2, size = (n_cases, multilabel))
    
    return(X, y)
    
if __name__ == "__main__":
    print(create_totally_random(10, 10, 2))
