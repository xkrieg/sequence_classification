# -*- coding: utf-8 -*-

#Import libraries
from numpy import floor
from keras.models import Sequential
from keras.initializers import TruncatedNormal
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, InputLayer
from keras.layers import Masking, TimeDistributed

#Build basic model architecture
def get_model(X, y, win_size, conv_node, dense_node, activation,
              last_activation, initialize = False, print_model = False):
    
    #Create model
    model = Sequential()
    model.add(InputLayer(input_shape = (X.shape[1], X.shape[2])))

    #Set up replicable initializers
    if initialize == True:
        my_init = TruncatedNormal(mean=0.0, stddev=0.05, seed=3223)
        model.add(Conv1D(conv_node, kernel_size = int(floor(win_size/10)),
                  strides = (2), activation = activation,
                  kernel_initializer = my_init, bias_initializer = my_init))
    else:
        model.add(Conv1D(conv_node, kernel_size = int(floor(win_size/10)),
                  strides = (2), activation = activation))
    
    #Add convolutions
    model.add(MaxPooling1D(2))
    model.add(Conv1D(conv_node, kernel_size = (2),
                  strides = (1), activation = activation))
    model.add(MaxPooling1D(2))
    model.add(Flatten())
    model.add(Dense(dense_node, activation = activation))
    model.add(Dense(y.shape[1], activation = last_activation))
    if print_model is True:
        print(model.summary())
    
    return model
