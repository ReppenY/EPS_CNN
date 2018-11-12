import numpy as np
import pandas as pd
from collections import Counter
import pickle
import gensim
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten,  Concatenate, Reshape
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.optimizers import SGD
from sklearn.utils import shuffle
from keras.callbacks import TensorBoard
import random
from datetime import datetime
from keras.utils.vis_utils import model_to_dot


def save_dict(name, filename):
    pickle.dump(name, open('{0}.p'.format(filename), 'wb'))

def load_dict(filename):
    return pickle.load(open('{0}.p'.format(filename), 'rb'))



feats = load_dict('table_input_tensor_train')
targs = load_dict('table_target_train')

feats_test = load_dict('table_input_tensor_test')
targs_test = load_dict('table_target_test')

batchSize = 64
TotalSize = feats.shape[0]
sequence_length = feats.shape[1]
vec_dim = feats.shape[2]
num_classes= targs.shape[1]

#len(np.bincount(targs[0]))
#myOpt = keras.optimizers.SGD(lr=0.005,  epsilon=1e-08, decay=0.0001) #rho=0.9,

num_filters_conv1 = 100
num_filters_conv2 = 20 
###################################################################
model = Sequential()
model.add(Conv2D(filters = num_filters_conv1,kernel_size = (1,vec_dim), padding = 'valid',strides=1, use_bias=True,  activation='relu', 
                 input_shape = (sequence_length, vec_dim,1)))
model.add(Reshape((sequence_length,num_filters_conv1,1)))
#model.add(Flatten())
model.add(Conv2D(filters = num_filters_conv2, kernel_size= (1,num_filters_conv1), strides =1, padding = 'valid',use_bias= True,activation = 'relu',input_shape = (sequence_length, num_filters_conv1,1)))
#model.add(Conv2D(filters = num_filters_conv2,kernel_size = (1,num_filters_conv1), padding = 'valid', strides=(,),use_bias=True,  activation='relu' ,input_shape = (sequence_length, num_filters_conv1)  ))
model.add(Reshape((sequence_length,num_filters_conv2,1)))

model.add(MaxPooling2D(pool_size = (1,num_filters_conv2)))

model.add(Flatten())
#model.add(Dropout(0.3))
#model.add(Dense(500, activation = 'tanh'))
#model.add(Dropout(0.15))
model.add(Dense(50, activation = 'relu'))
model.add(Dropout(0.15))
model.add(Dense(num_classes, activation = 'softmax'))
model.summary()
########################################################################
 
##submodels = []
##for f_size in (1,2,3):
##    submodel = Sequential()
##    submodel.add(Conv2D(filters = 10, kernel_size = (f_size, vec_dim), padding = 'valid', strides = 1, activation = 'tanh'))
##    submodel.add(MaxPooling2D(pool_size = (sequence_length,1)))
##    submodels.append(submodel)


##big_model = Sequential()
##big_model.add(Concatenate(submodels))
##big_model.add(Flatten())
##big_model.add(Dense(num_classes, activation = 'softmax'))

#########################################################################################


model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])


tensorboard = TensorBoard(log_dir='./cnn_logs', histogram_freq=1,
                          write_graph=True, write_images=True, write_grads=True )

Checkpoint = keras.callbacks.ModelCheckpoint("best2_tmp" , monitor='val_acc', verbose=1, save_best_only=True,
                                    save_weights_only=False, mode='auto', period=1)


for z in range(100000):
    print("----")
    #x,y = getNextBatch(features, targets, batchSize) 
    x, y = shuffle(feats, targs)

    hist = model.fit(x, y, batch_size=batchSize, epochs=1, callbacks=[Checkpoint, tensorboard], verbose=2 , validation_split = 0.15)
    
    if z%3 == 0:
        #pass
        score = model.evaluate(feats_test, targs_test, batch_size=100)
        print('__________________Score____________________: ' ,score)
        score1 = model.evaluate(feats, targs, batch_size=1000)
        print('__________________Score1____________________: ' ,score1)
