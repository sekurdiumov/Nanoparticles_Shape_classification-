# -*- coding: utf-8 -*-
"""
Created on Wed Jan 07, 2021

@author: sk1u19
"""

#%% Libraries:

from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras import layers

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio

#%% Shuffling data together:
def shuffle_in_unison(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)
    
    return a, b


#%% Downloading dataset:
N_input = 5600

N_out = 8

Npix = 256

N1 = 44

D0 = np.zeros((700, Npix, Npix))
GTA = np.zeros((700, 1))
size = np.zeros((700, 1))
a = np.zeros((700, 1))
b = np.zeros((700, 1))
#offset_x = np.zeros((700, 1))
#offset_y = np.zeros((700, 1))

data = np.zeros((700, 6))

D00 = np.zeros((8*N1, Npix, Npix))
data0 = np.zeros((8*N1, 6))

counter = 0

for i in range (8):
    for j in range (700):
        filename = "dp" + str(i) + '_' + str(j) + '.mat'
        print(filename)
        M1 = scio.loadmat(filename)
        D1 = M1['Ex']
        D1 = D1 - np.min(D1)
        D0[j,:,:] = D1[0::4, 0::4]
        GTA = M1['label']
        size = M1['size']
        a = M1['a']
        b = M1['b']
        

        data[j,0] = GTA[0,0]
        data[j,1] = size[0,0]
        data[j,2] = a[0,0]
        data[j,3] = b[0,0]
        

    D0, data = shuffle_in_unison(D0, data)

    D01 = D0[:N1,:,:]
    data01 = data[:N1,:]

    D00[i*N1 : (i+1) * N1, :, :] = D01
    data0[i*N1 : (i+1) * N1, :] = data01

#data = np.zeros((N_input, 5))

#data[:,0] = GTA[:,0]
#data[:,1] = size[:,0]
#data[:,2] = a[:,0]
#data[:,3] = b[:,0]
#data[:,4] = angle[:,0]

D00 = D00/np.max(D00)

D00, data0 = shuffle_in_unison(D00, data0)

N_use = len(D00)
print('N_use = ', N_use)
#D00 = D0[:N_use,:,:]
#data0 = data[:N_use,:]

N_train = np.floor(N_use*0.9).astype(int)
N_val = N_input - N_train

img_size = (Npix, Npix)
origin_size = 1
#img_size = resize_size_Big

N_pixels = img_size
N_pixels_orgn = origin_size

if K.image_data_format() == 'channels_first':
    print('Channels First')     
    D00 = D00.reshape(N_use, 1, img_size[0], img_size[1])
    In_shape = (1,img_size[0], img_size[1])

else:
    print('Channels Last')     
    D00 = D00.reshape(N_use, img_size[0], img_size[1], 1)
    In_shape = (img_size[0], img_size[1],1)

imgs_train = D00[:N_train]
imgs_val = D00[N_train:]

origins_train = data0[:N_train, 0]
origins_val = data0[N_train:, 0]

size_train = data0[:N_train, 1]
size_val = data0[N_train:, 1]

a_train = data0[:N_train, 2]
a_val = data0[N_train:, 2]

b_train = data0[:N_train, 3]
b_val = data0[N_train:, 3]




dp_hl = 0.1

#Size of train and test sets
N_input = np.size(D0, axis=0)
N_train = np.size(origins_train, axis=0)
N_val = np.size(origins_val, axis=0)
assert (N_train+N_val) == N_use \
    ,"The train,val separation has problem."
    
#%% Neural network:
    
## NN architecture:
def create_model():
    model = keras.Sequential()
    model.add(layers.Conv2D(128, (3, 3), activation = 'relu', input_shape = In_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation = 'relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation = 'relu'))
    model.add(layers.Dense(64, activation = 'relu'))
    model.add(layers.Dense(N_out, activation = 'softmax'))
    
    return model

## Model training + predictions
model = create_model()    

batch_size = 32
    
#model.compile(loss='mae', optimizer='adam', metrics=['mse','mae'])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

callback1 = keras.callbacks.EarlyStopping(monitor = 'val_accuracy', 
                                         patience = 70, 
                                         mode = 'auto',
                                         restore_best_weights = True)

filepath = 'M1x'

callback2 = keras.callbacks.ModelCheckpoint(filepath,
                                            save_weights_only = True,
                                            monitor = 'val_accuracy',
                                            mode = 'max',
                                            save_best_only = True,
                                            save_freq = 'epoch')
    
history = model.fit(imgs_train, 
                    origins_train, 
                    epochs = 350, 
                    batch_size = batch_size, 
                    callbacks = [callback1, callback2], 
                    validation_split = 0.1)

model1 = create_model()

model1.load_weights(filepath)

model1.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

test_loss, test_acc = model1.evaluate(imgs_val, origins_val, verbose=2)
#
print('\nTest loss:', test_loss)
#
print('\nTest accuracy:', test_acc)

predictions = model1.predict(imgs_val)

#%%
## Saving the results:

fn1 = 'Results_reg_' + str(N_use) + '.mat'
scio.savemat(fn1, {'acc': history.history['accuracy'], 
                   'val_acc': history.history['val_accuracy'], 
                   'GroundTruth': origins_val, 
                   'Predictions': predictions,
                   'size': size_val,
                   'a': a_val,
                   'b': b_val
                   })


        

print(' ')
