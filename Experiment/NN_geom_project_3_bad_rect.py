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
N_in = 27
N_input = 1206

Npix = 256

D0 = np.zeros((N_input, 1, Npix, Npix))

data = np.zeros((N_input, 8))

counter = 0
counter1 = 0

M0 = scio.loadmat('discard_bad_rect.mat')
discard = M0['discard']

M1 = scio.loadmat('Test_selection_bad_rect.mat')
Test_inds = M1['Test_inds']

for i in range (N_in):
    count = 0
    filename2 = 'GT_G' + str(i) + '.mat'
    M2 = scio.loadmat(filename2)
    
    a = M2['a']
    b = M2['b']
    GT = M2['GT']
    size = M2['size']
    
    
    for j in range (8):
        for k in range (8):
            if discard[0, counter1] == 0:

                data[counter,0] = GT[0,count]

                if GT[0, count] > 1:
                    data[counter, 0] = GT[0, count] - 1

                data[counter,1] = size[0, count]
                data[counter,2] = a[0, count]
                data[counter,3] = b[0, count]
                data[counter,4] = i
                data[counter,5] = j
                data[counter,6] = k
                data[counter,7] = Test_inds[0,counter]

                for n in range (1):
                    filename = "dp" + str(i) + '-' + str(n+1) + '-' + str(j) + '-' + str(k) + '.mat'
                    M1 = scio.loadmat(filename)
                    D1 = M1['E']
                    D1 = D1 - np.min(D1)
            
                
                    #D0[counter,:,:] = D1[128:-128, 128:-128].astype(float) 
                    D0[counter,n,:,:] = D1[0::2, 0::2].astype(float) 
                    D0[counter,n,:,:] = D0[counter,n,:,:] - np.min(D0[counter,n,:,:])
                    D0[counter,n,:,:] = D0[counter,n,:,:] / np.max(D0[counter,n,:,:])
                    #D0[counter,:,:] = D1.astype(float)      
              
                    
                
                counter = counter + 1
             
            counter1 = counter1 + 1   
            count = count + 1

print('counter = ', counter)

D0 = D0/np.max(D0)

D0, data = shuffle_in_unison(D0, data)

N_train = np.floor(N_input*0.9).astype(int)
N_val = N_input - N_train

inds1 = np.where(data[:,7] == 1)
inds1 = inds1[0]

inds0 = np.where(data[:,7] == 0)
inds0 = inds0[0]

D_train_a = D0[inds0]
data_train_a = data[inds0]
D_test = D0[inds1]
data_test = data[inds1]

scio.savemat('data_test_bad_rect.mat', {'data_test' : data_test})

D_train = D_train_a[:int(N_train*0.9)]
data_train = data_train_a[:int(N_train*0.9)]
D_val = D_train_a[int(N_train*0.9):]
data_val = data_train_a[int(N_train*0.9):]

scio.savemat('data_train_bad_rect.mat', {'data_train' : data_train})
scio.savemat('data_val_bad_rect.mat', {'data_val' : data_val})

Dtrain0 = D_train[:,0,:,:]
#Dtrain1 = D_train[:,1,:,:] 
#Dtrain2 = D_train[:,2,:,:]
#Dtrain3 = D_train[:,3,:,:] 

#Dtrain0 = np.concatenate((Dtrain0, 
#                          Dtrain1, 
#                          Dtrain2, 
#                          Dtrain3), 
#                         axis = 0)

#data_train = np.concatenate((data_train, 
#                             data_train, 
#                             data_train, 
#                             data_train), 
#                            axis = 0)

Dtrain0, data_train = shuffle_in_unison(Dtrain0, data_train)

Dval0 = D_val[:,0,:,:]
#Dval1 = D_val[:,1,:,:] 
#Dval2 = D_val[:,2,:,:]
#Dval3 = D_val[:,3,:,:] 

#Dval0 = np.concatenate((Dval0, 
#                        Dval1, 
#                        Dval2, 
#                        Dval3), 
#                       axis = 0)

#data_val = np.concatenate((data_val, 
#                           data_val, 
#                           data_val, 
#                           data_val), 
#                          axis = 0)


Dval0, data_val = shuffle_in_unison(Dval0, data_val)

Dtest0 = D_test[:,0,:,:]
#Dtest1 = D_test[:,1,:,:] 
#Dtest2 = D_test[:,2,:,:]
#Dtest3 = D_test[:,3,:,:] 

#Dtest0 = np.concatenate((Dtest0, 
#                          Dtest1, 
#                          Dtest2, 
#                          Dtest3), 
#                         axis = 0)

#data_test = np.concatenate((data_test, 
#                             data_test, 
#                             data_test, 
#                             data_test), 
#                            axis = 0)

Dtest0, data_test = shuffle_in_unison(Dtest0, data_test)

D00 = np.concatenate((Dtrain0, 
                      Dval0, 
                      Dtest0), 
                     axis = 0)

data0 = np.concatenate((data_train, 
                        data_val, 
                        data_test), 
                       axis = 0)

img_size = (Npix, Npix)
origin_size = 1
#img_size = resize_size_Big

N_pixels = img_size
N_pixels_orgn = origin_size

if K.image_data_format() == 'channels_first':
    print('Channels First')     
    D00 = D00.reshape(N_input, 1, img_size[0], img_size[1])
    In_shape = (4,img_size[0], img_size[1])

else:
    print('Channels Last')     
    D00 = D00.reshape(N_input, img_size[0], img_size[1], 1)
    In_shape = (img_size[0], img_size[1],1)

imgs_train = D00[:N_train]
imgs_val = D00[N_train:]

origins_train = data0[:N_train, 0]
origins_val = data0[N_train:, 0]

print('origin_train size = ', len(origins_train))

size_train = data0[:N_train, 1]
size_val = data0[N_train:, 1]

a_train = data0[:N_train, 2]
a_val = data0[N_train:, 2]

b_train = data0[:N_train, 3]
b_val = data0[N_train:, 3]

dp_hl = 0.1

#Size of train and test sets
N_input = np.size(D00, axis=0)
N_train = np.size(origins_train, axis=0)
N_val = np.size(origins_val, axis=0)
assert (N_train+N_val) == N_input \
    ,"The train,val separation has problem."
    
#%% Neural network:
    
## NN architecture:
def create_model():
    model = keras.Sequential()
    model.add(layers.Conv2D(256, (3, 3), activation = 'relu', input_shape = In_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    #model.add(layers.Conv2D(128, (3, 3), activation = 'relu'))
    #model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation = 'relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation = 'relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation = 'relu'))
    #model.add(layers.Dropout(0.2))
    model.add(layers.Dense(64, activation = 'relu'))
    #model.add(layers.Dropout(0.2))
    model.add(layers.Dense(5, activation = 'softmax'))
    
    return model

## Model training + predictions
model = create_model()    

batch_size = 32
    
#model.compile(loss='mae', optimizer='adam', metrics=['mse','mae'])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

callback1 = keras.callbacks.EarlyStopping(monitor = 'val_accuracy', 
                                         patience = 100, 
                                         mode = 'auto',
                                         restore_best_weights = True)

filepath = 'M2'

callback2 = keras.callbacks.ModelCheckpoint(filepath,
                                            save_weights_only = True,
                                            monitor = 'val_accuracy',
                                            mode = 'auto',
                                            save_best_only = True,
                                            save_freq = 'epoch')
    
history = model.fit(imgs_train, 
                    origins_train, 
                    epochs = 400, 
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

fn1 = 'Results_reg_5classes_' + str(N_input) + '.mat'
scio.savemat(fn1, {'acc': history.history['accuracy'], 
                   'val_acc': history.history['val_accuracy'], 
                   'GroundTruth': origins_val, 
                   'Predictions': predictions,
                   'size': size_val,
                   'a': a_val,
                   'b': b_val})

#%% Group dataset by sizes => separate predictions:
#N1 = 9

#A0 = [[] for xxx in range(N1)]
#GTA0 = [[] for xxx in range(N1)]
#a0 = [[] for xxx in range(N1)]
#b0 = [[] for xxx in range(N1)]
#mode0 = [[] for xxx in range(N1)]

#for j in range(N1):
#    for i in range (len(size_val[:,0])):
#        if size_val[i,0]*50 >= 1.25*j + 9.9 and size_val[i,0]*50 <= 1.25*j + 10.1:
#            A0[j].append(imgs_val[i,:,:,:])
#            GTA0[j].append(origins_val[i,0])
#            a0[j].append(a[i,0])
#            b0[j].append(b[i,0])
#            mode0[j].append(mode[i,0])
#            
#    A0[j] = np.array(A0[j])
#    GTA0[j] = np.array(GTA0[j])
#    a0[j] = np.array(a0[j])
#    b0[j] = np.array(b0[j])
#    mode0[j] = np.array(mode0[j])
#    
#   
#print(' ================================================================== ')

## Predictions:
#Acc = np.zeros(N1)
#preds_size = []
#GT_size = []
#a_size = []
#b_size = []
#mode_size = []

def pred_by_size(model, A0, GTA0):
    
    if GTA0.size == 0:
        
        print('array is empty')
        
        predictions = []
        
        test_acc01 = 1
        
    else:
        
        test_loss01, test_acc01 = model.evaluate(A0, GTA0, verbose=2)

        print('\nTest accuracy:', test_acc01)

        predictions = model.predict(A0)
    
    print(' ================================================================== ')

    return predictions, test_acc01

#for i in range(N1):
#    preds_sizes, Acc[i] = pred_by_size(model1, A0[i], GTA0[i])
#    preds_size.append(preds_sizes)
#    GT_size.append(GTA0[i])
#    a_size.append(a0[i])
#    b_size.append(b0[i])  
#    mode_size.append(mode0[i])

## Saving the results:
#sizes = np.linspace(0.25, 0.5, N1)
#fname = 'Result_sizes_'+ str(N_input) + '.mat'
#scio.savemat(fname, {'size': sizes, 
#                     'acc': Acc, 
#                     'preds': preds_size, 
#                     'GT': GT_size,
#                     'a' : a_size,
#                     'b' : b_size,
#                     'mode' : mode_size})

print(' ')
