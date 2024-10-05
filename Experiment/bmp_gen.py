# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 22:32:58 2021

@author: sk1u19
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from numpy.lib import scimath as SC
import time
import matplotlib.colors
import csv
import random as rnd
import scipy.io as scio
from PIL import Image
from scipy.ndimage import rotate
#import cv2
from Shapes import Shapes

tm = time.time()

matplotlib.rcParams.update({'font.size' : 16})
pi = np.pi

plt.style.use('default') 
## ============================================================================
## Generating bitmap of given size and given shape:
## ============================================================================
def gen_size_shape(inputField, PWL, size, i):
    
    p = 0
    
    #size = int(size * PWL)
    R = int(size/2)
    
    if i == 0:
        a = size   
        b = R
        IF, size, a, b = Shapes.triangle(inputField, PWL, size/PWL, a/PWL, b/PWL, p)
        
    elif i == 1:
        a = int(size/np.sqrt(2))
        b = int(size/np.sqrt(2))
        IF, size, a, b = Shapes.rectangle(inputField, PWL, size/PWL, a/PWL, b/PWL, p)
        
    elif i == 2:
        a = R
        b = R
        t = rnd.randint(0, 1)
        if t == 0:
            tmp = a
            a = b
            b = tmp
        IF, size, a, b = Shapes.ellipse(inputField, PWL, size/PWL, a/PWL, b/PWL, p)
        
    elif i == 3:
        a = int(size/np.sqrt(2))
        b = int(size/np.sqrt(2))
        a = a/2
        b = b/2
        IF, size, a, b = Shapes.zigzag(inputField, PWL, size/PWL, a/PWL, b/PWL, p)
        
    elif i == 4:
        a = size
        b = size
        a = a/2
        b = b/2
        IF, size, a, b = Shapes.plus(inputField, PWL, size/PWL, a/PWL, b/PWL, p)
        
    elif i == 5:
        a = R
        b = size - a
        IF, size, a, b = Shapes.y_shape(inputField, PWL, size/PWL, a/PWL, b/PWL, p)
    
    IF.astype(int)
    
    size = size / PWL
    a = a / PWL
    b = b / PWL
    
    return IF, size, a, b

## ============================================================================
## Row in bitmap - shapes of one size:
## ============================================================================
def one_size(size, rot):
    
    Nmax = 1000
    Nmay = 1000
    #L0 = 10
    #spacing = L0/(Nmax)   
    PWL = 64
    print ('Points per wavelength: ', PWL)                                                
    inputField = np.ones((Nmax, Nmay))*0
    
    NX2 = int(Nmax/2)
    
    #size = size * round(PWL)
    IF, size1, a, b = gen_size_shape(inputField, PWL, size, 0)
    IF0 = IF[NX2 - round(PWL/2):NX2 + round(PWL/2), 
              NX2 - round(PWL/2) : NX2 + round(PWL/2)]
    
    # IF0 = IF[NX2 - PWL:NX2 + PWL, 
    #           NX2 - PWL:NX2 + PWL]
    
    threshold = 0.1
    
    if rot:
        angle = rnd.randint(-180, 180)
        IF0 = rotate(IF0, angle, reshape = False)
        
        IF0[IF0 > threshold] = 1
        IF0[IF0 < threshold] = 0
    
    for i in range (1, 6):
        IF, size1, a, b = gen_size_shape(inputField, PWL, size, i)
        IF1 = IF[NX2 - round(PWL/2):NX2 + round(PWL/2), 
                  NX2 - round(PWL/2):NX2 + round(PWL/2)]
        
        # IF1 = IF[NX2 - PWL:NX2 + PWL, 
        #           NX2 - PWL:NX2 + PWL]
        
        if rot:
            angle = rnd.randint(-180, 180)
            IF1 = rotate(IF1, angle, reshape = False)
        
            IF1[IF1 > threshold] = 1
            IF1[IF1 < threshold] = 0
        
        IF0 = np.concatenate((IF0, IF1), axis = 1)

    
    return IF0

## ============================================================================
## Real bitmap:
## ============================================================================
def genBitmap(t, size, a, b, rot):
    
    Nmax = 1500
    Nmay = Nmax
    #L0 = 10
    #spacing = L0/(Nmax - 0)   
    PWL = 64
    print ('Points per wavelength: ', PWL)                                                
    inputField = np.ones((Nmax, Nmay))*0
    
    NX2 = int(Nmax/2)
    
    N = 9
    
    threshold = 0.1
    
    IF = Shapes.InputField_exp(t[0], size[0], inputField, PWL, a[0], b[0])
    IF0 = IF
    
    if rot:
        angle = rnd.randint(-180, 180)
        IF0 = rotate(IF0, angle, reshape = False)
        
    IF0[IF0 > threshold] = 1
    IF0[IF0 < threshold] = 0
    
    
    for i in range (1, N):
        IF = Shapes.InputField_exp(t[i], size[i], inputField, PWL, a[i], b[i])
        
        IF1 = IF
        
        if rot:
            angle = rnd.randint(-180, 180)
            IF1 = rotate(IF1, angle, reshape = False)
            
        IF1[IF1 > threshold] = 1
        IF1[IF1 < threshold] = 0
        
        IF0 = np.concatenate((IF0, IF1), axis = 1)
        
    counter = N
    #print(counter)
    
    for i in range(1, N):
        IF = Shapes.InputField_exp(t[counter], size[counter],
                inputField, PWL, a[counter], b[counter])
        IF2 = IF
        
        if rot:
            angle = rnd.randint(-180, 180)
            IF2 = rotate(IF2, angle, reshape = False)
            
        IF2[IF2 > threshold] = 1
        IF2[IF2 < threshold] = 0
        
        counter = counter + 1
        
        for i in range(1, N):
            
            IF = Shapes.InputField_exp(t[counter], size[counter],
                    inputField, PWL, a[counter], b[counter])
            
            IF1 = IF
            
            if rot:
                angle = rnd.randint(-180, 180)
                IF1 = rotate(IF1, angle, reshape = False)
                
            IF1[IF1 > threshold] = 1
            IF1[IF1 < threshold] = 0
            
            IF2 = np.concatenate((IF2, IF1), axis = 1)
            counter = counter + 1
    
        #print(counter)
    
        IF0 = np.concatenate((IF0, IF2), axis = 0)
        
    ## Alignment marks:
    # I = np.zeros((Nmax, Nmax))
    # I[NX2 - 8 : NX2 + 8, NX2 - 8 : NX2 + 8] = 1
    # I1 = np.zeros((Nmax, Nmax))
    
    # for i in range (N - 1):
    #     I = np.concatenate((I, I1), axis = 0)
        
    # IF0 = np.concatenate((I, IF0), axis = 1)
    
    return IF0

## ============================================================================
## Test bitmap:
## ============================================================================
def genTestBitmap(N, t, size, a, b, rot):
    
    Nmax = 200
    Nmay = Nmax
    #L0 = 10
    #spacing = L0/(Nmax - 0)   
    PWL = 64
    print ('Points per wavelength: ', PWL)                                                
    inputField = np.ones((Nmax, Nmay))*0
    
    NX2 = int(Nmax/2)
    
    #N = 8
    
    threshold = 0.1
    
    IF = Shapes.InputField_exp(t[0], size[0], inputField, PWL, a[0], b[0])
    IF0 = IF
    
    if rot:
        angle = rnd.randint(-180, 180)
        IF0 = rotate(IF0, angle, reshape = False)
        
    IF0[IF0 > threshold] = 1
    IF0[IF0 < threshold] = 0
    
    
    for i in range (1, N):
        IF = Shapes.InputField_exp(t[i], size[i], inputField, PWL, a[i], b[i])
        
        IF1 = IF
        
        if rot:
            angle = rnd.randint(-180, 180)
            IF1 = rotate(IF1, angle, reshape = False)
            
        IF1[IF1 > threshold] = 1
        IF1[IF1 < threshold] = 0
        
        IF0 = np.concatenate((IF0, IF1), axis = 1)
        
    counter = N
    #print(counter)
    
    for i in range(1, N):
        IF = Shapes.InputField_exp(t[counter], size[counter],
                inputField, PWL, a[counter], b[counter])
        IF2 = IF
        
        if rot:
            angle = rnd.randint(-180, 180)
            IF2 = rotate(IF2, angle, reshape = False)
            
        IF2[IF2 > threshold] = 1
        IF2[IF2 < threshold] = 0
        
        counter = counter + 1
        
        for i in range(1, N):
            
            IF = Shapes.InputField_exp(t[counter], size[counter],
                    inputField, PWL, a[counter], b[counter])
            
            IF1 = IF
            
            if rot:
                angle = rnd.randint(-180, 180)
                IF1 = rotate(IF1, angle, reshape = False)
                
            IF1[IF1 > threshold] = 1
            IF1[IF1 < threshold] = 0
            
            IF2 = np.concatenate((IF2, IF1), axis = 1)
            counter = counter + 1
    
        #print(counter)
    
        IF0 = np.concatenate((IF0, IF2), axis = 0)
        
    ## Alignment marks:
    I = np.zeros((Nmax, Nmax))
    I[NX2 - 8 : NX2 + 8, NX2 - 8 : NX2 + 8] = 1
    I1 = np.zeros((Nmax, Nmax))
    
    for i in range (N - 1):
        I = np.concatenate((I, I1), axis = 0)
        
    IF0 = np.concatenate((I, IF0), axis = 1)
    
    return IF0

## calibration sample:
def genCalibBitmap():
    
    inputField = np.zeros((1000, 1000))
    
    IF = Shapes.InputField_exp(2, 0.5, inputField, 64, 0.25, 0.25)
    
    IF = np.concatenate((IF, IF), axis = 0)
    IF = np.concatenate((IF, IF), axis = 1)
    
    return IF

#defining the steps
def genStepBitmap():
    
    inputField = np.zeros((1000, 1000))
    
    IF = Shapes.InputField_exp(2, 0.5, inputField, 64, 0.125, 0.125)
    
    IF = np.concatenate((IF, IF, IF), axis = 0)
    IF = np.concatenate((IF, IF, IF), axis = 1)
    
    return IF

def genDBitmap(N):
    
    inputField = np.zeros((N, N))
    
    IF1 = Shapes.InputField_exp(2, 0.5, inputField, 64, 0.25, 0.25)
    IF2 = Shapes.InputField_exp(2, 0.25, inputField, 64, 0.125, 0.125)
    
    IF1 = np.concatenate((IF1, IF2, IF1), axis = 0)
    
    IF = Shapes.InputField_exp(2, 0.5, inputField, 64, 0.25, 0.25)
    IF = np.concatenate((IF, IF, IF), axis = 0)
    
    IF = np.concatenate((IF, IF1, IF), axis = 1)
    
    return IF
    
## ============================================================================
## Main:
## ============================================================================
N = 81
#rot = 1

#for j in range(30):
#
#    t = np.zeros(N)
#    
#    for i in range (N):
#        t[i] = rnd.randint(0, 5)
#    
#    IF0, size = genBitmap(t, rot)
#    
#    #plt.figure()
#    #plt.pcolormesh(IF0, cmap = 'gray')
#    
#    ## Save .bmp file:
#    fn1 = 'A' + str(j) + '.bmp'
#    matplotlib.image.imsave(fn1, IF0, cmap = 'gray')
#    
#    img = Image.open(fn1)
#    newimg = img.convert(mode='RGB', colors=24)
#    newimg.save(fn1)
#    
#    fn2 = 'GT_A' + str(j) + '.mat'
#    scio.savemat(fn2, {'GT': t, 'size': size})
#    
#    print('j = ', j)
#
#sizes = np.arange(160, 340, 20)/640
#I1 = one_size(sizes[0], rot = 1)
#for i in range(len(sizes) - 1):
#    I2 = one_size(sizes[i + 1], rot = 1)
#    I1 = np.concatenate((I1, I2), axis = 0)
#    
### Save .bmp file:
#fn1 = 'test_map_rot.bmp'
#matplotlib.image.imsave(fn1, I1, cmap = 'gray')
#
#img = Image.open(fn1)
#newimg = img.convert('RGB', colors=24)
#newimg.save(fn1)
#
#I1 = one_size(0.4, rot = 1)
#
#fn1 = 'test_map_rot_04.bmp'
#matplotlib.image.imsave(fn1, I1, cmap = 'gray')
#
#img = Image.open(fn1)
#newimg = img.convert('RGB', colors=24)
#newimg.save(fn1)

#I1 = one_size(0.4, rot = 0)
#
#fn1 = 'test_map_no_rot_04.bmp'
#matplotlib.image.imsave(fn1, I1, cmap = 'gray')
#
#img = Image.open(fn1)
#newimg = img.convert('RGB', colors=24)
#newimg.save(fn1)

rot = 0

mode = 7
if mode == 4: 

    M = 1
    
    fname = 'data_train1.mat'
    M1 = scio.loadmat(fname)
    
    data_train = M1['data_train']
    
    for j in range(M):
    
        t = np.zeros(N)
        size = np.zeros(N)
        a = np.zeros(N)
        b = np.zeros(N)
        for i in range (N):
            t[i] = data_train[N*j + i, 0]
            size[i] = data_train[N*j + i, 1]
            a[i] = data_train[N*j + i, 2]
            b[i] = data_train[N*j + i, 3]
    
        IF0 = genBitmap(t, size, a, b, rot)
        
        IF1 = IF0[700:-700, 700:-700]
        
        ## Save .bmp file:
        fn1 = 'F' + str(j) + '.bmp'
        matplotlib.image.imsave(fn1, IF1, cmap = 'gray')
        
        img = Image.open(fn1)
        newimg = img.convert('RGB', colors=24)
        newimg.save(fn1)
        
        fn2 = 'GT_F' + str(j) + '.mat'
        scio.savemat(fn2, {'GT': t, 
                            'size': size,
                            'a' : a,
                            'b' : b})
        
        print('j = ', j)

mode = 9
if mode == 3: 

    M = 1
    
    fname = 'data_bad1.mat'
    M1 = scio.loadmat(fname)
    
    data_train = M1['data_train']
    
    for j in range(M):
    
        t = np.zeros(N)
        size = np.zeros(N)
        a = np.zeros(N)
        b = np.zeros(N)
        for i in range (N):
            t[i] = data_train[N*j + i, 0]
            size[i] = data_train[N*j + i, 1]
            a[i] = data_train[N*j + i, 2]
            b[i] = data_train[N*j + i, 3]
    
        IF0 = genBitmap(t, size, a, b, rot)
        
        IF1 = IF0[700:-700, 700:-700]
        
        ## Save .bmp file:
        fn1 = 'F' + str(j) + '.bmp'
        matplotlib.image.imsave(fn1, IF1, cmap = 'gray')
        
        img = Image.open(fn1)
        newimg = img.convert('RGB', colors=24)
        newimg.save(fn1)
        
        fn2 = 'GT_FF' + str(j) + '.mat'
        scio.savemat(fn2, {'GT': t, 
                            'size': size,
                            'a' : a,
                            'b' : b})
        
        print('j = ', j)
    
    # fname = 'data_val.mat'
    # M1 = scio.loadmat(fname)
    
    # data_val = M1['data_val']
    
    # M = 3
    
    # for j in range(3):
    
    #     t = np.zeros(N)
    #     size = np.zeros(N)
    #     a = np.zeros(N)
    #     b = np.zeros(N)
        
    #     for i in range (N):
    #         t[i] = data_val[N*j + i, 0]
    #         size[i] = data_val[N*j + i, 1]
    #         a[i] = data_val[N*j + i, 2]
    #         b[i] = data_val[N*j + i, 3]
        
    #     IF0 = genBitmap(t, size, a, b, rot)
        
    #     IF1 = IF0[400:-400, 400:-400]
        
    #     ## Save .bmp file:
    #     fn1 = 'C' + str(j + 27) + '.bmp'
    #     matplotlib.image.imsave(fn1, IF1, cmap = 'gray')
        
    #     img = Image.open(fn1)
    #     newimg = img.convert('RGB', colors=24)
    #     newimg.save(fn1)
        
    #     fn2 = 'GT_C' + str(j + 27) + '.mat'
    #     scio.savemat(fn2, {'GT': t, 
    #                         'size': size,
    #                         'a' : a,
    #                         'b' : b})
        
    #     print('j = ', j)
    
    # sizes = np.arange(14, 32, 1)
    # I1 = one_size(sizes[0], rot = 0)
    # for i in range(len(sizes) - 1):
    #     I2 = one_size(sizes[i + 1], rot = 0)
    #     I1 = np.concatenate((I1, I2), axis = 0)
    
    # # Save .bmp file:
    # fn1 = 'test_map_no_rot.bmp'
    # matplotlib.image.imsave(fn1, I1, cmap = 'gray')
    
    # img = Image.open(fn1)
    # newimg = img.convert('RGB', colors=24)
    # newimg.save(fn1)

mode = 0
if mode == 1:
    
    M1 = scio.loadmat('rectangles_test_data.mat')
    NN = M1['N']
    N = NN[0,0]
    data = M1['data']
    
    t = np.zeros(N**2)
    size = np.zeros(N**2)
    a = np.zeros(N**2)
    b = np.zeros(N**2)
    for i in range (N**2):
        t[i] = data[i, 0]
        size[i] = data[i, 1]
        a[i] = data[i, 2]
        b[i] = data[i, 3]
    
    IF0 = genTestBitmap(N, t, size, a, b, rot)
    
    ## Save .bmp file:
    fn1 = 'Test_rectangle.bmp'
    matplotlib.image.imsave(fn1, IF0, cmap = 'gray')
    
    img = Image.open(fn1)
    newimg = img.convert('RGB', colors=24)
    newimg.save(fn1)
    
    fn2 = 'GT_Test_rectangle.mat'
    scio.savemat(fn2, {'GT': t, 
                        'size': size,
                        'a' : a,
                        'b' : b})
    
    print ('mode == 1')

mode = 2
if mode == 6:
    IF0 = genCalibBitmap()
    
    ## Save .bmp file:
    fn1 = 'calib.bmp'
    matplotlib.image.imsave(fn1, IF0, cmap = 'gray')
    
    img = Image.open(fn1)
    newimg = img.convert('RGB', colors=24)
    newimg.save(fn1)
    
    IF1 = genStepBitmap()
    
    ## Save .bmp file:
    fn1 = 'step.bmp'
    matplotlib.image.imsave(fn1, IF1, cmap = 'gray')
    
    img = Image.open(fn1)
    newimg = img.convert('RGB', colors=24)
    newimg.save(fn1)
    
mode = 9
if mode == 7:
    
    D = np.arange(2) + 16
    
    for i in range(len(D)):
        IF0 = genDBitmap(100*D[i])
        
        ## Save .bmp file:
        fn1 = 'dist = ' + str(D[i]) + '.bmp'
        matplotlib.image.imsave(fn1, IF0, cmap = 'gray')
        
        img = Image.open(fn1)
        newimg = img.convert('RGB', colors=24)
        newimg.save(fn1)
        
        print('i = ', i)

mode = 15
if mode == 15:
    I1 = one_size(20, rot = 0)
    
    ## Save .bmp file:
    fn1 = 'one_size_test3.bmp'
    matplotlib.image.imsave(fn1, I1, cmap = 'gray')
    
    img = Image.open(fn1)
    newimg = img.convert('RGB', colors=24)
    newimg.save(fn1)

elapsed = time.time() - tm
print ('Time elapsed = ', elapsed)    