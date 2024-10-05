# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 18:43:11 2022

@author: sk1u19
"""
import numpy as np
import scipy.io as scio

PWL = 64
size = np.arange(10, 33, 1)
a0 = np.array([0])
b0 = np.array([0])

a = np.array([0])
b = np.array([0])
sz = np.array([0])

for i in range(len(size)):
    
    bmin = 2
    bmax = size[i] - 2
    
    bb = np.arange(bmin, bmax, 1)
    
    aa = size[i] - bb
    szz = np.ones_like(bb)*size[i]
    
    a = np.concatenate((a, aa), axis = 0)    
    b = np.concatenate((b, bb), axis = 0)
    sz = np.concatenate((sz, szz), axis = 0)
    
a = a/PWL
b = b/PWL
sz = sz/PWL

Ys = np.zeros((len(a), 5))
Ys[:,0] = 5
Ys[:,1] = sz
Ys[:,2] = a
Ys[:,3] = b

Ys = np.unique(Ys, axis = 0)

#Ys[0,:] = Ys[15,:]

for i in range(len(Ys)):
    
    if Ys[i,2] < 5/PWL or Ys[i,3] < 5/PWL:
        Ys[i,4] = 1

Ys_good = Ys[Ys[:,4] == 0]
Ys_bad = Ys[Ys[:,4] == 1]

fname = 'Ys_good_exp_data.mat'
fname1 = 'Ys_bad_exp_data.mat'

scio.savemat(fname, {'data' : Ys_good[1:,:]})
scio.savemat(fname1, {'data' : Ys_bad[1:,:]}) 