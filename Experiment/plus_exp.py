# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 14:53:15 2022

@author: sk1u19
"""
import numpy as np
import scipy.io as scio

PWL = 64
size = np.arange(10, 33, 1)
a0 = np.array([0])
b0 = np.array([0])
sz = np.array([0])

for i in range(len(size)):
    b = np.arange(2, size[i]+1, 2)
    a = np.zeros_like(b)
    szz = np.zeros_like(a)
    for j in range(len(a)):
    
        a[j] = int(size[i])    
        szz[j] = size[i]
    
    a0 = np.concatenate((a0, a), axis = 0)  
    a0 = np.concatenate((a0, b), axis = 0)  
    b0 = np.concatenate((b0, b), axis = 0)
    b0 = np.concatenate((b0, a), axis = 0)
    sz = np.concatenate((sz, szz), axis = 0)
    sz = np.concatenate((sz, szz), axis = 0)
    
a = a/PWL
b = b/PWL

sz = sz/PWL
a0 = a0/PWL
b0 = b0/PWL      
            
pluses = np.zeros((len(a0), 5))
pluses[:,0] = 4
pluses[:,1] = sz
pluses[:,2] = a0/2
pluses[:,3] = b0/2

for i in range(len(a0)):
    
    if pluses[i,2] < 5/PWL or pluses[i,3] < 5/PWL:
        pluses[i,4] = 1

pluses_good = pluses[pluses[:,4] == 0]
pluses_bad = pluses[pluses[:,4] == 1]

fname = 'pluses_good_exp_data.mat'
fname1 = 'pluses_bad_exp_data.mat'

scio.savemat(fname, {'data' : pluses_good[1:,:]})
scio.savemat(fname1, {'data' : pluses_bad[1:,:]}) 