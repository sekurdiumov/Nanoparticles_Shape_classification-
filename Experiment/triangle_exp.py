# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 13:17:02 2022

@author: sk1u19
"""
import numpy as np
import scipy.io as scio

PWL = 64
step = 20
size = np.arange(10, 33, 1)
a0 = np.array([0])
b0 = np.array([0])
c0 = np.array([0])
sz = np.array([0])

R = size/2

h = 0
a = 0
c = 0
b = 0

for i in range(len(size)):
    
    a = np.arange(2, int(size[i]), 1)
    
    for j in range(len(a)):
        
        h1 = np.sqrt(R[i]**2 - (a[j]/2)**2)
        
        b = int(h1 + R[i])
        
        a0 = np.concatenate((a0, np.array([a[j]])), axis = 0)
        b0 = np.concatenate((b0, np.array([b])), axis = 0)
        sz = np.concatenate((sz, np.array([size[i]])), axis = 0)
        
    bb = np.arange(2, int(R[i]), 1)
    
    for j in range(len(bb)):
        
        a0 = np.concatenate((a0, np.array([size[i]])), axis = 0)
        b0 = np.concatenate((b0, np.array([bb[j]])), axis = 0)
        sz = np.concatenate((sz, np.array([size[i]])), axis = 0)

sz = sz/PWL
a0 = a0/PWL
b0 = b0/PWL
            
triangles = np.zeros((len(a0), 5))
triangles[:,0] = 0
triangles[:,1] = sz
triangles[:,2] = a0
triangles[:,3] = b0

triangles = np.unique(triangles, axis = 0)

for i in range(len(a0)):
    
    if triangles[i,2] < 10/PWL or triangles[i,3] < 10/PWL:
        triangles[i,4] = 1

triangles_good = triangles[triangles[:,4] == 0]
triangles_bad = triangles[triangles[:,4] == 1]

fname = 'triangles_good_exp_data.mat'
fname1 = 'triangles_bad_exp_data.mat'

scio.savemat(fname, {'data' : triangles_good[1:,:]})
scio.savemat(fname1, {'data' : triangles_bad[1:,:]})