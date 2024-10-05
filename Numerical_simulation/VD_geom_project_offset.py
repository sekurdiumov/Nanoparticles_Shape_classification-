# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 09:46:30 2019

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
from scipy.ndimage import rotate

from Shapes import Shapes
from Vector_Diffraction import Vector_Diffraction as VD 

#plt.style.use('dark_background') 
plt.style.use('default') 
matplotlib.rcParams.update({'font.size' : 30})
pi = np.pi

   
## ============================================================================
## Main:
## ============================================================================
# constants
Nmax = 6000
Nmay = 6000
L0 = 60
PWL = 100
spacing = 1/PWL   
print ('Points per wavelength: ', PWL)                                                
inputField = np.ones((Nmax, Nmay))*0

NX2 = int(Nmax/2)
NY2 = int(Nmay/2)
x = np.linspace(-NX2, NX2, Nmax)*spacing
y = np.linspace(-NY2, NY2, Nmay)*spacing
X,Y = np.meshgrid(x,y)

z = np.array([2])
N0 = 0

doplot = 0

max_offset = int(1.0*PWL)

offilename = 'offsets_1.mat'
M2 = scio.loadmat(offilename)
offset_x = M2['offset_x']
offset_y = M2['offset_y']

fnames = ['ellipses_data.mat',
          'triangles_data.mat',
          'rectangles_data.mat',
          'hexagons_data.mat',
          'rings_data.mat',
          'stars_data.mat',
          'zigzags_data.mat',
          'Ys_data.mat']

shape_ind = 0

fname = fnames[shape_ind]

M1 = scio.loadmat(fname)
data = M1['data']

counter = 0
Nk = 4
for k in range(Nk):
    
    M1 = scio.loadmat(fnames[shape_ind])
    data = M1['data']
    
    Ni = len(data)
    #Ni = 1
    for i in range(700):
        
        
        t = data[i,0]
        size = data[i,1]
        a = data[i,2]
        b = data[i,3]
        
        tm = time.time()
        
        IF = Shapes.InputField(t, size, a, b, inputField, PWL)
        
        IF1 = IF[max_offset:-max_offset, max_offset:-max_offset]
        
        IF0 = np.zeros_like(inputField)
        
        dx = offset_x[k,shape_ind,i]
        dy = offset_y[k,shape_ind,i]
        
        IF0[int(max_offset + dy) : int(Nmax -max_offset + dy),
            int(max_offset + dx) : int(Nmay -max_offset + dx)] = IF1
         
        
        Ex = VD.propFS(IF0, spacing, z, 0)
        #Ey = VD.propFS(IF0, spacing, z, 1)
        
        img_x = np.abs(Ex[-1,:, :])
        #img_y = np.abs(Ey[-1,:, :])
        
        if doplot:
            
            plt.figure(figsize = [9,7])
            fig = plt.pcolormesh(X, Y, IF0, cmap = 'hot')
            plt.colorbar()
            lim = 0.5
            plt.xlim(-lim, lim)
            plt.ylim(-lim, lim)
            #fig.axes.get_xaxis().set_visible(False)
            #fig.axes.get_yaxis().set_visible(False)
            
            # plt.figure(figsize = [9, 7])
            # fig = plt.pcolormesh(X, Y, np.abs(img_x)**2,  cmap = 'hot')
            # plt.colorbar()
            # plt.xlabel('x, in λ')
            # plt.ylabel('y, in λ')
            # lim = 5
            # plt.xlim(-lim, lim)
            # plt.ylim(-lim, lim)
            #plt.clim(0.98, 1.02)
            #fig.axes.get_xaxis().set_visible(False)
            #fig.axes.get_yaxis().set_visible(False)
            
    #        diff = (img - img0)/img0
    #        
            # plt.figure(figsize = [9, 7])
            # fig = plt.pcolormesh(X, Y, np.abs(img_y**2),  cmap = 'hot')
            # plt.colorbar()
            # plt.xlabel('x, in λ')
            # plt.ylabel('y, in λ')
            # plt.xlim(-5, 5)
            # plt.ylim(-5, 5)
            #plt.clim(0.99, np.max(img))
            #fig.axes.get_xaxis().set_visible(False)
            #fig.axes.get_yaxis().set_visible(False)
        
        fname = 'dp' + str(shape_ind) + '_' + str(k*Ni + i) + '.mat'
        
        off_x = dx/PWL
        off_y = dy/PWL
    
        ## save maps 1024x1024 for convenient resizing (512x512, or 256x256):
    
        scio.savemat(fname, {'Ex' : np.abs(img_x[2488:3512, 2488:3512])**2, 
                             'label' : t, 
                             'size' : size,
                             'a' : a,
                             'b' : b,
                             'offset_x' : off_x,
                             'offset_y' : off_y})
        
        elapsed = time.time() - tm
        print ('Time elapsed = ', elapsed, ' i = ', k*Ni + i)
            
        
print('Game over!')