# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 22:25:17 2021

@author: sk1u19
"""

import numpy as np
from numpy.lib import scimath as SC
import csv

class Vector_Diffraction:
    
    ## Initialisation:
    def init (self, a = 1.0):
        self.a = a
        
    ## ============================================================================
    ## Direct Fourier transform of 2D array:
    ## ============================================================================
    def DFT_2D (inputField, spacing):
        Nmax = inputField[0,:].size
        Nmay = inputField[:,0].size
        
        sigma_step_x = 1/(spacing*Nmax)
        sigma_step_y = 1/(spacing*Nmay)
        sigma_x = sigma_step_x*np.arange(Nmax) 
        sigma_y = sigma_step_y*np.arange(Nmay) 
        
        A0 = np.fft.fft2(inputField)
        
        #'''
        sigma_nyquist_x = sigma_step_x*Nmax/2.0
        sigma_nyquist_y = sigma_step_y*Nmay/2.0
        indsx = np.nonzero(sigma_x >= (sigma_nyquist_x + sigma_step_x/3.0))
        indsy = np.nonzero(sigma_y >= (sigma_nyquist_y + sigma_step_y/3.0))
        sigma_x[indsx] = sigma_x[indsx] - 1/spacing
        sigma_y[indsy] = sigma_y[indsy] - 1/spacing
        #'''
        sigmax, sigmay = np.meshgrid(sigma_x, sigma_y)
        
        return A0, sigmax, sigmay

    ## ============================================================================
    ## Inverse Fourier transform of 2D array:
    ## ============================================================================
    def IDFT_2D(field, d_sigma_x, d_sigma_y):
        propF_z = np.fft.ifft2(field)
        return propF_z

    ## ============================================================================
    ## Including polarization terms:
    ## ============================================================================
    def Psi_ab(sigmax, sigmay):
        
        sigmaz = SC.sqrt(1 - sigmax**2 - sigmay**2)
        
        psi_xx = 1 - sigmax**2
        psi_xy = -sigmax * sigmay
        psi_xz = -sigmax * sigmaz
        psi_yx = -sigmax * sigmay
        psi_yy = 1 - sigmay**2
        psi_yz = -sigmay * sigmaz
        
        return psi_xx, psi_xy, psi_xz, psi_yx, psi_yy, psi_yz

    ## ============================================================================
    ## Propagation in Free space:
    ## ============================================================================
    def propFS(inputField, spacing, prop_D, mode):
        
        pi = np.pi                   # const
        
        sizex = inputField[0,:].size
        sizey = inputField[:,0].size
        
        #Ex = np.zeros((len(prop_D), sizex, sizey), dtype = complex)
        #Ey = np.zeros((len(prop_D), sizex, sizey), dtype = complex)
        #propF_z_x = np.zeros((len(prop_D), sizex, sizey), dtype = complex)
        #propF_z_y = np.zeros((len(prop_D), sizex, sizey), dtype = complex)
        propF_z = np.zeros((len(prop_D), sizex, sizey), dtype = complex)
        
        A0, sigmax, sigmay = Vector_Diffraction.DFT_2D(inputField, spacing)
        
        psi_xx, psi_xy, psi_xz, psi_yx, psi_yy, psi_yz = Vector_Diffraction.Psi_ab(sigmax, sigmay)
        
        ## Mansuripur 1989:
        if mode == 0:
            psi = psi_xx
        elif mode == 1:
            psi = psi_xy
            
        for i in np.arange(len(prop_D)):
            z0 = prop_D[i]
            psi = A0 * psi * np.exp(1j * 2*pi * z0  * SC.sqrt(1 - sigmax**2 - sigmay**2))
            propF_z[i] = np.fft.ifft2(psi)
        
        #propF_z = np.abs(propF_z)**2
        
        return propF_z


    ## ============================================================================
    ## Write a .csv file from the image:
    ## ============================================================================
    def csvWriter(fil_name, nparray):
        example = nparray.tolist()
        with open(fil_name+'.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerows(example)