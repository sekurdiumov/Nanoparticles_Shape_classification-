# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 22:02:33 2021

@author: sk1u19
"""

import numpy as np
import random as rnd

class Shapes:
    
    ## ========================================================================
    ## Initialisation:
    ## ========================================================================
    def __init__(self, a = 0):
        self.a = a
        
    
    ## Rectangle:
    def rectangle(inputField, PWL, size, a, b, p):
        Nmax = inputField[0,:].size
        Nmay = inputField[:,0].size
        IF = np.zeros_like(inputField)
        
        X0 = int(Nmax/2) + rnd.randint(-p, p)
        Y0 = int(Nmax/2) + rnd.randint(-p, p)
        
        size = int(size * PWL)
        a = int(a * PWL)
        b = int(b * PWL)
        
        X1 = int(a/2)
        Y1 = int(b/2)
        
        if a%2 == 0:
            X2 = int(a/2)
        else:
            X2 = int(a/2) + 1
            
        if b%2 == 0:
            Y2 = int(b/2)
        else:
            Y2 = int(b/2) + 1
        
        for i in range (Nmax):
            for j in range (Nmay):
                if (j > (X0 - X1) and j < (X0 + X2)) and (i > (Y0 - Y1) and i < (Y0 + Y2)):
                    IF[i,j] = 1
                    
        return IF, size, a, b
    
    ## Ellipse:
    def ellipse(inputField, PWL, size, a, b, p):
        Nmax = inputField[0,:].size
        Nmay = inputField[:,0].size
        IF = np.zeros_like(inputField)
        
        X0 = int(Nmax/2) + rnd.randint(-p, p)
        Y0 = int(Nmax/2) + rnd.randint(-p, p)
        
        # aa = int(2*a*PWL)
        # bb = int(2*b*PWL)
        
        # a = int(a * PWL)
        # b = int(b * PWL)
        
        # if aa%2 == 1:
        #     a1 = a + 1
        # else:
        #     a1 = a
            
        # if bb%2 == 1:
        #     b1 = b + 1
        # else:
        #     b1 = b
        
        a1 = a * PWL
        b1 = b * PWL
        
        for i in range(Nmax):
            for j in range(Nmax):
                if ((X0 - i)**2 / a1**2 + (Y0 - j)**2 / b1**2 < 1):
                    IF[i, j] = 1
        
        # for i in range(Nmax):
        #     for j in range (Nmay):
        #         if ((X0 - i)**2/a1**2 + (Y0 - j)**2/b1**2 < 1) and (j >= Nmax/2) and (i >= Nmax/2):
        #             IF[i,j] = 1
        #         elif ((X0 - i)**2/a1**2 + (Y0 - j)**2/b**2 < 1) and (j >= Nmax/2) and (i < Nmax/2):
        #             IF[i,j] = 1
        #         elif ((X0 - i)**2/a**2 + (Y0 - j)**2/b1**2 < 1) and (j < Nmax/2) and (i >= Nmax/2):
        #             IF[i,j] = 1
        #         elif ((X0 - i)**2/a**2 + (Y0 - j)**2/b**2 < 1) and (j < Nmax/2) and (i < Nmax/2):
        #             IF[i,j] = 1
       
        return IF, size, a, b
        
    ## Triangle:
    def triangle(inputField, PWL, size, a, b, p):
        Nmax = inputField[0,:].size
        Nmay = inputField[:,0].size
        IF = np.ones_like(inputField)*0
        
        spacing = 1/PWL

        NX2 = int(Nmax/2)
        NY2 = int(Nmay/2)
        x = np.linspace(-NX2, NX2, Nmax)*spacing
        y = np.linspace(-NY2, NY2, Nmay)*spacing
        X,Y = np.meshgrid(x,y)
        
        R = size/2
        
        if b > R:
            h = b - R
            Y1 = -h
            
        else:
            Y1 = 0

        Y2 = Y1
        Y3 = Y1 + b
        X1 = -a/2
        X3 = 0
        
        if int(PWL*a)%2 == 0:
            X2 = a/2   
            
        else:
            X2 = a/2 + spacing
            

        k1 = (Y3 - Y1) / (X3 - X1)
        b1 = Y1 - k1*X1

        k2 = (Y3 - Y2) / (X3 - X2)
        b2 = Y2 - k2*X2

        IF[(Y < k1*X + b1) & 
           (Y < k2*X + b2) & 
           (Y > Y1)] = 1    
        
        IF = np.flipud(IF)
        
        return IF, size, a, b
            
    ## Zigzag:
    def zigzag(inputField, PWL, size, a, b, p):
        
        Nmax = inputField[0,:].size
        Nmay = inputField[:,0].size
        IF = np.zeros_like(inputField)
        
        X0 = int(Nmax/2) + rnd.randint(-p, p)
        Y0 = int(Nmax/2) + rnd.randint(-p, p)
        
        a = int(a * PWL)
        b = int(b * PWL)
        
        d1 = int(PWL/64)
        d2 = int(PWL/64)
        
        X1 = a
        Y1 = b
        
        if a%2 == 0:
            X2 = a
        else:
            X2 = a + 1
            
        if b%2 == 0:
            Y2 = b
        else:
            Y2 = b + 1
        
        for i in range (Nmax):
            for j in range (Nmay):
                if ((j > X0 - X1 and j < X0 + d2 and i > Y0 - Y1 - d1 and i < Y0 - Y1 + d1)  or 
                    (j > X0 - d2 and j < X0 + d2 and i > Y0 - Y1 and i < Y0 + d1) or
                    (j > X0 and j < X0 + X2 and i > Y0 - d1 and i < Y0 + d1)  or
                    (j > X0 + X2 - d2 and j < X0 + X2 + d2 and i > Y0 - d1 and i < Y0 + Y2 + d1)):
                    IF[i,j] = 1
                    
                if ((j > X0 - X1  and j < X0 + d2 + 1 and i > Y0 - Y1 - d1 and i < Y0 - Y1 + d1 + 1)  or 
                    (j > X0 - d2 + 1 and j < X0 + d2 + 1 and i > Y0 - Y1 and i < Y0 + d1 + 1) or
                    (j > X0 - 1 and j < X0 + X2 + 1 and i > Y0 - d1 + 1 and i < Y0 + d1 + 1)  or
                    (j > X0 + X2 - d2 + 1 and j < X0 + X2 + d2 + 1 and i > Y0 - d1 and i < Y0 + Y2 + d1)):
                    IF[i,j] = 1
        
        return IF, size, a, b
    
    def plus(inputField, PWL, size, a, b, p):
        
        Nmax = inputField[0,:].size
        Nmay = inputField[:,0].size
        IF = np.zeros_like(inputField)
        
        X0 = int(Nmax/2) + rnd.randint(-p, p)
        Y0 = int(Nmax/2) + rnd.randint(-p, p)
        
        a = int(a * PWL)
        b = int(b * PWL)
        
        X1 = a
        Y1 = b
        
        if a%2 == 0:
            X2 = a
        else:
            X2 = a + 1
            
        if b%2 == 0:
            Y2 = b
        else:
            Y2 = b + 1
        
        for i in range (Nmax):
            for j in range (Nmay):
                if ((j > X0 - X1 and j < X0 + X2 and i == Y0)  or 
                    (i > Y0 - Y1 and i < Y0 + Y2 and j == X0)):
                    IF[i,j] = 1
                    
                if ((j > X0 - X1  and j < X0 + X2 and i == Y0 + 1)  or 
                    (i > Y0 - Y1 and i < Y0 + Y2 and j == X0 + 1)):
                    IF[i,j] = 1
        
        return IF, size, a, b
    
    def y_shape(inputField, PWL, size, a, b, p):
        
        Nmax = inputField[0,:].size
        Nmay = inputField[:,0].size
        IF = np.zeros_like(inputField)
        
        IF = np.zeros_like(inputField)
        
        X0 = int(Nmax/2) + rnd.randint(-p, p)
        Y0 = int(Nmax/2) + rnd.randint(-p, p)
        
        size = int(size * PWL)
        r = int(size/2)
        
        b = int(b * PWL)
        a = int(a * PWL)
        
        Y0 = Y0 + r - b
        
        X1 = X0 - int(b/np.sqrt(2))
        Y1 = Y0 - int(b/np.sqrt(2))
        X2 = X0 + int(b/np.sqrt(2))
        Y2 = Y1
        X3 = X0
        if size%2 == 0:
            Y3 = Y0 + a
        else:
            Y3 = Y0 + a + 1
        
        k1 = (Y1 - Y0)/(X1 - X0)
        b1 = Y1 - k1 * X1
        k2 = (Y2 - Y0)/(X2 - X0)
        b2 = Y2 - k2 * X2
        
        for j in range (Nmax):
            for i in range (Nmax):
                if (((i == k1*j + b1 or i == k2*j + b2) and 
                     (i >= Y1 and i <= Y0)) or  
                    (j == X3 and (i >= Y0 and i <= Y3))):
                    IF[i,j] = 1
                    
        for j in range (Nmax):
            for i in range (Nmax):
                if (((i == k1*j + b1 - 1 or i == k2*j + b2 + 1) and 
                     (i >= Y1 and i <= Y0)) or  
                    (j == X3 + 1 and (i >= Y0 and i <= Y3))):
                    IF[i,j] = 1
                    
        # for j in range (Nmax):
        #     for i in range (Nmax):
        #         if (((i == k1*j + b1 - 1 or i == k2*j + b2 - 1) and 
        #              (i >= Y1 and i <= Y0)) or  
        #             (j == X3 - 1 and (i >= Y0 and i <= Y3))):
        #             IF[i,j] = 1
        
        return IF, size, a, b
    
    def x_shape(inputField, PWL, size, a, b, p):
        
        Nmax = inputField[0,:].size
        Nmay = inputField[:,0].size
        IF = np.zeros_like(inputField)
        
        X0 = int(Nmax/2) + rnd.randint(-p, p)
        Y0 = int(Nmax/2) + rnd.randint(-p, p)
        
        size = int(size * PWL)
        a = int(a * PWL)
        b = int(b * PWL)
        
        r = int(size/2)
        
        if size%2 == 0:
            Y0 = Y0 + r - b
        else:
            Y0 = Y0 + r - b - 1
        
        X1 = X0 - int(b/np.sqrt(2))
        Y1 = Y0 + int(b/np.sqrt(2))
        X2 = X0 + int(b/np.sqrt(2))
        Y2 = Y1
        X3 = X0 + int(a/np.sqrt(2))
        Y3 = Y0 - int(a/np.sqrt(2))
        X4 = X0 - int(a/np.sqrt(2))
        Y4 = Y3
        
        k1 = (Y1 - Y0)/(X1 - X0)
        b1 = Y1 - k1 * X1
        k2 = (Y2 - Y0)/(X2 - X0)
        b2 = Y2 - k2 * X2
        
        for j in range (Nmax):
            for i in range (Nmax):
                if (((i == k1*j + b1 or i == k2*j + b2) and 
                     (i >= Y3 and i <= Y1))):
                    IF[i,j] = 1
                    
        for j in range (Nmax):
            for i in range (Nmax):
                if (((i == k1*j + b1 + 1 or i == k2*j + b2 + 1) and 
                     (i >= Y3 and i <= Y1))):
                    IF[i,j] = 1
                    
        # for j in range (Nmax):
        #     for i in range (Nmax):
        #         if (((i == k1*j + b1 - 1 or i == k2*j + b2 - 1) and 
        #              (i >= Y3 and i <= Y1))):
        #             IF[i,j] = 1
        
        #IF = np.transpose(IF)
        
        return IF, size, a, b
    
    ## InputField_exp:
    def InputField_exp(i, size, inputField, PWL, a, b):
        
        p = 0 * PWL / 2
        
        #i = rnd.randint(0,6)
        if i == 0:
            IF, size, a, b = Shapes.triangle(inputField, PWL, size, a, b, p)
        elif i == 1:
            IF, size, a, b = Shapes.rectangle(inputField, PWL, size, a, b, p)
        elif i == 2:
            IF, size, a, b = Shapes.ellipse(inputField, PWL, size, a, b, p)
        elif i == 3:
            IF, size, a, b = Shapes.zigzag(inputField, PWL, size, a, b, p)
        elif i == 4:
            IF, size, a, b = Shapes.plus(inputField, PWL, size, a, b, p)
        elif i == 5:
            IF, size, a, b = Shapes.y_shape(inputField, PWL, size, a, b, p)
        
        IF.astype(int)
        
        size = size/PWL
        a = a/PWL
        b = b/PWL
            
        return IF
        
