# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 22:02:33 2021

@author: sk1u19
"""

import numpy as np

class Shapes:
    
    ## ========================================================================
    ## Initialisation:
    ## ========================================================================
    def __init__(self, a = 0):
        self.a = a
      
    ## ========================================================================
    ## Shape classes:
    ## ========================================================================
    
    ## Triangle:
    def triangle(size, a, b, inputField, PWL):
        Nmax = inputField[0,:].size
        Nmay = inputField[:,0].size
        IF = np.ones_like(inputField)

        spacing = 1/PWL

        NX2 = int(Nmax/2)
        NY2 = int(Nmay/2)
        x = np.linspace(-NX2, NX2, Nmax)*spacing
        y = np.linspace(-NY2, NY2, Nmay)*spacing
        X,Y = np.meshgrid(x,y)

        R = size/2

        X1 = -int(PWL*a/2)/PWL

        if b > R:
            h = round(np.sqrt(np.abs((PWL*size)**2 - (PWL*a)**2))/2)
            h = h/PWL
            Y1 = -h
            
        else:
            Y1 = 0

        X2 = a - int(PWL*a/2)/PWL
        Y2 = Y1
        
        X3 = X1 + int(PWL*a/2)/PWL
        Y3 = round(PWL*(Y1 + b))/PWL
        
        X4 = X2 - int(PWL*a/2)/PWL
        Y4 = round(PWL*(Y1 + b))/PWL
        
        delta = 0.1*spacing
        
        k1 = (Y3 - Y1) / (X3 - X1)
        b1 = Y1 - k1*X1 

        k2 = (Y4 - Y2) / (X4 - X2)
        b2 = Y2 - k2*X2 

        IF[(Y <= k1*X + b1 + delta) & 
           (Y <= k2*X + b2 + delta) & 
           (Y >= Y1 - delta)] = 0
        
        ## Cleaning "tails":
        R = size/2
        
        if int(PWL*size)%2 == 1:
            R1 = R + spacing
        else:
            R1 = R
            
        IF[(X >= 0) & (Y >= 0) & (X**2/R1**2 + Y**2/R1**2 >= 1)] = 1
        IF[(X >= 0) & (Y < 0) & (X**2/R1**2 + Y**2/R**2 >= 1)] = 1
        IF[(X < 0) & (Y >= 0) & (X**2/R**2 + Y**2/R1**2 >= 1)] = 1
        IF[(X < 0) & (Y < 0) & (X**2/R**2 + Y**2/R**2 >= 1)] = 1
        
        return IF

    ## Hexagon:
    def hexagon(size, a, b, inputField, PWL):
        
        ## Input field mesh
        Nmax = inputField[0,:].size
        Nmay = inputField[:,0].size
        IF = np.ones_like(inputField)
        
        ## Step between mesh cells in wavelengths:
        spacing = 1/PWL
        
        ## Meshgrid:
        NX2 = int(Nmax/2)
        NY2 = int(Nmay/2)
        x = np.linspace(-NX2, NX2, Nmax)*spacing
        y = np.linspace(-NY2, NY2, Nmay)*spacing
        X,Y = np.meshgrid(x,y)
        
        ## Encoding angles coordinates of the hexagon:
        X1 = -int(PWL*size/2)/PWL
        Y1 = 0
        
        X2 = size - int(PWL*size/2)/PWL
        Y2 = Y1
        
        X3 = X1 + int(PWL*b/2)/PWL
        Y3 = int(PWL*b/2)/PWL
        
        X4 = X2 - (int(PWL*b/2)/PWL - b)
        Y4 = int(PWL*b/2)/PWL - b
        
        ## Setting lines forming the hexagon:    
        k1 = (Y3 - Y1)/(X3 - X1)
        b1 = Y3 - k1*X3
        k2 = (Y4 - Y2)/(X4 - X2)
        b2 = Y4 - k2*X4
        k3 = -k2
        b3 = -b2
        k4 = -k1
        b4 = -b1
        
        ## Taking into account inaccuracy of python:
        delta = 0.1*spacing
        
        #Hexagon itself:
        IF[(Y <= k1*X + b1 + delta) & 
           (Y <= k2*X + b2 + delta) & 
           (Y <= Y3 + delta) & 
           (Y >= k3*X + b3 - delta) & 
           (Y >= k4*X + b4 - delta) & 
           (Y >= Y4 - delta)] = 0
            
        return IF

    ## Ellipse:
    def ellipse(size, a, b, inputField, PWL):
        
        ## Input field mesh
        Nmax = inputField[0,:].size
        Nmay = inputField[:,0].size
        IF = np.ones_like(inputField)
        
        ## Step between mesh cells in wavelengths:
        spacing = 1/PWL
        
        ## Meshgrid:
        NX2 = int(Nmax/2)
        NY2 = int(Nmay/2)
        x = np.linspace(-NX2, NX2, Nmax)*spacing
        y = np.linspace(-NY2, NY2, Nmay)*spacing
        X,Y = np.meshgrid(x,y)
        
        # from half-axes to axes:
        a = 2*a
        b = 2*b
        
        aa = int(a*PWL/2)
        bb = int(b*PWL/2)
        
        a1 = aa/(PWL)
        b1 = bb/(PWL)
        
        a2 = a - a1
        b2 = b - b1
        
        ## Ellipse in the meshgrid:
        IF[(X >= 0) & (Y >= 0) & (X**2/a2**2 + Y**2/b2**2 <= 1)] = 0
        IF[(X >= 0) & (Y <= 0) & (X**2/a2**2 + Y**2/b1**2 <= 1)] = 0
        IF[(X <= 0) & (Y >= 0) & (X**2/a1**2 + Y**2/b2**2 <= 1)] = 0
        IF[(X <= 0) & (Y <= 0) & (X**2/a1**2 + Y**2/b1**2 <= 1)] = 0
       
        return IF

    ## Zigzag:
    def zigzag(size, a, b, inputField, PWL):
        
        Nmax = inputField[0,:].size
        IF = np.ones_like(inputField)
        
        a = int(2 * a * PWL)
        b = int(2 * b * PWL)
        
        a1 = int(a/2)
        a2 = a - a1
        
        b1 = int(b/2)
        b2 = b - b1
        
        X0 = int(Nmax/2)
        Y0 = int(Nmax/2)
        
        d = int(PWL/100)
        
        IF[Y0 + b2 - 2*d : Y0 + b2, X0 - a1: X0 + d] = 0
        IF[Y0 - d : Y0 + b1, X0 - d: X0 + d] = 0
        IF[Y0 - d : Y0 + d, X0 - d: X0 + a2] = 0
        IF[Y0 - b1 : Y0 + d, X0 + a2 - 2*d: X0 + a2] = 0

        return IF

    ## Rectangle:
    def rectangle(size, a, b, inputField, PWL):
        
        ## Input field mesh
        Nmax = inputField[0,:].size
        Nmay = inputField[:,0].size
        IF = np.ones_like(inputField)
        
        #Step between mesh cells in wavelengths
        spacing = 1/PWL
        
        ## Meshgrid
        NX2 = int(Nmax/2)
        NY2 = int(Nmay/2)
        x = np.linspace(-NX2, NX2, Nmax)*spacing
        y = np.linspace(-NY2, NY2, Nmay)*spacing
        X,Y = np.meshgrid(x,y)
        
        X1 = -int(PWL*a/2)/PWL
        Y1 = -int(PWL*b/2)/PWL
        
        X2 = a - int(PWL*a/2)/PWL
        Y2 = b - int(PWL*b/2)/PWL
        
        IF[(X >= X1) & (X <= X2) & (Y >= Y1) & (Y <= Y2)] = 0    

        return IF

    ## Ring:
    def ring(size, a, b, inputField, PWL):
        
        ## Input field mesh
        Nmax = inputField[0,:].size
        Nmay = inputField[:,0].size
        IF = np.ones_like(inputField)
        
        #Step between mesh cells in wavelengths
        spacing = 1/PWL
        
        ## Meshgrid
        NX2 = int(Nmax/2)
        NY2 = int(Nmay/2)
        x = np.linspace(-NX2, NX2, Nmax)*spacing
        y = np.linspace(-NY2, NY2, Nmay)*spacing
        X,Y = np.meshgrid(x,y)
        
        ## Outer radii
        R = int(PWL*size/2)/PWL
        R1 = size - R
        
        ## Inner semi-axes
        a1 = int(a*PWL)/PWL
        b1 = int(b*PWL)/PWL
        
        a2 = 2*a - a1
        b2 = 2*b - b1
        
        ## Outer ellipse
        IF[(X >= 0) & (Y >= 0) & (X**2/R1**2 + Y**2/R1**2 <= 1)] = 0
        IF[(X >= 0) & (Y < 0) & (X**2/R1**2 + Y**2/R**2 <= 1)] = 0
        IF[(X < 0) & (Y >= 0) & (X**2/R**2 + Y**2/R1**2 <= 1)] = 0
        IF[(X < 0) & (Y < 0) & (X**2/R**2 + Y**2/R**2 <= 1)] = 0
        
        ## Inner ellipse
        IF[(X >= 0) & (Y >= 0) & (X**2/a2**2 + Y**2/b2**2 <= 1)] = 1
        IF[(X >= 0) & (Y < 0) & (X**2/a2**2 + Y**2/b1**2 <= 1)] = 1
        IF[(X < 0) & (Y >= 0) & (X**2/a1**2 + Y**2/b2**2 <= 1)] = 1
        IF[(X < 0) & (Y < 0) & (X**2/a1**2 + Y**2/b1**2 <= 1)] = 1
         
        return IF 

    ## Star (David's Star):
    def star(size, a, b, inputField, PWL):
        
        ## Input field mesh
        Nmax = inputField[0,:].size
        Nmay = inputField[:,0].size
        IF = np.ones_like(inputField)
        
        #Step between mesh cells in wavelengths
        spacing = 1/PWL
        
        ## Meshgrid
        NX2 = int(Nmax/2)
        NY2 = int(Nmay/2)
        x = np.linspace(-NX2, NX2, Nmax)*spacing
        y = np.linspace(-NY2, NY2, Nmay)*spacing
        X,Y = np.meshgrid(x,y)
        
        ## Taking into account python inaccuracy
        delta = 0.01*spacing
        
        ## Upper triangle
        X1 = -int(PWL*a/2)/PWL
        X2 = a - int(PWL*a/2)/PWL
        
        h = round(np.sqrt((PWL*size)**2 - (PWL*a)**2)/2)    
        h = h/PWL
        
        X3 = 0
        Y1 = -h
        Y2 = Y1
        Y3 = Y1 + b
           
        k1 = (Y3 - Y1)/(X3 - X1)
        b1 = Y3 - k1 * X3
        k2 = (Y3 - Y2)/(X3 - X2)
        b2 = Y3 - k2 * X3
        
        ## Upper triangle
        IF[(Y <= k1*X + b1 + delta) & 
           (Y <= k2*X + b2 + delta) & 
           (Y >= Y2 - delta)] = 0
           
        ## Lower triangle:
        X4 = X1
        Y4 = -Y1
        X5 = X2
        Y5 = Y4
        X6 = X3
        Y6 = -Y3
        
        k3 = (Y6 - Y5)/(X6 - X5)
        b3 = Y6 - k3 * X6
        k4 = (Y6 - Y4)/(X6 - X4)
        b4 = Y6 - k4 * X6
        
        IF[(Y >= k3*X + b3 - delta) & 
           (Y >= k4*X + b4 - delta) & 
           (Y <= Y4 + delta)] = 0

        return IF 

    def y_shape(size, a, b, inputField, PWL):
        
        Nmax = inputField[0,:].size
        Nmay = inputField[:,0].size
        IF = np.ones_like(inputField)

        spacing = 1/PWL
        
        R = size/2
        
        if int(PWL*size)%2 == 1:
            R1 = R + spacing
        else:
            R1 = R 

        NX2 = int(Nmax/2)
        NY2 = int(Nmay/2)
        x = np.linspace(-NX2, NY2, Nmax)*spacing
        y = x
        X,Y = np.meshgrid(x,y)

        X0 = 0
        Y0 = int((-size/2 + a)*PWL)/PWL

        b0 = round(b*PWL/np.sqrt(2))/PWL

        X1 = -b0
        Y1 = Y0 + b0
        X2 = -X1
        Y2 = Y1

        delta = 0.1*spacing

        dx = np.abs(x[0] - x[1])

        IF[(Y >= X + Y0 - delta) & (Y <= X + Y0 + delta) & (X >= 0) & (X <= X2)] = 0
        IF[(Y >= X + Y0 - spacing - delta) & (Y <= X + Y0 - spacing + delta) & (X >= 0) & (X <= X2)] = 0
        IF[(Y >= -X + Y0 - delta) & (Y <= -X + Y0 + delta) & (X <= 0) & (X >= X1)] = 0
        IF[(Y >= -X + Y0 - spacing - delta) & (Y <= -X + Y0 - spacing + delta) & (X <= 0) & (X >= X1)] = 0
        IF[(X >= - dx - 2*delta) & (X <= dx + 2*delta) & (Y >= -size/2) & (Y <= Y0)] = 0
        
        return IF

    ## InputField:
    def InputField(i, size, a, b, inputField, PWL):
        
        if i == 0:
            IF = Shapes.ellipse(size, a, b, inputField, PWL)
        elif i == 1:
            IF = Shapes.triangle(size, a, b, inputField, PWL)
        elif i == 2:
            IF = Shapes.rectangle(size, a, b, inputField, PWL)
        elif i == 3:
            IF = Shapes.hexagon(size, a, b, inputField, PWL)
        elif i == 4:
            IF = Shapes.ring(size, a, b, inputField, PWL)
        elif i == 5:
            IF = Shapes.star(size, a, b, inputField, PWL)
        elif i == 6:
            IF = Shapes.zigzag(size, a, b, inputField, PWL)    
        elif i == 7:
            IF = Shapes.y_shape(size, a, b, inputField, PWL) 
            
        return IF
        
