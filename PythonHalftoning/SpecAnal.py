#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 18:54:16 2021

@author: Wenrui Li
"""

import numpy as np                 # Numpy is a library support computation of large, multi-dimensional arrays and matrices.
from PIL import Image              # Python Imaging Library (abbreviated as PIL) is a free and open-source additional library for the Python programming language that adds support for opening, manipulating, and saving many different image file formats.
import matplotlib.pyplot as plt    # Matplotlib is a plotting library for the Python programming language.

def BetterSpecAnal(x):
# Find center row
    num_rows = x.shape[0]
    num_cols = x.shape[1]
    center_row = int(num_rows/2)
    center_col = int(num_cols/2)
    psd = np.empty([64,64])
    psd_dft = np.empty([64,64])
    psd_dft_avg = np.zeros([64,64])
    #Hamming window does not need to be recalculated
    W = np.outer(np.hamming(64), np.hamming(64));
    # wind_row, wind_col are the window indices
    # cur_row, cur_col are the pixel indices
    for wind_row in range(-2,3):
        for wind_col in range(-2,3):
            for cur_row in range(64):
                for cur_col in range(64):
                    # Fill window with data
                    psd[cur_row][cur_col] = float(x[cur_row+wind_row*64+center_row][cur_col+wind_col*64+center_col])
                    # Multiply by Hamming window W
                    psd[cur_row][cur_col] = psd[cur_row][cur_col] * W[cur_row][cur_col];
            # Compute squared DFT magnitude
            psd_dft = (1/64**2)*np.abs(np.fft.fft2(psd))**2
            # Use fftshift to move the zero frequencies to the center of the plot.
            psd_dft = np.fft.fftshift(psd_dft)
            psd_dft_avg = psd_dft_avg + psd_dft
    psd_dft_avg = psd_dft_avg/25.0
    # Plot the result using a 3-D mesh plot and label the x and y axises properly. 
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    a = b = np.linspace(-np.pi, np.pi, num = 64)
    X, Y = np.meshgrid(a, b)
    # Compute the logarithm of the Power Spectrum.
    Zabs = np.log(psd_dft_avg)

    surf = ax.plot_surface(X, Y, Zabs, cmap=plt.cm.coolwarm)

    ax.set_xlabel('$\mu$ axis')
    ax.set_ylabel('$\\nu$ axis')
    ax.set_zlabel('Z Label')

    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

##    psd_out = Image.fromarray(psd)
##    psd_out.save('psd_out.tif');
# Read in a gray scale TIFF image.
im = Image.open('img04g.tif')
print('Read img04.tif.')
print('Image size: ', im.size)

# Display image object by PIL.
#im.show(title='image')

# Import Image Data into Numpy array.
# The matrix x contains a 2-D array of 8-bit gray scale values. 
x = np.array(im)
print('Data type: ', x.dtype)

# Test BetterSpecAnal
x = np.double(x)/255.0
BetterSpecAnal(x)

