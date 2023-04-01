import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from PIL import Image
from numpy import linalg as LA
import math

def convert_image_to_double_array(img):
    converted = np.array(img, dtype = float)
    return converted;

def rmse(array_orig, array_binary):
    # Work from inside out
    error_sum = 0
    for row_idx in range(array_orig.shape[0]):
        for col_idx in range(array_orig.shape[1]):
            error_sum += ((array_orig[row_idx, col_idx] -
                         array_binary[row_idx, col_idx]) *
                         (array_orig[row_idx, col_idx] -
                         array_binary[row_idx, col_idx]))
    error_sum /= array_orig.shape[0]
    error_sum /= array_orig.shape[1]
    rmse = np.sqrt(error_sum)
    return rmse

def ungamma_correct(g_array, gamma):
    for row_idx in range(g_array.shape[0]):
        for col_idx in range(g_array.shape[1]):
            g_array[row_idx, col_idx] = math.pow((g_array[row_idx, col_idx]/255), gamma)    
            g_array[row_idx, col_idx] *= 255
    return g_array

def calculate_lpf(lpf,sigma):
    scale_factor = 0
    for row_idx in range(lpf.shape[0]):
        for col_idx in range(lpf.shape[1]):
            filter_const = np.exp(-1*((row_idx-3)*(row_idx-3)+(col_idx-3)*(col_idx-3))/(2*sigma*sigma))
            scale_factor += filter_const    
            lpf[row_idx,col_idx] = filter_const
    lpf /= scale_factor
    # Check scale factor
    scale_check = 0
    for row_idx in range(lpf.shape[0]):
        for col_idx in range(lpf.shape[1]):
            scale_check += lpf[row_idx, col_idx]    
    return lpf

def filter_pixel(X, filt, row_idx, col_idx):
    pixel = 0
    for win_row_idx in range(-3,4):
        for win_col_idx in range(-3, 4):
            if ((win_row_idx + row_idx < 0) or
                (win_row_idx + row_idx >= X.shape[0]) or
                (win_col_idx + col_idx < 0) or
                (win_col_idx + col_idx >= X.shape[1])):
                pixel += 0
            else:
                pixel += (filt[win_row_idx+3,win_col_idx+3]*
                         X[win_row_idx+row_idx, win_col_idx+col_idx])
    return pixel

def apply_guassian_filter(filt, image_array):
    filtered_array = np.zeros((image_array.shape[0],image_array.shape[1]))
    for row_idx in range(image_array.shape[0]):
        for col_idx in range(image_array.shape[1]):
            filtered_array[row_idx, col_idx] = filter_pixel(image_array, filt, row_idx, col_idx)

    im_filtered = Image.fromarray(filtered_array.astype(np.uint8))
    #im_filtered.save("test.tif")    
    return filtered_array

def threshold_image(img, thresh):
    X = np.array(img)
    Y = np.zeros((X.shape))
    for row_idx in range(X.shape[0]):
        for col_idx in range(X.shape[1]):
            if X[row_idx, col_idx] > thresh:
                Y[row_idx, col_idx] = 255
            else:
                Y[row_idx, col_idx] = 0
    return Y

def apply_transformation(image_array):
    transform_array = np.zeros((image_array.shape[0],image_array.shape[1]))
    for row_idx in range(image_array.shape[0]):
        for col_idx in range(image_array.shape[1]):
            pixel = image_array[row_idx, col_idx]/255
            pixel = math.pow(pixel,1/3)
            transform_array[row_idx, col_idx] = 255*pixel
    return transform_array

def fidelity(f, b):
    N = f.shape[0]
    M = f.shape[1]
    sum = 0
    for row_idx in range(f.shape[0]):
        for col_idx in range(f.shape[1]):
            sum += math.pow((f[row_idx, col_idx] - b[row_idx, col_idx]),2)
    sum /= M
    sum /= N
    sum = math.pow(sum, 0.5)
    return sum

# Section 1
img_house = Image.open('house.tif')
#img14sp_plot = plt.imshow(img14sp)
#plt.show()

# Threshold house image
thresh_array = threshold_image(img_house, 127)
img_thresh = Image.fromarray(thresh_array.astype(np.uint8))
img_thresh.save("Thresholded house.tif") 

# Convert images to double
array_house_double = convert_image_to_double_array(img_house)
array_thresh_double = convert_image_to_double_array(img_thresh)

# Calculate rmse
house_rmse = rmse(array_house_double, array_thresh_double)
print("RMSE:",house_rmse)
# Ungamma initial image
array_house_double = ungamma_correct(array_house_double, 2.2)
# Fill low pass filter
gaussian_lpf = np.zeros((7,7))
gaussian_lpf = calculate_lpf(gaussian_lpf, math.sqrt(2))

#apply filter
array_house_double = apply_guassian_filter(gaussian_lpf,array_house_double)
array_thresh_double = apply_guassian_filter(gaussian_lpf,array_thresh_double)

#apply transformation
array_house_double = apply_transformation(array_house_double)
array_thresh_double = apply_transformation(array_thresh_double)

#calculate fidelity
fid = fidelity(array_house_double, array_thresh_double)
print("Fidelity:", fid)
