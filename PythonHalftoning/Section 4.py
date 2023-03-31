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

def generate_bayer(IN):
    new_bayer = np.block([
        [4*IN+1, 4*IN+2],
        [4*IN+3, 4*IN]
        ])
    return new_bayer

def create_threshold_matrix(index):
    N = index.shape[0]
    t = np.zeros((N,N))
    for row_idx in range(N):
        for col_idx in range(N):
            t[row_idx, col_idx] = 255 * (index[row_idx,col_idx] + 0.5)/(N*N)
    return t

def dither_one_tile(dither_array, thresh, cur_row, cur_col):
#    dither_array = np.zeros((image_array.shape[0],image_array.shape[1]))
    for row_idx in range(thresh.shape[0]):
        for col_idx in range(thresh.shape[1]):
            if (dither_array[(cur_row+row_idx),(cur_col+col_idx)] > thresh[row_idx,col_idx]):
                dither_array[(cur_row+row_idx),(cur_col+col_idx)] = 255
            else:
                dither_array[(cur_row+row_idx),(cur_col+col_idx)] = 0
    return dither_array


def dither_image(image_array, thresh, filename):
    dither_array= np.zeros((image_array.shape[0],image_array.shape[1]))
    for row_idx in range(image_array.shape[0]):
        for col_idx in range(image_array.shape[1]):
            dither_array[row_idx, col_idx] = image_array[row_idx, col_idx]
    N = thresh.shape[0]
    for row_idx in range(image_array.shape[0]):
        for col_idx in range(image_array.shape[1]):
            if ((row_idx % thresh.shape[0] == 0) and (col_idx % thresh.shape[1] == 0)):
                dither_array = dither_one_tile(dither_array, thresh, row_idx, col_idx)
    im_dithered = Image.fromarray(dither_array.astype(np.uint8))
    im_dithered.save(filename)    


# Section 4
img_house = Image.open('house.tif')

# Convert images to double
array_house_double = convert_image_to_double_array(img_house)

# Ungamma initial image
array_house_double = ungamma_correct(array_house_double, 2.2)
 
# Create Bayer index matrices
I_2 = np.zeros((2,2))
I_2[0,0] = 1
I_2[0,1] = 2
I_2[1,0] = 3
I_2[1,1] = 0

I_4 = generate_bayer(I_2)
I_8 = generate_bayer(I_4)

# Generate threshold matrices
T_2 = create_threshold_matrix(I_2)
T_4 = create_threshold_matrix(I_4)
T_8 = create_threshold_matrix(I_8)

dither_image(array_house_double, T_2, "DitherWith2by2.tif")
#array_house_double = convert_image_to_double_array(img_house)
#array_house_double = ungamma_correct(array_house_double, 2.2)
dither_image(array_house_double, T_4, "DitherWith4by4.tif")
#array_house_double = convert_image_to_double_array(img_house)
#array_house_double = ungamma_correct(array_house_double, 2.2)
dither_image(array_house_double, T_8, "DitherWith8by8.tif")


## Calculate rmse
#house_rmse = rmse(array_house_double, array_thresh_double)
#print("RMSE:",house_rmse)


## Fill low pass filter
#gaussian_lpf = np.zeros((7,7))
#gaussian_lpf = calculate_lpf(gaussian_lpf, 2)

##apply filter
#array_house_double = apply_guassian_filter(gaussian_lpf,array_house_double)


##apply transformation
#array_house_double = apply_transformation(array_house_double)
#array_thresh_double = apply_transformation(array_thresh_double)

#calculate fidelity
#fid = fidelity(array_house_double, array_thresh_double)
#print("Fidelity:", fid)

