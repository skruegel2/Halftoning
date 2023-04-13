import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from PIL import Image
import math

FILT_SIZE = 3
HALF_FILT = 1
ERR_DIFF_THRESH = 127

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
    g_output = np.zeros((g_array.shape[0],g_array.shape[1]),dtype=float)
    for row_idx in range(g_array.shape[0]):
        for col_idx in range(g_array.shape[1]):
            g_output[row_idx, col_idx] = math.pow((g_array[row_idx, col_idx]/255), gamma)    
            g_output[row_idx, col_idx] *= 255
    return g_output

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
    #print("Scale check: ",scale_check)
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

def apply_gaussian_filter(filt, image_array):
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

def threshold_pixel(pixel, thresh):
    if pixel > thresh:
        ret_val = 255
    else:
        ret_val  = 0
    return ret_val

def apply_transformation(image_array):
    transform_array = np.zeros((image_array.shape[0],image_array.shape[1]))
    for row_idx in range(image_array.shape[0]):
        for col_idx in range(image_array.shape[1]):
            pixel = image_array[row_idx, col_idx]/255
            pixel = math.pow(pixel,1/3)
            transform_array[row_idx, col_idx] = 255*pixel
    return transform_array

def fidelity(f, b):
    lpf = np.zeros((7,7))
    calculate_lpf(lpf,math.sqrt(2))
    f = apply_gaussian_filter(f, lpf)
    b = apply_gaussian_filter(b, lpf)
    f = apply_transformation(f)
    b = apply_transformation(b)
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


def init_err_diff_filter()  :
    h = np.zeros((FILT_SIZE, FILT_SIZE))
    h[1,2] = 7/16
    h[2,0] = 3/16
    h[2,1] = 5/16
    h[2,2] = 1/16
    return h

# Diffuse error
def add_scaled_error(f,h,error,row_idx, col_idx):
    for win_row_idx in range(-1,2):
        for win_col_idx in range(-1, 2):
            if ((win_row_idx + row_idx >= 0) and
                (win_row_idx + row_idx < f.shape[0]) and
                (win_col_idx + col_idx >= 0) and
                (win_col_idx + col_idx < f.shape[1])):
                f[win_row_idx+row_idx, win_col_idx+col_idx] += h[win_row_idx+1,win_col_idx+1]*error
    return f

def diffuse_error(f,filename):
    h = init_err_diff_filter()
    e = np.zeros((f.shape[0],f.shape[1]))
    b = np.zeros((f.shape[0],f.shape[1]))
    for row_idx in range(f.shape[0]):
        for col_idx in range(f.shape[1]):
            b[row_idx, col_idx] = threshold_pixel(f[row_idx,col_idx],ERR_DIFF_THRESH)
            error = f[row_idx,col_idx] - b[row_idx, col_idx]             
            f = add_scaled_error(f,h,error,row_idx,col_idx)
    diffused = Image.fromarray(b.astype(np.uint8))
    #plt.imshow(diffused,cmap='gray',interpolation='none')
    #plt.show()
    diffused.save(filename) 
    return b

def convert_to_double(input_array):
    output = np.zeros((input_array.shape[0],input_array.shape[1]),dtype=float)
    for row_idx in range(input_array.shape[0]):
        for col_idx in range(input_array.shape[1]):  
            output[row_idx, col_idx] = float(input_array[row_idx,col_idx])
    return output

# Section 5
img_house = Image.open('house.tif')

array_house = np.array(img_house)
# Ungamma initial image
array_house_ungamma = ungamma_correct(array_house, 2.2)
# Convert to double
ah_double = convert_to_double(array_house)
ah_ungamma_double = convert_to_double(array_house_ungamma)
diffused_array = diffuse_error(ah_ungamma_double,"diffused_error.tif")
# RMSE
rmse_diffused = rmse(ah_double, diffused_array)
print("RMSE diffused: ", rmse_diffused)
# Fidelity
fidelity_diffused = fidelity(ah_ungamma_double, diffused_array)
print("Fidelity diffused: ", fidelity_diffused)






