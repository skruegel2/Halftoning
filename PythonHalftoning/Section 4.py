import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from PIL import Image
import math

def convert_image_to_double_array(img):
    converted = np.array(img, dtype = float)
    return converted;

def convert_to_double(input_array):
    output = np.zeros((input_array.shape[0],input_array.shape[1]),dtype=float)
    for row_idx in range(input_array.shape[0]):
        for col_idx in range(input_array.shape[1]):  
            output[row_idx, col_idx] = float(input_array[row_idx,col_idx])
    return output

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
    plt.imshow(im_dithered,cmap='gray',interpolation='none')
    plt.show()
    im_dithered.save(filename)    
    return dither_array

# Section 4
 
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
print("Bayer 2x2:")
print(I_2)
T_4 = create_threshold_matrix(I_4)
print("Bayer 4x4:")
print(I_4)
T_8 = create_threshold_matrix(I_8)
print("Bayer 8x8:")
print(I_8)

img_house = Image.open('house.tif')

array_house = np.array(img_house)
# Ungamma initial image
array_house_ungamma = ungamma_correct(array_house, 2.2)
# Convert to double
ah_double = convert_to_double(array_house)
ah_ungamma_double = convert_to_double(array_house_ungamma)
# 2 x 2 dither
dither_2by2 = dither_image(ah_ungamma_double, T_2, "DitherWith2by2.tif")
rmse_2by2 = rmse(ah_double, dither_2by2)
print("2 x 2 RMSE:",rmse_2by2)
fid = fidelity(ah_ungamma_double, dither_2by2)
print("2 x 2 Fidelity:", fid)

# 4 x 4 dither
dither_4by4 = dither_image(ah_ungamma_double, T_4, "DitherWith4by4.tif")
rmse_4by4 = rmse(ah_double, dither_4by4 )
print("4 x 4 RMSE:",rmse_4by4)
fid = fidelity(ah_ungamma_double, dither_4by4 )
print("4 x 4 Fidelity:", fid)

# 8 x 8 dither
dither_8by8 = dither_image(ah_ungamma_double, T_8, "DitherWith8by8.tif")
rmse_8by8 = rmse(ah_double, dither_8by8)
print("8 x 8 RMSE:",rmse_8by8)
fid = fidelity(ah_ungamma_double, dither_8by8)
print("8 x 8 Fidelity:", fid)





