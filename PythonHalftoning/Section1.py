from itertools import zip_longest
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from PIL import Image
from numpy import linalg as LA
import math

def calculate_Y(img):
    y_list = []
    # Convert into array
    y_data = np.array(img)
    for row_idx in range(y_data.shape[0]):
        for col_idx in range(y_data.shape[1]):
            if ((row_idx % 20 == 0) and (col_idx % 20 == 0) and (row_idx > 0) and (col_idx > 0)):
                y_list.append(y_data[row_idx,col_idx]) 
    Y = np.asarray(y_list)
    return Y

def get_window_pixels(z_data, row_idx, col_idx):
    z_row = []
    for win_row_idx in range(-3,4):
        for win_col_idx in range(-3, 4):
            if (win_row_idx + row_idx < 0):
                z_row.append(0)
            elif (win_row_idx + row_idx >= z_data.shape[0]):
                z_row.append(0)
            elif (win_col_idx + col_idx < 0):
                z_row.append(0)
            elif (win_col_idx + col_idx >= z_data.shape[1]):
                z_row.append(0)
            else:
                z_row.append(z_data[row_idx+win_row_idx,col_idx+win_col_idx])
    return z_row

def calculate_Z(img, Y):
    Z = np.zeros((Y.shape[0],49))
    z_row_idx = 0
    # Convert into array
    z_data = np.array(img)
    for row_idx in range(z_data.shape[0]):
        for col_idx in range(z_data.shape[1]):
            if ((row_idx % 20 == 0) and (col_idx % 20 == 0) and (row_idx > 0) and (col_idx > 0)):
                z_row = get_window_pixels(z_data, row_idx, col_idx)
                for z_col_idx in range(49):
                    Z[z_row_idx,z_col_idx] = z_row[z_col_idx]
                z_row_idx += 1;
    return Z

def calculate_Rzz(Z):
    N = Z.shape[0]
    #print(N)
    Rzz = np.matmul(np.transpose(Z),Z)
    Rzz /= N
    return Rzz

def calculate_Rhat_zy(Y, Z):
    N = Z.shape[0]
    #print(N)
    Rhat_zy = np.matmul(np.transpose(Z),Y)
    Rhat_zy /= N
    return Rhat_zy

def calculate_theta_star(Rzz, Rhat_zy):
    theta_star = np.matmul(np.linalg.inv(Rzz),Rhat_zy)
    return theta_star

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
    im_filtered.save("test.tif")    


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
gaussian_lpf = calculate_lpf(gaussian_lpf, 2)
apply_guassian_filter(gaussian_lpf,array_house_double)
#Y = calculate_Y(img14g)
#Z = calculate_Z(img14bl,Y)
#Rzz = calculate_Rzz(Z)
#Rhat_zy = calculate_Rhat_zy(Y, Z)
#theta_star = calculate_theta_star(Rzz, Rhat_zy)
#theta_star_array = calculate_theta_star_array(theta_star)
#print(theta_star_array)
#apply_optimal_filter(theta_star_array, img14bl,"Filtered blurred image.tif")

#Y = calculate_Y(img14g)
#Z = calculate_Z(img14gn,Y)
#Rzz = calculate_Rzz(Z)
#Rhat_zy = calculate_Rhat_zy(Y, Z)
#theta_star = calculate_theta_star(Rzz, Rhat_zy)
#theta_star_array = calculate_theta_star_array(theta_star)
#print(theta_star_array)
#apply_optimal_filter(theta_star_array, img14gn,"Filtered noisy image 1 (img14gn).tif")

#Y = calculate_Y(img14g)
#Z = calculate_Z(img14sp,Y)
#Rzz = calculate_Rzz(Z)
#Rhat_zy = calculate_Rhat_zy(Y, Z)
#theta_star = calculate_theta_star(Rzz, Rhat_zy)
#theta_star_array = calculate_theta_star_array(theta_star)
#print(theta_star_array)
#apply_optimal_filter(theta_star_array, img14sp,"Filtered noisy image 2 (img14sp).tif")
