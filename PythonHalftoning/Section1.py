from itertools import zip_longest
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from PIL import Image
from numpy import linalg as LA

# Section 1
img14g = Image.open('img14g.tif')
#img14g_plot = plt.imshow(img14g)
#plt.show()
img14bl = Image.open('img14bl.tif')
#img14bl_plot = plt.imshow(img14bl)
#plt.show()
img14gn = Image.open('img14gn.tif')
#img14gn_plot = plt.imshow(img14gn)
#plt.show()
img14sp = Image.open('img14sp.tif')
#img14sp_plot = plt.imshow(img14sp)
#plt.show()

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

def calculate_theta_star_array(theta_star):
    theta_star_array = np.zeros((7,7))
    for row_idx in range(7):
        for col_idx in range(7):
            theta_star_array[row_idx,col_idx] = theta_star[col_idx + 7 * row_idx]
    return theta_star_array

def filter_pixel(X, theta_star_array, row_idx, col_idx):
    pixel = 0
    for win_row_idx in range(-3,4):
        for win_col_idx in range(-3, 4):
            if ((win_row_idx + row_idx < 0) or
                (win_row_idx + row_idx >= X.shape[0]) or
                (win_col_idx + col_idx < 0) or
                (win_col_idx + col_idx >= X.shape[1])):
                pixel += 0
            else:
                pixel += (theta_star_array[win_row_idx+3,win_col_idx+3]*
                         X[win_row_idx+row_idx, win_col_idx+col_idx])
    return pixel

def apply_optimal_filter(theta_star_array, im, filename):
    X = np.array(im)
    Y = np.zeros((X.shape))
    for row_idx in range(X.shape[0]):
        for col_idx in range(X.shape[1]):
            Y[row_idx, col_idx] = filter_pixel(X, theta_star_array, row_idx, col_idx)
            if (Y[row_idx, col_idx] < 0):
                Y[row_idx, col_idx] = 0
            if (Y[row_idx, col_idx] > 255):
                Y[row_idx, col_idx] = 255;
    #Y = np.clip(Y,0,1)
    #Y= np.uint8(Y*255)
    im_filtered = Image.fromarray(Y.astype(np.uint8))
    im_filtered.save(filename)    
    #im_filtered.show()

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

Y = calculate_Y(img14g)
Z = calculate_Z(img14sp,Y)
Rzz = calculate_Rzz(Z)
Rhat_zy = calculate_Rhat_zy(Y, Z)
theta_star = calculate_theta_star(Rzz, Rhat_zy)
theta_star_array = calculate_theta_star_array(theta_star)
print(theta_star_array)
apply_optimal_filter(theta_star_array, img14sp,"Filtered noisy image 2 (img14sp).tif")
