import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from PIL import Image
from numpy import linalg as LA

def create_wavelength():
    wavelength = np.zeros((1, 31))
    for idx in range(31):
        wavelength[0][idx] = 400 + 10 * idx
    return wavelength

def plot_wavelength(data, wavelength):
    plt.clf
    plt.title("XYZ Color Matching Functions")
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Value")
    plt.plot(wavelength[0,:], data['x'][0,:])
    plt.plot(wavelength[0,:], data['y'][0,:])
    plt.plot(wavelength[0,:], data['z'][0,:])
    plt.legend(['x', 'y', 'z'])
    plt.show()

def plot_lms(lms, wavelength):
    plt.clf
    plt.title("LMS Color Matching Functions")
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Value")
    plt.plot(wavelength[0,:],lms[0,:])
    plt.plot(wavelength[0,:],lms[1,:])
    plt.plot(wavelength[0,:],lms[2,:])
    plt.legend(['l', 'm', 's'])
    plt.show()

def plot_illumination(data, wavelength):
    plt.clf
    plt.title("Illumination")
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Value")
    plt.plot(wavelength[0,:],data['illum1'][0,:])
    plt.plot(wavelength[0,:],data['illum2'][0,:])
    plt.legend(['D65', 'Fluorescent'])
    plt.show()

def plot_chromaticity(data,R_CIE_1931,G_CIE_1931,B_CIE_1931,R_709,G_709,B_709,
                      D65, EE):
    xchrom = data['x'][0,:] / (data['x'][0,:] + data['y'][0,:] +
                               data['z'][0,:] )
    ychrom = data['y'][0,:] / (data['x'][0,:] + data['y'][0,:] +
                               data['z'][0,:] )
    plt.clf()
    plt.style.use('default')
    plt.title("Pure Spectral Source Chromaticity")
    #plt.xlabel("Wavelength [nm]")
    plt.plot(xchrom, ychrom)
    # CIE
    x_cie = [R_CIE_1931[0],G_CIE_1931[0],B_CIE_1931[0],R_CIE_1931[0]]
    y_cie  = [R_CIE_1931[1],G_CIE_1931[1],B_CIE_1931[1],R_CIE_1931[1]]
    x_709 = [R_709[0], G_709[0], B_709[0], R_709[0]]
    y_709 = [R_709[1], G_709[1], B_709[1], R_709[1]]
    x_D65 = [D65[0]]
    y_D65 = [D65[1]]
    x_EE = [EE[0]]
    y_EE = [EE[1]]
    plt.plot(x_cie, y_cie,'y')
    plt.text(R_CIE_1931[0],R_CIE_1931[1],'R CIE')
    plt.text(G_CIE_1931[0],G_CIE_1931[1],'G CIE')
    plt.text(B_CIE_1931[0],B_CIE_1931[1],'B CIE')
    plt.plot(x_709, y_709,'g')
    plt.text(R_709[0],R_709[1],'R 709')
    plt.text(G_709[0],G_709[1],'G 709')
    plt.text(B_709[0],B_709[1],'B 709')
    plt.plot(x_D65, y_D65, 'co')
    plt.text(D65[0]-0.13,D65[1]-0.02,'D65 White')
    plt.plot(x_EE, y_EE, 'mo')
    plt.text(EE[0]+0.025, EE[1],'Equal Energy White')
    plt.show()

def compute_I_D65(data, reflect_data,wavelength):
    illum = data['illum1']
    reflect = reflect_data['R']
    I = reflect*illum
    return I

def compute_I_fluor(data, reflect_data,wavelength):
    illum = data['illum2']
    reflect = reflect_data['R']
    I = reflect*illum
    return I

def compute_XYZ_D65(I, data, wavelength):
    x = data['x'][0,:]
    y = data['y'][0,:]
    z = data['z'][0,:]
    XYZ = np.zeros((I.shape[0],I.shape[1], 3))
    XYZ[:,:,0] = np.dot(I,x)
    XYZ[:,:,1] = np.dot(I,y)
    XYZ[:,:,2] = np.dot(I,z)
    return XYZ

def compute_XYZ_fluor(I, data, wavelength):
    x = data['x'][0,:]
    y = data['y'][0,:]
    z = data['z'][0,:]
    XYZ = np.zeros((I.shape[0],I.shape[1], 3))
    XYZ[:,:,0] = np.dot(I,x)
    XYZ[:,:,1] = np.dot(I,y)
    XYZ[:,:,2] = np.dot(I,z)
    return XYZ

def compute_K(RGB_709, XYZ_WP):
    K = np.matmul(np.linalg.inv(RGB_709),XYZ_WP)
    return K

def compute_M(RGB_709, K):
    diag_K = np.diag(K)
    M = np.matmul(RGB_709,diag_K)
    return M

def compute_image_RGB(M,XYZ):
    RGB = XYZ
    for row_idx in range(XYZ.shape[0]):
        for col_idx in range(XYZ.shape[1]):
            RGB[row_idx,col_idx,:] = np.matmul(np.linalg.inv(M),XYZ[row_idx,col_idx,:])
    return RGB

def threshold_0_1(RGB):
    for row_idx in range(RGB.shape[0]):
        for col_idx in range(RGB.shape[1]):
            for color_idx in range(3):
                if RGB[row_idx,col_idx,color_idx] < 0:
                    RGB[row_idx,col_idx,color_idx] = 0
                if RGB[row_idx,col_idx,color_idx] > 1:
                    RGB[row_idx,col_idx,color_idx] = 1
    return RGB;

def gamma_correct(RGB, gamma):
    for row_idx in range(RGB.shape[0]):
        for col_idx in range(RGB.shape[1]):
            for color_idx in range(3):
                RGB[row_idx,col_idx,color_idx] = pow(RGB[row_idx,col_idx,color_idx],1/gamma)
    return RGB

def save_image(RGB, filename):
    RGB_int = np.zeros(RGB.shape,np.uint8)
    for row_idx in range(RGB.shape[0]):
        for col_idx in range(RGB.shape[1]):
            for color_idx in range(3):
                RGB_int[row_idx,col_idx,color_idx] = np.uint8(255*RGB[row_idx,col_idx,color_idx])
    im = Image.fromarray(RGB_int)
    im.save(filename)

# Section 4
# Load data.npy
data = np.load('./CIE_data/data.npy', allow_pickle=True)[()]

# Load reflect.npy
reflect_data = np.load('./reflect/reflect.npy', allow_pickle=True)[()]
wavelength = create_wavelength()
# Compute I
I = compute_I_D65(data, reflect_data, wavelength)
XYZ = compute_XYZ_D65(I, data, wavelength)


RGB_709 = [[0.640, 0.300, 0.150],
           [0.330, 0.600, 0.060],
           [0.030, 0.100, 0.790]]
D65_WP = [0.3127, 0.3290, 0.3583]
XYZ_WP = [0.3127/0.3290, 1, 0.3583/0.3290]
K = compute_K(RGB_709, XYZ_WP)
M = compute_M(RGB_709, K)
# Transform XYZ into RGB
RGB_d65 = compute_image_RGB(M,XYZ)
RGB_d65 = threshold_0_1(RGB_d65)
RGB_d65 = gamma_correct(RGB_d65,2.2)
save_image(RGB_d65, 'RGB_D65.tif')


I_fluor = compute_I_fluor(data, reflect_data, wavelength)
XYZ_fluor = compute_XYZ_fluor(I_fluor, data, wavelength)
RGB_fluor = compute_image_RGB(M, XYZ_fluor)
RGB_fluor = threshold_0_1(RGB_fluor)
RGB_fluor = gamma_correct(RGB_fluor,2.2)
save_image(RGB_d65, 'RGB_fluor.tif')



