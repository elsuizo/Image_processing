#! /usr/bin/env python

"""
Collection of image processing functions in Python

Author: Martin Noblía
License: MIT
"""
#*************************************************************************
# Imports
import numpy as np
import cv2
#*************************************************************************

def salt_and_pepper(image, n):
    """
    Inputs:
    ------
    image: image(numpy.array)
    n: number of noise samples(Int)

    Output:
    ------
    image: image + noise (numpy.array)

    """
    a, b = image.shape
    for i in xrange(0,n):
        num_row = np.random.randint(a)
        num_col = np.random.randint(b)
        image[num_row,num_col]=255

    return image

def imadjust(image, low_in, hig_in, low_out, hig_out, gamma):
    """
    Compute the pixels transformation(Matlab function)

    inputs:
    ------
    image: grayscale image

    output:
    ------
    modified image

    """
    image_mod = np.zeros(image.shape, dtype=np.uint8)

    image_mod = low_out + (hig_out - low_out) * ((image - low_in) / (hig_in - low_in)) ** gamma


    return image_mod


def histograma(image):
    """
    Compute the histogram of a uint8 image(grayscale)

    inputs
    -------
    image: a grayscale image

    output
    ------
    h: numpy array with the probabilities of each pixel
    """
    h = np.zeros(256, dtype=float)
    image = image.astype(np.uint8)
    for i in xrange(256):
        x = np.array([])
        x = np.argwhere(image == i)
        if (len(x) == 0):
            continue
        h[i] = len(x) / float(image.size)  # Normalizamos
    return h

def contrast_stretch(image, m, E):
    """
    compute the contrast stretch transformation(Gonzalez-Goods: pag:69)

    inputs:
    ------
    image: a grayscale image
    m: cut-off grayscale limit
    E: nonlinear parameter(control the slope of the function)

    output:
    ------
    g: transformed image
    """
    epsilon = np.finfo(np.float).eps

    g = 1 / (1 + (m / (image + epsilon)) ** E)

    return g


def false_color(img):
    """
    Function for calculate a false color of a image

    Input:
    -----
    img: grayscale image

    output:
    ------
    img_fc: 3-channel false color image

    """
    # red LUT
    red = np.zeros(256,dtype="uint8")
    red[0:43] =  255
    red[43:86] = np.arange(43,0,-1) * (255.0 / 43.0)
    red[172:215] = np.arange(0,43) * (255.0 / 43.0)
    red[214:] = 255
    # green LUT
    green = np.zeros(256,dtype="uint8")
    green[0:43] = np.arange(43) * (255.0 / 43.0)
    green[43:129] = 255
    green[129:172] = np.arange(43,0,-1) * (255.0 / 43.0)
    # blue LUT
    blue = np.zeros(256,dtype="uint8")
    blue[86:129] = np.arange(43) * (255.0 / 43.0)
    blue[129:213] = 255
    blue[213:] = np.arange(43,0,-1) * (255 / 43.0)

    m,n = img.shape
    img_fc = np.zeros((m,n,3))

    img_fc_r = cv2.LUT(img,red)
    img_fc_g = cv2.LUT(img,green)
    img_fc_b = cv2.LUT(img,blue)
    img_fc = cv2.merge((img_fc_r,img_fc_g,img_fc_b))
    return img_fc


def ker_prom(n):
    ker = np.ones((n,n))
    ker_promedio = ker/np.size(ker)
    return ker_promedio

def segmentacion_auto_Gonz(image):
    """
    Calculate a optimal segmentation value

    Entrada: image (grayscale)
    -------
    Salida: T (float segmentation value)
    ------
    """

    T = 0.5 * (np.min(image) + np.max(image))
    flag = False
    while not flag:
        g = image >= T
        T_next = 0.5 * (np.mean(image[g]) + np.mean(image[np.invert(g)]))
        flag = np.abs(T - T_next) < 0.5
        T = T_next
    return T


def dftuv(M,N):
    """
    Funcion para realizar meshgrids de frecuencias(Gonzalez, Goods, pag:128)
    Parametros:
    ------------------------------------------------------------------------
    M : entero
    Dimension en U
    N : entero
    Dimension en V
    Salida
    ------------------------------------------------------------------------
    meshgrid ---> U, V

    """
    u = np.arange(0,M)
    v = np.arange(0,N)

    idx = np.where(u >= M/2)
    u[idx] = u[idx] - M
    idy = np.where(v >= N/2)
    v[idy] = v[idy] - N

    V, U = np.meshgrid(v,u)

    return V,U

def lpfilter_ideal(im,D_0):
    """ """
    M,N = im.shape
    U,V = dftuv(M,N)

    D = np.sqrt(U**2 + V**2)

    H = (D <= D_0)
    H_shift = np.fft.fftshift(H)

    F = fftpack.fft2(im)
    F_shift = np.fft.fftshift(F)

    im_fil = H_shift * F_shift

    im_back = np.fft.ifft2(im_fil)

    im_back = np.abs(im_back)

    return im_back

def lpfilter_butterworth(im,D_0,n):
    """ """
    M,N = im.shape
    U,V = dftuv(M,N)

    D = np.sqrt(U**2 + V**2)

    H = 1/(1 + (D/D_0)**(2*n))
    H_shift = np.fft.fftshift(H)

    F = fftpack.fft2(im)
    F_shift = np.fft.fftshift(F)

    im_fil = H_shift * F_shift

    im_back = np.fft.ifft2(im_fil)

    im_back = np.abs(im_back)

    return im_back


def lpfilter_gaussian(im,D_0):
    """ """
    M,N = im.shape
    U,V = dftuv(M,N)

    D = np.sqrt(U**2 + V**2)
    #mascara gaussiana
    H = np.exp((-D**2 )/ (2 * D_0**2))
    H_shift = np.fft.fftshift(H)

    F = fftpack.fft2(im)
    F_shift = np.fft.fftshift(F)

    im_fil = H_shift * F_shift

    im_back = np.fft.ifft2(im_fil)

    im_back = np.abs(im_back)

    return im_back


def hpfilter_ideal(im,D_0):
    """ """
    M,N = im.shape
    U,V = dftuv(M,N)

    D = np.sqrt(U**2 + V**2)

    H_lp = (D <= D_0)
    H_hp = 1 - H_lp
    H_shift = np.fft.fftshift(H_hp)

    F = fftpack.fft2(im)
    F_shift = np.fft.fftshift(F)

    im_fil = H_shift * F_shift

    im_back = np.fft.ifft2(im_fil)

    im_back = np.abs(im_back)

    return im_back


def hpfilter_butterworth(im,D_0,n):
    """ """
    M,N = im.shape
    U,V = dftuv(M,N)

    D = np.sqrt(U**2 + V**2)

    H_lp = 1/(1 + (D/D_0)**(2*n))
    H_hp = 1 - H_lp
    H_shift = np.fft.fftshift(H_hp)

    F = fftpack.fft2(im)
    F_shift = np.fft.fftshift(F)

    im_fil = H_shift * F_shift

    im_back = np.fft.ifft2(im_fil)

    im_back = np.abs(im_back)

    return im_back


def hpfilter_gaussian(im,D_0):
    """ """
    M,N = im.shape
    U,V = dftuv(M,N)

    D = np.sqrt(U**2 + V**2)
    #mascara gaussiana
    H_lp = np.exp((-D**2 )/ (2 * D_0**2))
    H_hp = 1 - H_lp

    H_shift = np.fft.fftshift(H_hp)

    F = fftpack.fft2(im)
    F_shift = np.fft.fftshift(F)

    im_fil = H_shift * F_shift

    im_back = np.fft.ifft2(im_fil)

    im_back = np.abs(im_back)

    return im_back



def imnoise(im, sigma, mu):

    M, N = im.shape
    R = (sigma * np.random.rand(M,N) + mu) + im

    return R
