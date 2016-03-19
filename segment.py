import numpy as np
from itertools import *
from math import *
import cv2
import util

def circunference(img, blobs):
    masks = []
    for blob in blobs:
        x, y, r = blob
        mask = np.zeros((2*r + 1, 2*r + 1), np.uint8)
        for i, j in product(range(2*r + 1), range(2*r + 1)):
            if r  ** 2 > (r - i) ** 2  + (r - j) ** 2:
                mask[i][j] = 1
        masks.append(mask)
        
    return blobs, np.array(masks)

# ADT by ARG segmentation
def cos_angle(a, b):
    dot = 1.0 * (a[0] * b[0] + a[1] * b[1])
    len_a = sqrt(a[0] * a[0] + a[1] * a[1])
    len_b = sqrt(b[0] * b[0] + b[1] * b[1])

    if len_a * len_b == 0:
        return 0

    tmp = dot/(len_a * len_b)
    if tmp > 1:
        tmp = 1
    elif tmp < -1:
        tmp = -1
    
    # cos(acos(tmp)
    return tmp

def distance_thold(img, point, grad, dx, dy, t0=0,):
    point = (int(point[0]), int(point[1]))
    size = img.shape
    rmax = 25
    diam = 2 * rmax + 1
    tdelta = 1.7
    mask = np.full((diam, diam), 0, dtype=np.uint8)

    rx = np.linspace(-1 * rmax, rmax, 2 * rmax + 1)
    ry = np.linspace(-1 * rmax, rmax, 2 * rmax + 1)
    ry, rx = np.meshgrid(rx, ry)
    dist = rx ** 2 + ry ** 2
    dist = dist.astype(np.float64)
    ndist = dist / (rmax ** 2)
    ratio = (1.0 - np.exp(-1 * ndist)) / ( 1.0 - exp(-1))
    T = t0 + tdelta * ratio

    mask = dist < rmax ** 2
    mask = mask.astype(np.uint8)
    T = T * mask
    T[T == 0] = 1e10

    roi = img[point[0] - rmax:point[0] + rmax + 1, point[1] - rmax:point[1] + rmax + 1]
    s = roi > T
    s = s.astype(np.uint8)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # TODO:  Hole filling via morphological op ( it can be done with findConturs retrccomps )
    s = cv2.morphologyEx(s, cv2.MORPH_OPEN, kernel)

    smask = np.full((s.shape[0] + 2, s.shape[1] + 2), dtype=np.uint8, fill_value=0)
    s[rmax, rmax] = 1
    cv2.floodFill(s, smask, (rmax, rmax), 255, loDiff=0, upDiff=0, flags=4|cv2.FLOODFILL_FIXED_RANGE)
    s = smask[1:s.shape[0] + 1, 1:s.shape[1] + 1]
    
    # Calculate the ARG of t0
    rx = -1 * np.linspace(-1 * rmax, rmax, 2 * rmax + 1)
    ry = -1 * np.linspace(-1 * rmax, rmax, 2 * rmax + 1)
    ry, rx = np.meshgrid(rx, ry)
    rmag = (rx ** 2 + ry ** 2) ** 0.5

    rdx = dx[point[0] - rmax:point[0] + rmax + 1, point[1] - rmax:point[1] + rmax + 1]
    rdy = dy[point[0] - rmax:point[0] + rmax + 1, point[1] - rmax:point[1] + rmax + 1]
    rgrad = grad[point[0] - rmax:point[0] + rmax + 1, point[1] - rmax:point[1] + rmax + 1]

    rphase = (rx * rdx + ry * rdy) / ( rmag * rgrad + util.EPS)
    rarg = rgrad * rphase

    rarg = rarg * s
    rgrad = rgrad * s
    a = np.sum(rarg)
    b = np.sum(s)

    if b == 0: 
        return -1e10, s
    else:
        return (a * 1.0 / (b + util.EPS)), s

def _adaptive_thold(img, point, grad, dx, dy):
    coorners = [[1e10, 1e10], [-1e10, -1e10]]
    best_arg = -1e10
    best_t = 0
    for t in np.arange(0, -4, -0.1):
        arg, _ = distance_thold(img, point, grad, dx, dy, t)
        #print("{} {}".format(t, arg))      
        #util.imshow('mask', _)

        if best_arg < arg:
            best_arg = arg
            best_t = t

    _, mask = distance_thold(img, point, grad, dx, dy, best_t)

    return np.array(mask)

# TODO: replace by efficient fd
def finite_derivatives(img):
    size = img.shape
    dx = img.copy()
    dy = img.copy()

    for i, j in product(range(1, size[0] - 1), range(1, size[1] - 1)):
        dy[i, j] = (img[i, j + 1] - img[i, j - 1]) / 2.0
        dx[i, j] = (img[i + 1, j] - img[i - 1, j]) / 2.0
    mag = (dx ** 2 + dy ** 2) ** 0.5

    return mag, dx, dy

def adaptive_distance_thold(img, blobs):
    masks = []
    mag, dx, dy = finite_derivatives(img)

    for blob in blobs:
        x, y, r = blob
        mask = _adaptive_thold(img, (x, y), mag, dx, dy)
        masks.append(mask)

    return blobs, np.array(masks)
