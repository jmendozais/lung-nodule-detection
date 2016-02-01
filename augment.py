from __future__ import absolute_import

import numpy as np
import re
from scipy import ndimage
from scipy import linalg

from os import listdir
from os.path import isfile, join
import random, math
from six.moves import range

'''
    Fairly basic set of tools for realtime data augmentation on image data.
    Can easily be extended to include new transforms, new preprocessing methods, etc...
'''

def random_rotation(x, rg, fill_mode="nearest", cval=0.):
    angle = random.uniform(-rg, rg)
    x = ndimage.interpolation.rotate(x, angle, axes=(1,2), reshape=False, mode=fill_mode, cval=cval)
    return x

def random_shift(x, wrg, hrg, fill_mode="nearest", cval=0.):
    crop_left_pixels = 0
    crop_right_pixels = 0
    crop_top_pixels = 0
    crop_bottom_pixels = 0

    original_w = x.shape[1]
    original_h = x.shape[2]

    if wrg:
        crop = random.uniform(0., wrg)
        split = random.uniform(0, 1)
        crop_left_pixels = int(split*crop*x.shape[1])
        crop_right_pixels = int((1-split)*crop*x.shape[1])

    if hrg:
        crop = random.uniform(0., hrg)
        split = random.uniform(0, 1)
        crop_top_pixels = int(split*crop*x.shape[2])
        crop_bottom_pixels = int((1-split)*crop*x.shape[2])

    x = ndimage.interpolation.shift(x, (0, crop_left_pixels, crop_top_pixels), mode=fill_mode, cval=cval)
    return x

def horizontal_flip(x):
    for i in range(x.shape[0]):
        x[i] = np.fliplr(x[i])
    return x

def vertical_flip(x):
    for i in range(x.shape[0]):
        x[i] = np.flipud(x[i])
    return x


def random_barrel_transform(x, intensity):
    # TODO
    pass

def random_shear(x, intensity):
    # TODO
    pass

def random_channel_shift(x, rg):
    # TODO
    pass

def random_zoom(x, rg, fill_mode="nearest", cval=0.):
    zoom_w = random.uniform(1.-rg, 1.)
    zoom_h = random.uniform(1.-rg, 1.)
    x = ndimage.interpolation.zoom(x, zoom=(1., zoom_w, zoom_h), mode=fill_mode, cval=cval)
    return x # shape of result will be different from shape of input!

def gaussian_noise(x, mean=0, std=0.1):	
    noise = mean + random.randn(x.shape) * std
    scale_factor = np.max(x)
    return x + noise * scale_factor


def array_to_img(x, scale=True):
    from PIL import Image
    x = x.transpose(1, 2, 0) 
    if scale:
        x += max(-np.min(x), 0)
        x /= np.max(x)
        x *= 255
    if x.shape[2] == 3:
        # RGB
        return Image.fromarray(x.astype("uint8"), "RGB")
    else:
        # grayscale
        return Image.fromarray(x[:,:,0].astype("uint8"), "L")


def img_to_array(img):
    x = np.asarray(img, dtype='float32')
    if len(x.shape)==3:
        # RGB: height, width, channel -> channel, height, width
        x = x.transpose(2, 0, 1)
    else:
        # grayscale: height, width -> channel, height, width
        x = x.reshape((1, x.shape[0], x.shape[1]))
    return x


def load_img(path, grayscale=False):
    from PIL import Image
    img = Image.open(path)
    if grayscale:
        img = img.convert('L')
    else: # Assure 3 channel even when loaded image is grayscale
        img = img.convert('RGB')
    return img


def list_pictures(directory, ext='jpg|jpeg|bmp|png'):
    return [join(directory,f) for f in listdir(directory) \
        if isfile(join(directory,f)) and re.match('([\w]+\.(?:' + ext + '))', f)]

def offline_augment(X, y, ratio=0.1,
        featurewise_center=False, # set input mean to 0 over the dataset
        samplewise_center=False, # set each sample mean to 0
        featurewise_std_normalization=False, # divide inputs by std of the dataset
        samplewise_std_normalization=False, # divide each input by its std

        zca_whitening=False, # apply ZCA whitening
        rotation_range=0., # degrees (0 to 180)
        width_shift_range=0., # fraction of total width
        height_shift_range=0., # fraction of total height
        horizontal_flip=False,
        vertical_flip=False,
    ):

    aX = X.copy()
    ay = y.copy()

    factor = int((float(ratio * (len(y) - np.sum(y.T[1]))) / np.sum(y.T[1])))
    print 'factor {}'.format(factor)

    for i in range(X.shape[0]):
        if y[i][1] > 0:
            aXi = np.full((factor,) + X[i].shape, dtype=X[i].dtype, fill_value=X[i])
            aX = np.append(aX, aXi, axis=0)
            ayi = np.full((factor,) + y[i].shape, dtype=y[i].dtype, fill_value=y[i])
            ay = np.append(ay, ayi, axis=0)

    seed = random.randint(1, 10e6)
    np.random.seed(seed)
    np.random.shuffle(aX)
    np.random.seed(seed)
    np.random.shuffle(ay)

    X = np.copy(aX)
    y = np.copy(ay)

    for i in range(X.shape[0]):
        x = X[i]
        if y[i][1] > 0:
            x = random_transform(x.astype("float32"), rotation_range, width_shift_range, height_shift_range, horizontal_flip, vertical_flip)
        x = standardize(x, featurewise_center, featurewise_std_normalization, samplewise_center, samplewise_std_normalization, zca_whitening, 0, 0, [])
        X[i] = x

    return X, y


def random_transform(x, rotation_range, width_shift_range, height_shift_range, h_flip, v_flip):
    if width_shift_range or height_shift_range:
        x = random_shift(x, width_shift_range, height_shift_range)
    if h_flip:
        if random.random() < 0.5:
            x = horizontal_flip(x)
    if v_flip:
        if random.random() < 0.5:
            x = vertical_flip(x)

    # TODO:
    # zoom
    # barrel/fisheye
    # shearing
    # channel shifting
    return x

def standardize(x, featurewise_center, featurewise_std_normalization, samplewise_center, samplewise_std_normalization, zca_whitening, mean, std, principal_components):
        if featurewise_center:
            x -= mean
        if featurewise_std_normalization:
            x /= std

        if zca_whitening:
            flatx = np.reshape(x, (x.shape[0]*x.shape[1]*x.shape[2]))
            whitex = np.dot(flatx, principal_components)
            x = np.reshape(whitex, (x.shape[0], x.shape[1], x.shape[2]))

        if samplewise_center:
            x -= np.mean(x)
        if samplewise_std_normalization:
            x /= np.std(x)

        return x




