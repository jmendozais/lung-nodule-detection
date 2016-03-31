from __future__ import absolute_import 
from math import ceil
import numpy as np
import re
from scipy import ndimage
from scipy import linalg

from os import listdir
from os.path import isfile, join
import random, math
from six.moves import range
import util
from skimage import transform

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

class Preprocessor:
    def __init__(
        self,
        samplewise_center=False, # set each sample mean to 0
        samplewise_std_normalization=False, # divide each input by its std
        featurewise_rescaling=False,
        featurewise_rescaling_range=[0, 1],
        featurewise_center=False, # set input mean to 0 over the dataset
        featurewise_std_normalization=False, # divide inputs by std of the dataset
        zca_whitening=False,
        zca_fudge=10e-7
        ):

        self.samplewise_center = samplewise_center
        self.samplewise_std_normalization = samplewise_std_normalization
        self.featurewise_rescaling = featurewise_rescaling
        self.featurewise_rescaling_range = featurewise_rescaling_range
        self.featurewise_center = featurewise_center
        self.featurewise_std_normalization = featurewise_std_normalization
        self.zca_whitening = zca_whitening
        self.zca_fudge = zca_fudge

    def fit_transform(self, X, Y):
        X = np.copy(X)
        if self.samplewise_center:
            for i in range(len(X)):
                tmp = np.mean(X[i])
                X[i] -= tmp
        elif self.samplewise_std_normalization:
            for i in range(len(X)):
                tmp_mean = np.mean(X[i])
                tmp_std = np.std(X[i])
                X[i] -= tmp_mean
                X[i] /= tmp_std

        if self.featurewise_rescaling:
            self.amax = np.amax(X, axis=0)
            self.amin = np.amin(X, axis=0)
            X = (X - self.min) / (self.amax - self.amin + util.EPS)

        if self.featurewise_center or self.featurewise_std_normalization:
            self.mean = np.mean(X, axis=0)
            X -= self.mean
        if self.featurewise_std_normalization:
            self.std = np.std(X, axis=0)
            X /= self.std + util.EPS

        if self.zca_whitening:
            flat_X = np.reshape(X, (X.shape[0], X.shape[1] * X.shape[2] * X.shape[3]))
            if not (self.featurewise_center or self.featurewise_std_normalization):
                flat_mean = np.reshape(self.mean, X.shape[1] * X.shape[2] * X.shape[3])
                flat_X -= flat_mean

            sigma = np.dot(flatX.T, flatX) / flatX.shape[1]
            U, S, V = linalg.svd(sigma)
            self.principal_components = np.dot(np.dot(U, np.diag(1. / np.sqrt(S + self.zca_fudge))), U.T)
            whitex = np.dot(flat_X, self.principal_components)
            X = np.reshape(whitex, (X.shape[0], X.shape[1], X.shape[2], X.shape[3]))
        return X
            
    def transform(self, X):
        if self.samplewise_center:
            for i in range(len(X)):
                tmp = np.mean(X[i])
                X[i] -= tmp
        elif self.samplewise_std_normalization:
            for i in range(len(X)):
                tmp_mean = np.mean(X[i])
                tmp_std = np.std(X[i])
                X[i] -= tmp_mean
                X[i] /= tmp_std

        if self.featurewise_rescaling:
            X = (X - self.min) / (self.amax - self.amin + util.EPS)

        if self.featurewise_center or self.featurewise_std_normalization:
            X -= self.mean
        if self.featurewise_std_normalization:
            X /= self.std + util.EPS

        if self.zca_whitening:
            flat_X = np.reshape(X, (X.shape[0], X.shape[1] * X.shape[2] * X.shape[3]))
            if not (self.featurewise_center or self.featurewise_std_normalization):
                flat_mean = np.reshape(self.mean, X.shape[1] * X.shape[2] * X.shape[3])
                flat_X -= flat_mean
            whitex = np.dot(flat_X, self.principal_components)
            X = np.reshape(whitex, (X.shape[0], X.shape[1], X.shape[2], X.shape[3]))
        return X

import math
class ImageDataGenerator:

    def __init__(self, 
        rotation_range=(0.,0.),
        translation_range=(0.,0.), 
        flip=False,
        zoom_range=(1.,1.),

        batch_size=32,
        ratio=1.,
        mode='balance_batch'):

        self.rotation_range = rotation_range
        self.translation_range = translation_range
        self.flip = flip
        self.zoom_range = zoom_range

        self.batch_size = batch_size
        self.ratio = ratio
        self.mode = mode
    
    def build_augmentation_transform(self, zoom=(1.0, 1.0), rotation=0, shear=0, translation=(0, 0), flip=False): 
        if flip:
            shear += 180
            rotation += 180

        tform_augment = transform.AffineTransform(scale=(1/zoom[0], 1/zoom[1]), rotation=np.deg2rad(rotation), shear=np.deg2rad(shear), translation=translation)
        return tform_augment 

    def random_perturbation_transform(self, zoom_range, rotation_range, shear_range, translation_range, do_flip=True, allow_stretch=False, rng=np.random):
        shift_x = rng.uniform(*translation_range)
        shift_y = rng.uniform(*translation_range)
        translation = (shift_x, shift_y)

        rotation = rng.uniform(*rotation_range)
        shear = rng.uniform(*shear_range)

        if do_flip:
            flip = (rng.randint(2) > 0) # flip half of the time
        else:
            flip = False

        # random zoom
        log_zoom_range = [np.log(z) for z in zoom_range]
        if isinstance(allow_stretch, float):
            log_stretch_range = [-np.log(allow_stretch), np.log(allow_stretch)]
            zoom = np.exp(rng.uniform(*log_zoom_range))
            stretch = np.exp(rng.uniform(*log_stretch_range))
            zoom_x = zoom * stretch
            zoom_y = zoom / stretch
        elif allow_stretch is True: # avoid bugs, f.e. when it is an integer
            zoom_x = np.exp(rng.uniform(*log_zoom_range))
            zoom_y = np.exp(rng.uniform(*log_zoom_range))
        else:
            zoom_x = zoom_y = np.exp(rng.uniform(*log_zoom_range))

        # the range should be multiplicatively symmetric, so [1/1.1, 1.1] instead of [0.9, 1.1] makes more sense.
        return self.build_augmentation_transform((zoom_x, zoom_y), rotation, shear, translation, flip)

    def fast_warp(self, img, tf, output_shape=(50, 50), mode='constant', order=1):
        m = tf.params # tf._matrix is
        assert len(img) == 1, 'Input shape should be 1xMxN'
        img = transform._warps_cy._warp_fast(img[0], m, output_shape=output_shape, mode=mode, order=order)
        return np.array([img])

    def perturb(self, x):
        tform_augment = self.random_perturbation_transform(
            zoom_range=self.zoom_range, rotation_range=self.rotation_range,
            shear_range=(0., 0.), translation_range=self.translation_range, 
            do_flip=self.flip)
        return self.fast_warp(x, tform_augment, output_shape=x.shape, mode='constant').astype('float32')

    def augment(self, X, y):
        assert self.batch_size % 2 == 0, 'Batch size should be even (batch_size = {}).'.format(self.batch_size)
        factor = int((float(self.ratio * (len(y) - np.sum(y.T[1]))) / np.sum(y.T[1])))
        print 'factor: {}'.format(factor)

        # balance
        print 'Mode: {} ...'.format(self.mode)
        if self.mode == 'balance_batch':
            idx = y.T[1]
            positives = X[idx > 0]
            negatives = X[idx == 0]
            l_pos = y[idx > 0][0]
            l_neg = y[idx == 0][0]
            aX = np.zeros((0,) + X[0].shape).astype(X[0].dtype)
            ay = np.zeros((0,) + y[0].shape).astype(y[0].dtype)
            n_batches = int(ceil(float(2 * negatives.shape[0]) / self.batch_size))
            for batch_idx in range(n_batches):
                begin = batch_idx * (self.batch_size / 2)
                end = min(begin + (self.batch_size / 2), negatives.shape[0]) 
                aX = np.append(aX, negatives[begin:end], axis=0)    
                ay = np.append(ay, np.full((end - begin,) + l_neg.shape, dtype=l_neg.dtype, fill_value=l_neg), axis=0)
            
                b_idx = np.random.uniform(size=(end - begin,))
                b_idx = np.floor(b_idx * positives.shape[0]).astype(np.int)
                aX = np.append(aX, positives[b_idx], axis=0)
                ay = np.append(ay, np.full((end - begin,) + l_pos.shape, dtype=l_pos.dtype, fill_value=l_pos), axis=0)
            X = np.copy(aX)
            y = np.copy(ay)

        elif self.mode == 'balance_dataset':
            aX = X.copy()
            ay = y.copy()
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

        # transform
        for i in range(X.shape[0]):
            x = X[i]
            if y[i][1] > 0:
               # x = random_transform(x.astype("float32"), self.rotation_range, self.width_shift_range, self.height_shift_range, self.horizontal_flip, self.vertical_flip)
                x = self.perturb(x.astype("float32"))
            X[i] = x

        return X, y

def bootstraping_augment(X, y, ratio=0.1, batch_size=32,
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
    assert batch_size % 2 == 0, 'Batch size should be even'

    principal_components = []
    fmean = np.mean(X, axis=0)
    fstd = np.std(X, axis=0)

    if zca_whitening:
        flatX = np.reshape(X, (X.shape[0], X.shape[1] * X.shape[2] * X.shape[3]))
        flatX -= np.reshape(fmean, (fmean.shape[0], fmean.shape[1] * fmean.shape[2] * fmean.shape[3]))
        print 'mean {}'.format(np.mean(flatX, axis=0)) 
        print 'std {}'.format(np.std(flatX, axis=0)) 
        sigma = np.dot(flatX.T, flatX) / flatX.shape[1]
        U, S, V = linalg.svd(sigma)
        principal_components = np.dot(np.dot(U, np.diag(1. / np.sqrt(S + 10e-7))), U.T)


    factor = int((float(ratio * (len(y) - np.sum(y.T[1]))) / np.sum(y.T[1])))
    print 'factor {}'.format(factor)

    idx = y.T[1]
    positives = X[idx > 0]
    negatives = X[idx == 0]
    l_pos = y[idx > 0][0]
    l_neg = y[idx == 0][0]

    aX = np.zeros((0,) + X[0].shape).astype(X[0].dtype)
    ay = np.zeros((0,) + y[0].shape).astype(y[0].dtype)
    n_batches = int(ceil(float(2 * negatives.shape[0]) / batch_size))
    for batch_idx in range(n_batches):
        begin = batch_idx * (batch_size / 2)
        end = min(begin + (batch_size / 2), negatives.shape[0]) 
        aX = np.append(aX, negatives[begin:end], axis=0)    
        ay = np.append(ay, np.full((end - begin,) + l_neg.shape, dtype=l_neg.dtype, fill_value=l_neg), axis=0)
	
        b_idx = np.random.uniform(size=(end - begin,))
        b_idx = np.floor(b_idx * positives.shape[0]).astype(np.int)
        aX = np.append(aX, positives[b_idx], axis=0)
        ay = np.append(ay, np.full((end - begin,) + l_pos.shape, dtype=l_pos.dtype, fill_value=l_pos), axis=0)
     
    X = np.copy(aX)
    y = np.copy(ay)

    for i in range(X.shape[0]):
        x = X[i]
        if y[i][1] > 0:
            x = random_transform(x.astype("float32"), rotation_range, width_shift_range, height_shift_range, horizontal_flip, vertical_flip)
        x = standardize(x, featurewise_center, featurewise_std_normalization, samplewise_center, samplewise_std_normalization, zca_whitening, 0, 0, principal_components) 
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

import time
def standardize(x, featurewise_center, featurewise_std_normalization, samplewise_center, samplewise_std_normalization, zca_whitening, mean, std, principal_components):
        if featurewise_center:
            x -= mean
        if featurewise_std_normalization:
            x /= std

        if zca_whitening:
            #name = time.clock()
            #print 'zca {} ...'.format(name)
            #util.imwrite('{}_roi.jpg'.format(name), x[0])
            flatx = np.reshape(x, (x.shape[0]*x.shape[1]*x.shape[2]))
            whitex = np.dot(flatx, principal_components)
            x = np.reshape(whitex, (x.shape[0], x.shape[1], x.shape[2]))
            #util.imwrite('{}_zca.jpg'.format(name), x[0])

        if samplewise_center:
            x -= np.mean(x)
        if samplewise_std_normalization:
            x /= np.std(x)

        return x



if __name__ == '__main__':
    fname = '501.49_roi.jpg' 
    import cv2
    img = cv2.imread(fname)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print 'shape {}'.format(img.shape)
    img = np.array([img])

    augment_params = {'ratio':1, 'batch_size':32, 'rotation_range':(-15,15), 'translation_range':(-0.1, 0.1), 'flip':True, 'mode':'balance_batch'}
    gen = ImageDataGenerator(**augment_params)

    for i in range(10):
        rnd_img = gen.perturb(img)
        util.imwrite('RND_{}_idx{}'.format(fname, i), rnd_img[0])
