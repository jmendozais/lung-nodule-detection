from __future__ import absolute_import 

from os import listdir
from os.path import isfile, join

import random, math
import re
import gc
from six.moves import range

import numpy as np
from scipy import ndimage
from scipy import linalg
from skimage import transform

import cv2
import util

def gaussian_noise(x, mean=0, std=0.1):	
    noise = np.random.normal(loc=mean, scale=std, size=x.shape)
    scale_factor = np.max(x)
    return noise * scale_factor

def gaussian_smooth(x, ksize=5, sigma=0.5):	
    ksize = (ksize, ksize)
    sigma *= np.random.uniform()
    smt = cv2.GaussianBlur(x, ksize, sigma)
    return smt

class Preprocessor:
    def __init__(
        self,
        samplewise_center=False, # set each sample mean to 0
        samplewise_std_normalization=False, # divide each input by its std
        featurewise_rescaling=False,
        featurewise_rescaling_range=[-1, 1],
        featurewise_center=False, # set input mean to 0 over the dataset
        featurewise_std_normalization=False, # divide inputs by std of the dataset
        zca_whitening=False,
        zca_fudge=10e-7,
        data_rescaling=False,
        data_rescaling_range=[0, 1],
        zmuv=False
        ):

        self.samplewise_center = samplewise_center
        self.samplewise_std_normalization = samplewise_std_normalization
        self.featurewise_rescaling = featurewise_rescaling
        self.featurewise_rescaling_range = featurewise_rescaling_range
        self.featurewise_center = featurewise_center
        self.featurewise_std_normalization = featurewise_std_normalization
        self.zca_whitening = zca_whitening
        self.zca_fudge = zca_fudge
        self.data_rescaling = data_rescaling
        self.data_rescaling_range = data_rescaling_range
        self.zmuv=zmuv

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
            X = (X - self.amin) / (self.amax - self.amin + util.EPS)
            X = X * (self.featurewise_rescaling_range[1] + self.featurewise_rescaling_range[0]) + self.featurewise_rescaling_range[0]

        if self.featurewise_center or self.featurewise_std_normalization:
            self.mean = np.mean(X, axis=0)
            X -= self.mean

        if self.featurewise_std_normalization:
            self.std = np.std(X, axis=0)
            X /= self.std + util.EPS

        if self.zca_whitening:
            flat_X = np.reshape(X, (X.shape[0], X.shape[1] * X.shape[2] * X.shape[3]))
            if not (self.featurewise_center or self.featurewise_std_normalization):
                self.mean = np.mean(X, axis=0)
                flat_mean = np.reshape(self.mean, X.shape[1] * X.shape[2] * X.shape[3])
                flat_X -= flat_mean

            sigma = np.dot(flat_X.T, flat_X) / flat_X.shape[1]
            U, S, V = linalg.svd(sigma)
            self.principal_components = np.dot(np.dot(U, np.diag(1. / np.sqrt(S + self.zca_fudge))), U.T)
            whitex = np.dot(flat_X, self.principal_components)
            X = np.reshape(whitex, (X.shape[0], X.shape[1], X.shape[2], X.shape[3]))

        if self.data_rescaling:
            self.dmin = mp.amin(X)
            self.dmax = mp.amax(X)
            X = (X - self.dmin) / (self.dmax - self.dmin)
            X = X * (self.data_rescaling_range[1] + self.data_rescaling_range[0]) + self.data_rescaling_range[0]

        if self.zmuv:
            self.zmuv_mean = X.mean()
            self.zmuv_std = X.std()
            X -= self.zmuv_mean
            X /= self.zmuv_std + util.EPS
            
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
            X = (X - self.amin) / (self.amax - self.amin + util.EPS)

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

        if self.data_rescaling:
            X = (X - self.dmin) / (self.dmax - self.dmin)
            X = X * (self.data_rescaling_range[1] + self.data_rescaling_range[0]) + self.data_rescaling_range[0]

        if self.zmuv:
            X -= self.zmuv_mean
            X /= self.zmuv_std + util.EPS
            
        return X

class ImageDataGenerator:

    def __init__(self, 
        rotation_range=(0.,0.),
        translation_range=(0.,0.), 
        flip=False,
        zoom_range=(1.,1.),
        intensity_shift_std=True,
        output_shape=(64, 64),

        batch_size=32,
        ratio=1.,
        gn_mean=.0,
        gn_std=.0,
        gs_size=0,
        gs_sigma=.0,
        mode='balance_batch'):

        self.rotation_range = rotation_range
        self.translation_range = translation_range
        self.flip = flip
        self.zoom_range = zoom_range
        self.intensity_shift_std = intensity_shift_std
        self.output_shape = output_shape

        self.batch_size = batch_size
        self.ratio = ratio
        self.gn_mean = gn_mean
        self.gn_std = gn_std
        self.gs_size = gs_size
        self.gs_sigma = gs_sigma
        self.mode = mode

        self.rng = np.random.RandomState(113)

    def build_centering_transform(self, image_shape, target_shape=(50, 50)):
        rows, cols = image_shape
        trows, tcols = target_shape
        shift_x = (cols - tcols) / 2.0
        shift_y = (rows - trows) / 2.0
        return transform.SimilarityTransform(translation=(shift_x, shift_y))

    def build_center_uncenter_transforms(self, image_shape):
        center_shift = np.array([image_shape[1], image_shape[0]]) / 2.0 - 0.5 # need to swap rows and cols here apparently! confusing!
        tform_uncenter = transform.SimilarityTransform(translation=-center_shift)
        tform_center = transform.SimilarityTransform(translation=center_shift)
        return tform_center, tform_uncenter
    
    def build_augmentation_transform(self, zoom=(1.0, 1.0), rotation=0, shear=0, translation=(0, 0), flip=False): 
        if flip:
            shear += 180
            rotation += 180

        tform_augment = transform.AffineTransform(scale=(1/zoom[0], 1/zoom[1]), rotation=np.deg2rad(rotation), shear=np.deg2rad(shear), translation=translation)
        return tform_augment 

    def random_perturbation_transform(self, zoom_range, rotation_range, shear_range, translation_range, do_flip=True, allow_stretch=False):
        shift_x = self.rng.uniform(*translation_range)
        shift_y = self.rng.uniform(*translation_range)
        translation = (shift_x, shift_y)

        rotation = self.rng.uniform(*rotation_range)
        shear = self.rng.uniform(*shear_range)

        if do_flip:
            flip = (self.rng.randint(2) > 0) # flip half of the time
        else:
            flip = False

        # random zoom
        log_zoom_range = [np.log(z) for z in zoom_range]
        if isinstance(allow_stretch, float):
            log_stretch_range = [-np.log(allow_stretch), np.log(allow_stretch)]
            zoom = np.exp(self.rng.uniform(*log_zoom_range))
            stretch = np.exp(self.rng.uniform(*log_stretch_range))
            zoom_x = zoom * stretch
            zoom_y = zoom / stretch
        elif allow_stretch is True: # avoid bugs, f.e. when it is an integer
            zoom_x = np.exp(self.rng.uniform(*log_zoom_range))
            zoom_y = np.exp(self.rng.uniform(*log_zoom_range))
        else:
            zoom_x = zoom_y = np.exp(self.rng.uniform(*log_zoom_range))

        # the range should be multiplicatively symmetric, so [1/1.1, 1.1] instead of [0.9, 1.1] makes more sense.
        return self.build_augmentation_transform((zoom_x, zoom_y), rotation, shear, translation, flip)

    def fast_warp(self, img, tf, output_shape=(50, 50), mode='constant', order=3):
        m = tf.params # tf._matrix is
        img = transform._warps_cy._warp_fast(img, m, output_shape=output_shape, mode=mode, order=order)
        return img

    def perturb(self, x):
        shape = x[0].shape

        if abs(self.translation_range[0]) < 1.:
            side = max(shape)
            self.translation_range = (self.translation_range[0] * side, self.translation_range[1] * side)

        tform_centering = self.build_centering_transform(shape, self.output_shape)
        tform_center, tform_ucenter = self.build_center_uncenter_transforms(shape)
        tform_augment = self.random_perturbation_transform(
            zoom_range=self.zoom_range, rotation_range=self.rotation_range,
            shear_range=(0., 0.), translation_range=self.translation_range, 
            do_flip=self.flip)
        tform_augment = tform_ucenter + tform_augment + tform_center

        intensity_shift = np.random.uniform(*self.intensity_shift_range)

        new_x = np.full(shape=(x.shape[0], self.output_shape[0], self.output_shape[1]), dtype=np.float32, fill_value=0)
        for i in range(x.shape[0]):
            new_x[i] = self.fast_warp(x[i], tform_centering + tform_augment, output_shape=self.output_shape, mode='constant').astype('float32')
            new_x[i] += intensity_shift
            if self.gs_sigma != .0 and self.gs_size != 0:
                new_x[i] = gaussian_smooth(new_x[i], self.gs_size, self.gs_sigma)
            if self.gn_std != .0:
                new_x[i] += gaussian_noise(new_x[i], self.gn_mean, self.gn_std)
        '''
        tmp = np.random.randint(10000)
        for i in range(len(new_x)):
            util.imwrite('aug_trfs_{}_{}.jpg'.format(tmp, i), new_x[i])
            print 'aug {} {} : {} {}'.format(tmp, i, np.min(new_x[i]), np.max(new_x[i]))
        '''
         
        return np.array(new_x)

    def centering_crop(self, X):
        assert len(X) > 0

        new_X = []
        tform_centering = self.build_centering_transform(X[0][0].shape, self.output_shape)
        for i in range(len(X)):
            new_x = []
            for k in range(len(X[i])):
                new_x.append(self.fast_warp(X[i][k], tform_centering, output_shape=self.output_shape, mode='constant').astype('float32'))
            new_X.append(np.array(new_x))

        return np.array(new_X)

    def augment(self, X, y, cropped_shape, disable_perturb=False):
        assert self.batch_size % 2 == 0, 'Batch size should be even (batch_size = {}).'.format(self.batch_size)
        if self.intensity_shift_std:
            std = np.std(X)
            self.intensity_shift_range = (-std, std)
        
        factor = int((float(self.ratio * (len(y) - np.sum(y.T[1]))) / np.sum(y.T[1])))
        # balance
        print 'Mode: {} ...'.format(self.mode)
        if self.mode == 'balance_batch':
            '''
            idx = y.T[1].astype(np.int)
            '''
            idx = y.T[1]
            '''
            positives = X[idx > 0]
            negatives = X[idx == 0]
            l_pos = y[idx > 0][0]
            l_neg = y[idx == 0][0]
            '''
            thold = 0.8
            positives = X[idx >= thold]
            negatives = X[idx < thold]
            l_pos = y[idx >= thold][0]
            l_neg = y[idx < thold][0]
            print 'len pos {}, len neg {}'.format(len(positives), len(negatives))

            aX = np.zeros((2 * negatives.shape[0],) + X[0].shape, dtype=X[0].dtype)
            ay = np.zeros((2 * negatives.shape[0],) + y[0].shape, dtype=y[0].dtype)
            n_batches = int(math.ceil(float(2 * negatives.shape[0]) / self.batch_size))
            for batch_idx in range(n_batches):
                begin = batch_idx * (self.batch_size / 2)
                end = min(begin + (self.batch_size / 2), negatives.shape[0]) 
                chunk = end - begin
                aX[2*begin:2*begin + chunk] = negatives[begin:end]
                ay[2*begin:2*begin + chunk] = np.full((chunk,) + l_neg.shape, dtype=l_neg.dtype, fill_value=l_neg)
                b_idx = self.rng.uniform(size=(chunk,))
                b_idx = np.floor(b_idx * positives.shape[0]).astype(np.int)
                aX[2*begin + chunk:2*(begin + chunk)] = positives[b_idx]
                ay[2*begin + chunk:2*(begin + chunk)] = np.full((end - begin,) + l_pos.shape, dtype=l_pos.dtype, fill_value=l_pos)

        elif self.mode == 'balance_dataset':
            aX = X.copy()
            ay = y.copy()
            for i in range(X.shape[0]):
                if y[i][1] > 0:
                    aXi = np.full((factor,) + X[i].shape, dtype=X[i].dtype, fill_value=X[i])
                    aX = np.append(aX, aXi, axis=0)
                    ayi = np.full((factor,) + y[i].shape, dtype=y[i].dtype, fill_value=y[i])
                    ay = np.append(ay, ayi, axis=0)
            seed = self.rng.randint(1, 10e6) 
            np.random.seed(seed) 
            np.random.shuffle(aX) 
            np.random.seed(seed)
            np.random.shuffle(ay)

        if disable_perturb:
            return aX, ay
        else:
            # transform
            new_X = []
            for i in range(aX.shape[0]):
                '''
                if i < 96:
                    for k in range(len(aX[i])):
                        util.imwrite('img_{}_{}.jpg'.format(i, k), aX[i][k])
                '''
                x = self.perturb(aX[i])
                '''
                if i < 96:
                    for k in range(len(aX[i])):
                        util.imwrite('img_{}_{}_aug.jpg'.format(i, k), x[k])
                '''
                new_X.append(x)
            aX = None
            gc.collect()
            return np.array(new_X), ay    
    '''
    def augment2(self, X, y, output_shape):
        assert self.batch_size % 2 == 0, 'Batch size should be even (batch_size = {}).'.format(self.batch_size)

        factor = int((float(self.ratio * (len(y) - np.sum(y.T[1]))) / np.sum(y.T[1])))
        # balance
        print 'Mode: {} ...'.format(self.mode)
        if self.mode == 'balance_batch':
            idx = y.T[1]
            positives = X[idx > 0]
            negatives = X[idx == 0]
            l_pos = y[idx > 0][0]
            l_neg = y[idx == 0][0]
            aX = np.zeros((2 * negatives.shape[0],) + output_shape, dtype=X[0].dtype)
            ay = np.zeros((2 * negatives.shape[0],) + y[0].shape, dtype=y[0].dtype)
            n_batches = int(math.ceil(float(2 * negatives.shape[0]) / self.batch_size))

            for batch_idx in range(n_batches):
                begin = batch_idx * (self.batch_size / 2)
                end = min(begin + (self.batch_size / 2), negatives.shape[0]) 
                chunk = end - begin
                b_idx = self.rng.uniform(size=(chunk,))
                b_idx = np.floor(b_idx * positives.shape[0]).astype(np.int)
                for i in range(chunk):
                    aX[2*begin + i] = self.perturb(negatives[begin + i])
                    aX[2*begin + chunk + i] = self.perturb(positives[b_idx[i]])

                ay[2*begin:2*begin + chunk] = np.full((chunk,) + l_neg.shape, dtype=l_neg.dtype, fill_value=l_neg)
                ay[2*begin + chunk:2*(begin + chunk)] = np.full((end - begin,) + l_pos.shape, dtype=l_pos.dtype, fill_value=l_pos)

        elif self.mode == 'balance_dataset':
            aX = X.copy()
            ay = y.copy()
            for i in range(X.shape[0]):
                if y[i][1] > 0:
                    aXi = np.full((factor,) + X[i].shape, dtype=X[i].dtype, fill_value=X[i])
                    aX = np.append(aX, aXi, axis=0)
                    ayi = np.full((factor,) + y[i].shape, dtype=y[i].dtype, fill_value=y[i])
                    ay = np.append(ay, ayi, axis=0)
            seed = self.rng.randint(1, 10e6) 
            np.random.seed(seed) 
            np.random.shuffle(aX) 
            np.random.seed(seed)
            np.random.shuffle(ay)
            # transform
            new_X = []
            for i in range(aX.shape[0]):
                x = self.perturb(aX[i])
                new_X.append(x)
                aX = np.array(new_X)

        return aX, ay
    def online_augment(self, X, y):
        assert self.batch_size % 2 == 0, 'Batch size should be even (batch_size = {}).'.format(self.batch_size)
        factor = int((float(self.ratio * (len(y) - np.sum(y.T[1]))) / np.sum(y.T[1])))
        # balance
        print 'Mode: {} ...'.format(self.mode)
        if self.mode == 'balance_batch':
            idx = y.T[1]
            positives = X[idx > 0]
            negatives = X[idx == 0]
            l_pos = y[idx > 0][0]
            l_neg = y[idx == 0][0]
            n_batches = int(math.ceil(float(2 * negatives.shape[0]) / self.batch_size))
            for batch_idx in range(n_batches):
                begin = batch_idx * (self.batch_size / 2)
                end = min(begin + (self.batch_size / 2), negatives.shape[0]) 
                chunk = end - begin

                aX = np.zeros((2 * chunk,) + X[0].shape, dtype=X[0].dtype)
                aY = np.zeros((2 * chunk,) + y[0].shape, dtype=y[0].dtype)

                aX[0:chunk] = negatives[begin:end]
                ay[0:chunk] = np.full((chunk,) + l_neg.shape, dtype=l_neg.dtype, fill_value=l_neg)
            
                b_idx = self.rng.uniform(size=(chunk,))
                b_idx = np.floor(b_idx * positives.shape[0]).astype(np.int)

                aX[chunk:2*chunk] = positives[b_idx]
                ay[chunk:2*chunk] = np.full((end - begin,) + l_pos.shape, dtype=l_pos.dtype, fill_value=l_pos)
                yield aX, ay
        elif self.mode == 'balance_dataset':
            print "ERR: Not supported"
        '''

# Test
if __name__ == '__main__':
    fname = 'grid2.jpg'
    #fname = '501.49_roi.jpg' 

    import cv2
    img = cv2.imread(fname)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.array([img])

    augment_params = {'output_shape':(50, 50), 'ratio':1, 'batch_size':32, 'rotation_range':(-5, 5), 'translation_range':(-0.05, 0.05), 'flip':True, 'mode':'balance_batch', 'zoom_range':(1.0, 1.2)}
    gen = ImageDataGenerator(**augment_params)

    for i in range(10):
        print 'rnd_{}_{}'.format(i, fname)
        rnd_img = random_transform(img.astype("float32"), 15, 0.1, 0.1, True, False)
        rnd_img = gen.perturb(img)
        util.imwrite('neg_{}_{}'.format(i, fname), rnd_img[0])

    ans = gen.centering_crop([img])
    util.imwrite('pos_{}'.format(fname), ans[0][0])
    
