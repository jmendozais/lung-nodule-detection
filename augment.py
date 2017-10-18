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
import h5py
import util
import time
import math

def gaussian_noise(x, mean=0, std=0.1):	
    noise = np.random.normal(loc=mean, scale=std, size=x.shape)
    scale_factor = np.max(x)
    return noise * scale_factor

def gaussian_smooth(x, ksize=5, sigma=0.5):	
    ksize = (ksize, ksize)
    sigma *= np.random.uniform()
    smt = cv2.GaussianBlur(x, ksize, sigma)
    return smt

class NumpyPreprocessor:
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

''' Numpy & HDF5 compatible functions ''' 

def _mean(X, low, high):
    chunksize = None
    if isinstance(X, np.ndarray):
        chunksize = X.shape[0] 
    else:
        chunksize = X.chunks[0]

    i = 0
    sum_ = .0
    while True:
        start = low + i * chunksize
        end = min((i + 1) * chunksize, high)
        X_chunk = X[start:end]
        sum_ += X_chunk.sum()
        i += 1
        if end == high:
            break

    return sum_/(high - low)

def _sum_squared_differences(X, low, high, mean):
    chunksize = None
    if isinstance(X, np.ndarray):
        chunksize = X.shape[0] 
    else:
        chunksize = X.chunks[0]

    i = 0
    sum_ = .0
    while True:
        start = low + i * chunksize
        end = min((i + 1) * chunksize, high)
        X_chunk = X[start:end]
        sum_ += np.sum(np.power(X_chunk - mean, 2))
        i += 1
        if end == high:
            break
    return sum_

def _mean_std_balancing_pos_neg(X, Y):
    split_pos = int(np.sum(Y.T[1]))
    mean_pos = _mean(X, 0, split_pos)
    mean_neg = _mean(X, split_pos, X.shape[0])
    mean = (mean_pos + mean_neg)/2.0

    ssd_pos = _sum_squared_differences(X, 0, split_pos, mean)
    ssd_neg = _sum_squared_differences(X, split_pos, X.shape[0], mean)

    len_neg = X.shape[0] - split_pos
    pos_weight = len_neg * 1.0 / split_pos
    
    item_size = 1.0
    for dim in X.shape:
        item_size *= dim
    item_size /= X.shape[0]

    var_pos = ssd_pos * pos_weight / (len_neg * item_size - 1)
    var_neg = ssd_neg / (len_neg * item_size - 1)
    std = math.sqrt((var_pos + var_neg)/2.0)
    std_pos = math.sqrt(ssd_pos / (split_pos * item_size - 1))
    std_neg = math.sqrt(ssd_neg / (len_neg * item_size - 1))

    return mean, std

class Preprocessor:
    def __init__(self, zmuv=True, **kwargs):
        self.zmuv = zmuv

    def fit(self, X, Y):
        if self.zmuv == True:
            self.zmuv_mean, self.zmuv_std = _mean_std_balancing_pos_neg(X, Y)
    
    def transform(self, X):
        if isinstance(X, np.ndarray):
            return (X - self.zmuv_mean)/self.zmuv_std
        else:
            chunksize = X.chunks[0]
            if self.zmuv == True:
                i = 0
                while True:
                    start = i * chunksize
                    end = min((i + 1) * chunksize, X.shape[0])
                    X[start:end] = (X[start:end] - self.zmuv_mean)/self.zmuv_std
                    i += 1
                    if end == X.shape[0]:
                        break
                return X

    def fit_transform(self, X, Y):
        self.fit(X, Y)
        return self.transform(X)

def get_preprocessor(config='zmuv'):
    params = None
    if config == 'zmuv':
        params = {'zmuv': True}
    return Preprocessor(**params)

def preprocess_dataset(preprocessor, X_train, Y_train, X_test=None, Y_test=None, streams=False):
    gc.collect()
    if streams:
        num_streams = len(X_train)
        num_channels = len(X_train[0][0])

        print 'Preprocess train set ...'
        print preprocessor.__dict__
        X_train[0] = preprocessor.fit_transform(X_train[0], Y_train)
        for i in range(1, num_streams):
            X_train[i] = preprocessor.transform(X_train[i])
        gc.collect()

        print "Preprocess test set ..."
        if X_test != None:
            X_test[0] = preprocessor.transform(X_test[0])
            for k in range(1, num_streams):
                X_test[k] = preprocessor.transform(X_test[k])
            gc.collect()
    else:
        num_channels = len(X_train[0])

        print "Preprocess train set ..."
        print preprocessor.__dict__
        print type(X_train)
        X_train = preprocessor.fit_transform(X_train, Y_train)
        gc.collect()

        print "Preprocess test set ..."
        if X_test is not None:
            X_test = preprocessor.transform(X_test)
            gc.collect()

    return X_train, Y_train, X_test, Y_test

class ImageDataGenerator:

    def __init__(self, 
        rotation_range=(0.,0.),
        translation_range=(0.,0.), 
        flip=False,
        zoom_range=(1.,1.),
        intensity_shift_std=0.1,
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
        self.intensity_shift_range = None
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
        translation = (round(shift_x), round(shift_y))

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

        '''
        print "tr {}".format(translation)
        print "zoom x, y {}".format((zoom_x, zoom_y))
        print "shear {}".format(shear)
        print "rot {}".format(rotation)
        print "flip {}".format(flip)
        '''

        return self.build_augmentation_transform((zoom_x, zoom_y), rotation, shear, translation, flip)

    def fast_warp(self, img, tf, output_shape=(50, 50), mode='constant', order=3):
        m = tf.params # tf._matrix is
        img = transform._warps_cy._warp_fast(img, m, output_shape=output_shape, mode=mode, order=order)
        return img

    def perturb(self, x):
        assert self.translation_range[1] >= 0

        shape = x[0].shape
        if self.translation_range[1]< 1.:
            side = max(self.output_shape)
            self.translation_range = (self.translation_range[0] * side, self.translation_range[1] * side)
            self.translation_range = (math.ceil(self.translation_range[0]), math.ceil(self.translation_range[1]))

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

        return new_x

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
        if isinstance(X, h5py.Dataset):
            return self._augment_hdf5(X, y, cropped_shape, disable_perturb)
        else:
            return self._augment_numpy(X, y, cropped_shape, disable_perturb)

    def _shuffle(source, target, indexes):
        data_size = source.shape[0]
        chunk_size = source.chunks[0]
        nb_chunks = math.ceil(data_size/chunk_size)
        source_bool_map = np.full(shape=(data_size,), dtype=bool, fill_value=False)
        target_bool_map = np.full(shape=(data_size,), dtype=bool, fill_value=False)
        for i in range(nb_chunks):
            begin = i*chunk_size
            end = min(begin + chunk_size, data_size)
            index = indexes[begin:end]
            for j in range(nb_chunks):
                begin = j*chunk_size
                end = min(begin + chunk_size, data_size)
                mask = np.logical_and(index >= begin, index < end)
                target_bool_map[:] = False
                target_bool_map[begin:end] = mask
                source_bool_map[:] = False
                source_bool_map[index[mask]] = True
                target[target_bool_map] = source[source_bool_map]

    def _augment_hdf5(self, X, y, cropped_shape, disable_perturb=False):
        assert self.batch_size % 2 == 0, 'Batch size should be even (batch_size = {}).'.format(self.batch_size)
        
        factor = int((float(self.ratio * (len(y) - np.sum(y.T[1]))) / np.sum(y.T[1])))
        # balance
        print 'Mode: {} ...'.format(self.mode)

        if self.mode == 'balance_batch':
            raise Exception("Mode {} not implemented".format(self.mode))

        elif self.mode == 'balance_dataset':
            current_size = X.shape[0]
            delta_size = np.sum(y.T[1]) * factor
            X.resize((current_size + delta_size,) + X.shape[1:])
            idx = current_size
            for i in range(y.shape[0]):
                if y[i][1] > 0:
                    aXi = np.full((factor,) + X[i].shape, dtype=X[i].dtype, fill_value=X[i])
                    X[idx:idx+factor] = aXi
                    ayi = np.full((factor,) + y[i].shape, dtype=y[i].dtype, fill_value=y[i])
                    y = np.append(y, ayi, axis=0)
                    idx += factor
            assert idx == X.shape[0]
            indexes = list(range(X.shape[0]))
            seed = self.rng.randint(1, 10e6) 
            np.random.seed(seed) 
            np.random.shuffle(indexes) 
            y_shuffled = y[indexes]

            f = h5py.File("array{}.h5".format(time.time()), "w")
            begin = time.time()
            X_shuffled = f.create_dataset("X_shuffled", X.shape, chunks=X.chunks, dtype='float32')
            X_shuffled[:] = X[indexes]
            print("Shuffle overhead {} secs".format(time.time() - begin))

        # Perturn
        if disable_perturb:
            return X_shuffled, y_shuffled
        else:
            for i in range(aX.shape[0]):
                X_shuffled[i] = self.perturb(X_shuffled[i])
            return X_shuffled, y_shuffled

    def _augment_numpy(self, X, y, cropped_shape, disable_perturb=False):
        assert self.batch_size % 2 == 0, 'Batch size should be even (batch_size = {}).'.format(self.batch_size)
        
        factor = int((float(self.ratio * (len(y) - np.sum(y.T[1]))) / np.sum(y.T[1])))
        # balance
        print 'Mode: {} ...'.format(self.mode)
        if self.mode == 'balance_batch':
            idx = y.T[1]
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
                #print("{} img mean {} min,max {}".format(i, np.mean(aX[i]), (np.min(aX[i]), np.max(aX[i]))))
                x = self.perturb(aX[i])
                #print("pert mean {} min,max {}".format(np.mean(x), (np.min(x), np.max(x))))
                '''
                if i < 96:
                    for k in range(len(aX[i])):
                        util.imwrite('img_{}_{}_aug.jpg'.format(i, k), x[k])
                '''
                new_X.append(x)
            aX = None
            gc.collect()
            return np.array(new_X), ay    

    def fit(self, X):
        assert isinstance(X, np.ndarray), "Perturbation object fit function only supports numpy arrays"
        # TODO: std should be only on the crop area (tr prove to be perjudicial)

        std = np.std(X)
        self.intensity_shift_range = (-self.intensity_shift_std*std, self.intensity_shift_std*std)

class DataGenerator:
    def __init__(self, input, output, batch_size, perturb_func, data_shape):
        self.input = input
        self.output = output
        self.batch_size = batch_size
        self.perturb_func = perturb_func

        self.step = int(batch_size/2)
        self.len_pos = len(input[input.keys()[0]][0])
        self.len_neg = len(input[input.keys()[0]][1])
        print("Data generator len pos {} len neg {} data_shape {}".format(self.len_pos, self.len_neg, data_shape))
        
        '''
        if isinstance(input[input.keys()[0]], np.ndarray):
            for k in self.input:
                input[k] = np.hstack(input[k], input[k])
        '''

        self.data_shape = data_shape
        self.offset = 0

    def next(self):
        if self.offset + self.step > self.len_neg: 
            self.offset = 0

        batch_input = {}
        pos_idx = np.random.randint(0, self.len_pos, self.step)
        for k in self.input:
            input_ = np.zeros(shape = (2*self.step,) + self.data_shape, dtype=self.input[k][0][0].dtype)
            pos = self.input[k][0][pos_idx]
            neg = self.input[k][1][self.offset:(self.offset + self.step)]
            for i in range(self.step):
                input_[i] = self.perturb_func(pos[i])
                '''
                # print perturbs
                input_[i] = self.perturb_func(pos[0])
                util.imwrite('pos{}-ptb.jpg'.format(i), input_[i][0])
                tmp = pos[0].copy()
                diff = tmp[0].shape[0] - input_[i][0].shape[0]
                isize = tmp[0].shape[0]
                osize = input_[i][0].shape[0]
                tmp = tmp[0][diff/2:diff/2 + osize,diff/2:diff/2 + osize]
                assert tmp.shape[0] == osize, '{}'.format((tmp.shape, osize))
                util.imwrite('pos{}.jpg'.format(i), tmp)
                '''
            for i in range(self.step):
                input_[self.step + i] = self.perturb_func(neg[i])
            batch_input[k] = input_

        batch_output = {}
        for k in self.output:
            output_shape = self.output[k][0][0].shape
            output_ = np.zeros(shape = (2*self.step,) + output_shape, dtype=self.output[k][0][0].dtype)
            output_[:self.step] = self.output[k][0][pos_idx]
            output_[self.step:] = self.output[k][1][self.offset:(self.offset + self.step)]
            batch_output[k] = output_

        self.offset += self.step

        # Show blobs and histograms here ! 
        '''
        mean = [[], []]
        std = [[], []]
        for k in self.input:
            for i in range(self.step):
                pos = batch_input[k][i][0]
                neg = batch_input[k][self.step + i][0]
                mean[0].append(np.mean(pos))
                std[0].append(np.std(pos))
                mean[1].append(np.mean(neg))
                std[1].append(np.std(neg))
                util.imshow('input p', batch_input[k][i][0], display_shape=(256, 256))
                util.imshow('perturbed p', batch_input[k][i][0], display_shape=(256, 256))
                util.imshow('input n', batch_input[k][self.step + i][0], display_shape=(256, 256))
                util.imshow('perturbed n', batch_input[k][self.step + i][0], display_shape=(256, 256))

        #print 'batch p:  mm {:.4f}, ms {:.4f} ss {:.4f}, n: mm {:.4f}, ms {:.4f}, ss {:.4f}'.format(np.mean(mean[0]), np.mean(std[0]), np.std(std[0]), np.mean(mean[1]), np.mean(std[1]), np.std(std[1]))
        '''

        return batch_input, batch_output 

    def __iter__(self):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)

class DataGeneratorOnMemory:
    def __init__(self, input, output, batch_size, perturb_func, data_shape):
        self.input = input
        self.output = output
        self.batch_size = batch_size
        self.perturb_func = perturb_func

        self.step = int(batch_size)
        self.length= len(input[input.keys()[0]][0])
        
        self.data_shape = data_shape
        self.offset = 0

    def next(self):
        if self.offset + self.step > self.length: 
            self.offset = 0

        batch_input = {}
        for k in self.input:
            input_ = np.zeros(shape = (self.step,) + self.data_shape, dtype=self.input[k][0][0].dtype)
            data_ = self.input[k][0][self.offset:(self.offset + self.step)]
            for i in range(self.step):
                #input_[i] = self.perturb_func(data_[i])
                input_[i] = data_[i]
                #print ("{} input mean {} max min {}, perturbed mean {} max min {}".format(i, np.mean(data_[i]), (np.max(data_[i]), np.min(data_[i])), np.mean(input_[i]), (np.max(input_[i]), np.min(input_[i]))))
            batch_input[k] = input_

        batch_output = {}
        for k in self.output:
            batch_output[k] = self.output[k][0][self.offset:(self.offset + self.step)]

        self.offset += self.step
            
        return batch_input, batch_output 

    def __iter__(self):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)

# default augmentation

factor = 1.4
shape = (64, 64)
default_augment_params = {'output_shape': shape, 'ratio':1, 'batch_size':32, 'rotation_range':(-18, 18), 'translation_range':(-0.12, 0.12), 'flip':True, 'intensity_shift_std':0.2, 'mode':'balance_batch', 'zoom_range':(1.0, 1.25)}

def get_default_generator(shape):
    default_augment_params['output_shape'] = shape
    gen = ImageDataGenerator(**default_augment_params)
    return gen
 
def balance_and_perturb(X, Y, gen):
    gen.fit(X)
    (X_pos, X_neg), (Y_pos, Y_neg) = util.split_data_pos_neg(X, Y)
    X_pos_aug, Y_pos_aug = [], []
    X_neg_pert = []

    idx = np.random.randint(0, len(X_pos), len(X_neg))

    for i in range(len(idx)):
        X_pos_aug.append(gen.perturb(X_pos[idx[i]]))
        Y_pos_aug.append(Y_pos[idx[i]])

    for i in range(len(X_neg)):
        X_neg_pert.append(gen.perturb(X_neg[i]))
    
    X_pos_aug = np.array(X_pos_aug)
    X_aug = np.concatenate((X_pos_aug, X_neg_pert))
    Y_aug = np.concatenate((Y_pos_aug, Y_neg))

    idx = np.array(range(len(Y_aug)))
    np.random.shuffle(idx)

    return X_aug[idx], Y_aug[idx]

# Test
if __name__ == '__main__':
    fname = 'grid2.jpg'

    import cv2
    img = cv2.imread(fname)
    print img.shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.array([img])
    util.imwrite('or_{}'.format(fname), img[0])
    augment_params = {'output_shape':(img.shape[1]*(1.0/factor), img.shape[1]*(1.0/factor)), 'ratio':1, 'batch_size':32, 'rotation_range':(-18, 18), 'translation_range':(-0.12, 0.12), 'flip':True, 'intensity_shift_std':0.2, 'mode':'balance_batch', 'zoom_range':(1.0, 1.25)}
    gen = ImageDataGenerator(**augment_params)
    gen.intensity_shift_range = (-0.1, 0.1)

    for i in range(20):
        print 'neg_{}_{}'.format(i, fname)
        #rnd_img = random_transform(img.astype("float32"), 15, 0.1, 0.1, True, False)
        rnd_img = gen.perturb(img)
        util.imwrite('neg_{}_{}'.format(i, fname), rnd_img[0])

    ans = gen.centering_crop([img])
    util.imwrite('pos_{}'.format(fname), ans[0][0])
