import cv2 
import sys

import skimage.io as io
from skimage.exposure import equalize_hist
from skimage.restoration import denoise_nl_means

from sklearn import lda
from sklearn import svm
from sklearn import decomposition
from sklearn import preprocessing
from sklearn import feature_selection as selection
from sklearn.externals import joblib
from sklearn.feature_selection import RFE

from time import *
from os import path

#from __future__ import print_function as print_f

from keras.models import Sequential
from keras.models import model_from_json
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils, layer_utils
from six.moves import range

import classify
from detect import *
from extract import * 
from preprocess import *
from segment import *
from augment import *
from util import *
import neural

import jsrt

#TODO: up classifier

def input(img_path, ll_path, lr_path):
    img = np.load(img_path)
    img = img.astype(np.float)
    ll_mask = cv2.imread(ll_path)
    lr_mask = cv2.imread(lr_path)
    lung_mask = ll_mask + lr_mask
    dsize = (512, 512)
    lung_mask = cv2.resize(lung_mask, dsize, interpolation=cv2.INTER_CUBIC)
    lung_mask = cv2.cvtColor(lung_mask, cv2.COLOR_BGR2GRAY)
    lung_mask.astype(np.uint8)

    return img, lung_mask

def adjacency_rule(blobs, probs):
    # candidate cue adjacency rule: 22 mm
    filtered_blobs = []
    filtered_probs = []
    for j in range(len(blobs)):
        valid = True
        for k in range(len(blobs)):
            dist2 = (blobs[j][0] - blobs[k][0]) ** 2 + (blobs[j][1] - blobs[k][1]) ** 2
            if dist2 < 988 and probs[j] + EPS < probs[k]:
                valid = False
                break

        if valid:
            filtered_blobs.append(blobs[j])
            filtered_probs.append(probs[j])

    return filtered_blobs, filtered_probs

import keras
class StageScheduler(keras.callbacks.Callback):
    def __init__(self, stages=[], decay=0.1):
        sorted(stages)
        self.stages = stages
        self.idx = 0
        self.decay = decay
    
    def on_epoch_end(self, epoch, logs={}):
        if self.idx < len(self.stages):
            if epoch + 1 == self.stages[self.idx]:
                lr = self.model.optimizer.lr.get_value()
                self.model.optimizer.lr.set_value(float(lr * self.decay))
                self.idx += 1
        print 'lr {}'.format(self.model.optimizer.lr.get_value())
                
# Baseline Model
class BaselineModel:
    def __init__(self, name='default'):
        self.name = name

        self.clf = lda.LDA()
        self.scaler = preprocessing.StandardScaler()
        self.selector = None
        self.transform = None
        self.extractor = HardieExtractor()
        self.feature_set = None
        self.keras_model = None
        self.roi_size = 64

    def load(self, name):
        # Model
        if path.isfile('{}_extractor.pkl'.format(name)):
            self.extractor = joblib.load('{}_extractor.pkl'.format(name))
        if path.isfile('{}_clf.pkl'.format(name)):
            self.clf = joblib.load('{}_clf.pkl'.format(name))
        if path.isfile('{}_scaler.pkl'.format(name)):
            self.scaler = joblib.load('{}_scaler.pkl'.format(name))
        if path.isfile('{}_selector.pkl'.format(name)):
            self.selector = joblib.load('{}_selector.pkl'.format(name))
        if path.isfile('{}_transform.pkl'.format(name)):
            self.transform = joblib.load('{}_transform.pkl'.format(name))
        if path.isfile('{}_arch.json'.format(name)):
            self.keras_model.load(name)
            '''
            self.keras_model = model_from_json(open('{}_arch.json'.format(name)).read())
            self.keras_model.load_weights('{}_weights.h5'.format(name))
            '''
        '''
        if path.isfile('{}_fs.npy'.format(self.extractor.name)):
            self.feature_set = np.load('{}_fs.npy'.format(self.extractor.name))
        '''
    def load_cnn(self, name):
        if path.isfile('{}_arch.json'.format(name)):
            self.keras_model = neural.NetModel()
            self.keras_model.load(name)
            '''
            self.keras_model = model_from_json(open('{}_arch.json'.format(name)).read())
            self.keras_model.load_weights('{}_weights.h5'.format(name))
            '''

    def save(self, name):
        if self.extractor != None:
            joblib.dump(self.extractor, '{}_extractor.pkl'.format(name))
        if self.clf != None:
            joblib.dump(self.clf, '{}_clf.pkl'.format(name))
        if self.scaler != None:
            joblib.dump(self.scaler, '{}_scaler.pkl'.format(name))
        if self.selector != None:
            joblib.dump(self.selector, '{}_selector.pkl'.format(name))
        if self.transform != None:
            joblib.dump(self.transform, '{}_transform.pkl'.format(name))
        if self.keras_model != None:
            self.keras_model.save(name)
            '''
            json_string = self.keras_model.to_json()
            open('{}_arch.json'.format(name), 'w').write(json_string)
            self.keras_model.save_weights('{}_weights.h5'.format(name), overwrite=True)
            '''

        '''
        if self.feature_set != None:
            np.save(self.feature_set, '{}_fs.npy'.format(self.extractor.name))
        '''
    def preprocess_rois(self, rois):
        if self.preprocessor == 'heq':
            for i in range(len(rois)):
                rois[i] = equalize_hist(rois[i])
        elif self.preprocessor == 'nlm':
            for i in range(len(rois)):
                rois[i] = denoise_nl_means(rois[i])

    def create_rois(self, data, blob_set, mode=None):
        dsize = (int(self.roi_size * 1.15), int(self.roi_size * 1.15))
        print 'dsize: {} rsize: {}'.format(dsize, self.roi_size)
        # Create image set
        img_set = []
        for i in range(len(data)):
            img, lung_mask = data.get(i)
            sampled, lce, norm = preprocess(img, lung_mask)
            img_set.append(lce)

        print 'preproc: {}'.format(self.preprocessor)
        # Create roi set
        roi_set = []
        for i in range(len(img_set)):
            img = img_set[i]
            rois = []
            masks = []
            if mode == 'mask':
                _, masks = adaptive_distance_thold(img, blob_set[i])
            
            for j in range(len(blob_set[i])):
                x, y, r = blob_set[i][j]
                r = int(r * 1.15)
                shift = 0 
                side = 2 * shift + 2 * r + 1

                tl = (x - shift - r, y - shift - r)
                ntl = (max(0, tl[0]), max(0, tl[1]))
                br = (x + shift + r + 1, y + shift + r + 1)
                nbr = (min(img.shape[0], br[0]), min(img.shape[1], br[1]))

                roi = img[ntl[0]:nbr[0], ntl[1]:nbr[1]]
                roi = cv2.resize(roi, dsize, interpolation=cv2.INTER_CUBIC)

                if mode == 'mask':
                    mask = cv2.resize(masks[j], dsize, interpolation=cv2.INTER_CUBIC)
                    roi *= mask.astype(np.float32)

                rois.append(roi)

            self.preprocess_rois(rois)
            '''
            for i in range(len(rois)):
                util.imwrite('after_heq_{}.jpg'.format(i), rois[i])
            '''
            roi_set.append(np.array(rois))

        return np.array(roi_set)

    def detect_blobs(self, img, lung_mask, threshold=0.5):
        sampled, lce, norm = preprocess(img, lung_mask)

        blobs, ci = wmci(lce, lung_mask, threshold)

        return blobs, norm, lce, ci

    def detect_blobs_proba(self, img, lung_mask, threshold=0.5, method='wmci'):
        sampled, lce, norm = preprocess(img, lung_mask)
        
        if method == 'wmci':
            blobs, ci, proba = wmci_proba(lce, lung_mask, threshold)
        elif method == 'log':
            blobs, proba = log_(lce, lung_mask, threshold, proba=True)
            ci = norm
        elif method == 'dog':
            blobs, proba = dog(lce, lung_mask, threshold, proba=True)
            ci = norm
        elif method == 'doh':
            blobs, proba = doh(lce, lung_mask, threshold, proba=True)
            ci = norm

        return blobs, norm, lce, ci, proba

    def segment(self, img, blobs):
        blobs, nod_masks = adaptive_distance_thold(img, blobs)

        return blobs, nod_masks

    def extract(self, norm, lce, wmci, lung_mask, blobs, nod_masks):
        return self.extractor.extract(norm, lce, wmci, lung_mask, blobs, nod_masks)

    def extract_feature_set(self, data):
        feature_set = []
        blob_set = []
        print "extract_feature_set"
        print '[',
        for i in range(len(data)):
            if i % (len(data)/10) == 0:
                print ".",
                sys.stdout.flush()

            img, lung_mask = data.get(i)
            blobs, norm, lce, ci = self.detect_blobs(img, lung_mask)
            blobs, nod_masks = self.segment(lce, blobs)
            feats = self.extract(norm, lce, ci, lung_mask, blobs, nod_masks)
            feature_set.append(feats)
            blob_set.append(blobs)
        print ']'
        return np.array(feature_set), np.array(blob_set)

    def extract_feature_set_proba(self, data, detecting_proba=0.3):
        feature_set = []
        blob_set = []
        proba_set = []
        print '[',
        for i in range(len(data)):
            if i % (len(data)/10) == 0:
                print ".",
                sys.stdout.flush()

            img, lung_mask = data.get(i)
            blobs, norm, lce, ci, proba = self.detect_blobs_proba(img, lung_mask, detecting_proba)
            blobs, nod_masks = self.segment(lce, blobs)
            feats = self.extract(norm, lce, ci, lung_mask, blobs, nod_masks)
            feature_set.append(feats)
            blob_set.append(blobs)
            proba_set.append(proba)

        print ']'
        return np.array(feature_set), np.array(blob_set), proba_set

    def predict_proba_one(self, blobs, feature_vectors):
        feature_vectors = self.scaler.transform(feature_vectors)

        if self.selector != None:
            feature_vectors = self.selector.transform(feature_vectors)

        probs = self.clf.predict_proba(feature_vectors)
        probs = probs.T[1]
        blobs = np.array(blobs)

        return blobs, probs

    def predict_proba_one_keras(self, blobs, rois):
        img_rows, img_cols = rois[0].shape
        X = rois.reshape(rois.shape[0], 1, img_rows, img_cols)
        X = X.astype('float32')

        print 'Predict proba one keras'
        print X.shape

        probs = self.keras_model.predict_proba(X)

        probs = probs.T[1]
        blobs = np.array(blobs)

        return blobs, probs

    def extract_features_one_keras(self, rois, layer=-1):
        img_rows, img_cols = rois[0].shape
        X = rois.reshape(rois.shape[0], 1, img_rows, img_cols)
        X = X.astype('float32')

        layers = len(self.keras_model.network.layers)

        feats = np.array([])

        if layer < 0:
            feats = neural.get_activations(self.keras_model.network, layers - 1, X)
        else:
            feats = neural.get_activations(self.keras_model.network, layer, X)

        return feats

    def _classify(self, blobs, feature_vectors, thold=0.012):
        blobs, probs = self.predict_proba_one(blobs, feature_vectors)
        blobs = blobs[probs>thold]

        return blobs, probs[probs>thold]

    '''     
        Input: images & masks (data provider), blobs
        Returns: classifier, scaler

    def train(self, data, blobs):
        X, Y = classify.create_training_set(data, blobs)
        
        self.clf, self.scaler = classify.train(X, Y, self.clf, self.scaler)

        self.save(self.name)

        return self.clf, self.scaler
    '''

    def train_with_feature_set(self, feature_set, pred_blobs, real_blobs, feat_weight=False):
        X, Y = classify.create_training_set_from_feature_set(feature_set, pred_blobs, real_blobs)

        return classify.train(X, Y, self.clf, self.scaler, self.selector, feat_weight)
        
    def fit_vgg(self, X_train, Y_train):
        np.random.seed(1337)  # for reproducibility
        batch_size = 32
        nb_classes = 2
        nb_epoch = 40
        data_augmentation = True

        # input image dimensions
        img_rows, img_cols = 32, 32
        # the CIFAR10 images are RGB
        img_channels = 1

        #print('X_train shape:', X_train.shape)
        #print(X_train.shape[0], 'train samples')

        _init = 'orthogonal'
        _activation = 'linear' #LeakyReLU(alpha=.333)
        _filters = 64
        _dropout = 0.1

        model = Sequential()

        model.add(Convolution2D(_filters, 3, 3, border_mode='same',
                    input_shape=(img_channels, img_rows, img_cols), init=_init))
        #model.add(Activation(_activation))
        model.add(LeakyReLU(alpha=.333))
        model.add(Convolution2D(_filters, 3, 3, init=_init))
        #model.add(Activation(_activation))
        model.add(LeakyReLU(alpha=.333))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(_dropout))

        model.add(Convolution2D(2 * _filters, 3, 3, border_mode='same', init=_init))
        #model.add(Activation(_activation))
        model.add(LeakyReLU(alpha=.333))
        model.add(Convolution2D(2 * _filters, 3, 3, init=_init))
        #model.add(Activation(_activation))
        model.add(LeakyReLU(alpha=.333))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(2 * _dropout))

        model.add(Convolution2D(3 * _filters, 3, 3, border_mode='same', init=_init))
        #model.add(Activation(_activation))
        model.add(LeakyReLU(alpha=.333))
        model.add(Convolution2D(3 * _filters, 3, 3, init=_init))
        #model.add(Activation(_activation))
        model.add(LeakyReLU(alpha=.333))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(3 * _dropout))

        model.add(Flatten())
        model.add(Dense(512, init=_init))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))

        # let's train the model using SGD + momentum (how original).
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd)

        lr_scheduler = StageScheduler([20, 30])
        if not data_augmentation:
            print('Not using data augmentation or normalization')
            model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, callbacks=[lr_scheduler])
            #score = model.evaluate(X_test, Y_test, batch_size=batch_size)
            #print('Test score
        else:
            print('Using data augmentation')
            X_train, Y_train = offline_augment(X_train, Y_train, ratio=1, 
                        rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)

            print 'Negatives: {}'.format(np.sum(Y_train.T[0]))
            print 'Positives: {}'.format(np.sum(Y_train.T[1]))
            model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, callbacks=[lr_scheduler])

        self.keras_model = model

    def fit_graham(self, X_train, Y_train):
        Y_train = np.array([Y_train.T[1]]).T
        print 'labels shape {}'.format(Y_train)
        np.random.seed(1337)  # for reproducibility
        batch_size = 32
        nb_classes = 2
        nb_epoch = 40
        data_augmentation = True

        # input image dimensions
        img_rows, img_cols = X_train[0][0].shape
        print img_rows, img_cols
        # the CIFAR10 images are RGB
        img_channels = 1

        #print('X_train shape:', X_train.shape)
        #print(X_train.shape[0], 'train samples')

        _init = 'orthogonal'
        _activation = 'linear' #LeakyReLU(alpha=.333)
        _filters = 64
        _dropout = 0.1

        model = Sequential()

        model.add(Convolution2D(_filters, 3, 3, border_mode='same',
                    input_shape=(img_channels, img_rows, img_cols), init=_init))
        #model.add(Activation(_activation))
        model.add(LeakyReLU(alpha=.333))
        model.add(Convolution2D(_filters, 3, 3, init=_init))
        #model.add(Activation(_activation))
        model.add(LeakyReLU(alpha=.333))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Convolution2D(2 * _filters, 2, 2, border_mode='same', init=_init))
        #model.add(Activation(_activation))
        model.add(LeakyReLU(alpha=.333))
        model.add(Dropout(_dropout))
        model.add(Convolution2D(2 * _filters, 2, 2, init=_init))
        #model.add(Activation(_activation))
        model.add(LeakyReLU(alpha=.333))
        model.add(Dropout(_dropout))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Convolution2D(3 * _filters, 2, 2, border_mode='same', init=_init))
        #model.add(Activation(_activation))
        model.add(LeakyReLU(alpha=.333))
        model.add(Dropout(2 * _dropout))
        model.add(Convolution2D(3 * _filters, 2, 2, init=_init))
        #model.add(Activation(_activation))
        model.add(LeakyReLU(alpha=.333))
        model.add(Dropout(2 * _dropout))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Convolution2D(4 * _filters, 2, 2, border_mode='same', init=_init))
        #model.add(Activation(_activation))
        model.add(LeakyReLU(alpha=.333))
        model.add(Dropout(3 * _dropout))
        model.add(Convolution2D(4 * _filters, 2, 2, init=_init))
        #model.add(Activation(_activation))
        model.add(LeakyReLU(alpha=.333))
        model.add(Dropout(3 * _dropout))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Convolution2D(5 * _filters, 2, 2, border_mode='same', init=_init))
        #model.add(Activation(_activation))
        model.add(LeakyReLU(alpha=.333))
        model.add(Dropout(4 * _dropout))
        model.add(Convolution2D(5 * _filters, 2, 2, init=_init))
        #model.add(Activation(_activation))
        model.add(LeakyReLU(alpha=.333))
        model.add(Dropout(4 * _dropout))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(512, init=_init))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('softmax'))

        # let's train the model using SGD + momentum (how original).
        sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='binary_crossentropy', optimizer=sgd)

        lr_scheduler = StageScheduler([15, 30])
        if not data_augmentation:
            print('Not using data augmentation or normalization')
            model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, callbacks=[lr_scheduler])
            #score = model.evaluate(X_test, Y_test, batch_size=batch_size)
            #print('Test score
        else:
            print('Using data augmentation')
            X_train, Y_train = offline_augment(X_train, Y_train, ratio=1, 
                        rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)

            model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, callbacks=[lr_scheduler])

        self.keras_model = model


    def train_with_feature_set_keras(self, feats_tr, pred_blobs_tr, real_blobs_tr,
                                        feats_test=None, pred_blobs_test=None, real_blobs_test=None,
                                        model='shallow_1'):
        img_rows, img_cols = feats_tr[0][0].shape
        nb_classes = 2

        X_tr, y_tr = classify.create_training_set_from_feature_set(feats_tr, pred_blobs_tr, real_blobs_tr)
        X_tr = X_tr.reshape(X_tr.shape[0], 1, img_rows, img_cols)
        X_tr = X_tr.astype('float32')
        Y_tr= np_utils.to_categorical(y_tr, nb_classes)

        X_test, Y_test = None, None
        if feats_test != None:
            X_test, y_test = classify.create_training_set_from_feature_set(feats_test, pred_blobs_test, real_blobs_test)
            X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
            X_test = X_test.astype('float32')
            Y_test = np_utils.to_categorical(y_test, nb_classes)
        
        self.keras_model, history = neural.fit(X_tr, Y_tr, X_test, Y_test, model)

        #self.save(self.name)
        return history

    def predict_proba(self, data):
        #self.load(self.name)

        data_blobs = []
        data_probs = []
        for i in range(len(data)):
            img, lung_mask = data.get(i)
            blobs, norm, lce, ci = self.detect_blobs(img, lung_mask)
            blobs, nod_masks = self.segment(lce, blobs)
            feats = self.extract(norm, lce, ci, lung_mask, blobs, nod_masks)
            blobs, probs = self.predict_proba_one(blobs, feats)

            # candidate cue adjacency rule: 22 mm
            filtered_blobs = []
            filtered_probs = []
            for j in range(len(blobs)):
                valid = True
                for k in range(len(blobs)):
                    dist2 = (blobs[j][0] - blobs[k][0]) ** 2 + (blobs[j][1] - blobs[k][1]) ** 2
                    if dist2 < 988 and probs[j] + EPS < probs[k]:
                        valid = False
                        break

                if valid:
                    filtered_blobs.append(blobs[j])
                    filtered_probs.append(probs[j])

            #show_blobs("Predict result ...", lce, filtered_blob)
            data_blobs.append(np.array(filtered_blobs)) 
            data_probs.append(np.array(filtered_probs))

        return np.array(data_blobs), np.array(data_probs)

    '''
        Input: data provider
        Returns: blobs
    '''
    def predict(self, data):
        data_blobs = []
        for i in range(len(data)):
            img, lung_mask = data.get(i)
            blobs, norm, lce, ci = self.detect_blobs(img, lung_mask)
            blobs, nod_masks = self.segment(lce, blobs)
            feats = self.extract(norm, lce, ci, lung_mask, blobs, nod_masks)
            blobs, probs = self._classify(blobs, feats)

            # candidate cue adjacency rule: 22 mm
            filtered_blobs = []
            for j in range(len(blobs)):
                valid = True
                for k in range(len(blobs)):
                    dist2 = (blobs[j][0] - blobs[k][0]) ** 2 + (blobs[j][1] - blobs[k][1]) ** 2
                    if dist2 < 988 and probs[j] + EPS < probs[k]:
                        valid = False
                        break

                if valid:
                    filtered_blobs.append(blobs[j])

            #show_blobs("Predict result ...", lce, filtered_blob)
            data_blobs.append(np.array(filtered_blobs))     

        return np.array(data_blobs)

    def predict_from_feature_set(self, feature_set, blob_set, thold=0.012):
        data_blobs = []
        for i in range(len(feature_set)):
            blobs, probs = self._classify(blob_set[i], feature_set[i], thold)

            # candidate cue adjacency rule: 22 mm
            filtered_blobs = []
            for j in range(len(blobs)):
                valid = True
                for k in range(len(blobs)):
                    dist2 = (blobs[j][0] - blobs[k][0]) ** 2 + (blobs[j][1] - blobs[k][1]) ** 2
                    if dist2 < 988 and probs[j] + EPS < probs[k]:
                        valid = False
                        break

                if valid:
                    filtered_blobs.append(blobs[j])

            #show_blobs("Predict result ...", lce, filtered_blob)
            data_blobs.append(np.array(filtered_blobs))     

        return np.array(data_blobs)

    def predict_proba_from_feature_set(self, feature_set, blob_set):
        #self.load(self.name)
        DIST2 = 987.755
        
        data_blobs = []
        data_probs = []
        for i in range(len(feature_set)):
            blobs, probs = self.predict_proba_one(blob_set[i], feature_set[i])

            ## candidate cue adjacency rule: 22 mm
            filtered_blobs = []
            filtered_probs = []
            for j in range(len(blobs)):
                valid = True
                for k in range(len(blobs)):
                    dist2 = (blobs[j][0] - blobs[k][0]) ** 2 + (blobs[j][1] - blobs[k][1]) ** 2
                    if dist2 < DIST2 and probs[j] + EPS < probs[k]:
                        valid = False
                        break

                if valid:
                    filtered_blobs.append(blobs[j])
                    filtered_probs.append(probs[j])

            #show_blobs("Predict result ...", lce, filtered_blob)

            data_blobs.append(np.array(filtered_blobs)) 
            data_probs.append(np.array(filtered_probs))

        return np.array(data_blobs), np.array(data_probs)

    def predict_proba_from_feature_set_keras(self, feature_set, blob_set):
        #self.load(self.name)
        DIST2 = 987.755

        data_blobs = []
        data_probs = []
        for i in range(len(feature_set)):
            blobs, probs = self.predict_proba_one_keras(blob_set[i], feature_set[i])

            ## candidate cue adjacency rule: 22 mm
            filtered_blobs = []
            filtered_probs = []
            for j in range(len(blobs)):
                valid = True
                for k in range(len(blobs)):
                    dist2 = (blobs[j][0] - blobs[k][0]) ** 2 + (blobs[j][1] - blobs[k][1]) ** 2
                    if dist2 < DIST2 and probs[j] + EPS < probs[k]:
                        valid = False
                        break

                if valid:
                    filtered_blobs.append(blobs[j])
                    filtered_probs.append(probs[j])

            #show_blobs("Predict result ...", lce, filtered_blob)
            data_blobs.append(np.array(filtered_blobs)) 
            data_probs.append(np.array(filtered_probs))

        return np.array(data_blobs), np.array(data_probs)

    def extract_features_from_keras_model(self, roi_set, layer):
        #self.load(self.name)
        data_feats = []

        print '[',
        for i in range(len(roi_set)):
            if i % (len(roi_set)/10) == 0:
                print '.',
                sys.stdout.flush()
            feats = self.extract_features_one_keras(roi_set[i], layer)
            data_feats.append(np.array(feats))
        print ']'

        return np.array(data_feats)

    # Join filter and eval on the same function and vectorize
    def filter_by_proba(self, blob_set, prob_set, thold = 0.012):
        data_blobs = []
        data_probs = []
        for i in range(len(blob_set)):
            probs = prob_set[i]
            filtered_blobs = blob_set[i][probs > thold]
            filtered_probs = prob_set[i][probs > thold]
            '''
            probs = prob_set[i]
            blobs = blob_set[i]
            filtered_blobs = []
            filtered_probs = []
            for j in range(len(blobs)):
                if probs[j] > thold:
                    filtered_blobs.append(blobs[j])
                    filtered_probs.append(probs[j])
            '''

            #show_blobs("Predict result ...", lce, filtered_blob)
            data_blobs.append(np.array(filtered_blobs)) 
            data_probs.append(np.array(filtered_probs))

        return np.array(data_blobs), np.array(data_probs)

# optimized
opt_classifiers = {'svm':svm.SVC(probability=True, C=0.0373, gamma=0.002), 'lda':lda.LDA()}
# default
classifiers = {'linear-svm':svm.SVC(kernel='linear', probability=True), 'svm':svm.SVC(kernel='rbf', probability=True, C=1.0, gamma=0.01), 'lda':lda.LDA()}
reductors = {'none':None, 'pca':decomposition.PCA(n_components=0.99999999999, whiten=True), 'lda':selection.SelectFromModel(lda.LDA())}

