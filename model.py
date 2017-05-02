import cv2 
import sys

import skimage.io as io
from skimage.exposure import equalize_hist
from skimage.restoration import denoise_nl_means

from sklearn import neighbors
from sklearn import lda
from sklearn import svm
from sklearn import decomposition
from sklearn import preprocessing
from sklearn import ensemble
from sklearn import feature_selection as selection
from sklearn.externals import joblib
from sklearn.feature_selection import RFE

import pykernels as pk

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
import gc

import classify
from detect import *
from extract import * 
from preprocess import *
from segment import *
from augment import *
from util import *
import neural
import ae
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
    MAX_DIST2 = 987.755
    filtered_blobs = []
    filtered_probs = []
    for j in range(len(blobs)):
        valid = True
        for k in range(len(blobs)):
            dist2 = (blobs[j][0] - blobs[k][0]) ** 2 + (blobs[j][1] - blobs[k][1]) ** 2
            if dist2 < MAX_DIST2 and probs[j] + EPS < probs[k]:
                valid = False
                break

        if valid:
            filtered_blobs.append(blobs[j])
            filtered_probs.append(probs[j])

    return np.array(filtered_blobs), np.array(filtered_probs)

# Baseline Model
class BaselineModel(object):
    def __init__(self, name='default'):
        self.clf = lda.LDA()
        self.scaler = preprocessing.StandardScaler()
        self.selector = None
        self.transform = None
        self.extractor = HardieExtractor()
        self.feature_set = None
        self.network = None

        self.name = name
        self.roi_size = 64
        self.dataset_type = 'numpy'
        self.file = None # datasets_file

    def __del__(self):
        if self.file != None:
            self.file.close()

    @property
    def roi_size(self):
        return self._roi_size

    @roi_size.setter
    def roi_size(self, value):
        self._roi_size = value
        self.dsize = (int(value * 1.15), int(value * 1.15))

    def create_datasets_file(self, datasets_filename):
        if self.file != None:
            self.file.close()
        self.file = h5py.File('data/' + datasets_filename + '.hdf5', 'w')

    def open_datasets_file(self, datasets_filename):
        if self.file != None:
            self.file.close()
        print('Opening file {}'.format('data/' + datasets_filename + '.hdf5'))
        self.file = h5py.File('data/' + datasets_filename + '.hdf5', 'r')

    def get_roiset(self, size):
        rois = []
        pred_blobs = []
        for i in range(size):
            rois.append(self.file['rois_{}'.format(i)])
            pred_blobs.append(self.file['pred_blobs_{}'.format(i)][:])
            print("roi len {}, pred_blobs {}".format(len(rois[0]), len(pred_blobs[0])))
        return np.array(rois), np.array(pred_blobs)

    def get_datasets(self, fold):
        X_train_pos = self.file['fold_{}'.format(fold)]['X_train_pos']
        X_train_pos = X_train_pos[:]
        X_train_neg = self.file['fold_{}'.format(fold)]['X_train_neg']
        Y_train_pos = np.zeros(shape=(len(X_train_pos),2), dtype=np.int)
        Y_train_pos.T[1] = 1
        Y_train_neg = np.zeros(shape=(len(X_train_neg),2), dtype=np.int)
        Y_train_neg.T[0] = 1

        X_test_pos = self.file['fold_{}'.format(fold)]['X_test_pos']
        X_test_pos = X_test_pos[:]
        X_test_neg = self.file['fold_{}'.format(fold)]['X_test_neg']
        Y_test_pos = np.zeros(shape=(len(X_test_pos),2), dtype=np.int)
        Y_test_pos.T[1] = 1
        Y_test_neg = np.zeros(shape=(len(X_test_neg),2), dtype=np.int)
        Y_test_neg.T[0] = 1
        return (X_train_pos, X_train_neg), (Y_train_pos, Y_train_neg), (X_test_pos, X_test_neg), (Y_test_pos, Y_test_neg)
        
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
            self.network.load(name)
        '''
        if path.isfile('{}_fs.npy'.format(self.extractor.name)):
            self.feature_set = np.load('{}_fs.npy'.format(self.extractor.name))
        '''
    def load_cnn(self, name):
        if path.isfile('{}_arch.json'.format(name)):
            self.network = neural.NetModel()
            self.network.load(name)
        else:
            raise Exception('{} model not found'.format(name))

    def load_cnn_weights(self, name):
            self.network.network.load_weights('{}_weights.h5'.format(name))

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
        if self.network != None:
            self.network.save(name)
        '''
        if self.feature_set != None:
            np.save(self.feature_set, '{}_fs.npy'.format(self.extractor.name))
        '''

    def preprocess_rois(self, rois):
        assert self.preprocess_roi != 'none'
        result = []
        for j in range(len(rois)): 
            result.append([])
            for k in range(len(rois[j])):
                if 'norm' in self.preprocess_roi:
                    result[-1].append(normalize(rois[j][k], np.ones(shape=rois[j][k].shape)))
                if 'norm3' in self.preprocess_roi:
                    norm = normalize(rois[j][k], np.ones(shape=rois[j][k].shape))
                    result[-1].append(norm)
                    result[-1].append(norm)
                    result[-1].append(norm)
                if 'heq' in self.preprocess_roi:
                    result[-1].append(equalize_hist(rois[j][k]))
                if 'nlm' in self.preprocess_roi:
                    result[-1].append(denoise_nl_means(rois[j][k]))
            result[-1] = np.array(result[-1])
        return np.array(result)

    def create_rois(self, data, blob_set, mode=None, pad=True, save=False):
        pad_size = 1.15
        if pad:
            dsize = (int(self.roi_size * pad_size), int(self.roi_size * pad_size))
        else:
            dsize = (self.roi_size, self.roi_size)

        print 'dsize: {} rsize: {}'.format(dsize, self.roi_size)

        img_set = []
        num_blobs = 0
        for i in range(len(data)):
            num_blobs += len(blob_set[i])
            if i % 10 == 0:
                print 'Create rois. Preprocess image {}.'.format(i)
            img, lung_mask = data.get(i, downsample=self.downsample) # img 2048 side, lung mask downsampled
            img = antialiasing_dowsample(img, self.downsample) # img downsampled

            results = []
            lce_ = None

            # Single-channel preprocessing
            if 'original' in self.preprocess_lung:
                results.append(img) 
            if 'norm' in self.preprocess_lung:
                results.append(normalize(img, lung_mask))
            if 'lce' in self.preprocess_lung:
                lce_ = lce(img, self.downsample)
                results.append(lce_) 
            if 'wmci' in self.preprocess_lung:
                sampled_lce = lce_
                sampled_mask = lung_mask
                if not self.downsample:
                    sampled_img, sampled_mask = data.get(i, downsample=True)
                    sampled_lce = lce(sampled_img, downsample=True)
                _, ci, _ = wmci_proba(sampled_lce, sampled_mask, 0.5)
                if not self.downsample:
                    ci = cv2.resize(ci, img.shape, interpolation=cv2.INTER_CUBIC)
                results.append(ci)
            img_set.append(np.array(results))

        # Create roi set
        roi_set = []
        '''
        # NOT WORKING
        if self.streams == 'trf':
            roi_set = [[], [], []]
        elif self.streams == 'seg':
            roi_set = [[], []]
        elif self.streams == 'fovea':
            roi_set = [[], []]
        '''

        print 'preprocess rois {}'.format(self.preprocess_roi)
        channels = len(img_set[0])
        range_values = np.zeros(shape=(channels, 2), dtype=np.float)
        for i in range(channels):
            range_values[i][0] = 1e9
            range_values[i][1] = -1e9

        for i in range(len(img_set)):
            img = img_set[i]
            if i % 10 == 0:
                print 'extract rois {} img shape {}'.format(i, img.shape)
            rois = []
            masks = []

            if self.streams == 'seg' :
                _, masks = adaptive_distance_thold(img[LCE_POS], blob_set[i])
            
            rois = []
            for j in range(len(blob_set[i])):
                sample_factor = 1
                if not self.downsample:
                    sample_factor = 4

                x, y, _ = blob_set[i][j]

                r = self.blob_rad

                x *= sample_factor
                y *= sample_factor
                r *= sample_factor

                r = int(r * pad_size)

                shift = 0 
                #side = 2 * shift + 2 * r + 1

                tl = (x - shift - r, y - shift - r)
                ntl = (max(0, tl[0]), max(0, tl[1]))
                br = (x + shift + r + 1, y + shift + r + 1)
                nbr = (min(img.shape[1], br[0]), min(img.shape[2], br[1]))

                roi = []
                for k in range(img.shape[0]):
                    tmp = img[k][ntl[0]:nbr[0], ntl[1]:nbr[1]]
                    tmp = cv2.resize(tmp, dsize, interpolation=cv2.INTER_CUBIC)
                    roi.append(tmp)
                    
                '''
                if mode == 'mask':
                    mask = cv2.resize(masks[j], dsize, interpolation=cv2.INTER_CUBIC)
                    for k in range(img.shape[0]):
                        roi[k] *= mask.astype(np.float32)
                '''
                rois.append(roi)

            rois = np.array(rois)

            if save:
                rois = self.file.create_dataset('tmp_{}'.format(i), data=rois)

            roi_set.append(rois)
            
            # Update range values
            min_by_channels = np.min(rois, axis=(0, 2, 3))
            max_by_channels = np.max(rois, axis=(0, 2, 3))
            for i in range(channels):
                range_values[i][0] = min(range_values[i][0], min_by_channels[i])
                range_values[i][1] = max(range_values[i][1], max_by_channels[i])
            
            '''
            for i in range(len(rois)):
                util.imwrite('after_heq_{}.jpg'.format(i), rois[i])
            '''
            gc.collect()

        if self.preprocess_roi != 'none':
            for i in range(len(roi_set)):
                if i % 10 == 0: 
                    print 'preprocess rois {} '.format(i)
                if save:
                    roi_set[i] = self.file.create_dataset('rois_{}'.format(i), data=self.preprocess_rois(roi_set[i], range_values))
                    self.file.create_dataset('pred_blobs_{}'.format(i), data=blob_set[i])
                else:
                    roi_set[i] = self.preprocess_rois(roi_set[i])

        '''
            # Fix segment
            if self.streams == 'trf':
                #rois = np.swapaxes(np.swapaxes(np.swapaxes(rois, 2, 3), 1, 2), 0, 1)
                rois = np.swapaxes(rois, 0, 1)
                rois = rois.reshape(rois.shape[0], rois.shape[1], 1, rois.shape[2], rois.shape[3])
                assert rois.shape[0] == len(roi_set)
                for k in range(rois.shape[0]):
                    roi_set[k].append(rois[k])
            elif self.streams == 'seg':
                masked_rois = np.copy(rois)
                for j in xrange(len(masked_rois)):
                    mask = cv2.resize(masks[j], dsize, interpolation=cv2.INTER_CUBIC)
                    for k in xrange(len(masked_rois[j])): 
                        masked_rois[j][k] *= mask.astype(np.float32)
                roi_set[0].append(rois)
                roi_set[1].append(masked_rois)
            elif self.streams == 'fovea':
                FOVEA_FACTOR = 2.5
                tsize = (int(dsize[0] * FOVEA_FACTOR), int(dsize[1] * FOVEA_FACTOR))
                offset = (tsize[0] - dsize[0]) / 2
                fovea_rois = np.copy(rois)
                for j in xrange(len(rois)):
                    for k in xrange(len(rois[j])): 
                        tmp = cv2.resize(rois[j][k], tsize, interpolation=cv2.INTER_CUBIC)
                        fovea_rois[j][k] = tmp[offset:offset+dsize[0], offset:offset+dsize[1]]
                roi_set[0].append(rois)
                roi_set[1].append(fovea_rois)
            else:
                roi_set.append(rois)
        '''
        return np.array(roi_set)
    
    def detect_blobs(self, img, lung_mask, threshold=0.5):
        print("img {} {} {}".format(img.shape, np.max(img), np.min(img)))
        sampled, lce, norm = preprocess(img, lung_mask)
        blobs, ci = wmci(lce, lung_mask, threshold)
        return blobs, norm, lce, ci

    def segment(self, img, blobs):
        blobs, nod_masks = adaptive_distance_thold(img, blobs)
        return blobs, nod_masks

    def extract(self, norm, lce, wmci, lung_mask, blobs, nod_masks):
        return self.extractor.extract(norm, lce, wmci, lung_mask, blobs, nod_masks)

    def extract_feature_set(self, data):
        feature_set = []
        blob_set = []
        print "extract feature set"
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

        probs = None
        if hasattr(self.clf, 'predict_proba'):
            probs = self.clf.predict_proba(feature_vectors)
            probs = np.max(probs.T[1:], axis=0)
        else:
            probs = self.clf.predict(feature_vectors)
            
        blobs = np.array(blobs)
        blobs, probs = adjacency_rule(blobs, probs)
        return blobs, probs

    def predict_proba_one_keras(self, blobs, rois):
        probs = self.network.predict_proba(rois, self.streams != 'none')
        #probs = probs.T[1]
        probs = np.max(probs.T[1:], axis=0)
        blobs = np.array(blobs)

        blobs, probs = adjacency_rule(blobs, probs)
        return blobs, probs

    def extract_features_one_keras(self, rois, layer=-1):
        X = rois.astype('float32')
        layers = len(self.network.network.layers)
        feats = np.array([])
        if layer < 0:
            feats = neural.get_activations(self.network.network, layers - 1, X)
        else:
            feats = neural.get_activations(self.network.network, layer, X)

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
        
    def train_with_feature_set_keras(self, feats_tr, pred_blobs_tr, real_blobs_tr,
                                        feats_test=None, pred_blobs_test=None, real_blobs_test=None,
                                        model='shallow_1', model_suffix=None, network_init=None):
        print("{} {} {} {} {} {}".format(len(feats_tr), len(pred_blobs_tr), len(real_blobs_tr), len(feats_test), len(pred_blobs_test), len(real_blobs_test)))
        X_train, Y_train, X_test, Y_test = neural.create_train_test_sets(feats_tr, pred_blobs_tr, real_blobs_tr, 
                                                feats_test, pred_blobs_test, real_blobs_test, streams=self.streams )

        print "X_train shape {}".format(X_train.shape)
        self.network = neural.create_network(model, (X_train.shape[1], self.roi_size, self.roi_size), self.streams) 
        if network_init is not None:
            if self.args.transfer:
                self.load_cnn_weights('data/{}_{}'.format(network_init, model_suffix))
            else:
                self.load_cnn_weights(network_init)

        name =  'data/{}_{}'.format(model, model_suffix)
        history = self.network.fit(X_train, Y_train, X_test, Y_test, streams=(self.streams != 'none'), cropped_shape=(self.roi_size, self.roi_size), checkpoint_prefix=name, checkpoint_interval=2)
        return history

    def predict_proba(self, data):
        data_blobs = []
        data_probs = []
        for i in range(len(data)):
            img, lung_mask = data.get(i)
            blobs, norm, lce, ci = self.detect_blobs(img, lung_mask)
            blobs, nod_masks = self.segment(lce, blobs)
            feats = self.extract(norm, lce, ci, lung_mask, blobs, nod_masks)
            blobs, probs = self.predict_proba_one(blobs, feats)
            data_blobs.append(blobs) 
            data_probs.append(probs)

        return np.array(data_blobs), np.array(data_probs)

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
        data_blobs = []
        data_probs = []

        for i in range(len(feature_set)):
            blobs, probs = self.predict_proba_one(blob_set[i], feature_set[i])
            data_blobs.append(blobs) 
            data_probs.append(probs)
        return np.array(data_blobs), np.array(data_probs)

    def predict_proba_from_feature_set_keras(self, feature_set, blob_set):
        #self.load(self.name)
        DIST2 = 987.755

        data_blobs = []
        data_probs = []
        if self.streams != 'none':
            feature_set = np.swapaxes(feature_set, 0, 1)

        for i in range(len(feature_set)):
            blobs, probs = self.predict_proba_one_keras(blob_set[i], feature_set[i])
            data_blobs.append(blobs) 
            data_probs.append(probs)
        return np.array(data_blobs), np.array(data_probs)

    def extract_features_from_network(self, roi_set, layer):
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
            #show_blobs("Predict result ...", lce, filtered_blob)
            data_blobs.append(np.array(filtered_blobs)) 
            data_probs.append(np.array(filtered_probs))

        return np.array(data_blobs), np.array(data_probs)

    def pretrain(self, model, X, fold=None, streams=False, nb_epoch=1):
        cropped_shape = (self.roi_size, self.roi_size)
        self.network = neural.create_network(model, (X.shape[1],) + cropped_shape, fold, streams)
        ae.pretrain_layerwise(self.network.network, X, nb_epoch=nb_epoch)

    def joint_training(self, model, X_train, Y_train, X_test, Y_test, fold=None):
        cropped_shape = (self.roi_size, self.roi_size)
        self.network = neural.create_network(model, (X_train.shape[1],) + cropped_shape, fold, self.streams)

        X_train, Y_train, X_test, Y_test = self.network.preprocess_augment(X_train, Y_train, X_test, Y_test, streams=(self.streams != 'none'), cropped_shape=cropped_shape)
        self.helper_model = ae.swwae_train(self.network.network, X_train, Y_train, X_test, Y_test, finetune_epochs=self.all_epochs, multipliers=self.multipliers, layerwise_epochs=self.lw_epochs, decoder_epochs=self.dec_epochs, lr=self.lr, model_name=self.init)

    def _split_data_pos_neg(self, X, Y):
        is_pos = Y.T[1]
        end_pos = 0
        for i in range(len(is_pos)):
            if is_pos[i] < 1:
                end_pos = i
                break

        return (X[:end_pos], X[end_pos:]), (Y[:end_pos], Y[end_pos:])
            
    def unsupervised_augment(self, model, X_train, Y_train, X_test, Y_test, fold=None, streams=False, nb_epoch=1, pos_neg_ratio=1.0, mode=None):
        cropped_shape = (self.roi_size, self.roi_size)
        channels = X_train.shape[1] if self.args.datasets == None else X_train[0].shape[1]
        self.network = neural.create_network(model, (channels,) + cropped_shape, fold, streams)
        #self.load_cnn_weights('data/{}_fold_{}'.format(model, fold))
        self.load_cnn('data/{}_fold_{}'.format(model, fold))

        if mode in {'add-random'}:
            self.network.generator.ratio = pos_neg_ratio
            self.network.generator.mode = 'balance_dataset'

        if self.args.datasets == None:
            X_train, Y_train, X_test, Y_test = self.network.preprocess_augment(X_train, Y_train, X_test, Y_test, streams=(self.streams != 'none'), cropped_shape=cropped_shape, disable_perturb=False)
            '''
            X_train = (np.array(X_train), np.array(X_train))
            X_test = (np.array(X_test), np.array(X_test))
            Y_train = (np.array(Y_train), np.array(Y_train))
            Y_test = (np.array(Y_test), np.array(Y_test))
            X_train, Y_train, X_test, Y_test = preprocess_dataset(X_train, Y_train, X_test, Y_test, streams=(self.streams != 'none'), config=self.args.preprocess_dataset, mode='hdf5')
            X_train, Y_train = self._split_data_pos_neg(X_train, Y_train)
            X_test, Y_test = self._split_data_pos_neg(X_test, Y_test)
            std = np.std(X_train[0])
            '''
            #self.helper_model = ae.swwae_augment(self.network.network, X_train, Y_train, X_test, Y_test, finetune_epochs=nb_epoch, multipliers=self.multipliers, layerwise_epochs=self.lw_epochs, decoder_epochs=self.dec_epochs, lr=self.lr, model_name=self.init)
            self.helper_model = ae.swwae_augment_hdf5(self.network.network, self.network.generator, X_train, Y_train, X_test, Y_test, finetune_epochs=nb_epoch, multipliers=self.multipliers, layerwise_epochs=self.lw_epochs, decoder_epochs=self.dec_epochs, lr=self.lr, model_name=self.init)

        else:
            print "Preprocessor {}".format(self.network.preprocessor)
            print "Generator {}".format(self.network.generator)
            self.helper_model = ae.swwae_augment_hdf5(self.network.network, self.network.generator, X_train, Y_train, X_test, Y_test, finetune_epochs=nb_epoch, multipliers=self.multipliers, layerwise_epochs=self.lw_epochs, decoder_epochs=self.dec_epochs, lr=self.lr, model_name=self.init)

    def fit_transform_bovw(self, rois_tr, pred_blobs_tr, blobs_tr, rois_te, pred_blobs_te, blobs_te,  model):
        X_tr, Y_tr = classify.create_training_set_from_feature_set(rois_tr, pred_blobs_tr, blobs_tr)
        X_te, Y_te = classify.create_training_set_from_feature_set(rois_te, pred_blobs_te, blobs_te)
        model.fit(X_tr)
        V_tr = []
        for rois in rois_tr:
            V_tr.append(model.transform(rois))
        V_te = []
        for rois in rois_te:
            V_te.append(model.transform(rois))
        return np.array(V_tr), np.array(V_te)

# optimized
opt_classifiers = {'svm':svm.SVC(probability=True, C=0.0373, gamma=0.002), 'lda':lda.LDA()}
# default
classifiers = {'linear-svm':svm.SVC(kernel='linear', C=1, probability=True), 'svm':svm.SVC(kernel='rbf', probability=True), 'lda':lda.LDA(), 'knn': neighbors.KNeighborsRegressor(), 'rf': ensemble.RandomForestClassifier(), 'hik':svm.SVC(kernel=pk.regular.Min(), C=1, probability=True)}
reductors = {'none':None, 'pca':decomposition.PCA(n_components=0.99999999999, whiten=True), 'lda':selection.SelectFromModel(lda.LDA())}
