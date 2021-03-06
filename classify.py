import sys
import time
import h5py

import numpy as np
from sklearn.lda import LDA
from sklearn import svm
from sklearn.metrics import classification_report
import sklearn.cross_validation as cross_val
from sklearn.externals import joblib
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.base import clone
from sklearn.feature_selection import SelectPercentile, f_classif

import model
import util

import theano
import theano.tensor as T

class Min:
    def __init__(self):
        self.h1 = T.dmatrix('h1')
        self.h2 = T.dmatrix('h2')
        self.kernel = T.dmatrix('k')
    
    def _compute(self, data_1, data_2):

        if np.any(data_1 < 0) or np.any(data_2 < 0):
            warnings.warn('Min kernel requires data to be strictly positive!')

        for d in range(data_1.shape[1]):
            column_1 = data_1[:, d].reshape(-1, 1)
            column_2 = data_2[:, d].reshape(-1, 1)
            self.kernel += T.min(column_1, column_2.T)

        return kernel

    def dim(self):
        return None

def create_uniform_trset(out_file):
    MAX_DIST = 35

    paths, locs, rads = jsrt.jsrt(set='jsrt140')
    left_masks = jsrt.left_lung(set='jsrt140')
    right_masks = jsrt.right_lung(set='jsrt140')

    size = len(paths)
    assert size == len(locs)
    assert size == len(rads)
    assert size == len(left_masks)
    assert size == len(right_masks)

    X = []
    Y = []
    
    # create positives
    print "Creating positives ..."
    for i in range(size):
        if rads[i] != -1:
            print " " + paths[i] + "..."
            x = pipeline_features(paths[i], [[locs[i][0], locs[i][1], rads[i]]], left_masks[i], right_masks[i])
            assert len(x) == 1
            X.append(x[0])
            Y.append(1)
            print "feats: " + str(np.array(X[-1]))
    
    # create negatives
    print "Creating negatives ..."
    for i in range(size):
        blobs = pipeline_blobs(paths[i], left_masks[i], right_masks[i])

        if len(blobs) == 0:
            continue

        print " " + paths[i] + "..."
        if rads[i] != -1:
            idx = -1
            for j in range(1234):
                idx = randint(0, len(blobs)-1)
                if ((blobs[idx][0] - locs[i][0]) ** 2 + (blobs[idx][1] - blobs[idx][1]) ** 2) > (MAX_DIST ** 2):
                    break

            features = pipeline_features(paths[i], [blobs[idx]], left_masks[i], right_masks[i])
            X.append(features[0])
            Y.append(0)
        else:
            idx = randint(0, len(blobs)-1)

            features = pipeline_features(paths[i], [blobs[idx]], left_masks[i], right_masks[i])
            X.append(features[0])
            Y.append(0)

        print "feats: " + str(np.array(X[-1]))
    np.save(out_file, [X, Y])

'''
def create_training_set(data, y_blobs):
    MAX_DIST = 35

    size = len(y_blobs)

    X = []
    Y = []

    # create positives
    print "Creating positives ..."
    for i in range(size):
        if y_blobs[i][2] == -1:
            continue
        print " " + data.img_paths[i], 
        #t = time.clock()
        img, lung_mask = data.get(i)
        #print time.clock() - t
        #t = time.clock()
        blobs, norm, lce, ci = model.detect_blobs(img, lung_mask)
        #print time.clock() - t
        #t = time.clock()
        blobs, nod_masks = model.segment(lce, blobs)
        #print time.clock() - t

        nb = []
        tmp = []
        for j in range(len(blobs)):
            x, y, z = blobs[j]#
            dst = ((x - y_blobs[i][0]) ** 2 + (y - y_blobs[i][1]) ** 2) ** 0.5
            if dst < MAX_DIST:
                nb.append((dst, j))
                tmp.append(blobs[j])

        print "nearest blobs {} ...".format(len(nb))
        nb = sorted(nb)
        if len(nb) == 0:
            continue

        feats = model.extract(norm, lce, ci, lung_mask, [blobs[nb[0][1]]], [nod_masks[nb[0][1]]])
        X.append(feats[0])
        Y.append(1)
    
    # create negatives
    print "Creating negatives ..."
    for i in range(size):
        print " " + data.img_paths[i],
        img, lung_mask = data.get(i)
        blobs, norm, lce, ci = model.detect_blobs(img, lung_mask)
        blobs, nod_masks = model.segment(lce, blobs)

        neg_idx = []
        for j in range(len(blobs)):
            x, y, z = blobs[j]
            dst = ((x - y_blobs[i][0]) ** 2 + (y - y_blobs[i][1]) ** 2) ** 0.5
            if dst > MAX_DIST:
                neg_idx.append(j)

        print "blobs {} ...".format(len(neg_idx))
        feats = model.extract(norm, lce, ci, lung_mask, blobs[neg_idx], nod_masks[neg_idx])
        for fv in feats:
            X.append(fv)
            Y.append(0)

    X = np.array(X)
    Y = np.array(Y)
    return X, Y
'''

def create_training_set_from_feature_set(real_blobs, pred_blobs, feature_set, array_class='numpy', container=None, suffix=None):
    if array_class == 'numpy':
        return _create_training_set_from_feature_set_numpy(real_blobs, pred_blobs, feature_set)
    elif array_class == 'hdf5':
        return _create_training_set_from_feature_set_hdf5(real_blobs, pred_blobs, feature_set, container=container, suffix=suffix)

def _create_training_set_from_feature_set_numpy(real_blobs, pred_blobs, feature_set):
    MAX_DIST = 35
    size = len(real_blobs)
    X = []
    Y = []

    print "Creating positives ..."
    for i in range(size):
        for j in range(len(real_blobs[i])):
            nb = []
            for k in range(len(pred_blobs[i])):
                x, y, z = pred_blobs[i][k]#
                dst = ((x - real_blobs[i][j][0]) ** 2 + (y - real_blobs[i][j][1]) ** 2) ** 0.5
                if dst < MAX_DIST:
                    nb.append((dst, k))
            nb = sorted(nb)
            if len(nb) == 0:
                continue

            X.append(feature_set[i][nb[0][1]])
            Y.append(1)
    
    print "Creating negatives ..."
    for i in range(size):
        for j in range(len(real_blobs[i])):
            neg_idx = []
            for k in range(len(pred_blobs[i])):
                x, y, z = pred_blobs[i][k]
                dst = ((x - real_blobs[i][j][0]) ** 2 + (y - real_blobs[i][j][1]) ** 2) ** 0.5
                if dst >= MAX_DIST:
                    neg_idx.append(k)

            for idx in neg_idx:
                X.append(feature_set[i][idx])
                Y.append(0)

    X = np.array(X)
    Y = np.array(Y)

    return X, Y

# FIX for multipe blobs
def _create_training_set_from_feature_set_hdf5(real_blobs, pred_blobs, feature_set, container, chunk_size=10000, suffix=None):
    MAX_DIST = 35
    size = len(real_blobs)

    total_rois = 0
    for i in range(size):
        total_rois += len(pred_blobs[i])
    output_shape = (total_rois,) + feature_set[0][0].shape
    chunk_shape = (min(chunk_size, size),) + feature_set[0][0].shape
    max_shape = (None,) + feature_set[0][0].shape
     
    print(container)

    print "Creating positives ..."
    print('X_{}'.format(suffix))
    X = container.create_dataset("X_{}".format(suffix), output_shape, chunks=chunk_shape, maxshape=max_shape, dtype='float32')
    Y = np.zeros(shape=(total_rois,))

    idx = 0
    for i in range(size):
        if real_blobs[i][2] == -1:
            continue
        nb = []
        for j in range(len(pred_blobs[i])):
            x, y, z = pred_blobs[i][j]#
            dst = ((x - real_blobs[i][0]) ** 2 + (y - real_blobs[i][1]) ** 2) ** 0.5
            if dst <= MAX_DIST:
                nb.append((dst, j))

        nb = sorted(nb)
        if len(nb) == 0:
            continue

        X[idx] = feature_set[i][nb[0][1]]
        Y[idx] = 1
        idx += 1

    print "Creating negatives ..."
    for i in range(size):
        neg_idx = []
        for j in range(len(pred_blobs[i])):
            x, y, z = pred_blobs[i][j]
            dst = ((x - real_blobs[i][0]) ** 2 + (y - real_blobs[i][1]) ** 2) ** 0.5
            if dst > MAX_DIST:
                neg_idx.append(j)

        for j in neg_idx:
            X[idx] = feature_set[i][j]
            idx += 1
    return X, Y

def create_training_set_for_detector(feature_set, pred_blobs, real_blobs):
    MAX_DIST = 35

    size = len(real_blobs)

    X = []
    Y = []

    # create positives
    print "Creating positives ..."
    for i in range(size):
        #print "real {} pred {} ft {}".format(real_blobs[i][2], len(pred_blobs), len(feature_set))
        if real_blobs[i][2] == -1:
            continue
        for j in range(len(pred_blobs[i])):
            x, y, z = pred_blobs[i][j]#
            dst = ((x - real_blobs[i][0]) ** 2 + (y - real_blobs[i][1]) ** 2) ** 0.5
            if dst < MAX_DIST:
                X.append(feature_set[i][j])
                Y.append(1.0 - ((MAX_DIST - dst) * 1.0 / MAX_DIST) ** 2)

    # create negatives
    print "Creating negatives ..."
    for i in range(size):

        neg_idx = []
        for j in range(len(pred_blobs[i])):
            x, y, z = pred_blobs[i][j]
            dst = ((x - real_blobs[i][0]) ** 2 + (y - real_blobs[i][1]) ** 2) ** 0.5
            if dst > MAX_DIST:
                neg_idx.append(j)

        for idx in neg_idx:
            X.append(feature_set[i][idx])
            Y.append(0.0)

    X = np.array(X)
    Y = np.array(Y)
    return X, Y

def train(X, Y, clf, scaler, selector, weights=False):
    iters = 1
    trs = int(0.7 * len(Y))
    tes = int(len(Y) - trs)
    seed = 113

    # hardcoded
    NX = []
    for xi in X:
        NX.append(np.array(xi))
    X = np.array(NX)

    ev = False
    if ev == True:
        folds = np.array(list(cross_val.StratifiedShuffleSplit(Y, iters, train_size=trs, test_size=tes, random_state=seed)))
        tr = folds[0][0]
        te = folds[0][1]
        eval_scaler = clone(scaler)
        if selector != None:
            eval_selector = clone(selector)

        eval_clf = clone(clf)
        Xt_tr = eval_scaler.fit_transform(X[tr])
        if selector != None:
            Xt_tr = eval_selector.fit_transform(Xt_tr, Y[tr])

        eval_clf.fit(Xt_tr, Y[tr])
        Xt_te = eval_scaler.transform(X[te])
        if selector != None:
            Xt_te = eval_selector.transform(Xt_te)
        pred = eval_clf.predict(Xt_te)
        print "Evaluate performance on patches"
        print "Classification report: " 
        print classification_report(Y[te].astype(int), pred.astype(int))
        
    Xt = scaler.fit_transform(X)
    if selector != None:
        print "selecting ..."
        print "before n_feats = {}".format(len(Xt[0]))
        Xt = selector.fit_transform(Xt, Y)
        
        #print "variance ratio sum: {}".format(selector.explained_variance_ratio_.sum())
        print "after n_feats = {}".format(len(Xt[0]))

    clf.fit(Xt, Y)
    
    if weights:
        print 'Ploting weights'

        selector = SelectPercentile(f_classif, percentile=10)
        selector.fit(Xt, Y)
        pvalues = selector.pvalues_
        pvalues[np.isnan(pvalues)] = 1.
        scores = -np.log10(pvalues)
        scores /= (scores.max() + util.EPS)

        weights_selected = (clf.coef_ ** 2).sum(axis=0)
        weights_selected /= weights_selected.max()

        return clf, scaler, selector, np.array([weights_selected, scores])
    else:   
        return clf, scaler, selector

if __name__ == '__main__':
    fname = sys.argv[1]
    data = np.load(fname)
    print "fname {}".format(fname)

    X = data[0]
    Y = data[1]
    clf = LDA()
    scaler = preprocessing.StandardScaler()
    train(X, Y, clf, scaler)

