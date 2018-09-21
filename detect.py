import sys
import os
import numpy as np
from math import sqrt
from skimage.feature import peak_local_max
from skimage.segmentation import find_boundaries


from sklearn.cross_validation import StratifiedKFold
from itertools import *
import cv2
from util import *
#TODO: add detection thinning
import skimage.io as io
import argparse
from operator import itemgetter

import preprocess
import jsrt
import lidc
import model
import eval
import util
import neural
import pickle
from segment import segment_func
from blob import *

from jsrt import DataProvider
import time
FOLDS_SEED = 113

from menpo.image import Image

''' 
Detection utils 
'''
DETECTOR_FPPI_RANGE = np.linspace(0.0, 200.0, 101)
def read_blobs(fname):
    fileh = open(fname, 'rb')
    blobs = pickle.load(fileh)
    fileh.close()
    return blobs

def write_blobs(blobs, fname):
    fileh = open(fname, 'wb')
    pickle.dump(blobs, fileh, -1)
    fileh.close()

def average_bppi(blobs):
    num_images = len(blobs)
    num_blobs = 0
    for i in range(len(blobs)):
        num_blobs += len(blobs[i])
    return num_blobs * 1.0 / num_images

def dst(a, b):
    return sqrt((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]))

def create_graph(points, thold):
    G = {}
    for i in range(len(points)):
        G[i] = set()
        for j in range(len(points)):
            if i != j and dst(points[i], points[j]) < thold:
                G[i].add(j)
    return G
        
def dfs(graph, start, visited):
    stack = [start]
    component = set()
    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.add(vertex)
            component.add(vertex)
            stack.extend(graph[vertex] - visited)
    return component

def detection_thining(points): # 5 mm thold
    G = create_graph(points, 7)
    visited = set()
    comps = []
    resp = []

    for i in range(len(points)):
        if i not in visited:    
            comps.append(dfs(G, i, visited))

    for comp in comps:
        avg = [0.0, 0.0]
        for v in comp:
            avg[0] += points[v][0]
            avg[1] += points[v][1]
        avg[0] /= len(comp)
        avg[1] /= len(comp)
        resp.append(avg)        

    return resp

''' 
Filtering blobs by rules
'''

def filter_by_size(blobs, lower=4, upper=32):
    ans = []
    for blob in blobs:
        x, y, r = blob
        if r >= lower and r <= upper:
            ans.append(blob)
    return np.array(ans)

def filter_by_masks(blobs, mask):
    print mask.shape
    ans = []
    for blob in blobs:
        x, y, r = blob
        found = False
        if mask[x][y] != 0:
            ans.append(blob)
    return np.array(ans)
    
def filter_by_margin(blobs, mask, margin=30):
    ans = []
    for blob in blobs:
        x, y, r = blob
        found = False
        if x > margin and y > margin and x < mask.shape[0] - margin and y < mask.shape[1] - margin:
            ans.append(blob)
    return np.array(ans)

def filter_blobs(blobs, probs, mask, size_range=(4, 32), margin=30):
    filtered_blobs = []
    filtered_probs = []
    shape = mask.shape

    for i in range(len(blobs)):
        x, y, r = blobs[i]
        in_range = (r >= size_range[0]) and (r <= size_range[1])
        in_margin = x > margin and y > margin and x < (shape[0] - margin) and y < (shape[1] - margin)

        if in_range and in_margin and mask[x][y] != 0:
            filtered_blobs.append(blobs[i])
            filtered_probs.append(probs[i])

    return np.array(filtered_blobs), np.array(filtered_probs)

def log_(img, mask, threshold=0.001):
    blobs, probs = blob_log(img, min_sigma=4,  max_sigma=32, num_sigma=10, log_scale=True, threshold=threshold, overlap=0.5)
    if len(blobs) > 0:
        blobs[:, 2] = blobs[:, 2] * sqrt(2)
    return filter_blobs(blobs, probs, mask)

def dog(img, mask, threshold=0.05):
    blobs, probs = blob_dog(img, max_sigma=20, threshold=threshold)
    if len(blobs) > 0:
        blobs[:, 2] = blobs[:, 2] * sqrt(2)
    return filter_blobs(blobs, probs, mask)

def doh(img, mask, threshold=0.0005):
    blobs, probs = blob_doh(1 - img, min_sigma=4, num_sigma=10, max_sigma=30, threshold=threshold)
    if len(blobs) > 0:
        blobs[:, 2] = blobs[:, 2] * sqrt(2)
    return filter_blobs(blobs, probs, mask)

'''
Convergence index based detectors
'''

def wmci(img, mask, threshold=0.5):
    min_distance = 7.143
    ans = preprocess.wmci(img, mask, threshold); 
    coords = peak_local_max(ans, min_distance)

    blobs = []
    for coord in coords:
        if ans[coord[0], coord[1]] >= threshold:
            blobs.append((coord[0], coord[1], 25))

    blobs = filter_by_margin(filter_by_size(filter_by_masks(blobs, mask)), mask)
    return blobs, ans

def wmci_proba(img, mask, threshold=0.5):
    min_distance = 7
    ans = preprocess.wmci(img, mask, threshold); 
    coords = peak_local_max(ans, min_distance)

    blobs = []
    proba = []
    for coord in coords:
        if ans[coord[0], coord[1]] >= threshold:
            blobs.append((coord[0], coord[1], 25))

    blobs = np.array(blobs)
    blobs = filter_by_margin(filter_by_size(filter_by_masks(blobs, mask)), mask)
    for blob in blobs:
        proba.append(ans[blob[0], blob[1]])

    return blobs, ans, np.array(proba)

def sbf(img, mask, threshold=0.5):
    min_distance = 7
    SBF_RANGE = (0, 1200)

    ans, rads = preprocess.sliding_band_filter(img); 
    print 'sbf stats min {}, max {}, mean {}, std {}'.format(ans.min(), ans.max(), ans.mean(), ans.std())
    coords = peak_local_max(ans, min_distance)
    threshold = SBF_RANGE[0] + threshold * (SBF_RANGE[1] - SBF_RANGE[0])
    print threshold

    blobs = []
    proba = []
    for coord in coords:
        if ans[coord[0], coord[1]] >= threshold:
            blobs.append((coord[0], coord[1], rads[coord[0], coord[1]]))

    print 'blobs limits {} {}'.format(np.max(blobs), np.min(blobs))

    blobs = np.array(blobs)
    blobs = filter_by_margin(filter_by_size(filter_by_masks(blobs, mask)), mask)
    for blob in blobs:
        proba.append(ans[blob[0], blob[1]])

    util.imwrite_with_blobs('det-nms', ans, blobs)

    return blobs, ans, np.array(proba)

''' 
Detection using convolutional neural networks
'''

from keras.layers import Convolution2D
from keras.layers import Dense

def copy_weights(from_network, to_network):
    from_layers = []
    for i in range(len(from_network.network.layers)):
        if isinstance(from_network.network.layers[i], Convolution2D) or isinstance(from_network.network.layers[i], Dense):
            from_layers.append(from_network.network.layers[i])
    
    to_layers = []
    for i in range(len(to_network.network.layers)):
        if isinstance(to_network.network.layers[i], Convolution2D) or isinstance(to_network.network.layers[i], Dense):
            to_layers.append(to_network.network.layers[i])

    assert len(from_layers) == len(to_layers)
    
    for i in range(len(from_layers)):
        W, b = from_layers[i].get_weights()
        nb_filter, prev_nb_filter, ax1, ax2 = to_layers[i].get_weights()[0].shape
        new_W = W.reshape((prev_nb_filter, ax1, ax2, nb_filter))
        new_W = new_W.transpose((3,0,1,2))
        new_W = new_W[:,:,::-1,::-1]
        to_layers[i].set_weights([new_W, b])

def detect_with_network(network, imgs, masks, threshold=0.5, fold=-1):
    #imgs = np.array([[imgs[0]]])
    length = len(imgs)
    outs = network.network.predict(imgs)
    print('Prob map output {}'.format(outs.shape))
    util.imwrite('data/probmap_{}.jpg'.format(fold), outs[0,0,:,:])

    side = outs[0][0].shape[0]
    scale_factor = int(512 / side)
    min_distance = int(side / 71.68)
    print 'img size {}, prob size {}, min_distance {}, scale_factor {}'.format(512, side, min_distance, scale_factor)

    blob_set = []
    prob_set = []
    for i in range(length):
        prob_map = outs[i][0]
        coords = peak_local_max(prob_map, min_distance)

        blobs = []
        for coord in coords:
            blobs.append([coord[0] * scale_factor, coord[1] * scale_factor, 25])

        blobs = filter_by_margin(filter_by_size(filter_by_masks(blobs, masks[i])), masks[i])

        print '# coords {}'.format(len(coords))
        print '# before blobs {}'.format(len(blobs))
        threshold = 0
        for blob in blobs:
            threshold += prob_map[blob[0] / scale_factor, blob[1] / scale_factor]
        threshold /= len(coords)

        tmp = []
        for blob in blobs:
            if prob_map[blob[0] / scale_factor, blob[1] / scale_factor] > threshold:
                tmp.append(blob)
        blobs = np.array(tmp)

        print 'after # blobs {}'.format(len(blobs))
        if len(blobs) == 0:
            raise Exception("No blobs founds at {}".format(i))
        blob_set.append(blobs)

        prob_set.append([])
        for blob in blobs:
            prob_set[-1].append(prob_map[blob[0] / scale_factor, blob[1] / scale_factor]) 
        prob_set[-1] = np.array(prob_set[-1])

    return np.array(blob_set), np.array(prob_set)
    
def eval_cnn_detector(data, blobs, augmented_blobs, rois, folds, model):
    fold = 1
    network_init = None
    roi_size=32
    streams = 'none'

    imgs = []
    masks = []
    for i in range(len(data)): 
        img, lung_mask = data.get(i, downsample=True)
        sampled, lce, norm = preprocess.preprocess_hardie(img, lung_mask, downsample=True)
        imgs.append([lce]) 
        masks.append(lung_mask)
    imgs = np.array(imgs)
    masks = np.array(masks)

    # Hardcoding blob set shapes
    blobs2 = blobs
    blobs = blobs.reshape((len(blobs), 3))

    frocs = []
    for tr_idx, te_idx in folds:
        print "Fold {} ...".format(fold)
        X_train, Y_train, X_test, Y_test = neural.create_train_test_sets(rois[tr_idx], augmented_blobs[tr_idx], blobs[tr_idx], 
                                                rois[te_idx], augmented_blobs[te_idx], blobs[te_idx], streams=streams, detector=True)

        network = neural.create_network(model, X_train.shape, fold, streams, detector=False) 
        if network_init is not None:
            network.network.load_weights('data/{}_fold_{}_weights.h5'.format(network_init, fold))
        
        # save network
        name =  'data/{}_fold_{}'.format(model, fold)
        history = network.fit(X_train, Y_train, X_test, Y_test, streams=(streams != 'none'), cropped_shape=(roi_size, roi_size), checkpoint_prefix=name, checkpoint_interval=2, loss='mse')
        network.save(name)

        # open network on detector mode
        network.network.summary()
        detector_network = neural.create_network(model, X_train.shape, fold, streams, detector=True) 
        detector_network.network.summary()
        copy_weights(network, detector_network)
        #network.network.load_weights('{}_weights.h5'.format(name))
        #network.load(name)

        # evaluate network on test
        blobs_te_pred, probs_te_pred = detect_with_network(detector_network, imgs[te_idx], masks[te_idx], fold=fold)

        froc = eval.froc(blobs2[te_idx], blobs_te_pred, probs_te_pred)
        frocs.append(froc)
        fold += 1

    av_froc = eval.average_froc(frocs, fppi_range)
    return av_froc

def froc_by_epochs(data, blobs, augmented_blobs, rois, folds, network_model, nb_epochs=30, epoch_interval=2):
    network_init = None
    roi_size=32
    streams = 'none'

    imgs = []
    masks = []
    for i in range(len(data)): 
        img, lung_mask = data.get(i, downsample=True)
        sampled, lce, norm = preprocess.preprocess_hardie(img, lung_mask, downsample=True)
        imgs.append([lce]) 
        masks.append(lung_mask)
    imgs = np.array(imgs)
    masks = np.array(masks)

    # Hardcoding blob set shapes
    blobs2 = blobs
    blobs = blobs.reshape((len(blobs), 3))

    nb_checkpoints = int(nb_epochs / epoch_interval)
    epochs = np.linspace(epoch_interval, nb_checkpoints * epoch_interval, nb_checkpoints).astype(np.int)

    av_frocs = []
    names = []
    aucs1 = []
    aucs2 = []
    for epoch in epochs:
        frocs = []
        fold = 1
        for tr_idx, te_idx in folds:
            print "Fold {} ...".format(fold)
            X_train, Y_train, X_test, Y_test = neural.create_train_test_sets(rois[tr_idx], augmented_blobs[tr_idx], blobs[tr_idx], 
                                                    rois[te_idx], augmented_blobs[te_idx], blobs[te_idx], streams=streams, detector=True)

            # load network
            network = neural.create_network(network_model, X_train.shape, fold, streams, detector=False) 
            name =  'data/{}_fold_{}.epoch_{}'.format(network_model, fold, epoch)
            network.network.load_weights('{}_weights.h5'.format(name))

            # open network on detector mode
            detector_network = neural.create_network(network_model, X_train.shape, fold, streams, detector=True) 
            copy_weights(network, detector_network)

            # evaluate network on test
            blobs_te_pred, probs_te_pred = detect_with_network(detector_network, imgs[te_idx], masks[te_idx], fold=fold)

            froc = eval.froc(blobs2[te_idx], blobs_te_pred, probs_te_pred)
            frocs.append(froc)
            fold += 1

        names.append('{}, epoch {}'.format(network_model, epoch))
        ops = eval.average_froc(frocs, fppi_range)
        av_frocs.append(ops)
        aucs1.append(util.auc(ops, range(0, 60)))
        aucs2.append(util.auc(ops, range(0, 40)))
        util.save_auc(np.array(range(1, len(aucs1)+1)) * epoch_interval, aucs1, 'data/{}-auc-0-60'.format(network_model))
        util.save_auc(np.array(range(1, len(aucs2)+1)) * epoch_interval, aucs2, 'data/{}-auc-0-40'.format(network_model))
 
    return av_frocs, names

def eval_models(model_instance, network_set, save_history=True):
    paths, locs, rads, subs, sizes, kinds = jsrt.jsrt(set='jsrt140')
    left_masks = jsrt.left_lung(set='jsrt140')
    right_masks = jsrt.right_lung(set='jsrt140')
    data = DataProvider(paths, left_masks, right_masks)

    size = len(paths)
    blobs = []
    for i in range(size):
        blobs.append([[locs[i][0], locs[i][1], rads[i]]])
    blobs = np.array(blobs)
 
    methods = [] #['wmci']
    thresholds = [0.5]
    frocs = []
    legend = []

    folds = StratifiedKFold(subs, n_folds=10, shuffle=True, random_state=FOLDS_SEED)
    for i in range(len(methods)):
        pred_blobs, proba = detect_blobs_with_dataprovider(data, methods[i], thresholds[i])
        frocs.append(util.froc_folds(blobs, pred_blobs, proba, folds))
        legend.append('{}, t={}'.format(methods[i], thresholds[i]))
    
    augmented_blobs = add_random_blobs(data, blobs, blobs_by_image=512, rng=np.random)
    rois = model_instance.create_rois(data, augmented_blobs, model_instance.downsample)
    #train_cnn_detector(data, blobs, augmented_blobs, rois, folds)
    for network_name in network_set:
        frocs.append(eval_cnn_detector(data, blobs, augmented_blobs, rois, folds, model=network_name))
        legend.append(network_name)
        util.save_froc(frocs, 'data/cmp-detectors-10-epochs', legend, with_std=False, fppi_max=fppi_range[-1])

def eval_network_by_epoch(model_instance, network_name, save_history=True):
    paths, locs, rads, subs, sizes, kinds = jsrt.jsrt(set='jsrt140')
    left_masks = jsrt.left_lung(set='jsrt140')
    right_masks = jsrt.right_lung(set='jsrt140')
    data = DataProvider(paths, left_masks, right_masks)

    size = len(paths)
    blobs = []
    for i in range(size):
        blobs.append([[locs[i][0], locs[i][1], rads[i]]])
    blobs = np.array(blobs)
 
    methods = [] #['wmci']
    thresholds = [0.5]
    frocs = []
    legend = []

    folds = StratifiedKFold(subs, n_folds=10, shuffle=True, random_state=FOLDS_SEED)
    for i in range(len(methods)):
        pred_blobs, proba = detect_blobs_with_dataprovider(data, methods[i], thresholds[i])
        frocs.append(util.froc_folds(blobs, pred_blobs, proba, folds))
        legend.append('{}, t={}'.format(methods[i], thresholds[i]))

    
    augmented_blobs = add_random_blobs(data, blobs, blobs_by_image=512, rng=np.random)
    rois = model_instance.create_rois(data, augmented_blobs, model_instance.downsample)
    #train_cnn_detector(data, blobs, augmented_blobs, rois, folds)
    frocs, legend = froc_by_epochs(data, blobs, augmented_blobs, rois, folds, network_model=network_name)
    util.save_froc(frocs, 'data/{}-by-epoch'.format(network_name), legend, with_std=False, fppi_max=fppi_range[-1])

''' 
Detect function for LIDC-JSRT !
'''

def detect_blobs_(image, mask, threshold, method):
    #lce = preprocess.lce(image)
    blobs, ci, proba = generic_detect_blobs(image, mask, threshold, method)
    return blobs, proba

def detect_blobs(images, masks, threshold=0.5, real_blobs=None, method='wmci'):
    assert len(images) == len(masks)
    blob_set = []
    prob_set = []

    print "detect blobs with probs ..."
    print '[',

    for i in range(len(images)):
        if i % (len(images)/10) == 0:
            print ".",
            sys.stdout.flush()
        
        lce = preprocess.lce(images[i][0])
        blobs, ci, proba = generic_detect_blobs(lce, masks[i], threshold, method)
        print ci.min(), ci.max(), ci.mean(), ci.std()

        blob_set.append(blobs)
        prob_set.append(proba)
    '''
    # Measure tp, av fp per image
    fppi = np.full((len(images),), fill_value=0, dtype=float)
    p = 0
    tp = 0
    for i in range(len(images)):
        p += len(real_blobs[i])
        tmp = tp
        for real_blob in real_blobs[i]:
            found = False
            for pred_blob in blob_set[i]:
                dist = ((real_blob[0] - pred_blob[0]) ** 2 + (real_blob[1] - pred_blob[1]) ** 2)**0.5
                if dist < 31.43:
                    found = True
            if found:
                tp += 1

        print "{} -> # nodules {}, # missed nodules {}".format(i, len(real_blobs[i]), len(real_blobs[i]) - (tp - tmp))
        for pred_blob in blob_set[i]:
            found = False
            for real_blob in real_blobs[i]:
                dist = ((real_blob[0] - pred_blob[0]) ** 2 + (real_blob[1] - pred_blob[1]) ** 2)**0.5
                if dist < 31.43:
                    found = True
            if not found:
                fppi[i] += 1 

    print "sensitivity {} {} {}".format(tp, p, tp * 1.0 / p)
    '''

    '''
    plt.violinplot(np.array([np.array(real_cis), np.array(pred_cis)]), showmeans=True, showmedians=True)
    plt.ylim((-1.0, 1.5))
    ax = plt.gca()
    ax.set_xticklabels(['', 'real', '', 'pred'])
    plt.show()
    '''

    print ']'
    #print '-> average of min, max, mean, std on ci imgs {}, {}, {}, {}'.format(np.mean(mins), np.mean(maxs), np.mean(means), np.mean(stds))

    return np.array(blob_set), np.array(prob_set)

'''
Generic blob detection function function
'''

def generic_detect_blobs(img, lung_mask, threshold=0.5, method='wmci'):
    print img.shape
    #util.imwrite_as_pdf('det-input2', image)
    sampled, lce, norm = preprocess.preprocess_hardie(img, lung_mask)
    blobs = None
    probs = np.array([])
    ci = norm
    lce = lce.astype(np.float64)
    if method == 'wmci':
        blobs, ci, probs = wmci_probs(lce, lung_mask, threshold)
    elif method == 'sbf':
        #util.imwrite_as_pdf('det-lce', lce)
        blobs, ci, probs = sbf(lce, lung_mask, threshold)
        #util.imwrite_as_pdf('det-sbf', ci) 
    elif method == 'log':
        blobs, probs = log_(lce, lung_mask, threshold)
    elif method == 'dog':
        blobs, probs = dog(lce, lung_mask, threshold)
    elif method == 'doh':
        blobs, probs = doh(lce, lung_mask, threshold)
    else:
        raise Exception("Undefined detection method")
    return blobs, ci, probs

def detection_vs_distance(method, args):
    images, blobs = lidc.load()
    masks = np.load('data/aam-lidc-pred-masks.npy')
    pred_blobs, probs = detect_blobs(images, masks, args.threshold, real_blobs=blobs, method=method)
    write_blobs(pred_blobs, 'data/{}-{}-aam-lidc-pred-blobs.pkl'.format(method, args.threshold))

    froc = eval.froc(blobs, pred_blobs, probs, distance='rad')
    froc = eval.average_froc([froc], DETECTOR_FPPI_RANGE)

def save_blobs(method, args):
    images, blobs = lidc.load()
    masks = np.load('data/aam-lidc-pred-masks.npy')
    print 'masks shape {}'.format(masks.shape)
    pred_blobs, probs = detect_blobs(images, masks, args.threshold, real_blobs=blobs, method=method)
    write_blobs(pred_blobs, 'data/{}-{}-aam-lidc-pred-blobs.pkl'.format(method, args.threshold))

    froc = eval.froc(blobs, pred_blobs, probs, distance='rad')
    froc = eval.average_froc([froc], DETECTOR_FPPI_RANGE)
    abpi = average_bppi(pred_blobs)

    util.save_froc([froc], 'data/{}-{}-lidc-blobs-froc'.format(method, args.threshold), ['{} FROC on LIDC dataset'.format(method)], fppi_max=100)
    np.savetxt('data/{}-{}-lidc-blobs-abpi.txt'.format(method, args.threshold), np.array([abpi]))
    print('Average blobs per image LIDC {}'.format(abpi))

    images, blobs = jsrt.load(set_name='jsrt140p')
    masks = np.load('data/aam-jsrt140p-pred-masks.npy')
    pred_blobs, probs = detect_blobs(images, masks, args.threshold, real_blobs=blobs, method=method) 
    write_blobs(pred_blobs, 'data/{}-{}-aam-jsrt140p-pred-blobs.pkl'.format(method, args.threshold))

    froc = eval.froc(blobs, pred_blobs, probs, distance='rad')
    froc = eval.average_froc([froc], DETECTOR_FPPI_RANGE)
    abpi = average_bppi(pred_blobs)

    util.save_froc([froc], 'data/{}-{}-jsrt140p-blobs-froc'.format(method, args.threshold), ['{} FROC on JSRT140 positives dataset'.format(method)], fppi_max=100)
    np.savetxt('data/{}-{}-jsrt140p-blobs-abpi.txt'.format(method, args.threshold), np.array([abpi]))
    print('Average blobs per image JSRT positives {}'.format(abpi))

def detect_func(image, detector, segmentator, threshold):
    mask = segment_func(image, segmentator, display=False)
    blobs, probs = detect_blobs_(image, mask[0], threshold, detector)
    img = cv2.resize(image, (512, 512), interpolation=cv2.INTER_CUBIC)
    imwrite_with_blobs('data/detected', img, blobs)
    return blobs, probs, mask

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='detect.py')
    parser.add_argument('file', nargs='?', default=None, type=str)
    parser.add_argument('---blobs', help='Use the detector and segmentator to generate blobs', action='store_true')
    parser.add_argument('--detector', help='Method used to extract candidates', type=str, default='sbf')
    parser.add_argument('--threshold', help='threshold', default=0.5, type=float)
    parser.add_argument('--segmentator', help='Method used to segment lung area used on candidate filtering', type=str, default='aam')
    parser.add_argument('--mode', help='', default='models', type=str)
    parser.add_argument('--roi-size', help='Roi size', default=32, type=int)
    args = parser.parse_args()
 
    model_instance = model.BaselineModel()
    #model_instance.extractor = model.extractors[args.descriptor]
    model_instance.preprocess_lung = 'lce'
    model_instance.preprocess_roi = 'none'
    model_instance.roi_size = args.roi_size
    model_instance.use_transformations = False # args.trf_channels
    model_instance.streams = 'none'
    model_instance.label = 'nodule'
    model_instance.augment = 'bt'
    model_instance.downsample = True

    '''
    network_set = ['d1p_a', 'd2p_a', 'd3p_a',
                   'd1p_b', 'd2p_b', 'd3p_b',
                   'd1p_c', 'd2p_c', 'd3p_c']
    '''

    t = time.time()
    network_set = ['d2p_a']
    if args.file:
        image = np.load(args.file).astype('float32')
        detect_func(image, args.detector, args.segmentator, args.threshold)
    elif args.save_blobs:
        save_blobs(args.detector, args)
    elif args.mode == 'models':
        eval_models(model_instance, network_set)
    elif args.mode == 'epochs':
        eval_network_by_epoch(model_instance, network_set[0])
    print 'finished in {:.2f} secs'.format(time.time() - t)
