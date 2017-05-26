#!/usr/bin/env python

import sys
sys.setrecursionlimit(10000)
import time
from itertools import product
import os
from os import path
import argparse

import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from sklearn.cross_validation import StratifiedKFold
from sklearn import svm
from sklearn import lda
from sklearn import decomposition
from sklearn import feature_selection as selection
from sklearn import ensemble
from sklearn import preprocessing
from sklearn.metrics import auc

import baseline
import model
import eval
import util
import jsrt
from jsrt import DataProvider
import neural
import bovw
import augment
import detect
import segment
from segment import MeanShape

import pdb

# Globals
FOLDS_SEED = 113
step = 10
fppi_range = np.linspace(0.0, 10.0, 101)

def get_froc_on_folds(detection_model, paths, left_masks, right_masks, blobs, pred_blobs, feats, folds, save_fw=False, model_name='undefined'):
    print 'feature shape {}'.format(feats[0].shape)
    fold = 0
    valid = True
    frocs = []
    feature_weights = []
    tr_val_folds, tr_val, te = folds
    for tr_idx, te_idx in tr_val_folds:    
        print "Fold {}".format(fold + 1),
        #data_te = DataProvider(paths[te_idx], left_masks[te_idx], right_masks[te_idx])
        paths_te = paths[te_idx]

        #blobs_te = blobs[te_idx].reshape((1,) + blobs.shape).swapaxes(0, 1)
        for bl in blobs[te_idx]:
            blobs_te.append([bl])
        blobs_te = np.array(blobs_te)

        print 'Train with {}...'.format(detection_model.clf)
        ret = detection_model.train_with_feature_set(feats[tr_idx], pred_blobs[tr_idx], blobs[tr_idx], save_fw)

        print 'Predict ...'
        blobs_te_pred, probs_te_pred = detection_model.predict_proba_from_feature_set(feats[te_idx], pred_blobs[te_idx])

        print 'Get froc ...'
        froc = eval.froc(blobs_te, blobs_te_pred, probs_te_pred)

        if save_fw == True:
            print 'w at fold -> {}'.format(ret[3].shape)
            feature_weights.append(ret[3])

        frocs.append(froc)
        sys.stdout.flush()
        fold += 1
    
    if save_fw == True:
        feature_weights = np.array(feature_weights)
        util.save_weights(feature_weights, '{}_fw'.format(model_name))

    av_froc = eval.average_froc(frocs, fppi_range)
    return av_froc

def _split_rois(rois, tr_idx, te_idx, streams):
    rois_tr = []
    rois_te = []
    if detection_model.streams != 'none':
        for i in range(len(rois)):
            rois_tr.append(rois[i][tr_idx])
            rois_te.append(rois[i][te_idx])
    else:
        rois_tr = rois[tr_idx]
        rois_te = rois[te_idx]
    return rois_tr, rois_te

def froc_with_model(detection_model, network_model, rois_tr, pred_blobs_tr, blobs_tr, pred_blobs_te, blobs_te, data_te, model_suffix, network_init=None, rois_te=None):
    if rois_te == None:
        rois_te = detection_model.create_rois(data_te, pred_blobs_te)

    history = detection_model.train_with_feature_set_keras(rois_tr, pred_blobs_tr, 
        blobs_tr, rois_te, 
        pred_blobs_te, blobs_te, 
        model=network_model, model_suffix=model_suffix, 
        network_init=network_init)

    blobs_te_pred, probs_te_pred = detection_model.predict_proba_from_feature_set_keras(rois_te, pred_blobs_te)
    froc = eval.froc(blobs_te, blobs_te_pred, probs_te_pred)

    detection_model.save('data/{}_{}'.format(network_model, model_suffix))

    return froc
 
def get_froc_on_folds_keras(detection_model, paths, left_masks, right_masks, blobs, pred_blobs, rois, folds, network_model, network_init=None, mode=None):
    fold = 1
    valid = True
    frocs = []
    frocs_ = []
    tr_val_folds, tr_val, te = folds 
    print "tr val {}".format(len(tr_val))

    for tr_idx, te_idx in tr_val_folds:    
        print "Fold {}".format(fold),
        print "len tr {}, len te {}".format(len(tr_idx), len(te_idx))
        rois_tr, _ = _split_rois(rois, tr_idx, te_idx, detection_model.streams)
        pred_blobs_te = detect.read_blobs('data/{}-blobs-f{}.pkl'.format(detection_model.args.blob_detector, fold))
        data_te = DataProvider(paths[te_idx], left_masks[te_idx], right_masks[te_idx])
        assert len(data_te) == len(pred_blobs_te), 'len(data_te) == len(pred_blobs_te), but {} != {}'.format(len(data_te), len(pred_blobs_te))

        froc = froc_with_model(detection_model, network_model, rois_tr, pred_blobs[tr_idx], blobs[tr_idx], pred_blobs_te, blobs[te_idx], data_te, model_suffix='fold_{}'.format(fold))
        frocs.append(froc)
        frocs_.append(eval.average_froc([froc], np.linspace(0.0, 10.0, 101))) 

        legend_ = ['Fold {}'.format(i + 1) for i in range(len(frocs))] 
        util.save_froc(frocs_, 'data/{}_froc_folds'.format(network_model), legend_)

        fold += 1

    froc_on_val_sets = eval.average_froc(frocs, np.linspace(0.0, 10.0, 101))
    rois_tr_val, _ = _split_rois(rois, tr_val, te, detection_model.streams)
    pred_blobs_te = detect.read_blobs('data/{}-blobs-te.pkl'.format(detection_model.args.blob_detector, fold))
    data_te = DataProvider(paths[te], left_masks[te], right_masks[te])
    rois_te = detection_model.create_rois(data_te, pred_blobs_te)

    frocs_on_test_sets = []

    if detection_model.args.bootstrap:
        history = detection_model.train_with_feature_set_keras(rois_tr_val, pred_blobs[tr_val], 
            blobs[tr_val], rois_te, 
            pred_blobs_te, blobs[te], 
            model=network_model, model_suffix="tr_val", 
            network_init=network_init)

        blobs_te_pred, probs_te_pred = detection_model.predict_proba_from_feature_set_keras(rois_te, pred_blobs_te)
        idxs = util.bootstrap_sets(len(rois_te), num_sets=1000)

        for i in range(len(idxs)):
            blobs_bt = (blobs[te])[idxs[i]]
            blobs_bt_pred = blobs_te_pred[idxs[i]]
            probs_bt_pred = probs_te_pred[idxs[i]]
            froc = eval.froc(blobs_bt, blobs_bt_pred, probs_bt_pred, verbose=False)
            frocs_on_test_sets.append(froc)
    else: 
        for i in range(10):
            froc = froc_with_model(detection_model, network_model, rois_tr_val, pred_blobs[tr_val], blobs[tr_val], pred_blobs_te, blobs[te], data_te, model_suffix='tr_val', rois_te=rois_te)
            frocs_on_test_sets.append(froc)

    froc_on_test_set, ci_low, ci_up = eval.average_froc_with_ci(frocs_on_test_sets, np.linspace(0.0, 10.0, 101)) 
    print "CI low {} up {}".format(ci_low, ci_up)
    legend_ = ['{} val sets'.format(network_model), '{} test set, CI 95% = [{:.2f} - {:.2f}]'.format(network_model, ci_low, ci_up)]
    print network_model, ci_low, ci_up
    util.save_froc([froc_on_val_sets, froc_on_test_set], 'data/{}_froc_val_test'.format(network_model), legend_, with_std=True)
    return froc_on_test_set

def bovw_folds(detection_model, fname, config, save_fw=False):
    # DATA
    paths, locs, rads, subs, sizes, kinds = jsrt.jsrt(set='jsrt140')
    left_masks = jsrt.left_lung(set='jsrt140')
    right_masks = jsrt.right_lung(set='jsrt140')
    size = len(paths)

    blobs = []
    for i in range(size):
        blobs.append([locs[i][0], locs[i][1], rads[i]])
    blobs = np.array(blobs)
    print "Loading blobs & features ..."
    data = DataProvider(paths, left_masks, right_masks)
    pred_blobs = np.load('data/{}_pred.blb.npy'.format(fname))
    rois = detection_model.create_rois(data, pred_blobs)

    av_cpi = 0
    for tmp in pred_blobs:
        av_cpi += len(tmp)
    print "Average blobs per image {} ...".format(av_cpi * 1.0 / len(pred_blobs))

    Y = (140 > np.array(range(size))).astype(np.uint8)
    folds = StratifiedKFold(subs, n_folds=10, shuffle=True, random_state=FOLDS_SEED)
    print "save_fw {}".format(save_fw)

    fold = 1
    valid = True
    frocs = []
    frocs_ = []
    for tr_idx, te_idx in folds:    
        print "Fold {}".format(fold),
            
        data_te = DataProvider(paths[te_idx], left_masks[te_idx], right_masks[te_idx])
        paths_te = paths[te_idx]

        rois_tr, rois_te = _split_rois(rois, tr_idx, te_idx, detection_model.streams)
        
        bovwdetection_model = bovw.createdetection_model(config)
        V_tr, V_te = detection_model.fit_transform_bovw(rois_tr, pred_blobs[tr_idx], blobs[tr_idx], rois_te, pred_blobs[te_idx], blobs[te_idx], model=bovwdetection_model)
        util.save_dataset(V_tr, pred_blobs[tr_idx], blobs[tr_idx], V_te, pred_blobs[te_idx], blobs[te_idx], 'data/{}-fold-{}'.format(config, fold))
        fold += 1

def classify_foldwise(detection_model, fname, config, save_fw=False, save_froc=True):
    paths, locs, rads, subs, sizes, kinds = jsrt.jsrt(set='jsrt140')
    left_masks = jsrt.left_lung(set='jsrt140')
    right_masks = jsrt.right_lung(set='jsrt140')
    size = len(paths)

    blobs = []
    for i in range(size):
        blobs.append([locs[i][0], locs[i][1], rads[i]])
    blobs = np.array(blobs)

    print "Loading blobs & features ..."
    data = DataProvider(paths, left_masks, right_masks)
    feats = np.load('data/{}.fts.npy'.format(fname))
    
    pred_blobs = np.load('data/{}_pred.blb.npy'.format(fname))

    av_cpi = 0
    for tmp in pred_blobs:
        av_cpi += len(tmp)
    print "Average blobs per image {} ...".format(av_cpi * 1.0 / len(pred_blobs))

    Y = (140 > np.array(range(size))).astype(np.uint8)
    folds = StratifiedKFold(subs, n_folds=10, shuffle=True, random_state=FOLDS_SEED)

    fold = 1
    valid = True
    frocs = []
    feature_weights = []

    for tr_idx, te_idx in folds:    
        print "Fold {}".format(fold),
        feats_tr, pred_blobs_tr, blobs_tr, feats_te, pred_blobs_te, blobs_te = util.load_dataset('data/{}-fold-{}'.format(config, fold))

        blobs_te2 = []
        for bl in blobs_te:
            blobs_te2.append([bl])
        blobs_te2 = np.array(blobs_te2)

        print 'Train with {}...'.format(detection_model.clf)
        ret = detection_model.train_with_feature_set(feats_tr, pred_blobs_tr, blobs_tr)

        print 'Predict ...'
        blobs_te_pred, probs_te_pred = detection_model.predict_proba_from_feature_set(feats_te, pred_blobs_te)

        print 'Get froc ...'
        froc = eval.froc(blobs_te2, blobs_te_pred, probs_te_pred)

        if save_fw == True:
            print 'w at fold -> {}'.format(ret[3].shape)
            feature_weights.append(ret[3])

        frocs.append(froc)
        sys.stdout.flush()
        fold += 1
    
    if save_fw == True:
        feature_weights = np.array(feature_weights)
        util.save_weights(feature_weights, '{}_fw'.format(model_name))

    ops = eval.average_froc(frocs, fppi_range)

    step=1
    range_ops = ops[step * 2:step * 4 + 1]
    print 'auc 2 - 4 fppis {}'.format(auc(range_ops.T[0], range_ops.T[1]))
    
    if save_froc == True:
        legend = []
        legend.append('Hardie et al.')
        legend.append(detection_model.name)
        util.save_froc([baseline.hardie, ops], '{}'.format(detection_model.name), legend, with_std=True)

    return ops

def extract_features_cnn(detection_model, fname, network_model, layer):
    print "Extract cnn features"
    paths, locs, rads, subs, sizes, kinds = jsrt.jsrt(set='jsrt140')
    left_masks = jsrt.left_lung(set='jsrt140')
    right_masks = jsrt.right_lung(set='jsrt140')

    size = len(paths)

    blobs = []
    for i in range(size):
        blobs.append([locs[i][0], locs[i][1], rads[i]])
    blobs = np.array(blobs)

    print "Loading blobs ..."
    data = DataProvider(paths, left_masks, right_masks)
    pred_blobs = np.load('data/{}_pred.blb.npy'.format(fname))
    rois = detection_model.create_rois(data, pred_blobs)

    Y = (140 > np.array(range(size))).astype(np.uint8)
    folds = StratifiedKFold(subs, n_folds=10, shuffle=True, random_state=FOLDS_SEED)

    fold = 1
    valid = True
    frocs = []
    for tr_idx, te_idx in folds:    
        print "Fold {}".format(fold),
        data_te = DataProvider(paths[te_idx], left_masks[te_idx], right_masks[te_idx])
        paths_te = paths[te_idx]
        blobs_te = []

        for bl in blobs[te_idx]:
            blobs_te.append([bl])
        blobs_te = np.array(blobs_te)

        print 'load model ...'
        detection_model.load_cnn('data/{}_fold_{}'.format(network_model, fold))
        if fold == 1:
            detection_model.network.network.summary()
        print 'extract ...'
        network_feats = detection_model.extract_features_from_kerasdetection_model(rois, layer)
        print 'save ...'
        np.save('data/{}_l{}_fold_{}.fts.npy'.format(network_model, layer, fold), network_feats)
        fold += 1

def froc_classify_cnn(detection_model, paths, left_masks, right_masks, blobs, pred_blobs, rois, folds, network_model, subs=None, sizes=None, kinds=None, ci=False):
    fold = 1
    valid = True
    frocs = []
    frocs_subs = []
    frocs_sizes = []
    frocs_kinds = []
    frocs_for_ci = []
    lens = [len(set(subs)), len(set(sizes)), len(set(kinds))]

    # Bootstrap confidence interval
    n_bootstraps = 1000
    rng_seed = 13
    rng = np.random.RandomState(rng_seed)
    
    for tr_idx, te_idx in folds:    
        print "Fold {}".format(fold),
        data_te = DataProvider(paths[te_idx], left_masks[te_idx], right_masks[te_idx])
        paths_te = paths[te_idx]
        blobs_te = []

        for bl in blobs[te_idx]:
            blobs_te.append([bl])
        blobs_te = np.array(blobs_te)

        print 'load model {} ...'.format('data/{}_fold_{}'.format(network_model, fold))
        detection_model.load_cnn('data/{}_fold_{}'.format(network_model, fold))

        if fold == 1:
            detection_model.network.network.summary()

        print "predict ..."
        rois_te = []
        if detection_model.streams != 'none':
            for i in range(len(rois)):
                rois_te.append(rois[i][te_idx])
        else:
            rois_te = rois[te_idx]
 
        blobs_te_pred, probs_te_pred = detection_model.predict_proba_from_feature_set_keras(rois_te, pred_blobs[te_idx])
        
        print "eval ..."
        froc = []
        if detection_model.streams != 'none':
            froc = eval.froc(blobs_te, blobs_te_pred, probs_te_pred, rois_te[1], data_te, te_idx)
        else:
            froc = eval.froc(blobs_te, blobs_te_pred, probs_te_pred, rois_te, data_te, te_idx)

        frocs.append(froc)
        frocs_subs.append(eval.froc_stratified(blobs_te, blobs_te_pred, probs_te_pred, subs[te_idx], lens[0]))
        frocs_sizes.append(eval.froc_stratified(blobs_te, blobs_te_pred, probs_te_pred, sizes[te_idx], lens[1]))
        frocs_kinds.append(eval.froc_stratified(blobs_te, blobs_te_pred, probs_te_pred, kinds[te_idx], lens[2]))
        fold += 1

        frocs_for_ci.append([])
        if ci:
            for i in range(n_bootstraps):
                idx = rng.random_integers(0, len(te_idx) - 1, len(te_idx))

                data_te = DataProvider(paths[idx], left_masks[idx], right_masks[idx])
                blobs_te = []
                for bl in blobs[idx]:
                    blobs_te.append([bl])
                blobs_te = np.array(blobs_te)

                rois_te = []
                if detection_model.streams != 'none':
                    rois_te = rois[0][idx]
                else:
                    rois_te = rois[idx]

                blobs_te_pred, probs_te_pred = detection_model.predict_proba_from_feature_set_keras(rois_te, pred_blobs[idx])
                froc = eval.froc(blobs_te, blobs_te_pred, probs_te_pred, rois_te, data_te, idx)
                frocs_for_ci[-1].append(froc)

    av_froc = eval.average_froc(frocs, np.linspace(0.0, 10.0, 101))
    av_froc_subs = []
    av_froc_sizes = []
    av_froc_kinds = []

    print "frocs_subs ..."

    frocs_subs = np.array(frocs_subs).swapaxes(0, 1)
    for i in range(lens[0]):
        av_froc_subs.append(eval.average_froc(frocs_subs[i], np.linspace(0.0, 10.0, 101)))

    frocs_sizes = np.array(frocs_sizes).swapaxes(0, 1)
    for i in range(lens[1]):
        av_froc_sizes.append(eval.average_froc(frocs_sizes[i], np.linspace(0.0, 10.0, 101)))

    frocs_kinds = np.array(frocs_kinds).swapaxes(0, 1)
    for i in range(lens[2]):
        av_froc_kinds.append(eval.average_froc(frocs_kinds[i], np.linspace(0.0, 10.0, 101)))

    if ci:
        frocs_for_ci = np.array(frocs_for_ci)
        frocs_for_ci = np.swap_axes(frocs_for_ci, 0, 1)

        assert len(frocs_for_ci) == n_bootstraps

        bootstrapped_aucs = []
        for i in range(n_bootstraps):
            av_froc = eval.average_froc(frocs_for_ci[i], np.linspace(0.0, 10.0, 101))
            auc_ = auc(np.array(av_froc), range=(0.0, 10.0))
            bootstrapped_aucs.append(auc_)
            
        sorted_scores = np.array(bootstrapped_aucs)
        sorted_scores.sort()
        confidence_lower = sorted_scores[int(0.05 * len(sorted_scores))]
        confidence_upper = sorted_scores[int(0.95 * len(sorted_scores))]

        auc_ = auc(np.array(av_froc), range=(0.0, 10.0))
        return av_froc , np.array(av_froc_subs), np.array(av_froc_sizes), np.array(av_froc_kinds), (auc_, confidence_lower, confidence_upper)
    else:
        return av_froc , np.array(av_froc_subs), np.array(av_froc_sizes), np.array(av_froc_kinds)


def froc_by_epoch(detection_model, paths, left_masks, right_masks, blobs, pred_blobs, rois, folds, network_model, epoch):
    tr_val_folds, tr_val, te = folds 
    fold = 1
    valid = True
    frocs = []
    for tr_idx, te_idx in tr_val_folds:
        print "Fold {}".format(fold),
        rois_tr, _ = _split_rois(rois, tr_idx, te_idx, detection_model.streams)
        pred_blobs_te = detect.read_blobs('data/{}-blobs-f{}.pkl'.format(detection_model.args.blob_detector, fold))
        data_te = DataProvider(paths[te_idx], left_masks[te_idx], right_masks[te_idx])
        assert len(data_te) == len(pred_blobs_te)

        name = 'data/{}_fold_{}.epoch_{}'.format(network_model, fold,epoch)
        print 'load model {} ...'.format(name)
        detection_model.load_cnn('data/{}_fold_{}'.format(network_model, fold))
        detection_model.load_cnn_weights(name)
        
        print "predict ..."
        rois_te = detection_model.create_rois(data_te, pred_blobs_te)
        blobs_te_pred, probs_te_pred = detection_model.predict_proba_from_feature_set_keras(rois_te, pred_blobs_te)
        froc = eval.froc(blobs[te_idx], blobs_te_pred, probs_te_pred)
        frocs.append(froc)
        fold += 1

    av_froc = eval.average_froc(frocs, np.linspace(0.0, 10.0, 101))
    return av_froc

def frocs_by_epoch(detection_model, fname, network_model):
    print "classify with cnn"
    paths, locs, rads, subs, sizes, kinds = jsrt.jsrt(set='jsrt140')
    left_masks = jsrt.left_lung(set='jsrt140')
    right_masks = jsrt.right_lung(set='jsrt140')

    size = len(paths)

    blobs = []
    for i in range(size):
        blobs.append([locs[i][0], locs[i][1], rads[i]])
    blobs = np.array(blobs)

    print "Loading blobs ..."
    data = DataProvider(paths, left_masks, right_masks)
    pred_blobs = detect.read_blobs('data/{}-blobs-gt.pkl'.format(detection_model.args.blob_detector))
    rois = detection_model.create_rois(data, pred_blobs)

    Y = (140 > np.array(range(size))).astype(np.uint8)
    folds = util.stratified_kfold_holdout(subs, n_folds=5)

    legends = []
    legends.append('Hardie et al.')
    frocs = []
    frocs.append(baseline.hardie)

    epoch_interval = 2
    aucs1 = []
    aucs2 = []
    epoch = 0
    while True:
        epoch += epoch_interval
        print 'data/{}.fold-1.epoch-{}.h5'.format(network_model, epoch)
        if path.isfile('data/{}_fold_1.epoch_{}_weights.h5'.format(network_model, epoch)):
            ops = froc_by_epoch(detection_model, paths, left_masks, right_masks, blobs, pred_blobs, rois, folds, network_model, epoch)
            frocs.append(ops)
            legends.append('{}, epoch {}'.format(network_model, epoch))
            aucs1.append(util.auc(ops, range(2, 4)))
            aucs2.append(util.auc(ops, range(0, 5)))
        else:
            break

    util.save_froc(frocs, 'data/{}-on-epochs'.format(network_model), legends, with_std=False, use_markers=False)
    util.save_auc(np.array(range(1, len(aucs1)+1)) * epoch_interval, aucs1, 'data/{}-auc-2-4'.format(network_model))
    util.save_auc(np.array(range(1, len(aucs2)+1)) * epoch_interval, aucs2, 'data/{}-auc-0-5'.format(network_model))
    return frocs

def pretrain_cnn(detection_model, fname, network_model, init_name=None, nb_epoch=1):
    print "pretrain with cnn"
    paths, locs, rads, subs, sizes, kinds = jsrt.jsrt(set='jsrt140')
    left_masks = jsrt.left_lung(set='jsrt140')
    right_masks = jsrt.right_lung(set='jsrt140')

    size = len(paths)

    blobs = []
    for i in range(size):
        blobs.append([locs[i][0], locs[i][1], rads[i]])
    blobs = np.array(blobs)

    print "Loading blobs ..."
    data = DataProvider(paths, left_masks, right_masks)
    pred_blobs = np.load('data/{}_pred.blb.npy'.format(fname))
    rois = detection_model.create_rois(data, pred_blobs)

    Y = (140 > np.array(range(size))).astype(np.uint8)
    folds = StratifiedKFold(subs, n_folds=10, shuffle=True, random_state=FOLDS_SEED)

    fold = 1
    for tr_idx, te_idx in folds:
        data_tr = DataProvider(paths[tr_idx], left_masks[tr_idx], right_masks[tr_idx])
        #data_te = DataProvider(paths[te_idx], left_masks[te_idx], right_masks[te_idx])
         
        X = util.extract_random_rois(data_tr, (detection_model.roi_size, detection_model.roi_size))
        print 'Pretraining Fold {}'.format(fold)
        print 'Pretraining dataset len: {} ...'.format(len(X))    
        detection_model.pretrain(network_model, X, nb_epoch=nb_epoch)
        print 'Save ...'
        if init_name == None:
            init_name = network + '_init'

        print 'data/{}_fold_{}'.format(init_name)
        detection_model.save('data/{}_fold_{}'.format(init_name, fold))
        fold += 1

def joint_training(detection_model, fname, network_model, init_name):
    paths, locs, rads, subs, sizes, kinds = jsrt.jsrt(set='jsrt140')
    left_masks = jsrt.left_lung(set='jsrt140')
    right_masks = jsrt.right_lung(set='jsrt140')

    size = len(paths)

    blobs = []
    for i in range(size):
        blobs.append([locs[i][0], locs[i][1], rads[i]])
    blobs = np.array(blobs)

    print "Loading blobs ..."
    data = DataProvider(paths, left_masks, right_masks)
    pred_blobs = np.load('data/{}_pred.blb.npy'.format(fname))

    rois = detection_model.create_rois(data, pred_blobs)

    Y = (140 > np.array(range(size))).astype(np.uint8)
    folds = StratifiedKFold(subs, n_folds=10, shuffle=True, random_state=FOLDS_SEED)

    fold = 1
    frocs = []
    frocs_ = []
    for tr_idx, te_idx in folds:
        print "Fold {}".format(fold)
        data_te = DataProvider(paths[te_idx], left_masks[te_idx], right_masks[te_idx])
        paths_te = paths[te_idx]

        rois_tr, rois_te = _split_rois(rois, tr_idx, te_idx, detection_model.streams)

        X_train, Y_train, X_test, Y_test = neural.create_train_test_sets(rois_tr, pred_blobs[tr_idx], blobs[tr_idx], 
                                                rois_te, pred_blobs[te_idx], blobs[te_idx], streams=detection_model.streams, dataset_type=detection_model.dataset_type)

        detection_model.joint_training(network_model, X_train, Y_train, X_test, Y_test, fold)

        print 'Save ...'
        if init_name == 'none':
            init_name = network_model + '_us'

        detection_model.save('data/{}_fold_{}'.format(init_name, fold))

        blobs_te = []
        for bl in blobs[te_idx]:
            blobs_te.append([bl])
        blobs_te = np.array(blobs_te)

        blobs_te_pred, probs_te_pred = detection_model.predict_proba_from_feature_set_keras(rois_te, pred_blobs[te_idx])
        froc = eval.froc(blobs_te, blobs_te_pred, probs_te_pred)
        frocs.append(froc)

        legend_ = ['Fold {}'.format(i + 1) for i in range(len(frocs))]
        frocs_.append(eval.average_froc([froc], np.linspace(0.0, 10.0, 101)))
        util.save_froc(frocs_, 'data/{}_froc_kfold'.format(init_name), legend_)
        fold += 1

    av_froc = eval.average_froc(frocs, np.linspace(0.0, 10.0, 101))

    legend = []
    legend.append('Hardie et al.')
    legend.append(init_name)

    util.save_froc([baseline.hardie, av_froc], 'data/{}-FROC'.format(init_name), legend, with_std=True)
 
    return av_froc

def unsupervised_augment(detection_model, fname, network_model, init_name=None, nb_epoch=1, mode=None, rois_per_image=1000, pos_neg_ratio=0.1):
    paths, locs, rads, subs, sizes, kinds = jsrt.jsrt(set='jsrt140')
    left_masks = jsrt.left_lung(set='jsrt140')
    right_masks = jsrt.right_lung(set='jsrt140')

    size = len(paths)

    blobs = []
    for i in range(size):
        blobs.append([locs[i][0], locs[i][1], rads[i]])
    blobs = np.array(blobs)

    print "Loading blobs ..."
    data = DataProvider(paths, left_masks, right_masks)

    pred_blobs = np.load('data/{}_pred.blb.npy'.format(fname))
    pred_blobs_na = pred_blobs

    rois = None
    rois_na = None
    if detection_model.args.datasets == None:
        if mode == 'add-random':
            pred_blobs = util.add_random_blobs(data, blobs, blobs_by_image=rois_per_image, pred_blobs=pred_blobs, blob_rad=detection_model.blob_rad)
            rois = detection_model.create_rois(data, pred_blobs, save=detection_model.args.create_datasets != None)
    elif detection_model.args.datasets != None:
        rois_na = detection_model.create_rois(data, pred_blobs_na, save=False)
        rois, pred_blobs = detection_model.get_roiset(size)

    Y = (140 > np.array(range(size))).astype(np.uint8)
    folds = StratifiedKFold(subs, n_folds=10, shuffle=True, random_state=FOLDS_SEED)

    fold = 1
    frocs = []
    frocs_ = []
    for tr_idx, te_idx in folds:
        print "Fold {}".format(fold)
        data_te = DataProvider(paths[te_idx], left_masks[te_idx], right_masks[te_idx])
        paths_te = paths[te_idx]

        rois_tr, rois_te = None, None
        X_train, Y_train, X_test, Y_test = None, None, None, None

        rois_tr, rois_te = _split_rois(rois, tr_idx, te_idx, detection_model.streams)

        if detection_model.args.datasets != None:
            X_train, Y_train, X_test, Y_test = detection_model.get_datasets(fold)
        else:
            X_train, Y_train, X_test, Y_test = neural.create_train_test_sets(rois_tr, pred_blobs[tr_idx], blobs[tr_idx], 
                                                rois_te, pred_blobs[te_idx], blobs[te_idx], streams=detection_model.streams, dataset_type=detection_model.dataset_type)
            '''
            # Uncomment when online augment will be ready
            X_train, Y_train, X_test, Y_test = augment.preprocess_dataset(X_train, Y_train, X_test, Y_test, 
                                                    detection_model.streams!='none', detection_model.args.preprocess_dataset, detection_model.dataset_type)
            '''
        detection_model.unsupervised_augment(network_model, X_train, Y_train, X_test, Y_test, fold, detection_model.streams, nb_epoch=nb_epoch, pos_neg_ratio=pos_neg_ratio, mode=mode)

        print 'Save ...'
        if init_name == 'none':
            init_name = network_model + '_ua'

        detection_model.save('data/{}_fold_{}'.format(init_name, fold))

        blobs_te = []
        for bl in blobs[te_idx]:
            blobs_te.append([bl])
        blobs_te = np.array(blobs_te)

        blobs_te_pred, probs_te_pred = None, None
        if detection_model.args.datasets != None:
            rois_tr_na, rois_te_na = _split_rois(rois_na, tr_idx, te_idx, detection_model.streams)
            blobs_te_pred, probs_te_pred = detection_model.predict_proba_from_feature_set_keras(rois_te_na, pred_blobs_na[te_idx])
        else:
            blobs_te_pred, probs_te_pred = detection_model.predict_proba_from_feature_set_keras(rois_te, pred_blobs[te_idx])

        froc = eval.froc(blobs_te, blobs_te_pred, probs_te_pred)
        frocs.append(froc)

        legend_ = ['Fold {}'.format(i + 1) for i in range(len(frocs))]
        frocs_.append(eval.average_froc([froc], np.linspace(0.0, 10.0, 101)))
        util.save_froc(frocs_, 'data/{}_froc_kfold'.format(init_name), legend_)
        fold += 1

    av_froc = eval.average_froc(frocs, np.linspace(0.0, 10.0, 101))

    legend = []
    legend.append('Hardie et al.')
    legend.append(init_name)

    util.save_froc([baseline.hardie, av_froc], 'data/{}-FROC'.format(init_name), legend, with_std=True)
 
    return av_froc

def classify_cnn(detection_model, fname, network_model):
    print "classify with cnn"
    paths, locs, rads, subs, sizes, kinds = jsrt.jsrt(set='jsrt140')
    left_masks = jsrt.left_lung(set='jsrt140')
    right_masks = jsrt.right_lung(set='jsrt140')

    size = len(paths)

    blobs = []
    for i in range(size):
        blobs.append([locs[i][0], locs[i][1], rads[i]])
    blobs = np.array(blobs)

    print "Loading blobs ..."
    data = DataProvider(paths, left_masks, right_masks)
    pred_blobs = np.load('data/{}_pred.blb.npy'.format(fname))
    rois = detection_model.create_rois(data, pred_blobs)

    Y = (140 > np.array(range(size))).astype(np.uint8)
    folds = StratifiedKFold(subs, n_folds=10, shuffle=True, random_state=FOLDS_SEED)

    #ops, ops_sub, ops_siz, ops_kind, ci_out = froc_classify_cnn(detection_model, paths, left_masks, right_masks, blobs, pred_blobs, rois, folds, network_model, subs, sizes, kinds, ci=True)
    ops, ops_sub, ops_siz, ops_kind = froc_classify_cnn(detection_model, paths, left_masks, right_masks, blobs, pred_blobs, rois, folds, network_model, subs, sizes, kinds, ci=False)

    legend = []
    legend.append('Hardie et al.')

    #legend.append('{}, AUC = {} with 95% CI = {} - {}'.format(network_model, ci_out[0], ci_out[1], ci_out[2]))
    legend.append(network_model)

    #util.save_froc([baseline.hardie, ops], 'data/{}-FROC'.format(network_model), legend, with_std=True, with_auc=False)
    util.save_froc([baseline.hardie, ops], 'data/{}-FROC'.format(network_model), legend, with_std=False, with_auc=True)
    util.save_froc(ops_sub, 'data/{}-sublety'.format(network_model), jsrt.sublety_labels, with_std=False)
    util.save_froc(ops_siz, 'data/{}-size'.format(network_model), jsrt.size_labels, with_std=False)
    util.save_froc(ops_kind, 'data/{}-severity'.format(network_model), jsrt.severity_labels, with_std=False)

    return ops

def get_froc_on_folds_hybrid(detection_model, paths, left_masks, right_masks, blobs, pred_blobs, feats, rois, folds, network_model, use_feats=False, layer=-1):
    fold = 1
    valid = True
    frocs = []
    for tr_idx, te_idx in folds:    
        print "Fold {}".format(fold),
        data_te = DataProvider(paths[te_idx], left_masks[te_idx], right_masks[te_idx])
        paths_te = paths[te_idx]
        blobs_te = []

        for bl in blobs[te_idx]:
            blobs_te.append([bl])
        blobs_te = np.array(blobs_te)

        detection_model.load_cnn('data/{}_fold_{}'.format(network_model, fold))
        network_feats = np.load('data/{}_l{}_fold_{}.fts.npy'.format(network_model, layer, fold))

        hybrid_feats = []

        if use_feats:
            # stack features
            assert len(feats) == len(network_feats)
            for i in range(len(feats)):
                hybrid_feats.append(np.hstack([feats[i], network_feats[i]]))
            hybrid_feats = np.array(hybrid_feats)   
        else:
            hybrid_feats = network_feats;

        print 'Hybrid feats shape {}...'.format(hybrid_feats[0].shape)
        if use_feats == True:
            #scaler = detection_model.scaler;
            scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
            print "Normalize features {}...".format(scaler)
            feats_tr = hybrid_feats[tr_idx]
            feats_tr_flat = np.full(shape=(0, feats_tr[0].shape[1]), fill_value=0, dtype=np.float32)
            for fvs in feats_tr:
                feats_tr_flat = np.append(feats_tr_flat, fvs, axis=0)
            scaler.fit(feats_tr_flat)
            for i in range(len(hybrid_feats)):
                hybrid_feats[i] = scaler.transform(hybrid_feats[i])
        
        print 'Train with {}...'.format(detection_model.clf)
        detection_model.train_with_feature_set(hybrid_feats[tr_idx], pred_blobs[tr_idx], blobs[tr_idx])
        print 'Predict ...'
        blobs_te_pred, probs_te_pred = detection_model.predict_proba_from_feature_set(hybrid_feats[te_idx], pred_blobs[te_idx])
        print 'Get froc ...'

        # Append froc
        froc = eval.froc(blobs_te, blobs_te_pred, probs_te_pred)
        frocs.append(froc)
        fold += 1

    av_froc = eval.average_froc(frocs, np.linspace(0.0, 10.0, 101))

    return av_froc

def protocol_two_stages():
    paths, locs, rads, subs, sizes, kinds = jsrt.jsrt(set='jsrt140')
    left_masks = jsrt.left_lung(set='jsrt140')
    right_masks = jsrt.right_lung(set='jsrt140')
    size = len(paths)

    # blobs detection
    print "Detecting blobs ..."
    blobs = []
    for i in range(size):
        blobs.append([locs[i][0], locs[i][1], rads[i]])
    blobs = np.array(blobs)

    # feature extraction
    print "Extracting features ..."
    data = DataProvider(paths, left_masks, right_masks)
    feats, pred_blobs = model.extract_feature_set(data)

    Y = (140 > np.array(range(size))).astype(np.uint8)
    folds = StratifiedKFold(subs, n_folds=10, shuffle=True, random_state=FOLDS_SEED)
    fold = 0

    sens = []
    fppi_mean = []
    fppi_std = []

    for tr_idx, te_idx in folds:
        fold += 1
        print "Fold {}".format(fold), 

        model.train_with_feature_set(feats[tr_idx], pred_blobs[tr_idx], blobs[tr_idx])
        blobs_te_pred = model.predict_from_feature_set(feats[te_idx], pred_blobs[te_idx])

        paths_te = paths[te_idx]
        for i in range(len(blobs_te_pred)):
            util.print_detection(paths_te[i], blobs_te_pred[i])

        blobs_te = []
        for bl in blobs[te_idx]:
            blobs_te.append([bl])
        blobs_te = np.array(blobs_te)

        s, fm, fs = eval.evaluate(blobs_te, blobs_te_pred, paths[te_idx])
        print "Result: sens {}, fppi mean {}, fppi std {}".format(s, fm, fs)

        sens.append(s)
        fppi_mean.append(fm)
        fppi_std.append(fs)

    sens = np.array(sens)
    fppi_mean = np.array(fppi_mean)
    fppi_std = np.array(fppi_std)

    print "Final: sens_mean {}, sens_std {}, fppi_mean {}, fppi_stds_mean {}".format(sens.mean(), sens.std(), fppi_mean.mean(), fppi_std.mean())

def protocol_froc_1(detection_model, fname):
    print '# {}'.format(fname)
    paths, locs, rads, subs, sizes, kinds = jsrt.jsrt(set='jsrt140')
    left_masks = jsrt.left_lung(set='jsrt140')
    right_masks = jsrt.right_lung(set='jsrt140')
    '''
    paths, locs, rads, subs = jsrt.jsrt(set=None)
    print "Lens"
    print len(paths)
    left_masks = jsrt.left_lung(set=None)
    print len(left_masks)
    right_masks = jsrt.right_lung(set=None)
    print len(right_masks)
    '''
    size = len(paths)

    # blobs detection
    blobs = []
    for i in range(size):
        blobs.append([locs[i][0], locs[i][1], rads[i]])
    blobs = np.array(blobs)

    # feature extraction
    print "Extracting blobs & features ..."
    data = DataProvider(paths, left_masks, right_masks)
    feats, pred_blobs = detection_model.extract_feature_set(data)

    av_cpi = 0
    for tmp in pred_blobs:
        av_cpi += len(tmp)
    print "Average blobs per image {} ...".format(av_cpi * 1.0 / len(pred_blobs))

    np.save('data/{}.fts.npy'.format(fname), feats)
    np.save('data/{}_pred.blb.npy'.format(fname), pred_blobs)

def protocol_froc_2(detection_model, fname, save_fw=False):
    paths, locs, rads, subs, sizes, kinds = jsrt.jsrt(set='jsrt140')
    left_masks = jsrt.left_lung(set='jsrt140')
    right_masks = jsrt.right_lung(set='jsrt140')
    size = len(paths)

    blobs = []
    for i in range(size):
        blobs.append([locs[i][0], locs[i][1], rads[i]])
    blobs = np.array(blobs)

    print "Loading blobs & features ..."
    data = DataProvider(paths, left_masks, right_masks)
    feats = np.load('data/{}.fts.npy'.format(fname))
    
    pred_blobs = np.load('data/{}_pred.blb.npy'.format(fname))

    av_cpi = 0
    for tmp in pred_blobs:
        av_cpi += len(tmp)
    print "Average blobs per image {} ...".format(av_cpi * 1.0 / len(pred_blobs))

    Y = (140 > np.array(range(size))).astype(np.uint8)
    folds = StratifiedKFold(subs, n_folds=10, shuffle=True, random_state=FOLDS_SEED)

    print "save_fw {}".format(save_fw)
    ops = get_froc_on_folds(detection_model, paths, left_masks, right_masks, blobs, pred_blobs, feats, folds, save_fw, fname)
    step=1
    range_ops = ops[step * 2:step * 4 + 1]
    print 'auc 2 - 4 fppis {}'.format(auc(range_ops.T[0], range_ops.T[1]))
    
    legend = []
    legend.append('Hardie et al.')
    legend.append(detection_model.name)

    util.save_froc([baseline.hardie, ops], '{}'.format(detection_model.name), legend, with_std=True)

    return ops

def eval_wmci_and_postprocessing(detection_model, fname):
    paths, locs, rads, subs = jsrt.jsrt(set='jsrt140')
    left_masks = jsrt.left_lung(set='jsrt140')
    right_masks = jsrt.right_lung(set='jsrt140')

    size = len(paths)

    blobs = []
    for i in range(size):
        blobs.append([locs[i][0], locs[i][1], rads[i]])
    blobs = np.array(blobs)

    print "Loading blobs & features ..."

    data = DataProvider(paths, left_masks, right_masks)
    '''
    feats = np.load('data/{}.fts.npy'.format(fname))
    pred_blobs = np.load('data/{}_pred.blb.npy'.format(fname))
    '''
    feats, pred_blobs, proba = detection_model.extract_feature_set_proba(data)

    av_cpi = 0
    for tmp in pred_blobs:
        av_cpi += len(tmp)
    print "Average blobs per image {} ...".format(av_cpi * 1.0 / len(pred_blobs))

    Y = (140 > np.array(range(size))).astype(np.uint8)
    folds = StratifiedKFold(subs, n_folds=10, shuffle=True, random_state=FOLDS_SEED)

    fold = 0
    valid = True
    frocs = []
    for tr_idx, te_idx in folds:    
        print "Fold {}".format(fold + 1),
        data_te = DataProvider(paths[te_idx], left_masks[te_idx], right_masks[te_idx])
        paths_te = paths[te_idx]
        blobs_te = []

        for bl in blobs[te_idx]:
            blobs_te.append([bl])
        blobs_te = np.array(blobs_te)

        last_opt = eval.fppi_sensitivity(blobs_te, pred_blobs[te_idx])
        print 'last opt {}'.format(last_opt)
        fold += 1
    
    all_blobs = []
    for blob in blobs:
        all_blobs.append([blob])
    all_blobs = np.array(all_blobs)

    last_opt = eval.fppi_sensitivity(all_blobs, pred_blobs)
    print 'all last opt {}'.format(last_opt)

def protocol_wmci_froc(detection_model, fname):
    paths, locs, rads, subs = jsrt.jsrt(set='jsrt140')
    left_masks = jsrt.left_lung(set='jsrt140')
    right_masks = jsrt.right_lung(set='jsrt140')
    size = len(paths)

    blobs = []
    for i in range(size):
        blobs.append([locs[i][0], locs[i][1], rads[i]])
    blobs = np.array(blobs)

    print "Loading   blobs & features ..."
    data = DataProvider(paths, left_masks, right_masks)
    feats, pred_blobs, proba = detection_model.extract_feature_set_proba(data)

    '''
    feats = np.load('data/{}.fts.npy'.format(fname))
    pred_blobs = np.load('data/{}_pred.blb.npy'.format(fname))
    '''

    av_cpi = 0
    for tmp in pred_blobs:
        av_cpi += len(tmp)
    print "Average blobs per image {} ...".format(av_cpi * 1.0 / len(pred_blobs))
    Y = (140 > np.array(range(size))).astype(np.uint8)
    folds = StratifiedKFold(subs, n_folds=10, shuffle=True, random_state=FOLDS_SEED)

    op_set = []
    op_set.append(baseline.hardie)
    detect_range = np.arange(0.3, 0.8, 0.1)
    for detect_thold in detect_range:
        selected_feats = []
        selected_blobs = []

        for i in range(len(feats)):
            probs = proba[i] > detect_thold
            selected_feats.append(feats[i][probs])
            selected_blobs.append(pred_blobs[i][probs])

        selected_feats = np.array(selected_feats)
        selected_blobs = np.array(selected_blobs)

        ops = get_froc_on_folds(detection_model, paths, left_masks, right_masks, blobs, selected_blobs, selected_feats, folds)
        op_set.append(ops)

    op_set = np.array(op_set)
    legend = []
    legend.append("baseline.hardie")
    for thold in detect_range:
        legend.append('wmci {}'.format(thold))

    util.save_froc(op_set, detection_model.name, legend)
    return op_set

def protocol_generic_froc(detection_model, fnames, components, legend, kind='descriptor', mode='feats'):
    paths, locs, rads, subs = jsrt.jsrt(set='jsrt140')
    left_masks = jsrt.left_lung(set='jsrt140')
    right_masks = jsrt.right_lung(set='jsrt140')
    size = len(paths)

    blobs = []
    for i in range(size):
        blobs.append([locs[i][0], locs[i][1], rads[i]])
    blobs = np.array(blobs)

    data = DataProvider(paths, left_masks, right_masks)
    Y = (140 > np.array(range(size))).astype(np.uint8)
    folds = StratifiedKFold(subs, n_folds=10, shuffle=True, random_state=FOLDS_SEED)
    
    op_set = []
    op_set.append(baseline.hardie)
    legend.insert(0, "baseline.hardie")

    for i in range(len(components)):
        print "Loading blobs & features ..."
        feats = np.load('data/{}.fts.npy'.format(fnames[i]))
        pred_blobs = np.load('data/{}_pred.blb.npy'.format(fnames[i]))

        print legend[i+1]
        if kind == 'descriptor':
            detection_model.descriptor = components[i]
        elif kind == 'selector':
            detection_model.selector = components[i]
        elif kind == 'classifier':
            detection_model.classifier = components[i]

        ops = get_froc_on_folds(detection_model, paths, left_masks, right_masks, blobs, pred_blobs, feats, folds)
        op_set.append(ops)

    op_set = np.array(op_set)
    util.save_froc(op_set, detection_model.name, legend) 
    return op_set


def protocol_selector_froc(detection_model, fname, selectors, legend):
    paths, locs, rads, subs = jsrt.jsrt(set='jsrt140')
    left_masks = jsrt.left_lung(set='jsrt140')
    right_masks = jsrt.right_lung(set='jsrt140')
    size = len(paths)

    blobs = []
    for i in range(size):
        blobs.append([locs[i][0], locs[i][1], rads[i]])
    blobs = np.array(blobs)

    print "Loading   blobs & features ..."
    data = DataProvider(paths, left_masks, right_masks)

    feats = np.load('data/{}.fts.npy'.format(fname))
    pred_blobs = np.load('data/{}_pred.blb.npy'.format(fname))

    av_cpi = 0
    for tmp in pred_blobs:
        av_cpi += len(tmp)
    print "Average blobs per image {} ...".format(av_cpi * 1.0 / len(pred_blobs))

    Y = (140 > np.array(range(size))).astype(np.uint8)
    folds = StratifiedKFold(subs, n_folds=10, shuffle=True, random_state=FOLDS_SEED)
    
    op_set = []
    
    op_set.append(baseline.hardie)
    legend.insert(0, "baseline.hardie")

    for i in range(len(selectors)):
        print legend[i+1]
        detection_model.selector = selectors[i]
        ops = get_froc_on_folds(detection_model, paths, left_masks, right_masks, blobs, pred_blobs, feats, folds)
        op_set.append(ops)

    op_set = np.array(op_set)
    util.save_froc(op_set, detection_model.name, legend)

    return op_set

def protocol_classifier_froc(detection_model, fname, classifiers, legend):
    paths, locs, rads, subs = jsrt.jsrt(set='jsrt140')
    left_masks = jsrt.left_lung(set='jsrt140')
    right_masks = jsrt.right_lung(set='jsrt140')
    size = len(paths)

    blobs = []
    for i in range(size):
        blobs.append([locs[i][0], locs[i][1], rads[i]])
    blobs = np.array(blobs)

    print "Loading   blobs & features ..."
    data = DataProvider(paths, left_masks, right_masks)

    feats = np.load('data/{}.fts.npy'.format(fname))
    pred_blobs = np.load('data/{}_pred.blb.npy'.format(fname))

    av_cpi = 0
    for tmp in pred_blobs:
        av_cpi += len(tmp)
    print "Average blobs per image {} ...".format(av_cpi * 1.0 / len(pred_blobs))

    Y = (140 > np.array(range(size))).astype(np.uint8)
    folds = StratifiedKFold(subs, n_folds=10, shuffle=True, random_state=FOLDS_SEED)
    
    op_set = []

    op_set.append(baseline.hardie)
    legend.insert(0, "baseline.hardie")

    for i in range(len(classifiers)):
        print legend[i+1]
        detection_model.clf = classifiers[i]
        ops = get_froc_on_folds(detection_model, paths, left_masks, right_masks, blobs, pred_blobs, feats, folds) 
        op_set.append(ops)
        
        # Somewhat free memory
        classifiers[i] = None
        detection_model.clf = None

    op_set = np.array(op_set)
    util.save_froc(op_set, '{}'.format(detection_model.name), legend)

    return op_set[1:]

def hog_impls(detection_model, fname, fts=False, clf=True):
    descriptors = []
    labels = []
    fnames = []

    for inp, mode in product(['lce', 'norm', 'wmci'], ['skimage_default', 'default']):
        fnames.append('{}_{}_{}'.format(fname, inp, mode))
        labels.append('{}_{}_{}'.format(fname, inp, mode))
        descriptors.append(model.HogExtractor(mode=mode, input=inp))
        if fts:
            detection_model.extractor = descriptors[-1]
            protocol_froc_1(detection_model, fnames[-1]) 

    if clf:
        protocol_generic_froc(detection_model, fnames, descriptors, labels, kind='descriptor')
        
# FIX: Dropout skimage impl and test type and modes
def hog_froc(detection_model, fname, fts=False, clf=True):
    descriptors = []
    descriptors.append(model.HogExtractor(mode='skimage_default'))
    descriptors.append(model.HogExtractor(mode='skimage_32x32'))
    descriptors.append(model.HogExtractor(mode='32x32'))
    descriptors.append(model.HogExtractor(mode='32x32_inner'))
    descriptors.append(model.HogExtractor(mode='32x32_inner_outer'))

    labels = []
    labels.append('skimage default')
    labels.append('skimage 32x32')
    labels.append('our impl 32x32')
    labels.append('our impl 32x32 inner')
    labels.append('our impl 32x32 inner + outer')

    # extract
    fnames = []
    for descriptor in descriptors:
        fnames.append('{}_{}'.format(fname, descriptor.mode))
        if fts:
            detection_model.extractor = descriptor
            protocol_froc_1(detection_model, fnames[-1])

    if clf:
        protocol_generic_froc(detection_model, fnames, descriptors, labels, kind='descriptor')

def hrg_froc(detection_model, fname, fts=False, clf=True):
    descriptors = []
    labels = []
    fnames = []

    for inp, mode in product(['lce', 'norm', 'wmci'], ['default', 'inner', 'inner_outer']):
        fnames.append('{}_{}_{}'.format(fname, inp, mode))
        labels.append('{}_{}_{}'.format(fname, inp, mode))
        descriptors.append(model.HRGExtractor(mode=mode, input=inp))
        if fts:
            detection_model.extractor = descriptors[-1]
            protocol_froc_1(detection_model, fnames[-1]) 

    if clf:
        protocol_generic_froc(detection_model, fnames, descriptors, labels, kind='descriptor')
        

def lbp_froc(detection_model, fname, fts=False, clf=True, mode='default'):
    descriptors = []
    labels = []

    for inp, method in product(['lce', 'norm', 'wmci'], ['default', 'uniform', 'nri_uniform']):
        descriptors.append(model.LBPExtractor(method=method, input=inp, mode=mode))
        labels.append('{}_{}_{}'.format(inp, method, mode))

    # extract
    fnames = []
    for descriptor in descriptors:
        fnames.append('{}_{}_{}_{}'.format(fname, descriptor.mode, descriptor.method, descriptor.input))
        if fts:
            detection_model.extractor = descriptor
            protocol_froc_1(detection_model, fnames[-1])

    if clf:
        protocol_generic_froc(detection_model, fnames, descriptors, labels, kind='descriptor')
    
def znk_froc(detection_model, fname, fts=False, clf=True):
    descriptors = []
    labels = []

    labels.append('mask')
    descriptors.append(model.ZernikeExtractor(input='lce', mode='mask'))

    for inp, mode in product(['lce', 'norm', 'wmci'], ['nomask', 'inner', 'inner_outer', 'contour']):
        descriptors.append(model.ZernikeExtractor(input=inp, mode=mode))
        labels.append('{}_{}'.format(inp, mode))

    # extract
    fnames = []
    for descriptor in descriptors:
        fnames.append('{}_{}_{}'.format(fname, descriptor.input, descriptor.mode))
        if fts:
            detection_model.extractor = descriptor
            protocol_froc_1(detection_model, fnames[-1])

    if clf:
        protocol_generic_froc(detection_model, fnames, descriptors, labels, kind='descriptor')
    

def protocol_clf_eval_froc(detection_model, fname):
    classifiers  = []
    classifiers.append(lda.LDA())
    classifiers.append(svm.SVC(kernel='linear', probability=True))
    classifiers.append(svm.SVC(kernel='rbf', probability=True))
    #classifiers.append(ensemble.RandomForestClassifier())
    #classifiers.append(ensemble.AdaBoostClassifier())
    #lassifiers.append(ensemble.GradientBoostingClassifier())

    labels = []
    labels.append('LDA')
    labels.append('Linear SVC')
    labels.append('RBF SVC')
    #labels.append('Random Forest')
    #labels.append('AdaBoost')
    #labels.append('Gradient Boosting')

    protocol_classifier_froc(detection_model, fname, classifiers, labels)

def protocol_svm_hp_search(detection_model, fname):
    C_set = np.logspace(-3, 4, 8)
    g_set = np.logspace(-3, -1, 9)
    #C_set = np.logspace(-2, 2, 9)
    #g_set = np.logspace(-4, -1, 10)
    #C_set = np.logspace(-2, 1, 10)
    #g_set = np.logspace(-4, -2, 9)
    #C_set = np.logspace(-0.7, 0.0, 9)
    #g_set = np.logspace(-3.5, -3.0, 9)
    classifiers = []
    legend = []
    for C, gamma in product(C_set, g_set):
        legend.append('C={}, g={}'.format(C, round(gamma, 5)))
        print 'SVM C = {}, g= {}'.format(C, round(gamma, 5))
        classifiers.append(svm.SVC(C=C, gamma=gamma, probability=True, cache_size=600, max_iter=10000))

    ops = protocol_classifier_froc(detection_model, fname, classifiers, legend)
    
    # compute the AUC from 2 to 4
    auc_grid = []
    for i in range(ops.shape[0]):
        range_ops = ops[i][step * 2:step * 4 + 1]
        auc_grid.append(auc(range_ops.T[0], range_ops.T[1]))
    auc_grid = np.array(auc_grid).reshape((C_set.shape[0], g_set.shape[0])) 
    print "AUC GRID"
    print auc_grid
    util.save_grid(auc_grid, detection_model.name, ['C', 'gamma'], [C_set, g_set], title='AUC between 2 and 4 FPPI\'s')

def protocol_pca_froc(detection_model, fname):
    selectors = []
    labels = []
    pca_var = np.arange(1, 10, 1)
    pca_var = -1 * pca_var
    pca_var = 2.0 ** pca_var
    pca_var = 1 - pca_var

    for var in pca_var:
        selector = decomposition.PCA(n_components=var, whiten=True)
        selectors.append(selector)
        labels.append('var {}'.format(var))

    protocol_selector_froc(detection_model, fname, selectors, labels)

def protocol_lda_froc(detection_model, fname):
    selectors = []
    labels = []
    var_set = [0.0125, 0.025, 0.05]
    var_set = np.append(var_set, np.arange(0.1, 0.9, 0.1))

    for var in var_set:
        selector = selection.SelectFromModel(lda.LDA(), threshold=var)
        selectors.append(selector)
        labels.append('thold {}'.format(var))

    protocol_selector_froc(detection_model, fname, selectors, labels)

def protocol_rlr_froc(detection_model, fname):
    selectors = []
    labels = []
    var_set = np.arange(5, 106, 10)

    for var in var_set:
        rlr = lineardetection_model.RandomizedLogisticRegression(C=var)
        selectors.append(rlr)
        labels.append('C {}'.format(var))

    protocol_selector_froc(detection_model, fname, selectors, labels)

def protocol_rfe_froc(detection_model, fname):
    nfeats = 136
    #var_set = 2 ** np.arange(2, 8, 1)
    #var_set = nfeats - var_set
    var_set = np.arange(70, 131, 5)
    selectors = []
    labels = []

    for var in var_set:
        svc = svm.SVC(kernel="linear", C=1)
        rfe = selection.RFE(estimator=svc, n_features_to_select=var, step=1)
        selectors.append(rfe)
        labels.append('n feats {}'.format(var))

    protocol_selector_froc(detection_model, fname, selectors, labels)

'''
Deep learning protocols
'''

def protocol_cnn_froc(detections_source, fname, network_model):
    paths, locs, rads, subs, sizes, kinds = jsrt.jsrt(set='jsrt140')
    left_masks = jsrt.left_lung(set='jsrt140')
    right_masks = jsrt.right_lung(set='jsrt140')

    size = len(paths)

    blobs = []
    for i in range(size):
        blobs.append(np.array([locs[i][0], locs[i][1], rads[i]]))
    blobs = np.array(blobs)

    print "Loading dataset ..."
    data = DataProvider(paths, left_masks, right_masks)
    pred_blobs = detect.read_blobs('data/{}-blobs-gt.pkl'.format(detection_model.args.blob_detector))
    rois = detections_source.create_rois(data, pred_blobs)

    av_cpi = 0
    for tmp in pred_blobs:
        av_cpi += len(tmp)
    print "Average blobs per image {} ...".format(av_cpi * 1.0 / len(pred_blobs))

    # Setting labels ...
    Y = None
    if detections_source.label == 'nodule':
        Y = (140 > np.array(range(size)))
    elif detections_source.label == 'sublety':
        Y = subs
    else:
        print "{} option doesnt exists. Using nodule labels".format(detections_source.labels)
        Y = (140 > np.array(range(size)))
    Y = Y.astype(np.uint8)
    
    folds = util.stratified_kfold_holdout(subs, n_folds=5) 
    ops = get_froc_on_folds_keras(detections_source, paths, left_masks, right_masks, blobs, pred_blobs, rois, folds, network_model)

    legend = []
    legend.append('Hardie et al.')
    legend.append(network_model)

    util.save_froc([baseline.hardie, ops], 'data/{}-FROC'.format(network_model), legend, with_std=True)

    return ops

def protocol_pretrained_cnn(detections_source, fname, network_model, network_init):
    paths, locs, rads, subs, sizes, kinds = jsrt.jsrt(set='jsrt140')
    left_masks = jsrt.left_lung(set='jsrt140')
    right_masks = jsrt.right_lung(set='jsrt140')
    size = len(paths)

    blobs = []
    for i in range(size):
        blobs.append([locs[i][0], locs[i][1], rads[i]])
    blobs = np.array(blobs)

    print "Loading dataset ..."
    data = DataProvider(paths, left_masks, right_masks)
    pred_blobs = np.load('data/{}_pred.blb.npy'.format(fname))
    rois = detections_source.create_rois(data, pred_blobs)

    av_cpi = 0
    for tmp in pred_blobs:
        av_cpi += len(tmp)
    print "Average blobs per image {} ...".format(av_cpi * 1.0 / len(pred_blobs))

    Y = (140 > np.array(range(size))).astype(np.uint8)
    folds = StratifiedKFold(subs, n_folds=10, shuffle=True, random_state=FOLDS_SEED)
    
    ops = get_froc_on_folds_keras(detections_source, paths, left_masks, right_masks, blobs, pred_blobs, rois, folds, network_model, network_init=network_init)

    legend = []
    legend.append('Hardie et al.')
    legend.append(network_model)

    util.save_froc([baseline.hardie, ops], 'data/{}-FROC'.format(network_model), legend, with_std=True)

    return ops


def protocol_cnn_froc_transforms(detections_source, fname, network_model):
    '''
    paths, locs, rads, subs = jsrt.jsrt(set=None)
    left_masks = jsrt.left_lung(set=None)
    right_masks = jsrt.right_lung(set=None)
    '''
    paths, locs, rads, subs = jsrt.jsrt(set='jsrt140')
    left_masks = jsrt.left_lung(set='jsrt140')
    right_masks = jsrt.right_lung(set='jsrt140')

    size = len(paths)

    blobs = []
    for i in range(size):
        blobs.append([locs[i][0], locs[i][1], rads[i]])
    blobs = np.array(blobs)

    print "Loading dataset ..."
    data = DataProvider(paths, left_masks, right_masks)
    pred_blobs = np.load('data/{}_pred.blb.npy'.format(fname))
    rois = detections_source.create_rois(data, pred_blobs)

    av_cpi = 0
    for tmp in pred_blobs:
        av_cpi += len(tmp)
    print "Average blobs per image {} ...".format(av_cpi * 1.0 / len(pred_blobs))

    Y = (140 > np.array(range(size))).astype(np.uint8)
    folds = StratifiedKFold(subs, n_folds=10, shuffle=True, random_state=FOLDS_SEED)
    
    ops = get_froc_on_folds_keras(detections_source, paths, left_masks, right_masks, blobs, pred_blobs, rois, folds, network_model)

    legend = []
    legend.append('Hardie et al.')
    legend.append(network_model)

    util.save_froc([baseline.hardie, ops], 'data/{}-FROC'.format(network_model), legend, with_std=True)

    return ops



def hybrid(detections_source, fname, network_model, layer):
    print 'CNN features with sklearn classifier'
    paths, locs, rads, subs = jsrt.jsrt(set='jsrt140')
    left_masks = jsrt.left_lung(set='jsrt140')
    right_masks = jsrt.right_lung(set='jsrt140')

    size = len(paths)

    blobs = []
    for i in range(size):
        blobs.append([locs[i][0], locs[i][1], rads[i]])
    blobs = np.array(blobs)

    print "Loading dataset ..."
    data = DataProvider(paths, left_masks, right_masks)
    feats = np.load('data/{}.fts.npy'.format(fname))
    pred_blobs = np.load('data/{}_pred.blb.npy'.format(fname))
    rois = detections_source.create_rois(data, pred_blobs)

    av_cpi = 0
    for tmp in pred_blobs:
        av_cpi += len(tmp)
    print "Average blobs per image {} ...".format(av_cpi * 1.0 / len(pred_blobs))

    Y = (140 > np.array(range(size))).astype(np.uint8)
    folds = StratifiedKFold(subs, n_folds=10, shuffle=True, random_state=FOLDS_SEED)
    
    ops = get_froc_on_folds_hybrid(detections_source, paths, left_masks, right_masks, blobs, pred_blobs, feats, rois, folds, network_model, use_feats=False, layer=layer)

    legend = []
    legend.append('Hardie et al.')
    legend.append('current')

    util.save_froc([baseline.hardie, ops], '{}_hybrid'.format(detection_model.name), legend, with_std=True)

    return ops

def compare_cnndetection_models(detections_source, fname, nw_names, nw_labels, exp_name):
    paths, locs, rads, subs, sizes, kinds = jsrt.jsrt(set='jsrt140')
    left_masks = jsrt.left_lung(set='jsrt140')
    right_masks = jsrt.right_lung(set='jsrt140')

    size = len(paths)

    blobs = []
    for i in range(size):
        blobs.append([locs[i][0], locs[i][1], rads[i]])
    blobs = np.array(blobs)

    print "Loading dataset ..."
    data = DataProvider(paths, left_masks, right_masks)
    pred_blobs = np.load('data/{}_pred.blb.npy'.format(fname))
    rois = detections_source.create_rois(data, pred_blobs)

    av_cpi = 0
    for tmp in pred_blobs:
        av_cpi += len(tmp)
    print "Average blobs per image {} ...".format(av_cpi * 1.0 / len(pred_blobs))

    Y = (140 > np.array(range(size))).astype(np.uint8)
    folds = StratifiedKFold(subs, n_folds=10, shuffle=True, random_state=FOLDS_SEED)

    op_set = []
    op_set.append(baseline.hardie)

    legend = []
    legend.append('Hardie et al.')

    for i in range(len(nw_names)):
        ops, _, _, _ = froc_classify_cnn(detections_source, paths, left_masks, right_masks, blobs, pred_blobs, rois, folds, nw_names[i], subs, sizes, kinds)
        op_set.append(ops)
        legend.append(nw_labels[i])

    util.save_froc(op_set, exp_name, legend)
    return ops
# Layer map
layer_idx_by_network = {'LND-A':[14, 17], 'LND-B':[20, 23], 'LND-C':[26, 29], 'LND-A-5P':[22, 25], 'LND-A-5P-ZMUV':[22, 25]}
def compare_cnn_sklearn_clfs(detections_source, fname, network):
    paths, locs, rads, subs = jsrt.jsrt(set='jsrt140')
    left_masks = jsrt.left_lung(set='jsrt140')
    right_masks = jsrt.right_lung(set='jsrt140')

    size = len(paths)

    blobs = []
    for i in range(size):
        blobs.append([locs[i][0], locs[i][1], rads[i]])
    blobs = np.array(blobs)

    print "Loading dataset ..."
    data = DataProvider(paths, left_masks, right_masks)
    pred_blobs = np.load('data/{}_pred.blb.npy'.format(fname))
    rois = detections_source.create_rois(data, pred_blobs, pre='histeq')

    av_cpi = 0
    for tmp in pred_blobs:
        av_cpi += len(tmp)
    print "Average blobs per image {} ...".format(av_cpi * 1.0 / len(pred_blobs))

    Y = (140 > np.array(range(size))).astype(np.uint8)
    folds = StratifiedKFold(subs, n_folds=10, shuffle=True, random_state=FOLDS_SEED)

    op_set = []
    op_set.append(baseline.hardie)


    classifiers = [svm.SVC(kernel='linear', probability=True), svm.SVC(kernel='linear', probability=True), svm.SVC(kernel='rbf', probability=True), svm.SVC(kernel='rbf', probability=True), lda.LDA(), lda.LDA()]
    layers = layer_idx_by_network[network]

    legend = ['Hardie et al.', 
                '{} only'.format(network),
                '{}, n-2 layer, Linear SVM'.format(network),
                '{}, n-1 layer, Linear SVM'.format(network),
                '{}, n-2 layer, RBF SVM'.format(network),
                '{}, n-1 layer, RBF SVM'.format(network),
                '{}, n-2 layer, LDA'.format(network),
                '{}, n-1 layer, LDA'.format(network)]

    ops = froc_classify_cnn(detections_source, paths, left_masks, right_masks, blobs, pred_blobs, rois, folds, network)
    op_set.append(ops)

    for i in range(len(layers)):
        detections_source.clf = classifiers[i]
        ops = get_froc_on_folds_hybrid(detections_source, paths, left_masks, right_masks, blobs, pred_blobs, None, rois, folds, network, use_feats=False, layer=layers[i%2])
        op_set.append(ops)
    util.save_froc(op_set, 'data/cnn-with-sklearn-clfs-{}'.format(network), legend)

    return ops

def compare_cnn_hybrid(detections_source, fname, network):
    paths, locs, rads, subs = jsrt.jsrt(set='jsrt140')
    left_masks = jsrt.left_lung(set='jsrt140')
    right_masks = jsrt.right_lung(set='jsrt140')

    size = len(paths)

    blobs = []
    for i in range(size):
        blobs.append([locs[i][0], locs[i][1], rads[i]])
    blobs = np.array(blobs)

    print "Loading dataset ..."
    data = DataProvider(paths, left_masks, right_masks)
    pred_blobs = np.load('data/{}_pred.blb.npy'.format(fname))
    feats = np.load('data/{}.fts.npy'.format(fname))
    rois = detections_source.create_rois(data, pred_blobs)

    av_cpi = 0
    for tmp in pred_blobs:
        av_cpi += len(tmp)

    print "Average blobs per image {} ...".format(av_cpi * 1.0 / len(pred_blobs))
    Y = (140 > np.array(range(size))).astype(np.uint8)
    folds = StratifiedKFold(subs, n_folds=10, shuffle=True, random_state=FOLDS_SEED)

    op_set = []
    op_set.append(baseline.hardie)

    classifiers = [svm.SVC(kernel='linear', probability=True), svm.SVC(kernel='rbf', probability=True), lda.LDA()]
    layers = layer_idx_by_network[network]
    layers_vect = [layers[1], layers[1], layers[1]]

    legend = ['Hardie et al.', 
                '{} only'.format(network),
                '{} layer n-1, linear SVM'.format(network),
                '{} layer n-2, rbf SVM'.format(network),
                '{} layer n-1, LDA'.format(network)]

    ops = froc_classify_cnn(detections_source, paths, left_masks, right_masks, blobs, pred_blobs, rois, folds, network)
    op_set.append(ops)

    for i in range(len(classifiers)):
        detections_source.clf = classifiers[i]
        print 'Layer {}'.format(layers_vect[i])
        ops = get_froc_on_folds_hybrid(detections_source, paths, left_masks, right_masks, blobs, pred_blobs, feats, rois, folds, network, use_feats=True, layer=layers_vect[i])
        op_set.append(ops)

    util.save_froc(op_set, 'data/{}-{}-HYB'.format(network, fname), legend)

    return ops

def compare_cnn_sota(detections_source, fname, nw_names, nw_labels, exp_name):
    paths, locs, rads, subs, sizes, kinds = jsrt.jsrt(set='jsrt140')
    left_masks = jsrt.left_lung(set='jsrt140')
    right_masks = jsrt.right_lung(set='jsrt140')

    size = len(paths)

    blobs = []
    for i in range(size):
        blobs.append([locs[i][0], locs[i][1], rads[i]])
    blobs = np.array(blobs)


    print "Loading dataset ..."
    data = DataProvider(paths, left_masks, right_masks)
    pred_blobs = np.load('data/{}_pred.blb.npy'.format(fname))
    rois = detections_source.create_rois(data, pred_blobs)

    av_cpi = 0
    for tmp in pred_blobs:
        av_cpi += len(tmp)
    print "Average blobs per image {} ...".format(av_cpi * 1.0 / len(pred_blobs))

    Y = (140 > np.array(range(size))).astype(np.uint8)
    folds = StratifiedKFold(subs, n_folds=10, shuffle=True, random_state=FOLDS_SEED)

    op_set = []
    legend = []

    for i in range(len(nw_names)):
        ops, ops_sub, ops_siz, ops_kind = froc_classify_cnn(detections_source, paths, left_masks, right_masks, blobs, pred_blobs, rois, folds, nw_names[i], subs, sizes, kinds, ci=False)
        #ops = froc_classify_cnn(detections_source, paths, left_masks, right_masks, blobs, pred_blobs, rois, folds, nw_names[i])
        op_set.append(ops)
        legend.append(nw_labels[i])

    util.save_froc_mixed(op_set, legend, baseline.sota_ops, baseline.sota_authors, exp_name)
    return None


def hyp_cnn_lsvm_hybrid(detections_source, fname, network):
    paths, locs, rads, subs = jsrt.jsrt(set='jsrt140')
    left_masks = jsrt.left_lung(set='jsrt140')
    right_masks = jsrt.right_lung(set='jsrt140')

    size = len(paths)

    blobs = []
    for i in range(size):
        blobs.append([locs[i][0], locs[i][1], rads[i]])
    blobs = np.array(blobs)

    print "Loading dataset ..."
    data = DataProvider(paths, left_masks, right_masks)
    pred_blobs = np.load('data/{}_pred.blb.npy'.format(fname))
    feats = np.load('data/{}.fts.npy'.format(fname))
    rois = detections_source.create_rois(data, pred_blobs)

    av_cpi = 0
    for tmp in pred_blobs:
        av_cpi += len(tmp)

    print "Average blobs per image {} ...".format(av_cpi * 1.0 / len(pred_blobs))
    Y = (140 > np.array(range(size))).astype(np.uint8)
    folds = StratifiedKFold(subs, n_folds=10, shuffle=True, random_state=FOLDS_SEED)

    legend = ['Hardie et al.', '{} only'.format(network)]
    op_set = []
    op_set.append(baseline.hardie)
    ops = froc_classify_cnn(detections_source, paths, left_masks, right_masks, blobs, pred_blobs, rois, folds, network)
    op_set.append(ops)

    layers = layer_idx_by_network[network]
    C_set = np.logspace(-3, 4, 8)
    for C in C_set:
        legend.append('{} layer n-1, linear SVM, C={}'.format(network, C))
        detections_source.clf = svm.SVC(kernel='linear', probability=True, C=C)
        print 'Layer {}'.format(layers[1])
        ops = get_froc_on_folds_hybrid(detections_source, paths, left_masks, right_masks, blobs, pred_blobs, feats, rois, folds, network, use_feats=True, layer=layers[1])
        op_set.append(ops)

    util.save_froc(op_set, 'data/{}-{}-GS-HYB'.format(network, fname), legend)
    return ops

# BOVW models

def compare_bovwdetection_models(detections_source, fname, model_names, labels, exp_name):
    op_set = []
    op_set.append(baseline.hardie)

    legend = []
    legend.append('Hardie et al.')

    for i in range(len(model_names)):
        ops = classify_foldwise(detections_source, fname, model_names[i], save_froc=False)
        op_set.append(ops)
        legend.append(labels[i])

    util.save_froc(op_set, exp_name, legend)
    return ops

# HDF5 support

def create_datasets(detection_model, blobs_filename, chunk_size=10000):
    paths, locs, rads, subs, sizes, kinds = jsrt.jsrt(set='jsrt140')
    left_masks = jsrt.left_lung(set='jsrt140')
    right_masks = jsrt.right_lung(set='jsrt140')
    dataset_size = len(paths)

    blobs = []
    for i in range(dataset_size):
        blobs.append([locs[i][0], locs[i][1], rads[i]])
    blobs = np.array(blobs)

    print "Loading blobs ..."
    data = DataProvider(paths, left_masks, right_masks)
    pred_blobs = np.load('data/{}_pred.blb.npy'.format(blobs_filename))

    if detection_model.args.rpi > 0:
        pred_blobs = util.add_random_blobs(data, blobs, blobs_by_image=detection_model.args.rpi, pred_blobs=pred_blobs, blob_rad=detection_model.blob_rad)

    rois = detection_model.create_rois(data, pred_blobs, save=True)

    Y = (140 > np.array(range(dataset_size))).astype(np.uint8)

    folds = StratifiedKFold(subs, n_folds=10, shuffle=True, random_state=FOLDS_SEED)

    fold = 1
    frocs = []
    frocs_ = []
    for tr_idx, te_idx in folds:
        print "Fold {}".format(fold)
        data_te = DataProvider(paths[te_idx], left_masks[te_idx], right_masks[te_idx])
        paths_te = paths[te_idx]

        group = detection_model.file.create_group('fold_{}'.format(fold))

        rois_tr, rois_te = _split_rois(rois, tr_idx, te_idx, detection_model.streams)

        X_train, Y_train, X_test, Y_test = neural.create_train_test_sets(rois_tr, pred_blobs[tr_idx], blobs[tr_idx], 
                                                rois_te, pred_blobs[te_idx], blobs[te_idx], streams=detection_model.streams, dataset_type='hdf5', container=group)
        
        print 'preprocess_dataset in (before call) {}'.format((type(X_train), type(Y_train), type(X_test), type(Y_test)))

        preprocessor = augment.get_preprocessor(detection_model.args.preprocess_dataset)
        X_train, Y_train, X_test, Y_test = augment.preprocess_dataset(preprocessor, X_train, Y_train, X_test, Y_test, streams=(detection_model.streams != 'none'))

        len_train_pos = int(np.sum(Y_train.T[1]))
        len_train_neg = len(Y_train) - len_train_pos
        X_train_pos = group.create_dataset('X_train_pos', (len_train_pos,) + X_train[0].shape, chunks=(chunk_size,) + X_train[0].shape, maxshape=(None,) + X_train[0].shape, dtype='float32')
        X_train_neg = group.create_dataset('X_train_neg', (len_train_neg,) + X_train[0].shape, chunks=(chunk_size,) + X_train[0].shape, maxshape=(None,) + X_train[0].shape, dtype='float32')
        X_train_pos[...] = X_train[0:len_train_pos]
        X_train_neg[...] = X_train[len_train_pos:len(Y_train)]

        len_test_pos = int(np.sum(Y_test.T[1]))
        len_test_neg = len(Y_test) - len_test_pos
        X_test_pos = group.create_dataset('X_test_pos', (len_test_pos,) + X_test[0].shape, chunks=(chunk_size,) + X_test[0].shape, maxshape=(None,) + X_test[0].shape, dtype='float32')
        X_test_neg = group.create_dataset('X_test_neg', (len_test_neg,) + X_test[0].shape, chunks=(chunk_size,) + X_test[0].shape, maxshape=(None,) + X_test[0].shape, dtype='float32')
        X_test_pos[...] = X_test[0:len_test_pos]
        X_test_neg[...] = X_test[len_test_pos:len(Y_test)]

        fold += 1

def detect_(detection_model, network_model, image, num_blobs=5):
    detector_segmentator = detection_model.args.blob_detector
    idx = detector_segmentator.find('-')
    detector = detector_segmentator[:idx]
    segmentator = detector_segmentator[idx+1:]
    print('Detect(clf:{}, det:{}, seg:{})'.format(network_model, detector, segmentator))

    cropped_shape = (detection_model.roi_size, detection_model.roi_size)
    detection_model.network = neural.create_network(network_model, (1,) + cropped_shape, fold=1, streams=False)
    detection_model.load_cnn('data/{}_fold_{}'.format(network_model, 1))
    print detection_model.network.preprocessor.__dict__
    print detection_model.network.generator.__dict__
    blobs, _ = detect.detect(image, detector, segmentator, display=False)

    rois_te = detection_model.create_rois(data_te, pred_blobs_te)

    blobs, probs = detection_model.predict_proba_one_keras(image, blobs)
    assert len(blobs) == len(probs)

    entries = []
    for i in range(len(blobs)):
        entries.append([blobs[i], probs[i]])
    entries = sorted(entries, key=itemgetter(1))
    entries = entries[:num_blobs]

    util.show_blobs_with_probs('Detect(clf:{}, det:{}, seg:{})'.format(network_model, detector, segmentator), image, entries.T[0], entries.T[1])

if __name__=="__main__": 
    parser = argparse.ArgumentParser(prog='lnd.py')
    parser.add_argument('file', nargs='?', default=None, type=str)
    # TRADITIONAL PIPELINES
    parser.add_argument('--preprocess-lung', help='Generates a bunch preprocessed images for the input. Example --preprocessor norm,lce,ci (generates 3 images)', default='norm')
    parser.add_argument('--preprocess-roi', help='Generates a bunch preprocessed images for the input. Example --preprocessor norm,lce,ci (generates 3 images)', default='norm')
    parser.add_argument('-b', '--blob-detector', help='Options: wmci-mean-shape, wmci-aam.', default='wmci-aam')
    parser.add_argument('--eval-wmci', help='Measure sensitivity and fppi without classification', action='store_true')
    parser.add_argument('-d', '--descriptor', help='Options: none, baseline.hardie, hog, hogio, lbpio, zernike, shape, all, set1, overf, overfin.', default=None, type=str)
    parser.add_argument('-c', '--classifier', help='Options: lda(default), svm.', default='lda')
    parser.add_argument('-r', '--reductor', help='Feature reductor or selector. Options: none(default), pca, lda, rfe, rlr.', default='none')
    
    parser.add_argument('--fts', help='Performs feature extraction.', action='store_true')
    parser.add_argument('--clf', help='Performs classification.', action='store_true')
    parser.add_argument('--hyp', help='Performs hyperparameter search. The target method to evaluate should be specified using -t.', action='store_true')
    parser.add_argument('-t', '--target', help='Method to be optimized. Options wmci, pca, lda, rlr, rfe, svm, ', default='svm')
    parser.add_argument('--cmp', help='Compares results of different models via froc. Options: hog, hog-impls, lbp, clf.', default='none')
    parser.add_argument('--fw', help='Plot the importance of individual features ( selected clf coef vs anova ) ', action='store_true')
    parser.add_argument('--label', help='Options: nodule, sublety.', default='nodule')

    # BAG OF VISUAL WORDS
    parser.add_argument('--bovw', help='Options: check available configs on bovw.py.', default='none')
    parser.add_argument('--clf-foldwise', help='Performs classification loading features foldwise.', default='none')
    
    # DEEP LEARNING

    # Commons
    parser.add_argument('--cnn', help='Evaluate convnet. Options: shallow_1, shallow_2.', default='none')
    parser.add_argument('-l', '--layer', help='Layer index used to extract feature from cnn model.', default=-1, type=int)
    parser.add_argument('--hybrid', help='Evaluate hybrid approach: convnet + descriptor.', action='store_true')
    parser.add_argument('--sota', help='Performs classification.', action='store_true')

    # Fussion
    parser.add_argument('--trf-channels', help='', action='store_true') # Early fussion channels
    parser.add_argument('--streams', help='Options: trf (transformations), seg (segmentation), fovea (center-scaling), none (default)', default='none') # Late fussion opts
    
    # Fussion : TODO
    parser.add_argument('--early', help='Options: trf, seg, fovea, none(lce, default)', default='none')
    parser.add_argument('--late', help='Options: trf, seg, fovea, none(lce, default)', default='none')

    # Comparing models
    parser.add_argument('--cmp-cnn', help = 'Compare models (mod), preprocessing (pre), regularization (reg), optimization (opt), \
                                            max-pooling stages (mp), number of feature maps (nfm), dropout (dp), \
                                            mlp width (clf-width, deprecated), common classifiers (skl), hybrid cnn + features (hyb), \
                                            hybrid model evaluated with linear SVM grid search on C (hyp-hyb)', \
                                            default='none') 

    # Unsupervised learning
    parser.add_argument('--pre-tr', help='Enable pretraining', action='store_true')
    parser.add_argument('--joint-tr', help='Joint supervised-unsupervised traininig', action='store_true')
    parser.add_argument('--uns-aug', help='Enable pretraining', action='store_true')
    parser.add_argument('--rpi', help='Number of random regions of interest per image you want to augment', default=0, type=int)
    parser.add_argument('--init', help='Enable initialization from a existing network', default='none')
    parser.add_argument('--pos-neg-ratio', help='Set the positive/negative ratio for unsupervised aug experiments', default=0.1, type=float)
    parser.add_argument('--multipliers', nargs='+', type=float, default=[3e-4, 1e-5, 1e-5])
    parser.add_argument('--lw-epochs', help='number of epochs used on layerwise initialization', default=6, type=int)
    parser.add_argument('--dec-epochs', help='number of epochs used on decoder training', default=6, type=int)
    parser.add_argument('--lr', help='Learning rate', default=0.01, type=float)

    # Transfer learning
    parser.add_argument('--transfer', help='Enable initialization from a existing network', action='store_true')

    # TODO: Optimization
    parser.add_argument('--opt', help='Select an optimization algorithm: sgd-nesterov, adagrad, adadelta, adam.', default='sgd-nesterov')

    # Options
    parser.add_argument('-a', '--augment', help='Augmentation configurations: bt, zcabt, xbt.', default='bt')
    parser.add_argument('--roi-size', help='Size of ROIs after scaling', default=64, type=int)
    parser.add_argument('--blob-rad', help='Radius used to extract blobs', default=32, type=int)
    parser.add_argument('--epochs', help='Number of epochs for pretraining.', default=1, type=int)

    # Evals
    parser.add_argument('--frocs-by-epoch', help='Generate a figure with froc curves every 5 epochs', action='store_true')
    parser.add_argument('--bootstrap', help='Bootstrap', action='store_true')

    # HDF5
    parser.add_argument('--create-datasets', help= 'Create rois set, train and test set for each fold, and store in file', default=None, type=str)
    parser.add_argument('--datasets', help= 'Create rois set, train and test set for each fold, and store in file', default=None, type=str)
    parser.add_argument('--preprocess-dataset', help= 'Create rois set, train and test set for each fold, and store in file', default='zmuv', type=str)

    args = parser.parse_args()
    opts = vars(args)
    extractor_key = args.descriptor
    detection_model = model.BaselineModel("data/default")
    if (args.descriptor):
        detection_model.extractor = model.extractors[args.descriptor]
    detection_model.args = args

    detection_model.preprocess_lung = args.preprocess_lung
    detection_model.preprocess_roi = args.preprocess_roi
    detection_model.roi_size = args.roi_size
    detection_model.blob_rad = args.blob_rad
    detection_model.use_transformations = args.trf_channels
    detection_model.streams = args.streams 
    detection_model.label = args.label
    detection_model.multipliers = args.multipliers

    detection_model.augment = args.augment
    detection_model.downsample = True
    detection_model.lw_epochs = args.lw_epochs
    detection_model.dec_epochs = args.dec_epochs
    detection_model.all_epochs = args.epochs
    detection_model.lr = args.lr
    detection_model.init = args.init

    if args.datasets != None:
        detection_model.dataset_type = 'hdf5'
        detection_model.open_datasets_file(args.datasets)
         
    elif args.eval_wmci:
        eval_wmci_and_postprocessing(detection_model, extractor_key)

    elif args.create_datasets != None:
        detection_model.create_datasets_file(args.create_datasets)
        detection_model.dataset_type = 'hdf5'
        create_datasets(detection_model, extractor_key)

    elif args.frocs_by_epoch:
        frocs_by_epoch(detection_model, extractor_key, args.cnn)

    elif args.bovw != 'none':
        detection_model.name = 'bovw-{}'.format(args.bovw)
        bovw_folds(detection_model, extractor_key, args.bovw)

    elif args.clf_foldwise != 'none':
        detection_model.name = 'data/bovw-{}-clf-{}'.format(args.clf_foldwise, args.classifier)
        detection_model.clf = model.classifiers[args.classifier]
        classify_foldwise(detection_model, extractor_key, args.clf_foldwise)

    elif args.cmp_cnn == 'opts':
        networks = ['6P-nesterov', '6P-adagrad', '6P-adadelta', '6P-adam']
        network_labels = ['6P, nesterov', '6P, adagrad', '6P, adadelta', '6P, adam']
        compare_cnndetection_models(detection_model, '{}'.format(extractor_key), networks, network_labels, 'data/models')

    elif args.cmp_cnn == 'caes':
        networks = ['3P', '3P-CAE1', '3P-CAE5', '3P-CAE10', '3P-CAE15', '3P-CAE20']
        network_labels = ['Orthogonal init.', 'CAE, 1 epoch', 'CAE, 5 epochs', 'CAE, 10 epochs', 'CAE, 15 epochs', 'CAE, 20 epochs']
        compare_cnndetection_models(detection_model, '{}'.format(extractor_key), networks, network_labels, 'data/models')

    elif args.cmp_cnn == 'caes2':
        networks = ['3PE40', '3P-CAE'] 
        network_labels = ['Orthogonal init.', 'CAE init']
        compare_cnndetection_models(detection_model, '{}'.format(extractor_key), networks, network_labels, 'data/models')

    elif args.cmp_cnn == 'mod':
        networks = ['LND-A', 'LND-B', 'LND-C', 'LND-A-5P-C2', 'LND-A-ALLCNN-5P', 'LND-C-5P']
        network_labels = ['LND-A', 'LND-B', 'LND-C', 'LND-A-5P, 2x2 conv on top', 'ALLCNN-A, 5 maxpool', 'LND-C, 5 maxpool']
        compare_cnndetection_models(detection_model, '{}'.format(extractor_key), networks, network_labels, 'data/models')

    elif args.cmp_cnn == 'pre':
        networks = ['LND-A-5P', 'LND-A-5P-GCN', 'LND-A-5P-ZCA', 'LND-A-5P-ZMUV']
        network_labels = ['LND-A-5P', 'LND-A-5P with GCN', 'LND-A-5P with ZCA', 'LND-A-5P with ZMUV']
        compare_cnndetection_models(detection_model, '{}'.format(extractor_key), networks, network_labels, 'data/pre')

    elif args.cmp_cnn == 'reg':
        networks = ['LND-A-5P', 'LND-A-5P-L2']
        network_labels = ['LND-A-5P', 'LND-A-5P, L2 regularization.']
        compare_cnndetection_models(detection_model, '{}'.format(extractor_key), networks, network_labels, 'data/reg')

    elif args.cmp_cnn == 'mp':
        networks = ['LND-A-3P', 'LND-A-4P', 'LND-A-5P']
        network_labels = ['LND-A, 3 pooling stages', 'LND-A, 4 pooling stages', 'LND-A, 5 pooling stages']
        compare_cnndetection_models(detection_model, '{}'.format(extractor_key), networks, network_labels, 'data/pool_stages')

    elif args.cmp_cnn == 'nfm':
        networks = ['LND-A-5P', 'LND-A-5P-64', 'LND-A-5P-96']
        network_labels = ['LND-A-5P, layer * 32 filters', 'LND-A-5P, layer * 64 filters', 'LND-A-5P, layer * 96 filters']
        compare_cnndetection_models(detection_model, '{}'.format(extractor_key), networks, network_labels, 'data/fmaps')

    elif args.cmp_cnn == 'dp':
        networks = ['LND-A-5P', 'LND-A-5P-LDP', 'LND-A-5P-LDP2']
        network_labels = ['LND-A-5P, 0.25 dp', 'LND-A-5P, 0.15 + layer * 0.05 dp', 'LND-A-5P, 0.1 + layer * 0.1 dp']
        compare_cnndetection_models(detection_model, '{}'.format(extractor_key), networks, network_labels, 'data/dps')

    elif args.cmp_cnn == 'clf-width':
        networks = ['LND-A-5P', 'LND-A-5P-MLP512', 'LND-A-5P-MLP1024']
        network_labels = ['LND-A-5P with MPL(512-256)', 'LND-A-5P with MPL(512, 512)', 'LND-A-5P with MPL(1024, 1024)']
        compare_cnndetection_models(detection_model, '{}'.format(extractor_key), networks, network_labels, 'data/widths')

    elif args.cmp_cnn == 'skl':
        assert args.cnn != 'none'
        compare_cnn_sklearn_clfs(detection_model, '{}'.format(extractor_key), args.cnn)

    elif args.cmp_cnn == 'hyb':
        compare_cnn_hybrid(detection_model, '{}'.format(extractor_key), args.cnn)

    elif args.cmp_cnn == 'hyp-hyb':
        hyp_cnn_lsvm_hybrid(detection_model, '{}'.format(extractor_key), args.cnn)

    elif args.pre_tr == 'conv':
        pretrain_convnet(detection_model, args.extractor_key, args.cnn, nb_epoch=args.epochs)

    elif args.cmp != 'none':
        if args.clf:
            detection_model.clf = model.classifiers[args.classifier]
        detection_model.name = 'data/{}'.format(args.cmp)

        if args.cmp == 'hog':
            hog_froc(detection_model, 'hog', args.fts, args.clf) 
        elif args.cmp == 'hog-impls':
            hog_impls(detection_model, 'hog-impls', args.fts, args.clf)  
        elif args.cmp == 'lbp':
            lbp_froc(detection_model, 'lbp', args.fts, args.clf, mode='default') 
        elif args.cmp == 'lbp-inner':
            lbp_froc(detection_model, 'lbp-inner', args.fts, args.clf, mode='inner') 
        elif args.cmp == 'lbp-io':
            lbp_froc(detection_model, 'lbp-io', args.fts, args.clf, mode='inner_outer')
        elif args.cmp == 'znk':
            znk_froc(detection_model, 'znk', args.fts, args.clf)     
        elif args.cmp == 'hrg':
            hrg_froc(detection_model, 'hrg', args.fts, args.clf)
        elif args.cmp == 'clf':
            detection_model.name = '{}_{}'.format(detection_model.name, extractor_key)
            protocol_clf_eval_froc(detection_model, '{}'.format(extractor_key))
        elif args.cmp == 'bovw':
            models = ['img-hard', 'img-soft', 'hog-hard', 'hog']
            labels = ['IMG features, hard pooling', 'IMG feature, soft pooling', 'HOG features, hard pooling', 'HOG features, soft pooling'] 
            detection_model.clf = model.classifiers[args.classifier]
            compare_bovwdetection_models(detection_model, extractor_key, models, labels, 'data/bovw-{}'.format(args.classifier))

        elif args.cmp == 'bovw-soft-k':
            models = ['hog-soft-16', 'hog-soft-32', 'hog-soft-64', 'hog-soft-256', 'hog-soft-512', 'hog-soft-1024']
            labels = ['hog-soft, k=16', 'hog-soft, k=32', 'hog-soft, k=64', 'hog-soft, k=256', 'hog-soft, k=512', 'hog-soft, k=1024']
            detection_model.clf = model.classifiers[args.classifier]
            compare_bovwdetection_models(detection_model, extractor_key, models, labels, 'data/bovw-hsoft-k-{}'.format(args.classifier))
            
    elif args.hybrid:
        assert args.cnn != 'none'
        detection_model.clf = model.classifiers[args.classifier]
        hybrid(detection_model, '{}'.format(extractor_key), args.cnn, args.layer)

    elif args.cnn != 'none':
        if args.joint_tr:
            joint_training(detection_model, extractor_key, args.cnn, args.init)
        elif args.uns_aug:
            unsupervised_augment(detection_model, extractor_key, args.cnn, args.init, args.epochs, mode='add-random', rois_per_image=args.rpi, pos_neg_ratio=args.pos_neg_ratio)
        elif args.pre_tr:
            pretrain_cnn(detection_model, extractor_key, args.cnn, args.init, nb_epoch=args.epochs)
        # extract features fron a specific layer. 
        elif args.fts:
            extract_features_cnn(detection_model, extractor_key, args.cnn, args.layer)
        # perform classification stage convnet only.
        elif args.clf:
            classify_cnn(detection_model, extractor_key, args.cnn)

        elif args.sota:
            compare_cnn_sota(detection_model, extractor_key, [args.cnn], [args.cnn], args.cnn)
 
        # perform detection pipeline with convnet model only
        else:
            if args.init != 'none':
                protocol_pretrained_cnn(detection_model, extractor_key, args.cnn, args.init)
            elif args.file:
                print 'args.file {}'.format(args.file)
                image = np.load(args.file).astype(np.float32)
                detect_(detection_model, args.cnn, image)
            else:
                protocol_cnn_froc(detection_model, '{}'.format(extractor_key), args.cnn)

    else:
        method = protocol_froc_2
        if args.fts:
            method = protocol_froc_1

        elif args.clf:
            method = protocol_froc_2
            detection_model.clf = model.classifiers[args.classifier]
            detection_model.selector = model.reductors[args.reductor]
            if args.reductor != 'none':
                detection_model.name += '-{}'.format(args.reductor)
            detection_model.name += '-{}'.format(args.classifier)
                
        elif args.hyp:
            if args.target == 'wmci':
                method = protocol_wmci_froc
            if args.target == 'pca':
                method = protocol_pca_froc
            if args.target == 'lda':
                method = protocol_lda_froc
            if args.target == 'svm':
                method = protocol_svm_hp_search
            if args.target == 'rfe':
                method = protocol_rfe_froc
            if args.target == 'rlr':
                method = protocol_rlr_froc
            if args.target == 'rfe':
                method = protocol_lda_froc
            detection_model.name = '{}_{}'.format(detection_model.name, args.target)

        print detection_model.extractor
        if args.clf == True:
            method(detection_model, '{}'.format(extractor_key), args.fw)
        else:
            method(detection_model, '{}'.format(extractor_key))


