#!/usr/bin/env python
import sys
import time
from itertools import product

import numpy as np
from scipy.interpolate import interp1d
from sklearn.cross_validation import StratifiedKFold
from sklearn import svm
from sklearn import lda
from sklearn import decomposition
from sklearn import feature_selection as selection
from sklearn import linear_model
from sklearn import ensemble
from sklearn.metrics import auc
import matplotlib.pyplot as plt

from data import DataProvider
import model
import eval
import util
import sys
import argparse

import jsrt

#fppi range
step = 10
fppi_range = np.linspace(0.0, 10.0, 101)

# Baselines
hardie = np.array([[0.0, 0.0], [0.1, 0.2], [0.2, 0.3],[0.3, 0.38], [0.4, 0.415], [0.5, 0.46], [0.6, 0.48], [0.7, 0.51], [0.9, 0.53], [1.0, 0.57], [1.5, 0.67], [2.0, 0.72], [2.5, 0.75],[3.0, 0.78], [4.0, 0.79], [5.0, 0.81], [6.0, 0.82], [7.0, 0.85], [8.0, 0.86], [9.0, 0.895], [10.0, 0.93]])
fun = interp1d(hardie.T[0], hardie.T[1], kind='linear', fill_value=0, bounds_error=False)
hardie = np.array([fppi_range, fppi_range.copy()])
hardie[1] = fun(hardie[0])
hardie = hardie.T

horvath = np.array([[0.0, 0.0], [0.5, 0.49], [1.0, 0.63], [1.5, 0.68], [2.0, 0.72], [2.5, 0.75], [3.0, 0.78], [3.5, 0.79], [4.0, 0.81], [10.0, 0.81]])
fun = interp1d(horvath.T[0], horvath.T[1], kind='linear', fill_value=0, bounds_error=False)
horvath = np.array([fppi_range, fppi_range.copy()])
horvath[1] = fun(horvath[0])
horvath = horvath.T

vde = np.array([[0.0, 0.0], [0.5, 0.56], [1.0, 0.66], [1.5, 0.68], [2.0, 0.78], [2.5, 0.79], [3.0, 0.81], [3.5, 0.84], [4.0, 0.85], [5.0, 0.86], [10.0, 0.86]])
fun = interp1d(vde.T[0], vde.T[1], kind='linear', fill_value=0, bounds_error=False)
vde = np.array([fppi_range, fppi_range.copy()])
vde[1] = fun(vde[0])
vde = vde.T
'''
returns: Free Receiving Operating Curve obtained given a fold set
'''


def get_froc_on_folds(_model, paths, left_masks, right_masks, blobs, pred_blobs, feats, folds):
	print 'feature shape {}'.format(feats[0].shape)

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

		print 'Train with {}...'.format(_model.clf)
		_model.train_with_feature_set(feats[tr_idx], pred_blobs[tr_idx], blobs[tr_idx])
		print 'Predict ...'
		blobs_te_pred, probs_te_pred = _model.predict_proba_from_feature_set(feats[te_idx], pred_blobs[te_idx])
		print 'Get froc ...'
		froc = eval.froc(blobs_te, blobs_te_pred, probs_te_pred)

		frocs.append(froc)
		fold += 1
		sys.stdout.flush()
	
	av_froc = eval.average_froc(frocs, fppi_range)
	return av_froc

def get_froc_on_folds_keras(_model, paths, left_masks, right_masks, blobs, pred_blobs, rois, folds, network_model):
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

		history = _model.train_with_feature_set_keras(rois[tr_idx], pred_blobs[tr_idx], blobs[tr_idx], 
											rois[te_idx], pred_blobs[te_idx], blobs[te_idx], 
											model=network_model)
		if fold == 1:
			model.layer_utils.model_summary(_model.keras_model)			

		blobs_te_pred, probs_te_pred = _model.predict_proba_from_feature_set_keras(rois[te_idx], pred_blobs[te_idx])

		froc = eval.froc(blobs_te, blobs_te_pred, probs_te_pred)
		frocs.append(froc)

		_model.save('data/{}_fold_{}'.format(network_model, fold))
		util.save_loss(history, 'data/{}_fold_{}'.format(network_model, fold))
		fold += 1

	av_froc = eval.average_froc(frocs, np.linspace(0.0, 10.0, 101))

	return av_froc

def extract_features_cnn(_model, paths, left_masks, right_masks, blobs, pred_blobs, feats, rois, folds, network_model, cut):
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
		_model.load_cnn('data/{}_fold_{}'.format(network_model, fold))
		print 'extract ...'
		network_feats = _model.extract_features_from_keras_model(rois, cut)
		print 'save ...'
		np.save('data/{}_fold_{}.fts.npy'.format(network_model, fold), network_feats)

def get_froc_on_folds_hybrid(_model, paths, left_masks, right_masks, blobs, pred_blobs, feats, rois, folds, network_model):
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

		_model.load_cnn('data/{}_fold_{}'.format(network_model, fold))
		network_feats = np.load('data/{}_folds_{}.fts.npy'.format(network_model, fold))

		# stack features
		assert len(feats) == len(network_feats)

		hybrid_feats = []
		for i in range(len(feats)):
			hybrid_feats.append(np.hstack([feats[i], network_feats[i]]))

		hybrid_feats = np.array(hybrid_feats)	
		print 'hybrid feats shape {}'.format(hybrid_feats[0].shape)
		print 'Train with {}...'.format(_model.clf)
		_model.train_with_feature_set(hybrid_feats[tr_idx], pred_blobs[tr_idx], blobs[tr_idx])
		print 'Predict ...'
		blobs_te_pred, probs_te_pred = _model.predict_proba_from_feature_set(hybrid_feats[te_idx], pred_blobs[te_idx])
		print 'Get froc ...'

		# Append froc
		froc = eval.froc(blobs_te, blobs_te_pred, probs_te_pred)
		frocs.append(froc)
		fold += 1

	av_froc = eval.average_froc(frocs, np.linspace(0.0, 10.0, 101))

	return av_froc

def protocol_two_stages():
	paths, locs, rads = jsrt.jsrt(set='jsrt140')
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
	skf = StratifiedKFold(Y, n_folds=10, shuffle=True, random_state=113)
	fold = 0

	sens = []
	fppi_mean = []
	fppi_std = []

	for tr_idx, te_idx in skf:
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

def protocol_froc_1(_model, fname):
	print '# {}'.format(fname)
	paths, locs, rads = jsrt.jsrt(set='jsrt140')
	left_masks = jsrt.left_lung(set='jsrt140')
	right_masks = jsrt.right_lung(set='jsrt140')
	size = len(paths)

	# blobs detection
	blobs = []
	for i in range(size):
		blobs.append([locs[i][0], locs[i][1], rads[i]])
	blobs = np.array(blobs)

	# feature extraction
	print "Extracting blobs & features ..."
	data = DataProvider(paths, left_masks, right_masks)
	feats, pred_blobs = _model.extract_feature_set(data)

	av_cpi = 0
	for tmp in pred_blobs:
		av_cpi += len(tmp)
	print "Average blobs per image {} ...".format(av_cpi * 1.0 / len(pred_blobs))

	np.save('data/{}.fts.npy'.format(fname), feats)
	np.save('data/{}_pred.blb.npy'.format(fname), pred_blobs)

def protocol_froc_2(_model, fname):
	paths, locs, rads = jsrt.jsrt(set='jsrt140')
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
	print feats[0].shape
	pred_blobs = np.load('data/{}_pred.blb.npy'.format(fname))

	av_cpi = 0
	for tmp in pred_blobs:
		av_cpi += len(tmp)
	print "Average blobs per image {} ...".format(av_cpi * 1.0 / len(pred_blobs))

	Y = (140 > np.array(range(size))).astype(np.uint8)
	skf = StratifiedKFold(Y, n_folds=10, shuffle=True, random_state=113)

	ops = get_froc_on_folds(_model, paths, left_masks, right_masks, blobs, pred_blobs, feats, skf)
	range_ops = ops[step * 2:step * 4 + 1]
	print 'auc 2 - 4 fppis {}'.format(auc(range_ops.T[0], range_ops.T[1]))
	

	legend = []
	legend.append('Hardie et al')
	legend.append(_model.name)

	util.save_froc([hardie, ops], '{}'.format(_model.name), legend)

	return ops

def protocol_wmci_froc(_model, fname):
	paths, locs, rads = jsrt.jsrt(set='jsrt140')
	left_masks = jsrt.left_lung(set='jsrt140')
	right_masks = jsrt.right_lung(set='jsrt140')
	size = len(paths)

	blobs = []
	for i in range(size):
		blobs.append([locs[i][0], locs[i][1], rads[i]])
	blobs = np.array(blobs)

	print "Loading	 blobs & features ..."
	data = DataProvider(paths, left_masks, right_masks)
	feats, pred_blobs, proba = _model.extract_feature_set_proba(data)

	'''
	feats = np.load('data/{}.fts.npy'.format(fname))
	pred_blobs = np.load('data/{}_pred.blb.npy'.format(fname))
	'''

	av_cpi = 0
	for tmp in pred_blobs:
		av_cpi += len(tmp)
	print "Average blobs per image {} ...".format(av_cpi * 1.0 / len(pred_blobs))
	Y = (140 > np.array(range(size))).astype(np.uint8)
	skf = StratifiedKFold(Y, n_folds=10, shuffle=True, random_state=113)

	op_set = []
	op_set.append(hardie)
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

		ops = get_froc_on_folds(_model, paths, left_masks, right_masks, blobs, selected_blobs, selected_feats, skf)
		op_set.append(ops)

	op_set = np.array(op_set)
	legend = []
	legend.append("hardie")
	for thold in detect_range:
		legend.append('wmci {}'.format(thold))

	util.save_froc(op_set, _model.name, legend)
	return op_set

def protocol_generic_froc(_model, fnames, components, legend, kind='descriptor', mode='feats'):
	paths, locs, rads = jsrt.jsrt(set='jsrt140')
	left_masks = jsrt.left_lung(set='jsrt140')
	right_masks = jsrt.right_lung(set='jsrt140')
	size = len(paths)

	blobs = []
	for i in range(size):
		blobs.append([locs[i][0], locs[i][1], rads[i]])
	blobs = np.array(blobs)

	data = DataProvider(paths, left_masks, right_masks)
	Y = (140 > np.array(range(size))).astype(np.uint8)
	skf = StratifiedKFold(Y, n_folds=10, shuffle=True, random_state=113)
	
	op_set = []
	op_set.append(hardie)
	legend.insert(0, "hardie")

	for i in range(len(components)):
		print "Loading blobs & features ..."
		feats = np.load('data/{}.fts.npy'.format(fnames[i]))
		pred_blobs = np.load('data/{}_pred.blb.npy'.format(fnames[i]))

		print legend[i+1]
		if kind == 'descriptor':
			_model.descriptor = components[i]
		elif kind == 'selector':
			_model.selector = components[i]
		elif kind == 'classifier':
			_model.classifier = components[i]

		ops = get_froc_on_folds(_model, paths, left_masks, right_masks, blobs, pred_blobs, feats, skf)
		op_set.append(ops)

	op_set = np.array(op_set)
	util.save_froc(op_set, _model.name, legend)

	return op_set


def protocol_selector_froc(_model, fname, selectors, legend):
	paths, locs, rads = jsrt.jsrt(set='jsrt140')
	left_masks = jsrt.left_lung(set='jsrt140')
	right_masks = jsrt.right_lung(set='jsrt140')
	size = len(paths)

	blobs = []
	for i in range(size):
		blobs.append([locs[i][0], locs[i][1], rads[i]])
	blobs = np.array(blobs)

	print "Loading	 blobs & features ..."
	data = DataProvider(paths, left_masks, right_masks)

	feats = np.load('data/{}.fts.npy'.format(fname))
	pred_blobs = np.load('data/{}_pred.blb.npy'.format(fname))

	av_cpi = 0
	for tmp in pred_blobs:
		av_cpi += len(tmp)
	print "Average blobs per image {} ...".format(av_cpi * 1.0 / len(pred_blobs))

	Y = (140 > np.array(range(size))).astype(np.uint8)
	skf = StratifiedKFold(Y, n_folds=10, shuffle=True, random_state=113)
	
	op_set = []
	
	op_set.append(hardie)
	legend.insert(0, "hardie")

	for i in range(len(selectors)):
		print legend[i+1]
		_model.selector = selectors[i]
		ops = get_froc_on_folds(_model, paths, left_masks, right_masks, blobs, pred_blobs, feats, skf)
		op_set.append(ops)

	op_set = np.array(op_set)
	util.save_froc(op_set, _model.name, legend)

	return op_set

def protocol_classifier_froc(_model, fname, classifiers, legend):
	paths, locs, rads = jsrt.jsrt(set='jsrt140')
	left_masks = jsrt.left_lung(set='jsrt140')
	right_masks = jsrt.right_lung(set='jsrt140')
	size = len(paths)

	blobs = []
	for i in range(size):
		blobs.append([locs[i][0], locs[i][1], rads[i]])
	blobs = np.array(blobs)

	print "Loading	 blobs & features ..."
	data = DataProvider(paths, left_masks, right_masks)

	feats = np.load('data/{}.fts.npy'.format(fname))
	pred_blobs = np.load('data/{}_pred.blb.npy'.format(fname))

	av_cpi = 0
	for tmp in pred_blobs:
		av_cpi += len(tmp)
	print "Average blobs per image {} ...".format(av_cpi * 1.0 / len(pred_blobs))

	Y = (140 > np.array(range(size))).astype(np.uint8)
	skf = StratifiedKFold(Y, n_folds=10, shuffle=True, random_state=113)
	
	op_set = []

	op_set.append(hardie)
	legend.insert(0, "hardie")

	for i in range(len(classifiers)):
		print legend[i+1]
		_model.clf = classifiers[i]
		ops = get_froc_on_folds(_model, paths, left_masks, right_masks, blobs, pred_blobs, feats, skf) 
		op_set.append(ops)
		
		# Somewhat free memory
		classifiers[i] = None
		_model.clf = None

	op_set = np.array(op_set)
	util.save_froc(op_set, '{}'.format(_model.name), legend)

	return op_set[1:]

def hog_impls(_model, fname, fts=False, clf=True):
	descriptors = []
	labels = []
	fnames = []

	for inp, mode in product(['lce', 'norm', 'wmci'], ['skimage_default', 'default']):
		fnames.append('{}_{}_{}'.format(fname, inp, mode))
		labels.append('{}_{}_{}'.format(fname, inp, mode))
		descriptors.append(model.HogExtractor(mode=mode, input=inp))
		if fts:
			_model.extractor = descriptors[-1]
			protocol_froc_1(_model, fnames[-1])	

	if clf:
		protocol_generic_froc(_model, fnames, descriptors, labels, kind='descriptor')
		
# FIX: Dropout skimage impl and test type and modes
def hog_froc(_model, fname, fts=False, clf=True):
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
			_model.extractor = descriptor
			protocol_froc_1(_model, fnames[-1])

	if clf:
		protocol_generic_froc(_model, fnames, descriptors, labels, kind='descriptor')

def hrg_froc(_model, fname, fts=False, clf=True):
	descriptors = []
	labels = []
	fnames = []

	for inp, mode in product(['lce', 'norm', 'wmci'], ['default', 'inner', 'inner_outer']):
		fnames.append('{}_{}_{}'.format(fname, inp, mode))
		labels.append('{}_{}_{}'.format(fname, inp, mode))
		descriptors.append(model.HRGExtractor(mode=mode, input=inp))
		if fts:
			_model.extractor = descriptors[-1]
			protocol_froc_1(_model, fnames[-1])	

	if clf:
		protocol_generic_froc(_model, fnames, descriptors, labels, kind='descriptor')
		

def lbp_froc(_model, fname, fts=False, clf=True, mode='default'):
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
			_model.extractor = descriptor
			protocol_froc_1(_model, fnames[-1])

	if clf:
		protocol_generic_froc(_model, fnames, descriptors, labels, kind='descriptor')
	
def znk_froc(_model, fname, fts=False, clf=True):
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
			_model.extractor = descriptor
			protocol_froc_1(_model, fnames[-1])

	if clf:
		protocol_generic_froc(_model, fnames, descriptors, labels, kind='descriptor')
	

def protocol_clf_eval_froc(_model, fname):
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

	protocol_classifier_froc(_model, fname, classifiers, labels)

def protocol_svm_hp_search(_model, fname):
	# Stage 1
	#C_set = np.logspace(-3, 4, 8)
	#g_set = np.logspace(-3, -1, 9)
	# Stage 2
	C_set = np.logspace(-1, 1, 9)
	g_set = np.logspace(-2.25, -1.75, 9)

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

	ops = protocol_classifier_froc(_model, fname, classifiers, legend)
	
	# compute the AUC from 2 to 4
	auc_grid = []
	for i in range(ops.shape[0]):
		range_ops = ops[i][step * 2:step * 4 + 1]
		auc_grid.append(auc(range_ops.T[0], range_ops.T[1]))
	auc_grid = np.array(auc_grid).reshape((C_set.shape[0], g_set.shape[0]))	
	print "AUC GRID"
	print auc_grid
	util.save_grid(auc_grid, _model.name, ['C', 'gamma'], [C_set, g_set], title='AUC between 2 and 4 FPPI\'s')

def protocol_pca_froc(_model, fname):
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

	protocol_selector_froc(_model, fname, selectors, labels)

def protocol_lda_froc(_model, fname):
	selectors = []
	labels = []
	var_set = [0.0125, 0.025, 0.05]
	var_set = np.append(var_set, np.arange(0.1, 0.9, 0.1))

	for var in var_set:
		selector = selection.SelectFromModel(lda.LDA(), threshold=var)
		selectors.append(selector)
		labels.append('thold {}'.format(var))

	protocol_selector_froc(_model, fname, selectors, labels)

def protocol_rlr_froc(_model, fname):
	selectors = []
	labels = []
	var_set = np.arange(5, 106, 10)

	for var in var_set:
		rlr = linear_model.RandomizedLogisticRegression(C=var)
		selectors.append(rlr)
		labels.append('C {}'.format(var))

	protocol_selector_froc(_model, fname, selectors, labels)

def protocol_rfe_froc(_model, fname):
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

	protocol_selector_froc(_model, fname, selectors, labels)

def protocol_cnn_froc(detections_source, fname, network_model):
	paths, locs, rads = jsrt.jsrt(set='jsrt140')
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
	rois = model.create_rois(data, pred_blobs)

	av_cpi = 0
	for tmp in pred_blobs:
		av_cpi += len(tmp)
	print "Average blobs per image {} ...".format(av_cpi * 1.0 / len(pred_blobs))

	Y = (140 > np.array(range(size))).astype(np.uint8)
	skf = StratifiedKFold(Y, n_folds=10, shuffle=True, random_state=113)
	
	ops = get_froc_on_folds_keras(detections_source, paths, left_masks, right_masks, blobs, pred_blobs, rois, skf, network_model)

	legend = []
	legend.append('Hardie et al')
	legend.append('current')

	util.save_froc([hardie, ops], '{}_cnn'.format(_model.name), legend)

	return ops

def hybrid(detections_source, fname, network_model):
	paths, locs, rads = jsrt.jsrt(set='jsrt140')
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
	rois = model.create_rois(data, pred_blobs)

	av_cpi = 0
	for tmp in pred_blobs:
		av_cpi += len(tmp)
	print "Average blobs per image {} ...".format(av_cpi * 1.0 / len(pred_blobs))

	Y = (140 > np.array(range(size))).astype(np.uint8)
	skf = StratifiedKFold(Y, n_folds=10, shuffle=True, random_state=113)
	
	ops = get_froc_on_folds_hybrid(detections_source, paths, left_masks, right_masks, blobs, pred_blobs, feats, rois, skf, network_model)

	legend = []
	legend.append('Hardie et al')
	legend.append('current')

	util.save_froc([hardie, ops], '{}_hybrid'.format(_model.name), legend)

	return ops



if __name__=="__main__":	
	parser = argparse.ArgumentParser(prog='lnd.py')
	parser.add_argument('-b', '--blob-detector', help='Options: wmci(default), TODO hog, log.', default='wmci')
	parser.add_argument('-d', '--descriptor', help='Options: hardie(default), hog, hogio, lbpio, zernike, shape, all, set1, overf, overfin.', default='hardie')
	parser.add_argument('-r', '--reductor', help='Feature reductor or selector. Options: none(default), pca, lda, rfe, rlr.', default='none')
	parser.add_argument('-c', '--classifier', help='Options: lda(default), svm.', default='lda')
	parser.add_argument('--fts', help='Performs feature extraction.', action='store_true')
	parser.add_argument('--clf', help='Performs classification.', action='store_true')
	parser.add_argument('--hyp', help='Performs hyperparameter search. The target method to evaluate should be specified using -t.', action='store_true')
	parser.add_argument('--cmp', help='Compare results of different models via froc. Options: hog, hog-impls, lbp, clf.', default='none')
	parser.add_argument('--cnn', help='Evaluate convnet. Options: shallow_1, shallow_2.', default='none')
	parser.add_argument('--hybrid', help='Evaluate hybrid approach: convnet + descriptor.', action='store_true')
	parser.add_argument('-t', '--target', help='Method to be optimized. Options wmci, pca, lda, rlr, rfe, svm, ', default='svm')

	args = parser.parse_args()
	opts = vars(args)
	extractor_key = args.descriptor
	_model = model.BaselineModel("data/default")
	_model.name = 'data/{}'.format(extractor_key)
	_model.extractor = model.extractors[args.descriptor]

	# default: clf -d hardie
	if args.cmp != 'none':
		if args.clf:
			_model.clf = model.classifiers[args.classifier]
		_model.name = 'data/{}'.format(args.cmp)

		if args.cmp == 'hog':
			hog_froc(_model, 'hog', args.fts, args.clf)	
		elif args.cmp == 'hog-impls':
			hog_impls(_model, 'hog-impls', args.fts, args.clf)	
		elif args.cmp == 'lbp':
			lbp_froc(_model, 'lbp', args.fts, args.clf, mode='default')	
		elif args.cmp == 'lbp-inner':
			lbp_froc(_model, 'lbp-inner', args.fts, args.clf, mode='inner')	
		elif args.cmp == 'lbp-io':
			lbp_froc(_model, 'lbp-io', args.fts, args.clf, mode='inner_outer')
		elif args.cmp == 'znk':
			znk_froc(_model, 'znk', args.fts, args.clf)		
		elif args.cmp == 'hrg':
			hrg_froc(_model, 'hrg', args.fts, args.clf)
		elif args.cmp == 'clf':
			_model.name = '{}_{}'.format(_model.name, extractor_key)
			protocol_clf_eval_froc(_model, '{}'.format(extractor_key))
			
	elif args.hybrid:
		assert args.cnn != 'none'
		_model.clf = model.classifiers[args.classifier]
		hybrid(_model, '{}'.format(extractor_key), args.cnn)
	elif args.cnn != 'none':
		protocol_cnn_froc(_model, '{}'.format(extractor_key), args.cnn)
	else:
		method = protocol_froc_2
		if args.fts:
			method = protocol_froc_1
		elif args.clf:
			method = protocol_froc_2
			_model.clf = model.classifiers[args.classifier]
			_model.selector = model.reductors[args.reductor]
			if args.reductor != 'none':
				_model.name += '-{}'.format(args.reductor)
			_model.name += '-{}'.format(args.classifier)
				
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
			_model.name = '{}_{}'.format(_model.name, args.target)

		print _model.extractor
		method(_model, '{}'.format(extractor_key))
