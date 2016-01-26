import sys
import time
from itertools import product

import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn import svm
from sklearn import lda
from sklearn import decomposition
from sklearn import feature_selection as selection
from sklearn import linear_model
import matplotlib.pyplot as plt

from data import DataProvider
import model
import eval
import util
import sys

import jsrt

DATA_LEN = 40

'''
returns: Free Receiving Operating Curve obtained given a fold set
'''

def get_froc_on_folds(_model, paths, left_masks, right_masks, blobs, pred_blobs, feats, folds, tholds):
	ops = []
	sen_set = []
	fppim_set = []
	fppis_set = []
	op_set = []
	fold = 0

	for tr_idx, te_idx in folds:	
		print "Fold {}".format(fold + 1),

		data_te = DataProvider(paths[te_idx], left_masks[te_idx], right_masks[te_idx])
		paths_te = paths[te_idx]
		blobs_te = []

		for bl in blobs[te_idx]:
			blobs_te.append([bl])
		blobs_te = np.array(blobs_te)

		_model.train_with_feature_set(feats[tr_idx], pred_blobs[tr_idx], blobs[tr_idx])
		blobs_te_pred, probs_te_pred = _model.predict_proba_from_feature_set(feats[te_idx], pred_blobs[te_idx])

		sen_set.append([])
		fppim_set.append([])
		fppis_set.append([])
		op_set.append([])

		for thold in tholds:
			fblobs_te_pred, fprobs_te_pred = _model.filter_by_proba(blobs_te_pred, probs_te_pred, thold)
			s, fm, fs = eval.evaluate(blobs_te, fblobs_te_pred, data_te)

			print "thold {}, sens {}, fppi mean {}, fppi std {}".format(thold, s, fm, fs)

			sen_set[-1].append(s)
			fppim_set[-1].append(fm)
			fppis_set[-1].append(fs)
			op_set[-1].append([fm, s])

		fold += 1
	
	# Threshold averaging operating points
	sen_set = np.array(sen_set).T
	fppim_set = np.array(fppim_set).T
	fppis_set = np.array(fppis_set).T
	ta_ops = []
	for i in range(len(tholds)):
		if fppim_set[i].mean() < 10 + util.EPS:
			ta_ops.append([fppim_set[i].mean(), sen_set[i].mean()])
	ta_ops = np.array(ta_ops)

	return ta_ops

def get_froc_on_folds_keras(_model, paths, left_masks, right_masks, blobs, pred_blobs, rois, folds, tholds):
	ops = []
	sen_set = []
	fppim_set = []
	fppis_set = []
	fold = 0

	for tr_idx, te_idx in folds:	
		print "Fold {}".format(fold + 1),
		data_te = DataProvider(paths[te_idx], left_masks[te_idx], right_masks[te_idx])
		paths_te = paths[te_idx]
		blobs_te = []

		for bl in blobs[te_idx]:
			blobs_te.append([bl])
		blobs_te = np.array(blobs_te)

		_model.train_with_feature_set_keras(rois[tr_idx], pred_blobs[tr_idx], blobs[tr_idx])
		blobs_te_pred, probs_te_pred = _model.predict_proba_from_feature_set_keras(rois[te_idx], pred_blobs[te_idx])

		sen_set.append([])
		fppim_set.append([])
		fppis_set.append([])
		for thold in tholds:
			fblobs_te_pred, fprobs_te_pred = _model.filter_by_proba(blobs_te_pred, probs_te_pred, thold)
			s, fm, fs = eval.evaluate(blobs_te, fblobs_te_pred, data_te)

			print "thold {}, sens {}, fppi mean {}, fppi std {}".format(thold, s, fm, fs)

			sen_set[-1].append(s)
			fppim_set[-1].append(fm)
			fppis_set[-1].append(fs)

		fold += 1

	sen_set = np.array(sen_set).T
	fppim_set = np.array(fppim_set).T
	fppis_set = np.array(fppis_set).T
	ta_ops = []
	for i in range(len(tholds)):
		if fppim_set[i].mean() < 10 + util.EPS:
			ta_ops.append([fppim_set[i].mean(), sen_set[i].mean()])
	ta_ops = np.array(ta_ops)
	
	# Vertical averaging operating points
	op_set = np.array(op_set)
	va_ops = eval.vertical_averaging_froc(op_set, np.arange(0.0, 10.0, 0.1))
	va_ops = np.mean(va_ops, axis=0)

	return ta_ops, va_ops

def protocol():
	paths, locs, rads = jsrt.jsrt(set='jsrt140')
	left_masks = jsrt.left_lung(set='jsrt140')
	right_masks = jsrt.right_lung(set='jsrt140')
	size = len(paths)

	# blobs detection
	blobs = []
	for i in range(size):
		blobs.append([locs[i][0], locs[i][1], rads[i]])
	blobs = np.array(blobs)

	Y = (140 > np.array(range(size))).astype(np.uint8)
	skf = StratifiedKFold(Y, n_folds=10, shuffle=True, random_state=113)
	
	fold = 0

	sens = []
	fppi_mean = []
	fppi_std = []
	for tr_idx, te_idx in skf:
		fold += 1
		print "Fold {}".format(fold)

		xtr = DataProvider(paths[tr_idx], left_masks[tr_idx], right_masks[tr_idx])
		model.train(xtr, blobs[tr_idx])

		xte = DataProvider(paths[te_idx], left_masks[te_idx], right_masks[te_idx])
		blobs_te_pred = model.predict(xte)

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
		blobs_te_pred = model.predict_from_feature_set(feats[te_idx], pred_blobs[te_idx], thold)

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


def protocol_froc(_model):
	paths, locs, rads = jsrt.jsrt(set='jsrt140')
	left_masks = jsrt.left_lung(set='jsrt140')
	right_masks = jsrt.right_lung(set='jsrt140')
	size = DATA_LEN #len(paths)

	# blobs detection
	print "Detecting blobs ..."
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

	Y = (140 > np.array(range(size))).astype(np.uint8)
	skf = StratifiedKFold(Y, n_folds=10, shuffle=True, random_state=113)

	tholds = np.hstack((np.arange(0.0, 0.02, 0.0005), np.arange(0.02, 0.06, 0.0025), np.arange(0.06, 0.66, 0.01)))

	ops = froc_on_folds(_model, paths, left_masks, right_masks, blobs, pred_blobs, feats, skf, tholds)

	x1 = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
	y1 = [0.0, 0.57, 0.72, 0.78, 0.79, 0.81, 0.82, 0.85, 0.86, 0.895, 0.93]

	x2 = []
	y2 = []
	for i in range(len(ops)):
		if ops[i][1] <= 10:
			x2.append(ops[i][1])
			y2.append(ops[i][0])

	plt.plot(x1, y1, 'yo-')
	plt.plot(x2, y2, 'bo-')
	plt.title('FROC')
	plt.ylabel('Sensitivity')
	plt.xlabel('Average FPPI')

	name='{}_{}'.format(_model.name, time.clock())
	np.savetxt('{}_ops.txt'.format(name), [x2, y2])
	plt.savefig('{}_froc.jpg'.format(name))

	return np.array(ops)

def protocol_froc_1(_model, fname):
	paths, locs, rads = jsrt.jsrt(set='jsrt140')
	left_masks = jsrt.left_lung(set='jsrt140')
	right_masks = jsrt.right_lung(set='jsrt140')
	size = DATA_LEN#len(paths)

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

	np.save('{}.fts.npy'.format(fname), feats)
	np.save('{}_pred.blb.npy'.format(fname), pred_blobs)

def protocol_froc_2(_model, fname):
	paths, locs, rads = jsrt.jsrt(set='jsrt140')
	left_masks = jsrt.left_lung(set='jsrt140')
	right_masks = jsrt.right_lung(set='jsrt140')

	#size = DATA_LEN
	size = len(paths)

	blobs = []
	for i in range(size):
		blobs.append([locs[i][0], locs[i][1], rads[i]])
	blobs = np.array(blobs)

	print "Loading blobs & features ..."
	data = DataProvider(paths, left_masks, right_masks)
	feats = np.load('{}.fts.npy'.format(fname))
	pred_blobs = np.load('{}_pred.blb.npy'.format(fname))

	av_cpi = 0
	for tmp in pred_blobs:
		av_cpi += len(tmp)
	print "Average blobs per image {} ...".format(av_cpi * 1.0 / len(pred_blobs))

	Y = (140 > np.array(range(size))).astype(np.uint8)
	skf = StratifiedKFold(Y, n_folds=10, shuffle=True, random_state=113)

	tholds = np.hstack((np.arange(0.0, 0.02, 0.00005), np.arange(0.02, 0.06, 0.0025), np.arange(0.06, 0.66, 0.01)))
	
	ops = get_froc_on_folds(_model, paths, left_masks, right_masks, blobs, pred_blobs, feats, skf, tholds)

	base_line = [[0.0, 0.0], [1.0, 0.57], [2.0, 0.72], [3.0, 0.78], [4.0, 0.79], [5.0, 0.81], [6.0, 0.82], [7.0, 0.85], [8.0, 0.86], [9.0, 0.895], [10.0, 0.93]]
	op_set = []
	legend = []

	op_set.append(base_line)
	legend.append('baseline')
	op_set.append(ops)
	legend.append('current')

	util.save_froc(op_set, _model.name, legend)

	return np.array(ops)

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
	feats = np.load('{}.fts.npy'.format(fname))
	pred_blobs = np.load('{}_pred.blb.npy'.format(fname))
	'''

	av_cpi = 0
	for tmp in pred_blobs:
		av_cpi += len(tmp)
	print "Average blobs per image {} ...".format(av_cpi * 1.0 / len(pred_blobs))
	Y = (140 > np.array(range(size))).astype(np.uint8)
	skf = StratifiedKFold(Y, n_folds=10, shuffle=True, random_state=113)

	tholds = np.hstack((np.arange(0.0, 0.02, 0.0005), np.arange(0.02, 0.06, 0.0025), np.arange(0.06, 0.66, 0.01)))
	
	base_line = [[0.0, 0.0], [1.0, 0.57], [2.0, 0.72], [3.0, 0.78], [4.0, 0.79], [5.0, 0.81], [6.0, 0.82], [7.0, 0.85], [8.0, 0.86], [9.0, 0.895], [10.0, 0.93]]
	op_set = []
	op_set.append(base_line)
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

		ops = get_froc_on_folds(_model, paths, left_masks, right_masks, blobs, selected_blobs, selected_feats, skf, tholds)
		op_set.append(ops)

	op_set = np.array(op_set)
	legend = []
	legend.append("baseline")
	for thold in detect_range:
		legend.append('wmci {}'.format(thold))

	util.save_froc(op_set, _model.name, legend)
	return op_set

def protocol_svm_froc(_model, fname):
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

	feats = np.load('{}.fts.npy'.format(fname))
	pred_blobs = np.load('{}_pred.blb.npy'.format(fname))

	av_cpi = 0
	for tmp in pred_blobs:
		av_cpi += len(tmp)
	print "Average blobs per image {} ...".format(av_cpi * 1.0 / len(pred_blobs))

	Y = (140 > np.array(range(size))).astype(np.uint8)
	skf = StratifiedKFold(Y, n_folds=10, shuffle=True, random_state=113)
	tholds = np.hstack((np.arange(0.0, 0.02, 0.0005), np.arange(0.02, 0.06, 0.0025), np.arange(0.06, 0.66, 0.01)))
	
	op_set = []
	legend = []
	base_line = [[0.0, 0.0], [1.0, 0.57], [2.0, 0.72], [3.0, 0.78], [4.0, 0.79], [5.0, 0.81], [6.0, 0.82], [7.0, 0.85], [8.0, 0.86], [9.0, 0.895], [10.0, 0.93]]
	op_set.append(base_line)
	legend.append("baseline")

	C_set = 10.0 ** np.arange(-1, 2, 1)
	g_set = 10.0 ** np.arange(-3, 1, 1)
	for C, gamma in product(C_set, g_set):
		print 'SVM C = {}, gamma = {}'.format(C, gamma)

		_model.clf = svm.SVC(C=C, gamma=gamma, probability=True)
		ops = get_froc_on_folds(_model, paths, left_masks, right_masks, blobs, pred_blobs, feats, skf, tholds)
		op_set.append(ops)
		legend.append('C={}, gamma={}'.format(C, gamma))

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

	feats = np.load('{}.fts.npy'.format(fname))
	pred_blobs = np.load('{}_pred.blb.npy'.format(fname))

	av_cpi = 0
	for tmp in pred_blobs:
		av_cpi += len(tmp)
	print "Average blobs per image {} ...".format(av_cpi * 1.0 / len(pred_blobs))

	Y = (140 > np.array(range(size))).astype(np.uint8)
	skf = StratifiedKFold(Y, n_folds=10, shuffle=True, random_state=113)
	tholds = np.hstack((np.arange(0.0, 0.02, 0.0005), np.arange(0.02, 0.06, 0.0025), np.arange(0.06, 0.66, 0.01)))
	
	op_set = []
	base_line = [[0.0, 0.0], [1.0, 0.57], [2.0, 0.72], [3.0, 0.78], [4.0, 0.79], [5.0, 0.81], [6.0, 0.82], [7.0, 0.85], [8.0, 0.86], [9.0, 0.895], [10.0, 0.93]]
	op_set.append(base_line)
	legend.insert(0, "baseline")

	for i in range(len(selectors)):
		print legend[i+1]
		_model.selector = selectors[i]
		ops = get_froc_on_folds(_model, paths, left_masks, right_masks, blobs, pred_blobs, feats, skf, tholds)
		op_set.append(ops)

	op_set = np.array(op_set)

	util.save_froc(op_set, _model.name, legend)
	return op_set


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

def protocol_cnn_froc(_model, fname):
	paths, locs, rads = jsrt.jsrt(set='jsrt140')
	left_masks = jsrt.left_lung(set='jsrt140')
	right_masks = jsrt.right_lung(set='jsrt140')

	#size = DATA_LEN
	size = len(paths)

	blobs = []
	for i in range(size):
		blobs.append([locs[i][0], locs[i][1], rads[i]])
	blobs = np.array(blobs)

	print "Loading dataset ..."
	data = DataProvider(paths, left_masks, right_masks)
	pred_blobs = np.load('{}_pred.blb.npy'.format(fname))
	rois = model.create_rois(data, pred_blobs)

	av_cpi = 0
	for tmp in pred_blobs:
		av_cpi += len(tmp)
	print "Average blobs per image {} ...".format(av_cpi * 1.0 / len(pred_blobs))

	Y = (140 > np.array(range(size))).astype(np.uint8)
	skf = StratifiedKFold(Y, n_folds=10, shuffle=True, random_state=113)

	#tholds = np.hstack((np.arange(0.0, 1e-7, 2e-9), np.arange(1e-7, 1e-6, 0.2e-8), np.arange(1e-6, 1e-5, 2e-7), np.arange(1e-5, 5e-5, 1e-6), np.arange(5e-5, 3e-4, 5e-6), np.arange(3e-4, 0.007, 0.00005), np.arange(0.007, 0.02, 0.0005), np.arange(0.02, 0.06, 0.0025), np.arange(0.06, 0.66, 0.01)))
	
	tholds = np.hstack(np.arange(0.49, 0.51, 1e-4))
	ta_ops, va_ops = get_froc_on_folds_keras(_model, paths, left_masks, right_masks, blobs, pred_blobs, rois, skf, tholds)

	base_line = [[0.0, 0.0], [1.0, 0.57], [2.0, 0.72], [3.0, 0.78], [4.0, 0.79], [5.0, 0.81], [6.0, 0.82], [7.0, 0.85], [8.0, 0.86], [9.0, 0.895], [10.0, 0.93]]
	
	legend = []
	legend.append('baseline')
	legend.append('current')

	util.save_froc([baseline, ta_ops], '{}_ta'.format(_model.name), legend)
	util.save_froc([baseline, va_ops], '{}_va'.format(_model.name), legend)

	return np.array(ops)

if __name__=="__main__":	
	model_type = sys.argv[1]
	stage = sys.argv[2]
	_model = model.BaselineModel("data/default")

	extractor = model.extractors.get(model_type)
	if extractor != None:
		_model.extractor = extractor()
		_model.name = 'data/{}'.format(model_type)
	else:
		_model.extractor = model.HardieExtractor()

	if stage == 'fts':
		method = protocol_froc_1
	if stage.find('clf') != -1:
		clf = sys.argv[3]
		if clf == 'svm':
			_model.clf = svm.SVC(probability=True)
		elif clf == 'lda':
			_model.clf = lda.LDA()
		method = protocol_froc_2
	if stage == 'red-clf':
		red = sys.argv[4]
		if red == 'pca':
			_model.selector = decomposition.PCA(n_components=0.99999999999, whiten=True)
		if red == 'lda':
			_model.selector = selection.SelectFromModel(lda.LDA())

		method = protocol_froc_2
	if stage == 'wmci':
		method = protocol_wmci_froc
	if stage == 'clf-pca':
		method = protocol_pca_froc
	if stage == 'clf-lda':
		method = protocol_lda_froc
	if stage == 'clf-svm':
		method = protocol_svm_froc
	if stage == 'clf-rlr':
		method = protocol_rlr_froc
	if stage == 'clf-rfe':
		method = protocol_rfe_froc
	if stage == 'cnn':
		method = protocol_cnn_froc
	_model.name += stage
	method(_model, '{}'.format(model_type))

