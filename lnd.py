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

import jsrt

#fppi range
fppi_range = np.linspace(0.0, 10.0, 101)

# baseline
baseline = np.array([[0.0, 0.0], [0.1, 0.2], [0.2, 0.3],[0.3, 0.38], [0.4, 0.415], [0.5, 0.46], [0.6, 0.48], [0.7, 0.51], [0.9, 0.53], [1.0, 0.57], [1.5, 0.67], [2.0, 0.72], [2.5, 0.75],[3.0, 0.78], [4.0, 0.79], [5.0, 0.81], [6.0, 0.82], [7.0, 0.85], [8.0, 0.86], [9.0, 0.895], [10.0, 0.93]])
fun = interp1d(baseline.T[0], baseline.T[1], kind='linear', fill_value=0, bounds_error=False)
baseline = np.array([fppi_range, fppi_range.copy()])
baseline[1] = fun(baseline[0])
baseline = baseline.T


'''
returns: Free Receiving Operating Curve obtained given a fold set
'''


def get_froc_on_folds(_model, paths, left_masks, right_masks, blobs, pred_blobs, feats, folds):
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

		print 'Train ...'
		_model.train_with_feature_set(feats[tr_idx], pred_blobs[tr_idx], blobs[tr_idx])
		print 'Predict ...'
		blobs_te_pred, probs_te_pred = _model.predict_proba_from_feature_set(feats[te_idx], pred_blobs[te_idx])
		print 'Get froc ...'
		froc = eval.froc(blobs_te, blobs_te_pred, probs_te_pred)

		frocs.append(froc)
		fold += 1
	
	av_froc = eval.average_froc(frocs, fppi_range)
	return av_froc

def get_froc_on_folds_keras(_model, paths, left_masks, right_masks, blobs, pred_blobs, rois, folds):
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

		_model.train_with_feature_set_keras(rois[tr_idx], pred_blobs[tr_idx], blobs[tr_idx])
		blobs_te_pred, probs_te_pred = _model.predict_proba_from_feature_set_keras(rois[te_idx], pred_blobs[te_idx])

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

	np.save('{}.fts.npy'.format(fname), feats)
	np.save('{}_pred.blb.npy'.format(fname), pred_blobs)

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
	feats = np.load('{}.fts.npy'.format(fname))
	pred_blobs = np.load('{}_pred.blb.npy'.format(fname))

	av_cpi = 0
	for tmp in pred_blobs:
		av_cpi += len(tmp)
	print "Average blobs per image {} ...".format(av_cpi * 1.0 / len(pred_blobs))

	Y = (140 > np.array(range(size))).astype(np.uint8)
	skf = StratifiedKFold(Y, n_folds=10, shuffle=True, random_state=113)

	ops = get_froc_on_folds(_model, paths, left_masks, right_masks, blobs, pred_blobs, feats, skf)

	legend = []
	legend.append('baseline')
	legend.append(_model.name)

	util.save_froc([baseline, ops], '{}'.format(_model.name), legend)

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
	feats = np.load('{}.fts.npy'.format(fname))
	pred_blobs = np.load('{}_pred.blb.npy'.format(fname))
	'''

	av_cpi = 0
	for tmp in pred_blobs:
		av_cpi += len(tmp)
	print "Average blobs per image {} ...".format(av_cpi * 1.0 / len(pred_blobs))
	Y = (140 > np.array(range(size))).astype(np.uint8)
	skf = StratifiedKFold(Y, n_folds=10, shuffle=True, random_state=113)

	op_set = []
	op_set.append(baseline)
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
	legend.append("baseline")
	for thold in detect_range:
		legend.append('wmci {}'.format(thold))

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
	
	op_set = []
	
	op_set.append(baseline)
	legend.insert(0, "baseline")

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

	feats = np.load('{}.fts.npy'.format(fname))
	pred_blobs = np.load('{}_pred.blb.npy'.format(fname))

	av_cpi = 0
	for tmp in pred_blobs:
		av_cpi += len(tmp)
	print "Average blobs per image {} ...".format(av_cpi * 1.0 / len(pred_blobs))

	Y = (140 > np.array(range(size))).astype(np.uint8)
	skf = StratifiedKFold(Y, n_folds=10, shuffle=True, random_state=113)
	
	op_set = []

	op_set.append(baseline)
	legend.insert(0, "baseline")

	for i in range(len(classifiers)):
		print legend[i+1]
		_model.clf = classifiers[i]
		ops = get_froc_on_folds(_model, paths, left_masks, right_masks, blobs, pred_blobs, feats, skf) 
		op_set.append(ops)

	op_set = np.array(op_set)
	util.save_froc(op_set, '{}_ta'.format(_model.name), legend)

	return op_set[1:]

def protocol_clf_eval_froc(_model, fname):
	classifiers  = []
	classifiers.append(lda.LDA())
	classifiers.append(svm.SVC(kernel='linear', probability=True))
	classifiers.append(svm.SVC(kernel='rbf', probability=True))
	classifiers.append(ensemble.RandomForestClassifier())
	classifiers.append(ensemble.AdaBoostClassifier())
	classifiers.append(ensemble.GradientBoostingClassifier())

	labels = []
	labels.append('LDA')
	labels.append('Linear SVC')
	labels.append('RBF SVC')
	labels.append('Random Forest')
	labels.append('AdaBoost')
	labels.append('Gradient Boosting')

	protocol_classifier_froc(_model, fname, classifiers, labels)

def protocol_svm_hp_search(_model, fname):
	C_set = np.logspace(-3, 4, 8)
	g_set = np.logspace(-3, 4, 8)
	classifiers = []
	legend = []
	for C, gamma in product(C_set, g_set):
		legend.append('C={}, g={}'.format(C, round(gamma, 5)))
		print 'SVM C = {}, g= {}'.format(C, round(gamma, 5))
		classifiers.append(svm.SVC(C=C, gamma=gamma, probability=True, max_iter=1000))

	ops = protocol_classifier_froc(_model, fname, classifiers, legend)
	
	# compute the AUC from 2 to 4
	step = 10
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

def protocol_cnn_froc(_model, fname):
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
	pred_blobs = np.load('{}_pred.blb.npy'.format(fname))
	rois = model.create_rois(data, pred_blobs)

	av_cpi = 0
	for tmp in pred_blobs:
		av_cpi += len(tmp)
	print "Average blobs per image {} ...".format(av_cpi * 1.0 / len(pred_blobs))

	Y = (140 > np.array(range(size))).astype(np.uint8)
	skf = StratifiedKFold(Y, n_folds=10, shuffle=True, random_state=113)
	
	ops = get_froc_on_folds_keras(_model, paths, left_masks, right_masks, blobs, pred_blobs, rois, skf)

	legend = []
	legend.append('baseline')
	legend.append('current')

	util.save_froc([baseline, ops], '{}_ta'.format(_model.name), legend)

	return ops

if __name__=="__main__":	
	model_type = sys.argv[1]
	stage = sys.argv[2]
	_model = model.BaselineModel("data/default")

	extractor = model.extractors.get(model_type)
	if extractor != None:
		_model.extractor = extractor
		_model.name = 'data/{}'.format(model_type)
	else:
		_model.extractor = model.HardieExtractor()

	if stage == 'fts':
		method = protocol_froc_1
	if stage.find('clf') != -1:
		clf = sys.argv[3]
		if clf == 'svm':
			_model.clf = svm.SVC(kernel='linear', probability=True)
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
		#method = protocol_svm_froc
		method = protocol_svm_hp_search
	if stage == 'clf-rlr':
		method = protocol_rlr_froc
	if stage == 'clf-rfe':
		method = protocol_rfe_froc
	if stage == 'clf-eval':
		method = protocol_clf_eval_froc
	if stage == 'cnn':
		method = protocol_cnn_froc
	_model.name += stage
	method(_model, '{}'.format(model_type))

