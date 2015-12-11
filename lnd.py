import sys
import time

import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn import svm

import matplotlib.pyplot as plt

from data import DataProvider
import model
import eval
import util
import sys

import jsrt

def run_individual(): 
	model.pipeline(sys.argv[1], sys.argv[2], sys.argv[3])

def run_on_dataset():
	paths, locs, rads = jsrt.jsrt(set='jsrt140')
	left_masks = jsrt.left_lung(set='jsrt140')
	right_masks = jsrt.right_lung(set='jsrt140')

	blobs_ = []
	for i in range(len(paths)):
		img, blobs = model.pipeline(paths[i], left_masks[i], right_masks[i])
		#img, blobs = model.pipeline_features(paths[i], [(locs[i][0], locs[i][1], 25)], left_masks[i], right_masks[i])
		blobs_.append(blobs)
		print_detection(paths[i], blobs)
		sys.stdout.flush()

def protocol():
	paths, locs, rads = jsrt.jsrt(set='jsrt140')
	left_masks = jsrt.left_lung(set='jsrt140')
	right_masks = jsrt.right_lung(set='jsrt140')
	size = len(paths)

	# blobs detection
	blobs = []
	imgs = []
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
	imgs = []
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
	size = 10#len(paths)

	# blobs detection
	print "Detecting blobs ..."
	blobs = []
	imgs = []
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
	
	ops = []
	sen_set = []
	fppim_set = []
	fppis_set = []

	fold = 0
	for tr_idx, te_idx in skf:	
		print "Fold {}".format(fold + 1),
		sys.stdout.flush()

		paths_te = paths[te_idx]
		blobs_te = []

		for bl in blobs[te_idx]:
			blobs_te.append([bl])
		blobs_te = np.array(blobs_te)

		_model.train_with_feature_set(feats[tr_idx], pred_blobs[tr_idx], blobs[tr_idx])
		blobs_te_pred, probs_te_pred = _model.predict_proba_from_feature_set(feats[te_idx], pred_blobs[te_idx])
		
		'''
		for i in range(len(blobs_te_pred)):
			util.print_detection(paths_te[i], blobs_te_pred[i])
		'''

		sen_set.append([])
		fppim_set.append([])
		fppis_set.append([])
		for thold in tholds:
			fblobs_te_pred, fprobs_te_pred = _model.filter_by_proba(blobs_te_pred, probs_te_pred, thold)

			s, fm, fs = eval.evaluate(blobs_te, fblobs_te_pred, paths_te)

			print "thold {}, sens {}, fppi mean {}, fppi std {}".format(thold, s, fm, fs)
			sys.stdout.flush()

			sys.stdout.flush()
			sen_set[-1].append(s)
			fppim_set[-1].append(fm)
			fppis_set[-1].append(fs)

		fold += 1

	sen_set = np.array(sen_set).T
	fppim_set = np.array(fppim_set).T
	fppis_set = np.array(fppis_set).T

	for i in range(len(tholds)):
		print "thold {}: sen mean {}, fppi mean {}, fppi std {}".format(tholds[i], sen_set[i].mean(), fppim_set[i].mean(), fppim_set[i].std())
		ops.append([sen_set[i].mean(), fppim_set[i].mean()])

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

if __name__=="__main__":	
	model_type = sys.argv[1]
	_model = model.BaselineModel("data/default")

	if model_type == 'hardie':
		_model.extractor = model.HardieExtractor()
		_model.name = 'data/hardie'
	elif model_type == 'hog':	
		_model.extractor = model.HogExtractor()
		_model.clf = svm.SVC(probability=True)
		_model.name = 'data/hog'

	protocol_froc(_model)
