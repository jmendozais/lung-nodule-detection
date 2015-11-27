import sys
import time

import numpy as np
from sklearn.lda import LDA
from sklearn import svm
from sklearn.metrics import classification_report
import sklearn.cross_validation as cross_val
from sklearn.externals import joblib
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.base import clone


import model
import util

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

def train(X, Y, clf, scaler):
	iters = 1
	tr_prop = 0.7
	te_prop = 1 - tr_prop
	seed = 113

	# hardcoded
	NX = []
	for xi in X:
		NX.append(np.array(xi))
	X = np.array(NX)

	folds = np.array(list(cross_val.StratifiedShuffleSplit(Y, iters, train_size=tr_prop, test_size=te_prop, random_state=seed)))

	tr = folds[0][0]
	te = folds[0][1]

	eval_scaler = clone(scaler)
	eval_clf = clone(clf)

	eval_scaler.fit(X[tr])
	eval_clf.fit(eval_scaler.transform(X[tr]), Y[tr])
	pred = eval_clf.predict(eval_scaler.transform(X[te]))

	print "Evaluate performance on patches"
	print "classification report: "	
	print classification_report(Y[te].astype(int), pred.astype(int))
	print "confusion matrix: "
	print confusion_matrix(Y[te].astype(int), pred.astype(int))

	scaler.fit(X)
	clf.fit(scaler.transform(X), Y)
	return clf, scaler

if __name__ == '__main__':
	fname = sys.argv[1]
	data = np.load(fname)
	print "fname {}".format(fname)

	X = data[0]
	Y = data[1]
	clf = LDA()
	scaler = preprocessing.StandardScaler()
	train(X, Y, clf, scaler)

