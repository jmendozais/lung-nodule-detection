import cv2
import sys
import skimage.io as io
from sklearn.lda import LDA
from sklearn import svm
from sklearn import preprocessing
from sklearn.externals import joblib
from sklearn.feature_selection import RFE
from time import *

import classify
from detect import *
from extract import * 
from preprocess import *
from segment import *
from util import *

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

def detect_blobs(img, lung_mask):
	sampled, lce, norm = preprocess(img, lung_mask)
	blobs, ci = wmci(lce, lung_mask)
	#ci = lce
	#blobs = log_(lce, lung_mask)

	return blobs, norm, lce, ci

def segment(img, blobs):
	blobs, nod_masks = adaptive_distance_thold(img, blobs)

	return blobs, nod_masks

def extract(norm, lce, wmci, lung_mask, blobs, nod_masks):

	return hardie(norm, lce, wmci, lung_mask, blobs, nod_masks)

def extract_feature_set(data):
	feature_set = []
	blob_set = []
	print '[',
	for i in range(len(data)):
		if i % (len(data)/10) == 0:
			print ".",
			sys.stdout.flush()

		img, lung_mask = data.get(i)
		blobs, norm, lce, ci = detect_blobs(img, lung_mask)
		blobs, nod_masks = segment(lce, blobs)
		feats = extract(norm, lce, ci, lung_mask, blobs, nod_masks)
		feature_set.append(feats)
		blob_set.append(blobs)
	print ']'
	return np.array(feature_set), np.array(blob_set)

def _classify(blobs, feature_vectors, thold=0.012):
	clf = joblib.load('data/clf.pkl')
	scaler = joblib.load('data/scaler.pkl')
	selector = joblib.load('data/selector.pkl')

	feature_vectors = scaler.transform(feature_vectors)
	feature_vectors = selector.transform(feature_vectors)
	probs = clf.predict_proba(feature_vectors)
	probs = probs.T[1]

	blobs = np.array(blobs)
	blobs = blobs[probs>thold]

	return blobs, probs[probs>thold]

''' 	
	Input: images & masks (data provider), blobs
	Returns: classifier, scaler
'''
def train(data, blobs):
	X, Y = classify.create_training_set(data, blobs)
	#clf = svm.SVC()
	clf = LDA()
	#scaler = preprocessing.MinMaxScaler()
	scaler = preprocessing.StandardScaler()
	
	clf, scaler = classify.train(X, Y, clf, scaler)

	joblib.dump(clf, 'data/clf.pkl')
	joblib.dump(scaler, 'data/scaler.pkl')

	return clf, scaler

def train_with_feature_set(feature_set, pred_blobs, real_blobs):
	X, Y = classify.create_training_set_from_feature_set(feature_set, pred_blobs, real_blobs)

	#scaler = preprocessing.MinMaxScaler()
	scaler = preprocessing.StandardScaler()
	selector = RFE(estimator=LDA(), n_features_to_select=56, step=1)
	#clf = svm.SVC()
	clf = LDA()
	clf, scaler, selector = classify.train(X, Y, clf, scaler, selector)

	joblib.dump(clf, 'data/clf.pkl')
	joblib.dump(scaler, 'data/scaler.pkl')
	joblib.dump(selector, 'data/selector.pkl')

	return clf, scaler, selector

'''
	Input: data provider
	Returns: blobs
'''
def predict(data):
	data_blobs = []
	for i in range(len(data)):
		img, lung_mask = data.get(i)
		blobs, norm, lce, ci = detect_blobs(img, lung_mask)
		blobs, nod_masks = segment(lce, blobs)
		feats = extract(norm, lce, ci, lung_mask, blobs, nod_masks)
		blobs, probs = _classify(blobs, feats)

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

def predict_from_feature_set(feature_set, blob_set, thold=0.012):
	data_blobs = []
	for i in range(len(feature_set)):
		blobs, probs = _classify(blob_set[i], feature_set[i], thold)

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

# PIPELINES 
def pipeline_features(img_path, blobs, ll_path, lr_path):
	img, lung_mask = input(img_path, ll_path, lr_path)
	_, norm, lce, ci = detect_blobs(img, lung_mask)
	blobs, nod_masks = segment(lce, blobs)
	feats = extract(norm, lce, ci, lung_mask, blobs, nod_masks)

	return lce, feats

def pipeline(img_path, ll_path, lr_path):
	img, lung_mask = input(img_path, ll_path, lr_path)
	blobs = predict(img, lung_mask)

	return img, blobs


