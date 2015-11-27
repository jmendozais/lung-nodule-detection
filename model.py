import cv2
import sys
import skimage.io as io
from sklearn.lda import LDA
from sklearn import svm
from sklearn import preprocessing
from sklearn.externals import joblib
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

def _classify(img, blobs, feature_vectors, thold=0.2):
	clf = joblib.load('data/clf.pkl')
	scaler = joblib.load('data/scaler.pkl')
	probs = clf.predict_proba(scaler.transform(feature_vectors))
	probs = probs.T[1]

	blobs = np.array(blobs)
	blobs = blobs[probs>thold]

	return blobs

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
		blobs = _classify(img, blobs, feats)

		#show_blobs("Predict result ...", lce, blobs)
		data_blobs.append(blobs)

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


