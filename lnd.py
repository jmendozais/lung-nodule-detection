import cv2
import sys
from sklearn.externals import joblib

from classify import *
from detect import *
from extract import * 
from preprocess import *
from segment import *
from util import *
import jsrt
import skimage.io as io
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
	sampled, lce, norm = preprocess(img)
	blobs, ci = wmci(lce, lung_mask)
	#ci = lce
	#blobs = log_(lce, lung_mask)

	return blobs, norm, lce, ci

def segment(img, blobs):
	blobs, nod_masks = circunference(lce, blobs)

	return blobs, nod_masks

def extract(norm, lce, wmci, lung_mask, blobs, nod_masks):

	return hardie(norm, lce, wmci, lung_mask, blobs, nod_masks)

def classify(img, blobs, feature_vectors):
	clf = joblib.load('data/clf.pkl') 
	scaler = joblib.load('data/scaler.pkl')
	results = clf.predict(scaler.transform(feature_vectors))
	blobs = np.array(blobs)
	blobs = blobs[results>0]
	return blobs

# pipelines 
def pipeline_blobs(img_path, ll_path, lr_path):
	img, lung_mask = input(img_path, ll_path, lr_path)
	blobs, norm, lce, ci = detect_blobs(img, lung_mask)

  #show_blobs("result", lce, blobs)

	return blobs

def pipeline_features(img_path, blobs, ll_path, lr_path):
	img, lung_mask = input(img_path, ll_path, lr_path)
	_, norm, lce, ci = detect_blobs(img, lung_mask)
	blobs, nod_masks = segment(img, blobs)
	feats = extract(norm, lce, ci, lung_mask, blobs, nod_masks)

	return feats
'''
def pipeline_blobs_features(img_path, ll_path, lr_path):
	img, lung_mask = input(img_path, ll_path, lr_path)
	blobs, norm, lce, ci = detect_blobs(img, lung_mask)
	blobs, nod_masks = segment(img, blobs)
	feats = extract(norm, lce, ci, lung_mask, blobs, nod_masks)

	return blobs, feats
'''
def pipeline(img_path, ll_path, lr_path):
	img, lung_mask = input(img_path, ll_path, lr_path)
	blobs, norm, lce, ci = detect_blobs(img, lung_mask)
	blobs, nod_masks = segment(img, blobs)
	feats = extract(norm, lce, ci, lung_mask, blobs, nod_masks)
	blobs = classify(img, blobs, feats)

	#show_blobs("result", lce, blobs)

	return img, blobs

def all():
	paths, locs, rads = jsrt.jsrt(set='jsrt140')
	left_masks = jsrt.left_lung(set='jsrt140')
	right_masks = jsrt.right_lung(set='jsrt140')

	blobs_ = []
	for i in range(len(paths)):
		img, blobs = pipeline(paths[i], left_masks[i], right_masks[i])
		blobs_.append(blobs)
		print_detection(paths[i], blobs)
		sys.stdout.flush()

if __name__=="__main__":
	all()
	#pipeline(sys.argv[1], sys.argv[2], sys.argv[3])
