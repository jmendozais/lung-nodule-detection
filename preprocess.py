import cv2
import numpy as np
import util

EPS = 1e-9

def downsample(img):
	# check a better low-pass anti-aliasing filter, boxfilter is better
	ksize = (11, 11)
	dsize = (512, 512)
	sigma = 0.5

	smt = cv2.GaussianBlur(img, ksize, sigma)
	resized = cv2.resize(smt, dsize, interpolation=cv2.INTER_CUBIC)
	
	return resized

def lce(img):
	hsize = (33, 33)
	hsigma = 16

	mu = cv2.GaussianBlur(img, hsize, hsigma)
	ro2 = cv2.GaussianBlur(pow(img, 2), hsize, hsigma) - pow(mu, 2) + EPS
	assert np.min(ro2) >= 0
	res = (img - mu) / pow(ro2, 0.5)
	return res

def normalize(img):
	mean, std = cv2.meanStdDev(img)
	normalized = (img - mean)/std
	
	return normalized

def preprocess(img):
	resized = downsample(img)
	enhanced = lce(resized)
	normalized = normalize(resized)

	return resized, enhanced, normalized



