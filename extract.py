import numpy as np
from itertools import *
from math import *
from time import *

import cv2
from scipy.stats import skew, kurtosis
import skimage.feature as feature
import mahotas
import overfeat

import util


_ge = ['max_diam', 'x_fract', 'dist_perim']
_in = ['max_value_in', 'mean_sep', 'contrast_1', 
	   'std_in', 'std_out', 'std_sep', 
	   'skew_in', 'kurt_in', 'moment_1_in',
	   'moment_4_in', 'moment_5_in', 'moment_7_in']
_gr = ['mag_mean_in', 'mag_std_in', 'rd_mean_in', 
	   'rd_mean_out', 'rd_mean_sep', 'rd_std_in',
	   'rd_std_out', 'rd_std_sep', 'rgrad_mean_in',
	   'rgrad_mean_out', 'rgrad_mean_sep', 'rgrad_std_in',
	   'rgrad_std_out', 'rgrad_std_sep', 'rgrad_mean_per',
	   'rgrad_std_per']

def finite_derivatives(img):
	size = img.shape

	dx = np.empty(img.shape, dtype=np.double)
	dx[0, :] = 0
	dx[-1, :] = 0
	dx[1:-1, :] = (img[2:, :] - img[:-2, :]) / 2.0

	dy = np.empty(img.shape, dtype=np.double)
	dy[:, 0] = 0
	dy[:, -1] = 0
	dy[:, 1:-1] = (img[:, 2:] - img[:, :-2]) / 2.0

	mag = (dx ** 2 + dy ** 2) ** 0.5
	return mag, dx, dy

def _mean_std_maxin_with_extended_mask(img, blob, mask):
	result = []

	shape = img.shape
	x, y, r = blob
	shift = 35 
	side = 2 * (shift + r) + 1

	# Vectorize ROI 	operations
	tl = (x - shift - r, y - shift - r)
	ntl = (max(0, tl[0]), max(0, tl[1]))
	br = (x + shift + r + 1, y + shift + r + 1)
	nbr = (min(img.shape[0], br[0]), min(img.shape[1], br[1]))

	roi = img[ntl[0]:nbr[0], ntl[1]:nbr[1]]
	ext_mask = np.full(roi.shape, dtype=np.uint8, fill_value=0)	
	ext_mask[(tl[0] + shift - ntl[0]):(tl[0] + shift + 2 * r + 1 - ntl[0]), (tl[1] + shift - ntl[1]):(tl[1] + shift + 2 * r + 1 - ntl[1])] = mask

	#util.imshow('mask', mask)
	#util.show_nodule(roi, ext_mask)

	ins = np.ma.array(data=roi, mask=1-ext_mask, fill_value=0)
	out = np.ma.array(data=roi, mask=ext_mask, fill_value=0)
	return (ins.mean(), out.mean(), ins.std(), out.std(), ins.max())


def angle(u, v):
	angle = atan2(v[1], v[0]) - atan2(u[1], u[0])
	if angle < 0:
		angle += 2 * pi;
	return angle

def angle2(u, v):
	angle = np.arctan2(v[1], v[0]) - np.arctan2(u[1], u[0])
	mask = angle > 0
	ans = np.ma.array(data=angle, mask=mask, fill_value=0)
	ans += 2 * pi
	return ans.data

def _mean_std_maxin_with_extended_mask_phase_rgrad(mag, dx, dy, blob, mask):
	result = []

	x, y, r = blob
	shift = 35 
	side = 2 * shift + 2 * r + 1

	tl = (x - shift - r, y - shift - r)
	ntl = (max(0, tl[0]), max(0, tl[1]))
	br = (x + shift + r + 1, y + shift + r + 1)
	nbr = (min(mag.shape[0], br[0]), min(mag.shape[1], br[1]))

	roi = mag[ntl[0]:nbr[0], ntl[1]:nbr[1]]
	dxr = dx[ntl[0]:nbr[0], ntl[1]:nbr[1]]
	dyr = dy[ntl[0]:nbr[0], ntl[1]:nbr[1]]
	rx = -1 * np.linspace(-1 * (side/2), side/2, side)
	ry = -1 * np.linspace(-1 * (side/2), side/2, side)

	ext_mask = np.full(roi.shape, dtype=np.uint8, fill_value=0)
	ext_mask[(tl[0] + shift - ntl[0]):(tl[0] + shift + 2 * r + 1 - ntl[0]), (tl[1] + shift - ntl[1]):(tl[1] + shift + 2 * r + 1 - ntl[1])] = mask

	#util.imshow('rmask', mask)
	#util.show_nodule(roi, ext_mask)

	# TODO: validate the order is correct on angle2 (1)
	# TODO: fix meshgrid
	ry, rx = np.meshgrid(rx, ry)
	rx = rx[(ntl[0]-tl[0]):side - (br[0] - nbr[0]), ntl[1]-tl[1]:side - (br[1] - nbr[1])]
	ry = ry[(ntl[0]-tl[0]):side - (br[0] - nbr[0]), ntl[1]-tl[1]:side - (br[1] - nbr[1])]

	phase = angle2((rx, ry), (dxr, dyr)) # (1)
	ins = np.ma.array(data=phase, mask=1-ext_mask, fill_value=0)
	out = np.ma.array(data=phase, mask=ext_mask, fill_value=0)
	phase_result = (ins.mean(), out.mean(), ins.std(), out.std())

	rgrad = phase * roi
	ins = np.ma.array(data=rgrad, mask=1-ext_mask, fill_value=0)
	out = np.ma.array(data=rgrad, mask=ext_mask, fill_value=0)
	rgrad_result = (ins.mean(), out.mean(), ins.std(), out.std())

	return phase_result, rgrad_result

def geometric(img, blob, mask, xrange, yrange, dt_img):
	results = {}
	contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	if len(contours) > 0:
		center, radius = cv2.minEnclosingCircle(contours[0])
	else :
		radius = 13.0
	# 1. Size
	# TODO: implemented double dfs exact max diameter
	results[_ge[0]] = radius

	# 2. X-fraction
	xfraction = (blob[0] - xrange[0]) * 1.0 / (xrange[1] - xrange[0])
	results[_ge[1]] = xfraction

	# 3. Distance to lung perimeter ( normalized )
	results[_ge[2]] = dt_img[blob[0], blob[1]]
	return results

def intensity(img, blob, mask):
	result = {}
	mean_in, mean_out, std_in, std_out, max_in = _mean_std_maxin_with_extended_mask(img, blob, mask)
	roi = img[(blob[0] - blob[2]):(blob[0] + blob[2] + 1), (blob[1] - blob[2]):(blob[1] + blob[2] + 1)]

	roi = np.multiply(roi, mask)

	moments = cv2.moments(roi)
	hu = cv2.HuMoments(moments)

	roi = roi.flatten()
	grays_in = np.trim_zeros(roi)

	# max in
	result[_in[0]] = max_in
	# mean separation
	result[_in[1]] = (mean_in - mean_out) / (mean_in + mean_out)
	# contrast 1
	result[_in[2]] = mean_in - mean_out
	# std in
	result[_in[3]] = std_in
	# std out
	result[_in[4]] = std_out
	# std separation
	result[_in[5]] = (std_in - std_out) / (std_in + std_out)
	# skew in
	result[_in[6]] = skew(grays_in)
	# kurtosis in
	result[_in[7]] = kurtosis(grays_in)
	# invariant moments
	result[_in[8]] = hu[0]
	result[_in[9]] = hu[3]
	result[_in[10]] = hu[4]
	result[_in[11]] = hu[6]

	return result	

def gradient(img, blob, mask, mag, dx, dy):
	result = {}
	x, y, r = blob
	shift = 35 # 25 mm
	mag_mean_in = 0.0
	mag_std_in = 0.0

	mag_mean_in, _, mag_std_in, _, _ = _mean_std_maxin_with_extended_mask(mag, blob, mask)
	phase_ans, rgrad_ans = _mean_std_maxin_with_extended_mask_phase_rgrad(mag, dx, dy, blob, mask)
	rd_mean_in, rd_mean_out, rd_std_in, rd_std_out = phase_ans
	rgrad_mean_in, rgrad_mean_out, rgrad_std_in, rgrad_std_out = rgrad_ans

	per_mask = np.full(mask.shape, dtype=mask.dtype, fill_value=0)
	contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  	cv2.drawContours(per_mask, contours, -1, 1, 1)
	_, rgrad_per = _mean_std_maxin_with_extended_mask_phase_rgrad(mag, dx, dy, blob, per_mask)
	rgrad_mean_per, _, rgrad_std_per, _ = rgrad_per

	result[_gr[0]] = mag_mean_in
	result[_gr[1]] = mag_std_in

	result[_gr[2]] = rd_mean_in
	result[_gr[3]] = rd_mean_out
	sep = (rd_mean_in - rd_mean_out) * 1.0 / (1e-9 + rd_mean_in + rd_mean_out)
	result[_gr[4]] = sep

	result[_gr[5]] = rd_std_in
	result[_gr[6]] = rd_std_out
	sep = (rd_std_in - rd_std_out) * 1.0 / (1e-9 + rd_std_in + rd_std_out)
	result[_gr[7]] = sep

	result[_gr[8]] = rgrad_mean_in
	result[_gr[9]] = rgrad_mean_out
	sep = (rgrad_mean_in - rgrad_mean_out) * 1.0 / (1e-9 + rgrad_mean_in + rgrad_mean_out)
	result[_gr[10]] = sep

	result[_gr[11]] = rgrad_std_in
	result[_gr[12]] = rgrad_std_out
	sep = (rgrad_std_in - rgrad_std_out) * 1.0 / (1e-9 + rgrad_std_in + rgrad_std_out)
	result[_gr[13]] = sep

	result[_gr[14]] = rgrad_mean_per
	result[_gr[15]] = rgrad_std_per

	return result
 
def hardie_blob(img, blob, mask, xrange, yrange, dt_mask, mag, dx, dy):
	geom = geometric(img, blob, mask, xrange, yrange, dt_mask)
	inte = intensity(img, blob, mask)
	grad = gradient(img, blob, mask, mag, dx, dy)

	feats = np.hstack((geom.values(), inte.values(), grad.values()))
	#feats = np.hstack((geom.values(),))

	return np.array(feats)

def hardie_blob_selected(img, blob, mask, xrange, yrange, dt_mask, mag, dx, dy, geom_feats, inten_feats, grad_feats):
	geom = geometric(img, blob, mask, xrange, yrange, dt_mask)
	inte = intensity(img, blob, mask)
	grad = gradient(img, blob, mask, mag, dx, dy)

	feats = []
	for feat_name in geom_feats:
		feats.append(geom[feat_name])
	for feat_name in inten_feats:
		feats.append(inte[feat_name])
	for feat_name in grad_feats:
		feats.append(grad[feat_name])

	return np.array(feats)

def hardie(norm, lce, wmci, lung_mask, blobs, masks):
	# Ranges for geometric features
	xsum = np.sum(lung_mask, axis=0)
	ysum = np.sum(lung_mask, axis=1)
	xrange = [1e10, -1e10]
	for i in range(len(xsum)):
		if xsum[i] > 0:
			xrange[0] = min(xrange[0], i)
			xrange[1] = max(xrange[1], i)

	yrange = [1e10, -1e10]
	for i in range(len(ysum)):
		if ysum[i] > 0:
			yrange[0] = min(yrange[0], i)
			yrange[1] = max(yrange[1], i)

	# Normalized distance transformed image
	dt_img = cv2.distanceTransform(lung_mask, cv2.cv.CV_DIST_L2, 5)
	_, max_dist, _, _ = cv2.minMaxLoc(dt_img, lung_mask)
	dt_img /= (1.0 * max_dist)


	# FLD selected features
	adt_geom = ['x_fract', 'dist_perim']
	
	norm_inte = ['mean_sep', 'skew_in', 
	   'moment_4_in', 'moment_5_in', 'moment_7_in']

	norm_grad = ['rd_mean_in', 'rd_std_in',
	   'rd_std_out', 'rd_std_sep', 'rgrad_mean_in',
	   'rgrad_mean_sep', 'rgrad_std_in',
	   'rgrad_std_out', 'rgrad_std_sep']

	lce_inte = ['mean_sep', 'contrast_1', 
	   'std_in', 'std_sep', 
	   'skew_in', 'kurt_in', 'moment_1_in']
	lce_grad = ['mag_mean_in', 'rd_std_in',
	   'rd_std_out', 'rgrad_mean_sep']

	wmci_inte = ['max_value_in', 'contrast_1',
	   'std_out', 'std_sep', 
	   'skew_in', 'kurt_in', 'moment_1_in',
	   'moment_5_in', 'moment_7_in']
	wmci_grad = ['mag_std_in', 'rd_mean_in', 
	   'rd_mean_out', 'rd_mean_sep', 
	   'rd_std_out', 'rgrad_mean_in',
	   'rgrad_mean_out', 'rgrad_mean_sep']

	# Derivatives for gradient features
	mag_norm, dx_norm, dy_norm = finite_derivatives(norm)
	mag_lce, dx_lce, dy_lce = finite_derivatives(lce)
	mag_wmci, dx_wmci, dy_wmci = finite_derivatives(wmci)

	# Feature vectors
	feature_vectors = []
	for i in range(len(blobs)):
		nf = hardie_blob_selected(norm, blobs[i], masks[i], xrange, yrange, dt_img, mag_norm, dx_norm, dy_norm, adt_geom, norm_inte, norm_grad)
		lf = hardie_blob_selected(lce, blobs[i], masks[i], xrange, yrange, dt_img, mag_lce, dx_lce, dy_lce, [], lce_inte, lce_grad)
		wf = hardie_blob_selected(wmci, blobs[i], masks[i], xrange, yrange, dt_img, mag_wmci, dx_wmci, dy_wmci, [], wmci_inte, wmci_grad)
		'''
		nf = hardie_blob_selected(norm, blobs[i], masks[i], xrange, yrange, dt_img, mag_norm, dx_norm, dy_norm, _ge, _in, _gr)
		lf = hardie_blob_selected(lce, blobs[i], masks[i], xrange, yrange, dt_img, mag_lce, dx_lce, dy_lce, _ge, _in, _gr)
		wf = hardie_blob_selected(wmci, blobs[i], masks[i], xrange, yrange, dt_img, mag_wmci, dx_wmci, dy_wmci, _ge, _in, _gr)
		'''
		#feats = np.hstack((nf,))
		feats = np.hstack((nf, lf, wf))
		feature_vectors.append(np.array(feats))

	return np.array(feature_vectors)

def _hog(img, mask, orientations=9, cell=(8,8)):
	mag, dx, dy = finite_derivatives(img)
	phase = np.arctan2(dy, dx)
	phase = phase.astype(np.float64)	
	#phase = np.abs(phase)

	size = img.shape
	size = (size[0] / cell[0], size[1] / cell[1])
	w = mask.astype(np.float64)
	w *= mag

	if np.sum(w) > util.EPS:
		w /= np.sum(w)

	ans = np.array([])
	for i, j in product(range(size[0]), range(size[1])):
		tl = (i * cell[0], j * cell[1])
		br = ((i + 1) * cell[0], (j + 1) * cell[1])
		roi = phase[tl[0]:br[0], tl[1]:br[1]]
		wroi = w[tl[0]:br[0], tl[1]:br[1]]
		hist, _ = np.histogram(roi, bins=orientations, range=(-np.pi, np.pi), weights=wroi, density=True)
		#hist /= (np.sum(hist) + util.EPS)

		if np.sum(wroi) < util.EPS:
			hist = np.zeros(hist.shape, dtype=hist.dtype)
		
		ans = np.hstack((ans, hist))
	ans /= (np.sum(ans) + util.EPS)
	return ans

import matplotlib.pyplot as plt
def hog_io(norm, lce, wmci, lung_mask, blobs, masks):
	feature_vectors = []
	for i in range(len(blobs)):
		x, y, r = blobs[i]
		shift = 0 
		side = 2 * shift + 2 * r + 1
		dsize = (32, 32)

		tl = (x - shift - r, y - shift - r)
		ntl = (max(0, tl[0]), max(0, tl[1]))
		br = (x + shift + r + 1, y + shift + r + 1)
		nbr = (min(lce.shape[0], br[0]), min(lce.shape[1], br[1]))

		lce_roi = lce[ntl[0]:nbr[0], ntl[1]:nbr[1]]
		lce_roi = cv2.resize(lce_roi, dsize, interpolation=cv2.INTER_CUBIC)
		mask = cv2.resize(masks[i], dsize, interpolation=cv2.INTER_CUBIC)
		#mask = np.ones(mask.shape, dtype=mask.dtype)
		
		feats = _hog(lce_roi, mask, orientations=9, cell=(32, 32))
		feats_outer = _hog(lce_roi, 1-mask, orientations=9, cell=(32, 32))
		feats = np.hstack((feats, feats_outer))
		feats /= 2
		'''
		print len(feats), max(feats), min(feats), np.mean(feats), np.sum(feats)
		plt.bar(range(len(feats)), feats)
		plt.show()
		util.imshow('roi', lce_roi)
		'''
		feature_vectors.append(np.array(feats))

	return np.array(feature_vectors)


def hog(norm, lce, wmci, lung_mask, blobs, masks):
	feature_vectors = []
	for i in range(len(blobs)):
		x, y, r = blobs[i]
		shift = 0 
		side = 2 * shift + 2 * r + 1
		dsize = (32, 32)

		tl = (x - shift - r, y - shift - r)
		ntl = (max(0, tl[0]), max(0, tl[1]))
		br = (x + shift + r + 1, y + shift + r + 1)
		nbr = (min(lce.shape[0], br[0]), min(lce.shape[1], br[1]))

		lce_roi = lce[ntl[0]:nbr[0], ntl[1]:nbr[1]]
		lce_roi = cv2.resize(lce_roi, dsize, interpolation=cv2.INTER_CUBIC)

		#util.imshow('hog lce roi', lce_roi)

		feats = feature.hog(lce_roi, orientations=9, pixels_per_cell=(32, 32), cells_per_block=(1, 1), visualise=False, normalise=False)
		'''
		print feats
		print len(feats), max(feats), min(feats), np.mean(feats), np.sum(feats)
		plt.bar(range(len(feats)), feats)
		plt.show()
		util.imshow('roi', lce_roi)
		'''
		feature_vectors.append(np.array(feats))

	return np.array(feature_vectors)

def lbpio(img, lung_mask, blobs, masks):
	P = 9
	R = 1
	feature_vectors = []
	for i in range(len(blobs)):
		x, y, r = blobs[i]
		shift = 0 
		side = 2 * shift + 2 * r + 1

		tl = (x - shift - r, y - shift - r)
		ntl = (max(0, tl[0]), max(0, tl[1]))
		br = (x + shift + r + 1, y + shift + r + 1)
		nbr = (min(img.shape[0], br[0]), min(img.shape[1], br[1]))

		img_roi = img[ntl[0]:nbr[0], ntl[1]:nbr[1]]
		#util.imshow('hog lce roi', lce_roi)
		lbp = feature.local_binary_pattern(img_roi, P, R, method='uniform')

		mask = masks[i].astype(np.float64)
		imask = 1 - mask

		#print "mask shape {} {}".format(mask.shape, mask.ravel().shape)
		#print "roi shape {} {}".format(lbp.shape, lbp.ravel().shape)
		bins = lbp.max() + 1
		hi, _ = np.histogram(lbp.ravel(), bins=bins, range=(0, bins), weights=mask.ravel(), density=True)
		ho, _ = np.histogram(lbp.ravel(), bins=bins, range=(0, bins), weights=imask.ravel(), density=True)
		
		#print "hi shape {} sum {}".format(hi.shape, util.EPS)
		hi /= (np.sum(hi) + util.EPS)
		ho /= (np.sum(hi) + util.EPS)
		hist = np.hstack((hi, ho))
		hist /= (np.sum(hist) + util.EPS)

		feature_vectors.append(np.array(hist))

	return np.array(feature_vectors)

class HardieExtractor:
	def extract(self, norm, lce, wmci, lung_mask, blobs, nod_masks):
		return hardie(norm, lce, wmci, lung_mask, blobs, nod_masks)

class HogExtractor:
	def extract(self, norm, lce, wmci, lung_mask, blobs, nod_masks):
		return hog(norm, lce, wmci, lung_mask, blobs, nod_masks)

class HogIOExtractor:
	def extract(self, norm, lce, wmci, lung_mask, blobs, nod_masks):
		return hog_io(norm, lce, wmci, lung_mask, blobs, nod_masks)

class LBPExtractor:
	def extract(self, norm, lce, wmci, lung_mask, blobs, nod_masks):
		fv_lce = lbpio(lce, lung_mask, blobs, nod_masks)
		fv_wmci = lbpio(wmci, lung_mask, blobs, nod_masks)
		hist = np.hstack((fv_lce, fv_wmci))
		hist /= 2.0

		return hist

class ZernikeExtractor:
	def __init__(self):
		self.radius = int(32 * 0.4)

	def extract_whole(self, img, blobs, masks):
		feature_vectors = []
		for i in range(len(blobs)):
			x, y, r = blobs[i]
			shift = 0 
			side = 2 * shift + 2 * r + 1

			tl = (x - shift - r, y - shift - r)
			ntl = (max(0, tl[0]), max(0, tl[1]))
			br = (x + shift + r + 1, y + shift + r + 1)
			nbr = (min(img.shape[0], br[0]), min(img.shape[1], br[1]))

			img_roi = img[ntl[0]:nbr[0], ntl[1]:nbr[1]]
			mask = masks[i].astype(np.float64)
			imask = 1 - mask

			feats = mahotas.features.zernike_moments(img_roi, int(r * 0.8), cm=(img_roi.shape[0]/2, img_roi.shape[1]/2))
			#feats /= (np.sum(feats) + util.EPS)
			feature_vectors.append(np.array(feats))

		return np.array(feature_vectors)

	def extract_inner(self, img, blobs, masks):
		feature_vectors = []
		for i in range(len(blobs)):
			x, y, r = blobs[i]
			shift = 0 
			side = 2 * shift + 2 * r + 1

			tl = (x - shift - r, y - shift - r)
			ntl = (max(0, tl[0]), max(0, tl[1]))
			br = (x + shift + r + 1, y + shift + r + 1)
			nbr = (min(img.shape[0], br[0]), min(img.shape[1], br[1]))

			img_roi = img[ntl[0]:nbr[0], ntl[1]:nbr[1]]
			mask = masks[i].astype(np.float64)

			feats = mahotas.features.zernike_moments(mask * img_roi, int(r * 0.8), cm=(img_roi.shape[0]/2, img_roi.shape[1]/2))
			#util.imshow('inner', mask * img_roi)
			#feats /= (np.sum(feats) + util.EPS)
			feature_vectors.append(np.array(feats))

		return np.array(feature_vectors)

	def extract_contour(self, img, blobs, masks):
		feature_vectors = []
		for i in range(len(blobs)):
			x, y, r = blobs[i]
			shift = 0 
			side = 2 * shift + 2 * r + 1

			tl = (x - shift - r, y - shift - r)
			ntl = (max(0, tl[0]), max(0, tl[1]))
			br = (x + shift + r + 1, y + shift + r + 1)
			nbr = (min(img.shape[0], br[0]), min(img.shape[1], br[1]))

			img_roi = img[ntl[0]:nbr[0], ntl[1]:nbr[1]]
			mask = masks[i].astype(np.float64)
			imask = 1 - mask
			per_mask = np.full(mask.shape, dtype=mask.dtype, fill_value=0)
			contours, _ = cv2.findContours(mask.astype(np.uint8).copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		  	cv2.drawContours(per_mask, contours, -1, 1, 1)

			feats = mahotas.features.zernike_moments(per_mask, int(r * 0.8), cm=(img_roi.shape[0]/2, img_roi.shape[1]/2))
			#feats /= (np.sum(feats) + util.EPS)
			feature_vectors.append(np.array(feats))

		return np.array(feature_vectors)

	def extract_mask(self, img, blobs, masks):
		feature_vectors = []
		for i in range(len(blobs)):
			x, y, r = blobs[i]
			shift = 0 
			side = 2 * shift + 2 * r + 1

			tl = (x - shift - r, y - shift - r)
			ntl = (max(0, tl[0]), max(0, tl[1]))
			br = (x + shift + r + 1, y + shift + r + 1)
			nbr = (min(img.shape[0], br[0]), min(img.shape[1], br[1]))

			img_roi = img[ntl[0]:nbr[0], ntl[1]:nbr[1]]
			mask = masks[i].astype(np.float64)
			feats = mahotas.features.zernike_moments(mask, int(r * 0.8), cm=(img_roi.shape[0]/2, img_roi.shape[1]/2))
			#feats /= (np.sum(feats) + util.EPS)
			feature_vectors.append(np.array(feats))

		return np.array(feature_vectors)


	def extract(self, norm, lce, wmci, lung_mask, blobs, nod_masks):
		return self.extract_mask(lce, blobs, nod_masks)

class ShapeExtractor:
	def finite_derivative(self, v):
		dv = np.empty(v.shape, dtype=np.double)

		dv[0] = [0, 0]
		dv[-1] = [0, 0]
		dv[1:-1, :] = (v[2:, :] - v[:-2, :]) / 2.0

		return dv

	def extract_(self, img, blobs, masks):
		feature_vectors = []
		for i in range(len(blobs)):
			x, y, r = blobs[i]
			shift = 0 
			side = 2 * shift + 2 * r + 1

			tl = (x - shift - r, y - shift - r)
			ntl = (max(0, tl[0]), max(0, tl[1]))
			br = (x + shift + r + 1, y + shift + r + 1)
			nbr = (min(img.shape[0], br[0]), min(img.shape[1], br[1]))

			img_roi = img[ntl[0]:nbr[0], ntl[1]:nbr[1]]
			mask = masks[i].astype(np.float64)
			per_mask = np.full(mask.shape, dtype=mask.dtype, fill_value=0)
			contours, _ = cv2.findContours(mask.astype(np.uint8).copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		  	
		  	cv2.drawContours(per_mask, contours, -1, 1, 1)

		  	if len(contours) > 0 and len(contours[0]) > 0:
		  		np.append(contours[0], [contours[0][0]])

		  	contours = np.array(contours[0])
		  	dv = self.finite_derivative(contours)
		  	d2v = self.finite_derivative(dv)
		  	dv = dv.T
		  	d2v = d2v.T
		  	L = len(contours)

		  	k = (dv[0] * d2v[1] - dv[1] * d2v[0]) / ((dv[0] ** 2 + dv[1] ** 2) ** 1.5 + util.EPS)
		  	k /= (np.mean(np.absolute(k)) + util.EPS)

		  	curvature = np.sum(np.absolute(k)) / (float(L) + util.EPS)

		  	deformation_energy = np.mean(k ** 2)

		  	perimeter = cv2.arcLength(contours, True)
		  	area = np.sum(mask)
		  	compactness = perimeter ** 2 / (float(area) + util.EPS)

		  	hull = cv2.convexHull(contours)
		  	hull_perimeter = cv2.arcLength(hull, True)
		  	convexity = hull_perimeter / (perimeter + util.EPS)

		  	hull_area = cv2.contourArea(hull)
		  	solidity = hull_area / (float(area) + util.EPS)

			feats = [curvature, deformation_energy, compactness, convexity, solidity]

			feature_vectors.append(np.array(feats))

		return np.array(feature_vectors)

	def extract(self, norm, lce, wmci, lung_mask, blobs, nod_masks):
		return self.extract_(lce, blobs, nod_masks)


# TODO: centrist
class AllExtractor:
	def __init__(self):
		self.extractors = []
		self.extractors.append(LBPExtractor())
		self.extractors.append(HogIOExtractor())
		self.extractors.append(HardieExtractor())
		self.extractors.append(ZernikeExtractor())
		self.extractors.append(ShapeExtractor())

	def extract(self, norm, lce, wmci, lung_mask, blobs, nod_masks):
		fv_set = []

		for extractor in self.extractors:
			fv_set.append(extractor.extract(norm, lce, wmci, lung_mask, blobs, nod_masks))
		fv = np.hstack(fv_set)
		
		return fv

class Set1Extractor:
	def __init__(self):
		self.extractors = []
		self.extractors.append(HardieExtractor())
		self.extractors.append(LBPExtractor())
		self.extractors.append(HogIOExtractor())
		self.extractors.append(ZernikeExtractor())
		self.extractors.append(ShapeExtractor())

	def extract(self, norm, lce, wmci, lung_mask, blobs, nod_masks):
		fv_set = []

		for extractor in self.extractors:
			fv_set.append(extractor.extract(norm, lce, wmci, lung_mask, blobs, nod_masks))
		fv = np.hstack(fv_set)
		
		return fv
# Register extractors

class OverfeatExtractor:
	def __init__(self, mode=None):
		overfeat.init('/home/juliomb/lnd-env/OverFeat/data/default/net_weight_0', 0)
		self.mode = mode

	def extract(self, norm, lce, wmci, lung_mask, blobs, nod_masks):
		feature_vectors = []
		for i in range(len(blobs)):
			x, y, r = blobs[i]
			shift = 0 
			side = 2 * shift + 2 * r + 1
			dsize = (231, 231)

			tl = (x - shift - r, y - shift - r)
			ntl = (max(0, tl[0]), max(0, tl[1]))
			br = (x + shift + r + 1, y + shift + r + 1)
			nbr = (min(lce.shape[0], br[0]), min(lce.shape[1], br[1]))

			lce_roi = lce[ntl[0]:nbr[0], ntl[1]:nbr[1]]
			lce_roi = cv2.resize(lce_roi, dsize, interpolation=cv2.INTER_CUBIC)

			if self.mode == 'inner':
				mask = cv2.resize(nod_masks[i], dsize, interpolation=cv2.INTER_CUBIC)
				image = mask.astype(np.float32) * lce_roi	
				image = np.array([image.copy(), image.copy(), image.copy()])
			else:	
				image = np.array([lce_roi.copy(), lce_roi.copy(), lce_roi.copy()])

			_ = overfeat.fprop(image.astype(np.float32))

			feats = overfeat.get_output(19)
			feats = feats.flatten()	

			feature_vectors.append(np.array(feats))

		return np.array(feature_vectors)


extractors = {'hardie':HardieExtractor(), 'hog':HogExtractor(), 'hogio':HogIOExtractor(), \
				'lbpio':LBPExtractor(), 'znk':ZernikeExtractor(), 'shape':ShapeExtractor(), \
				'all':AllExtractor(), 'set1':Set1Extractor(), 'overf':OverfeatExtractor(), \
				'overfin':OverfeatExtractor(mode='inner')}
