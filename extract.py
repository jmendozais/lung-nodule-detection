import numpy as np
from itertools import *
from math import *
from time import *

import cv2
from scipy.stats import skew, kurtosis
import skimage.feature as feature
import mahotas
import overfeat
import matplotlib.pyplot as plt


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

def sort_by_len(arr1, arr2):
	return len(arr2) - len(arr1)

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

	# Vectorize ROI	operations
	tl = (x - shift - r, y - shift - r)
	ntl = (max(0, tl[0]), max(0, tl[1]))
	br = (x + shift + r + 1, y + shift + r + 1)
	nbr = (min(img.shape[0], br[0]), min(img.shape[1], br[1]))

	roi = img[ntl[0]:nbr[0], ntl[1]:nbr[1]]
	ext_mask = np.full(roi.shape, dtype=np.uint8, fill_value=0)	
	ext_mask[(tl[0] + shift - ntl[0]):(tl[0] + shift + 2 * r + 1 - ntl[0]), (tl[1] + shift - ntl[1]):(tl[1] + shift + 2 * r + 1 - ntl[1])] = mask

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

# FIX: The shift should be in relation with the bounding box of the segmentation. It means the mask.
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

	rgrad = np.cos(phase) * roi
	ins = np.ma.array(data=rgrad, mask=1-ext_mask, fill_value=0)
	out = np.ma.array(data=rgrad, mask=ext_mask, fill_value=0)
	rgrad_result = (ins.mean(), out.mean(), ins.std(), out.std())

	return phase_result, rgrad_result

def geometric(img, blob, mask, xrange, yrange, dt_img):
	results = {}
	contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	contours = sorted(contours, cmp=sort_by_len)
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
	contours = sorted(contours, cmp=sort_by_len)
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
	# 0-15
	adt_geom = ['x_fract', 'dist_perim']
	
	norm_inte = ['mean_sep', 'skew_in', 
	   'moment_4_in', 'moment_5_in', 'moment_7_in']

	norm_grad = ['rd_mean_in', 'rd_std_in',
	   'rd_std_out', 'rd_std_sep', 'rgrad_mean_in',
	   'rgrad_mean_sep', 'rgrad_std_in',
	   'rgrad_std_out', 'rgrad_std_sep']
	# 16-26
	lce_inte = ['mean_sep', 'contrast_1', 
	   'std_in', 'std_sep', 
	   'skew_in', 'kurt_in', 'moment_1_in']
	lce_grad = ['mag_mean_in', 'rd_std_in',
	   'rd_std_out', 'rgrad_mean_sep']
	# 27-43
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


from skimage.exposure import equalize_hist
class HardieExtractor:
	def extract(self, norm, lce, wmci, lung_mask, blobs, nod_masks):
		return hardie(norm, lce, wmci, lung_mask, blobs, nod_masks)

class HogExtractor:
	def __init__(self, mode='default', input='norm'):
		self.mode = mode
		self.input = input

	def hog(self, img, mask, orientations=9, cell=(8,8)):
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

	def hog_mask(self, img, lung_mask, blobs, masks, mode='default', cell=(8,8)):
		feature_vectors = []
		for i in range(len(blobs)):
			x, y, r = blobs[i]
			shift = 0 
			side = 2 * shift + 2 * r + 1
			dsize = (32, 32)

			tl = (x - shift - r, y - shift - r)
			ntl = (max(0, tl[0]), max(0, tl[1]))
			br = (x + shift + r + 1, y + shift + r + 1)
			nbr = (min(img.shape[0], br[0]), min(img.shape[1], br[1]))

			img_roi = img[ntl[0]:nbr[0], ntl[1]:nbr[1]]
			img_roi = cv2.resize(img_roi, dsize, interpolation=cv2.INTER_CUBIC)
			mask = cv2.resize(masks[i], dsize, interpolation=cv2.INTER_CUBIC)
			if mode == 'default':
				mask = np.ones(mask.shape, dtype=mask.dtype)
				feats = self.hog(img_roi, mask, orientations=9, cell=cell)
			elif mode == 'inner':
				feats = self.hog(img_roi, mask, orientations=9, cell=cell)
			elif mode == 'inner_outer':
				feats = self.hog(img_roi, mask, orientations=9, cell=cell)
				feats_outer = self.hog(img_roi, 1-mask, orientations=9, cell=cell)
				feats = np.hstack((feats, feats_outer))
				feats /= 2

			feature_vectors.append(np.array(feats))

		return np.array(feature_vectors)


	def hog_skimage(self, img, lung_mask, blobs, masks, cell=None):
		feature_vectors = []
		for i in range(len(blobs)):
			x, y, r = blobs[i]
			shift = 0 
			side = 2 * shift + 2 * r + 1
			dsize = (32, 32)

			tl = (x - shift - r, y - shift - r)
			ntl = (max(0, tl[0]), max(0, tl[1]))
			br = (x + shift + r + 1, y + shift + r + 1)
			nbr = (min(img.shape[0], br[0]), min(img.shape[1], br[1]))

			img_roi = img[ntl[0]:nbr[0], ntl[1]:nbr[1]]
			img_roi = cv2.resize(img_roi, dsize, interpolation=cv2.INTER_CUBIC)

			if cell != None:
				feats = feature.hog(img_roi, orientations=9, pixels_per_cell=cell, cells_per_block=(1, 1), visualise=False, normalise=False)
			else:
				feats = feature.hog(img_roi, orientations=9, cells_per_block=(1, 1), visualise=False, normalise=False)
			feature_vectors.append(np.array(feats))

		return np.array(feature_vectors)

	def extract(self, norm, lce, wmci, lung_mask, blobs, nod_masks):
		img = lce
		if self.input == 'norm':
			img = norm
		elif self.input == 'wmci':
			img = wmci

		if self.mode == 'skimage_default':
			return self.hog_skimage(img, lung_mask, blobs, nod_masks)
		elif self.mode == 'skimage_32x32':
			return self.hog_skimage(img, lung_mask, blobs, nod_masks, cell=(32,32))
		elif self.mode == 'default':
			return  self.hog_mask(img, lung_mask, blobs, nod_masks, mode='default', cell=(8,8))
		elif self.mode == 'inner':
			return  self.hog_mask(img, lung_mask, blobs, nod_masks, mode='inner', cell=(8,8))
		elif self.mode == 'inner_outer':
			return  self.hog_mask(img, lung_mask, blobs, nod_masks, mode='inner_outer', cell=(8,8))
		elif self.mode == '32x32_default':
			return self.hog_mask(img, lung_mask, blobs, nod_masks, mode='default', cell=(32,32))
		elif self.mode == '32x32_inner':
			return self.hog_mask(img, lung_mask, blobs, nod_masks, mode='inner', cell=(32,32))
		elif self.mode == '32x32_inner_outer':
			return self.hog_mask(img, lung_mask, blobs, nod_masks, mode='inner_outer', cell=(32,32))


# Histogram of Second Order Gradients
# TODO: implement first order gradient magnitude with filter bank at diferente orientations
class HSOGExtractor:
	def __init__(self, mode='default', input='norm'):
		self.mode = mode
		self.input = input

	def hsog(self, img, mask, orientations=9, cell=(8,8)):
		mag, _, _ = finite_derivatives(img)
		mag, dx, dy = finite_derivatives(mag)
		phase = np.arctan2(dy, dx)
		phase = phase.astype(np.float64)	
		#phase = np.abs(phase)

		size = img.shape
		size = (size[0] / cell[0], size[1] / cell[1])
		w = mask.astype(np.float64)
		w *= mag

		ans = np.array([])
		for i, j in product(range(size[0]), range(size[1])):
			tl = (i * cell[0], j * cell[1])
			br = ((i + 1) * cell[0], (j + 1) * cell[1])
			roi = phase[tl[0]:br[0], tl[1]:br[1]]
			wroi = w[tl[0]:br[0], tl[1]:br[1]]
			if np.sum(wroi) > util.EPS:
				wroi /= np.sum(wroi)

			hist, _ = np.histogram(roi, bins=orientations, range=(-np.pi, np.pi), weights=wroi, density=True)
			#hist /= (np.sum(hist) + util.EPS)
			if np.sum(wroi) < util.EPS:
				hist = np.zeros(hist.shape, dtype=hist.dtype)
			
			ans = np.hstack((ans, hist))
		ans /= (np.sum(ans) + util.EPS)
		return ans

	def hsog_mask(self, img, lung_mask, blobs, masks, mode='default', cell=(8,8)):
		feature_vectors = []
		for i in range(len(blobs)):
			x, y, r = blobs[i]
			shift = 0 
			side = 2 * shift + 2 * r + 1
			dsize = (32, 32)

			tl = (x - shift - r, y - shift - r)
			ntl = (max(0, tl[0]), max(0, tl[1]))
			br = (x + shift + r + 1, y + shift + r + 1)
			nbr = (min(img.shape[0], br[0]), min(img.shape[1], br[1]))

			img_roi = img[ntl[0]:nbr[0], ntl[1]:nbr[1]]
			img_roi = cv2.resize(img_roi, dsize, interpolation=cv2.INTER_CUBIC)
			mask = cv2.resize(masks[i], dsize, interpolation=cv2.INTER_CUBIC)
			if mode == 'default':
				mask = np.ones(mask.shape, dtype=mask.dtype)
				feats = self.hsog(img_roi, mask, orientations=9, cell=cell)
			elif mode == 'inner':
				feats = self.hsog(img_roi, mask, orientations=9, cell=cell)
			elif mode == 'inner_outer':
				feats = self.hsog(img_roi, mask, orientations=9, cell=cell)
				feats_outer = self.hsog(img_roi, 1-mask, orientations=9, cell=cell)
				feats = np.hstack((feats, feats_outer))
				feats /= 2

			feature_vectors.append(np.array(feats))

		return np.array(feature_vectors)


	def extract(self, norm, lce, wmci, lung_mask, blobs, nod_masks):
		img = lce
		if self.input == 'norm':
			img = norm
		elif self.input == 'wmci':
			img = wmci

		if self.mode == 'default':
			return  self.hsog_mask(img, lung_mask, blobs, nod_masks, mode='default', cell=(8,8))
		elif self.mode == 'inner':
			return  self.hsog_mask(img, lung_mask, blobs, nod_masks, mode='inner', cell=(8,8))
		elif self.mode == 'inner_outer':
			return  self.hsog_mask(img, lung_mask, blobs, nod_masks, mode='inner_outer', cell=(8,8))
		elif self.mode == '32x32_default':
			return self.hsog_mask(img, lung_mask, blobs, nod_masks, mode='default', cell=(32,32))
		elif self.mode == '32x32_inner':
			return self.hsog_mask(img, lung_mask, blobs, nod_masks, mode='inner', cell=(32,32))
		elif self.mode == '32x32_inner_outer':
			return self.hsog_mask(img, lung_mask, blobs, nod_masks, mode='inner_outer', cell=(32,32))

# Histogram of Radial Gradients

class HRGExtractor:
	def __init__(self, mode='default', input='norm', method='deviation'):
		self.mode = mode
		self.input = input
		self.method = method
		self.max = -1

	def hist(self, phase, mag, mask, orientations=9, cell=(8,8)):
		size = phase.shape
		size = (size[0] / cell[0], size[1] / cell[1])
		w = mask.astype(np.float64)
		w *= mag


		ans = np.array([])
		for i, j in product(range(size[0]), range(size[1])):
			tl = (i * cell[0], j * cell[1])
			br = ((i + 1) * cell[0], (j + 1) * cell[1])
			roi = phase[tl[0]:br[0], tl[1]:br[1]]
			wroi = w[tl[0]:br[0], tl[1]:br[1]]

			if np.sum(wroi) > util.EPS:
				wroi /= np.sum(wroi)
			if self.max < np.max(wroi * roi):
				self.max = np.max(wroi * roi)
			# deviation
			range_ = (0, 2 * np.pi)
			if self.method == 'strenght':
				range_ = (0, 1)

			h, _ = np.histogram(roi, bins=orientations, range=range_, weights=wroi, density=True)

			if np.sum(wroi) < util.EPS:
				h = np.zeros(h.shape, dtype=h.dtype)

			ans = np.hstack((ans, h))
		ans /= (np.sum(ans) + util.EPS)
		return ans

	def hrg(self, img, mask, dx, dy, mag, orientations=9, cell=(8,8)):
		side = img.shape[0]
		rx = -1 * np.linspace(-1 * (side/2), side/2, side)
		ry = -1 * np.linspace(-1 * (side/2), side/2, side)
		ry, rx = np.meshgrid(rx, ry)
		phase = angle2((rx, ry), (dx, dy)) # (1)

		# deviation
		mag_ = np.ones(mag.shape, dtype=np.float)
		if self.method == 'strenght':
			mag_ = mag
		#rgrad = np.cos(phase) * mag
		return self.hist(phase, mag_, mask, orientations=9, cell=(8,8))

	def hrg_mask(self, img, lung_mask, blobs, masks, mode='default', cell=(8,8)):
		mag, dx, dy = finite_derivatives(img)

		feature_vectors = []
		for i in range(len(blobs)):
			x, y, r = blobs[i]
			shift = 0 #35
			side = 2 * shift + 2 * r + 1

			tl = (x - shift - r, y - shift - r)
			ntl = (max(0, tl[0]), max(0, tl[1]))
			br = (x + shift + r + 1, y + shift + r + 1)
			nbr = (min(img.shape[0], br[0]), min(img.shape[1], br[1]))

			roi = img[ntl[0]:nbr[0], ntl[1]:nbr[1]]
			dx_roi = dx[ntl[0]:nbr[0], ntl[1]:nbr[1]]
			dy_roi = dy[ntl[0]:nbr[0], ntl[1]:nbr[1]]
			mag_roi = mag[ntl[0]:nbr[0], ntl[1]:nbr[1]]
			ext_mask = np.full(roi.shape, dtype=np.uint8, fill_value=0)	
			ext_mask[(tl[0] + shift - ntl[0]):(tl[0] + shift + 2 * r + 1 - ntl[0]), (tl[1] + shift - ntl[1]):(tl[1] + shift + 2 * r + 1 - ntl[1])] = masks[i]

			dsize = (64, 64)
			roi = cv2.resize(roi, dsize, interpolation=cv2.INTER_CUBIC)
			mask = cv2.resize(ext_mask, dsize, interpolation=cv2.INTER_CUBIC)
			dx_roi = cv2.resize(dx_roi, dsize, interpolation=cv2.INTER_CUBIC)
			dy_roi = cv2.resize(dy_roi, dsize, interpolation=cv2.INTER_CUBIC)
			mag_roi = cv2.resize(mag_roi, dsize, interpolation=cv2.INTER_CUBIC)

			if mode == 'default':
				mask = np.ones(mask.shape, dtype=mask.dtype)
				feats = self.hrg(roi, mask, dx_roi, dy_roi, mag_roi, orientations=9, cell=cell)
			elif mode == 'inner':
				feats = self.hrg(roi, mask, dx_roi, dy_roi, mag_roi, orientations=9, cell=cell)
			elif mode == 'inner_outer':
				feats = self.hrg(roi, mask, dx_roi, dy_roi, mag_roi, orientations=9, cell=cell)
				feats_outer = self.hrg(roi, 1-mask, dx_roi, dy_roi, mag_roi, orientations=9, cell=cell)
				feats = np.hstack((feats, feats_outer))
				feats /= 2

			feature_vectors.append(np.array(feats))

		return np.array(feature_vectors)

	def extract(self, norm, lce, wmci, lung_mask, blobs, nod_masks):
		img = lce
		if self.input == 'norm':
			img = norm
		elif self.input == 'wmci':
			img = wmci

		if self.mode == 'skimage_default':
			return self.hrg_skimage(img, lung_mask, blobs, nod_masks)
		elif self.mode == 'skimage_32x32':
			return self.hrg_skimage(img, lung_mask, blobs, nod_masks, cell=(32,32))
		elif self.mode == 'default':
			return  self.hrg_mask(img, lung_mask, blobs, nod_masks, mode='default', cell=(8,8))
		elif self.mode == 'inner':
			return  self.hrg_mask(img, lung_mask, blobs, nod_masks, mode='inner', cell=(8,8))
		elif self.mode == 'inner_outer':
			return  self.hrg_mask(img, lung_mask, blobs, nod_masks, mode='inner_outer', cell=(8,8))
		elif self.mode == '32x32_default':
			return self.hrg_mask(img, lung_mask, blobs, nod_masks, mode='default', cell=(32,32))
		elif self.mode == '32x32_inner':
			return self.hrg_mask(img, lung_mask, blobs, nod_masks, mode='inner', cell=(32,32))
		elif self.mode == '32x32_inner_outer':
			return self.hrg_mask(img, lung_mask, blobs, nod_masks, mode='inner_outer', cell=(32,32))


class LBPExtractor:
	# TODO set optimal model
	def __init__(self, method='uniform', input='norm', mode='default'):
		self.method = method
		self.input = input
		self.mode = mode

	def lbpio(self, img, lung_mask, blobs, masks, method='uniform', mode='inner_outer'):
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
			lbp = feature.local_binary_pattern(img_roi, P, R, method=method)

			mask = masks[i].astype(np.float64)
			imask = 1 - mask

			bins = lbp.max() + 1
			hi = []
			ho = []
			if mode == 'inner' or mode == 'inner_outer':
				hi, _ = np.histogram(lbp.ravel(), bins=bins, range=(0, bins), weights=mask.ravel(), density=True)
				hi /= (np.sum(hi) + util.EPS)
			if mode == 'outer' or mode == 'inner_outer':
				ho, _ = np.histogram(lbp.ravel(), bins=bins, range=(0, bins), weights=imask.ravel(), density=True)
				ho /= (np.sum(hi) + util.EPS)
			
			#print "hi shape {} sum {}".format(hi.shape, util.EPS)
			hist = []
			if mode == 'inner_outer':
				hist = np.hstack((hi, ho))
				hist /= (np.sum(hist) + util.EPS)
			elif mode == 'inner':
				hist = hi
			elif mode == 'outer':
				hist = ho
			elif mode == 'default':
				hist, _ = np.histogram(lbp.ravel(), bins=bins, range=(0, bins), density=True)
				hist /= (np.sum(hist) + util.EPS)

			feature_vectors.append(np.array(hist))

		return np.array(feature_vectors)

	def extract(self, norm, lce, wmci, lung_mask, blobs, nod_masks):
		img = lce
		if self.input == 'wmci':
			img = wmci
		if self.input == 'norm':
			img = norm
		
		return self.lbpio(img, lung_mask, blobs, nod_masks, self.method, self.mode)

# PhaseLBP
class PLBPExtractor:
	# TODO set optimal model
	def __init__(self, method='uniform', input='norm', mode='default'):
		self.method = method
		self.input = input
		self.mode = mode

	def lbpio(self, img, lung_mask, blobs, masks, method='uniform', mode='inner_outer'):
		P = 9
		R = 1
		feature_vectors = []

		mag, dx, dy = finite_derivatives(img)

		for i in range(len(blobs)):
			x, y, r = blobs[i]
			shift = 0 
			side = 2 * shift + 2 * r + 1

			tl = (x - shift - r, y - shift - r)
			ntl = (max(0, tl[0]), max(0, tl[1]))
			br = (x + shift + r + 1, y + shift + r + 1)
			nbr = (min(img.shape[0], br[0]), min(img.shape[1], br[1]))

			img_roi = img[ntl[0]:nbr[0], ntl[1]:nbr[1]]
			dx_roi = dx[ntl[0]:nbr[0], ntl[1]:nbr[1]]
			dy_roi = dy[ntl[0]:nbr[0], ntl[1]:nbr[1]]
			mag_roi = mag[ntl[0]:nbr[0], ntl[1]:nbr[1]]

			side = img_roi.shape[0]
			rx = -1 * np.linspace(-1 * (side/2), side/2, side)
			ry = -1 * np.linspace(-1 * (side/2), side/2, side)
			ry, rx = np.meshgrid(rx, ry)
			phase = angle2((rx, ry), (dx_roi, dy_roi)) 

			util.imshow('roi', img_roi)
			util.imshow('phase', phase)
			util.imshow('mask', masks[i])

			lbp = feature.local_binary_pattern(phase, P, R, method=method)

			mask = masks[i].astype(np.float64)
			imask = 1 - mask

			bins = lbp.max() + 1
			hi = []
			ho = []
			if mode == 'inner' or mode == 'inner_outer':
				hi, _ = np.histogram(lbp.ravel(), bins=bins, range=(0, bins), weights=mask.ravel(), density=True)
				hi /= (np.sum(hi) + util.EPS)
			if mode == 'outer' or mode == 'inner_outer':
				ho, _ = np.histogram(lbp.ravel(), bins=bins, range=(0, bins), weights=imask.ravel(), density=True)
				ho /= (np.sum(hi) + util.EPS)
			
			#print "hi shape {} sum {}".format(hi.shape, util.EPS)
			hist = []
			if mode == 'inner_outer':
				hist = np.hstack((hi, ho))
				hist /= (np.sum(hist) + util.EPS)
			elif mode == 'inner':
				hist = hi
			elif mode == 'outer':
				hist = ho
			elif mode == 'default':
				hist, _ = np.histogram(lbp.ravel(), bins=bins, range=(0, bins), density=True)
				hist /= (np.sum(hist) + util.EPS)

			feature_vectors.append(np.array(hist))

		return np.array(feature_vectors)

	def extract(self, norm, lce, wmci, lung_mask, blobs, nod_masks):
		img = lce
		if self.input == 'wmci':
			img = wmci
		if self.input == 'norm':
			img = norm
		
		return self.lbpio(img, lung_mask, blobs, nod_masks, self.method, self.mode)

class ZernikeExtractor:
	def __init__(self, input='wmci', mode='nomask'):
		self.radius = int(32 * 0.4)
		self.input = input
		self.mode = mode

	def zernike(self, img, blobs, masks, mode='nomask'):
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

			feats = []
			if mode == 'nomask':
				feats = mahotas.features.zernike_moments(img_roi, int(r * 0.8), cm=(img_roi.shape[0]/2, img_roi.shape[1]/2))
			elif mode == 'mask':
				feats = mahotas.features.zernike_moments(mask, int(r * 0.8), cm=(img_roi.shape[0]/2, img_roi.shape[1]/2))
			elif mode == 'inner':
				feats = mahotas.features.zernike_moments(mask * img_roi, int(r * 0.8), cm=(img_roi.shape[0]/2, img_roi.shape[1]/2))
			elif mode == 'inner_outer':
				imask = 1 - mask
				fi = mahotas.features.zernike_moments(mask * img_roi, int(r * 0.8), cm=(img_roi.shape[0]/2, img_roi.shape[1]/2))
				fo = mahotas.features.zernike_moments(imask * img_roi, int(r * 0.8), cm=(img_roi.shape[0]/2, img_roi.shape[1]/2))
				feats = np.hstack((fi, fo))
			elif mode == 'contour':
				per_mask = np.full(mask.shape, dtype=mask.dtype, fill_value=0)
				contours, _ = cv2.findContours(mask.astype(np.uint8).copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
				contours = sorted(contours, cmp=sort_by_len)
				cv2.drawContours(per_mask, [contours[0]], -1, 1, 1)
				feats = mahotas.features.zernike_moments(per_mask, int(r * 0.8), cm=(img_roi.shape[0]/2, img_roi.shape[1]/2))

			#feats /= (np.sum(feats) + util.EPS)
			feature_vectors.append(np.array(feats))

		return np.array(feature_vectors)

	def extract(self, norm, lce, wmci, lung_mask, blobs, nod_masks):
		img = lce
		if self.input == 'norm':
			img = norm
		elif self.input == 'wmci':
			img = wmci

		return self.zernike(img, blobs, nod_masks, mode=self.mode)

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
			contours, _ = cv2.findContours(mask.astype(np.uint8).copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
			contours = sorted(contours, cmp=sort_by_len)
			cv2.drawContours(per_mask, [contours[0]], -1, 1, 1)

			has_inner = 0
			if len(contours) > 1 and len(contours[1]) > 25:
				has_inner = 1
			
			contours = contours[0]

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

			feats = [curvature, deformation_energy, compactness, convexity, solidity, has_inner]

			feature_vectors.append(np.array(feats))

		return np.array(feature_vectors)

	def extract(self, norm, lce, wmci, lung_mask, blobs, nod_masks):
		return self.extract_(lce, blobs, nod_masks)


# TODO: centrist
class AllExtractor:
	def __init__(self):
		self.extractors = []
		self.extractors.append(LBPExtractor())
		self.extractors.append(HogExtractor())
		self.extractors.append(HardieExtractor())
		self.extractors.append(ZernikeExtractor())
		self.extractors.append(ShapeExtractor())

	def extract(self, norm, lce, wmci, lung_mask, blobs, nod_masks):
		fv_set = []
		lce = equalize_hist(lce)
		for extractor in self.extractors:
			fv_set.append(extractor.extract(norm, lce, wmci, lung_mask, blobs, nod_masks))
		fv = np.hstack(fv_set)
		
		return fv

class Set1Extractor:
	def __init__(self):
		self.extractors = []
		self.extractors.append(LBPExtractor())
		self.extractors.append(HogExtractor())
		self.extractors.append(ZernikeExtractor())
		self.extractors.append(ShapeExtractor())

	def extract(self, norm, lce, wmci, lung_mask, blobs, nod_masks):
		fv_set = []

		for extractor in self.extractors:
			fv_set.append(extractor.extract(norm, lce, wmci, lung_mask, blobs, nod_masks))
		fv = np.hstack(fv_set)
		
		return fv

class Set2Extractor:
	def __init__(self):
		self.extractors = []
		self.extractors.append(HogExtractor())
		self.extractors.append(HRGExtractor())

	def extract(self, norm, lce, wmci, lung_mask, blobs, nod_masks):
		fv_set = []

		for extractor in self.extractors:
			fv_set.append(extractor.extract(norm, lce, wmci, lung_mask, blobs, nod_masks))
		fv = np.hstack(fv_set)
		
		return fv

class Set3Extractor:
	def __init__(self):
		self.extractors = []
		self.extractors.append(HRGExtractor())
		self.extractors.append(HogExtractor())
		self.extractors.append(ZernikeExtractor())
		self.extractors.append(ShapeExtractor())

	def extract(self, norm, lce, wmci, lung_mask, blobs, nod_masks):
		fv_set = []
		lce = equalize_hist(lce)
		for extractor in self.extractors:
			fv_set.append(extractor.extract(norm, lce, wmci, lung_mask, blobs, nod_masks))
		fv = np.hstack(fv_set)
		
		return fv

class OverfeatExtractor:
	def __init__(self, mode=None):
		#overfeat.init('/home/juliomb/lnd-env/OverFeat/data/default/net_weight_0', 0)
		self.mode = mode

	def extract(self, norm, lce, wmci, lung_mask, blobs, nod_masks):
		feature_vectors = []
		'''
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

		'''
		return np.array(feature_vectors)


# Register extractors
extractors = {'hardie':HardieExtractor(), 'hog':HogExtractor(), 'hsog':HSOGExtractor(), 'hogio':HogExtractor(mode='inner_outer'), \
				'lbp':LBPExtractor(), 'plbp': PLBPExtractor(), 'znk':ZernikeExtractor(), 'shape':ShapeExtractor(), \
				'all':AllExtractor(), 'set1':Set1Extractor(), 'set2':Set2Extractor(), 'set3':Set3Extractor(),'overf':OverfeatExtractor(), \
				'overfin':OverfeatExtractor(mode='inner'), 'hrg':HRGExtractor(), 'hrgs':HRGExtractor(method='strenght')}
