import numpy as np
from math import sqrt
from skimage.feature import blob_dog, blob_log, blob_doh, peak_local_max
from itertools import *
import cv2
from util import *
#TODO: add detection thinning
import skimage.io as io
# Detection thining
def dst(a, b):
	return sqrt((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]))

def create_graph(points, thold):
	G = {}
	for i in range(len(points)):
		G[i] = set()
		for j in range(len(points)):
			if i != j and dst(points[i], points[j]) < thold:
				G[i].add(j)
	return G
		
def dfs(graph, start, visited):
	stack = [start]
	component = set()
	while stack:
		vertex = stack.pop()
		if vertex not in visited:
			visited.add(vertex)
			component.add(vertex)
			stack.extend(graph[vertex] - visited)
	return component

def detection_thining(points): # 5 mm thold
	G = create_graph(points, 7)
	visited = set()
	comps = []
	resp = []

	for i in range(len(points)):
		if i not in visited:	
			comps.append(dfs(G, i, visited))

	for comp in comps:
		avg = [0.0, 0.0]
		for v in comp:
			avg[0] += points[v][0]
			avg[1] += points[v][1]
		avg[0] /= len(comp)
		avg[1] /= len(comp)
		resp.append(avg)		

	return resp


# Filtering
def filter_by_size(blobs, lower=4, upper=32):
	ans = []
	for blob in blobs:
		x, y, r = blob
		if r >= lower and r <= upper:
			ans.append(blob)
	return np.array(ans)

def filter_by_masks(blobs, mask):
	ans = []
	for blob in blobs:
		x, y, r = blob
		found = False
		if mask[x][y] != 0:
			ans.append(blob)
	return np.array(ans)
	
def filter_by_margin(blobs, mask, margin=30):
	ans = []
	for blob in blobs:
		x, y, r = blob
		found = False
		if x > margin and y > margin and x < mask.shape[0] - margin and y < mask.shape[1] - margin:
			ans.append(blob)
	return np.array(ans)

# Common blob detectors	
def log_(img, mask):
	blobs_log = blob_log(img, min_sigma=4,  max_sigma=32, num_sigma=10, log_scale=True, threshold=0.001, overlap=0.5)
	if len(blobs_log) > 0:
		blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)
	return filter_by_margin(filter_by_size(filter_by_masks(blobs_log, mask)), mask)

def dog(img, mask):
	blobs_dog = blob_dog(img, max_sigma=20, threshold=0.05)
	if len(blobs_dog) > 0:
		blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)
	return filter_by_margin(filter_by_size(filter_by_masks(blobs_dog, mask)), mask)

def doh(img, mask):
	blobs_doh = blob_doh(1 - img, min_sigma=4, num_sigma=10, max_sigma=30, threshold=0.0005)
	return filter_by_margin(filter_by_size(filter_by_masks(blobs_doh, mask)), mask)

# wmci detector
def finite_derivatives(img):
	size = img.shape
	dx = img.copy()
	dy = img.copy()

	for i, j in product(range(1, size[0] - 1), range(1, size[1] - 1)):
		dy[i, j] = (img[i, j + 1] - img[i, j - 1]) / 2.0
		dx[i, j] = (img[i + 1, j] - img[i - 1, j]) / 2.0
	mag = (dx ** 2 + dy ** 2) ** 0.5 + 1e-9
	return mag, dx, dy

def hardie_filters():
	sizes = [7, 10, 13]
	energy = [1.0, 0.47, 0.41]
	k = sizes[2] * 2 + 1
	filters = []

	for idx in range(3):
		filter = np.empty((k, k), dtype=np.float64)
		for i in range(k):
			for j in range(k):
				if ((i - k/2) * (i - k/2) + (j - k/2) * (j - k/2) <= sizes[idx] * sizes[idx]):
					filter[i, j] = 1
				else:
					filter[i, j] = 0

		filters.append(filter);	
		_sum = np.sum(filter);
		filter /= _sum
		filter *= energy[idx];
	
	filters[1] += filters[0];
	filters[2] += filters[1];
	return filters

def wci(img, filter):
	size = filter.shape
	magnitude, dx, dy = finite_derivatives(img)
	
	fx = np.empty(size, dtype=np.float64)
	fy = np.empty(size, dtype=np.float64)
	ax = np.empty(size, dtype=np.float64)
	ay = np.empty(size, dtype=np.float64)

	for i in range(size[0]):
		for j in range(size[1]):
			x = -1 * (i - size[0] / 2)
			y = -1 * (j - size[1] / 2)
			mu = sqrt(x * x + y * y) + 1e-9;	
			fx[i, j] = filter[i, j] * x * 1.0 / mu
			fy[i, j] = filter[i, j] * y * 1.0 / mu

	nx = dx / magnitude
	ny = dy / magnitude

	ax = cv2.filter2D(nx, -1, fx)
	ay = cv2.filter2D(ny, -1, fy)
	return ax + ay

def wmci(img, mask):
	filters = hardie_filters()
	min_distance = 7

	ans = wci(img, filters[0])
	for i in range(1, len(filters)):
		tmp = wci(img, filters[i])
		ans = np.maximum(tmp, ans)

	coords = peak_local_max(ans, min_distance)
	blobs = []
	for coord in coords:
		blobs.append((coord[0], coord[1], 25))

	blobs = filter_by_margin(filter_by_size(filter_by_masks(blobs, mask)), mask)
	#show_blobs("wci", ans, blobs)

	return blobs, ans


