import numpy as np
from itertools import *
def circunference(img, blobs):
	masks = []
	for blob in blobs:
		x, y, r = blob
		mask = np.zeros((2*r + 1, 2*r + 1), np.uint8)
		for i, j in product(range(2*r + 1), range(2*r + 1)):
			if r  ** 2 > (r - i) ** 2  + (r - j) ** 2:
				mask[i][j] = 1
		masks.append(mask)
		
	return blobs, np.array(masks)


# ADT by ARG segmentation
def cos_angle(a, b):
	dot = 1.0 * (a[0] * b[0] + a[1] * b[1])
	len_a = sqrt(a[0] * a[0] + a[1] * a[1])
	len_b = sqrt(b[0] * b[0] + b[1] * b[1])

	if len_a * len_b == 0:
		return 0

	tmp = dot/(len_a * len_b)
	if tmp > 1:
		tmp = 1
	elif tmp < -1:
		tmp = -1
	
	# cos(acos(tmp)
	return tmp

def distance_thold(img, point, grad, lung_mask, t0=0, nod_mask=None):
	point = (int(point[0]), int(point[1]))
	size = img.shape
	rmax = 25
	diam = 2 * rmax + 1
	tdelta = 1.7
	mask = np.full((diam, diam), 0)

	cors = [[1e10, 1e10], [-1e10, -1e10]]
	# Segment and calculate the ARG of t0

	area = 1
	arg = 0.0
	for i in range(diam):
		for j in range(diam):
			x = point[0] - rmax + i
			y = point[1] - rmax + j
			if x < 0 or y < 0 or x >= size[0] or y >= size[1]:
				continue

			dist = (i - rmax) * (i - rmax) + (j - rmax) * (j - rmax)
			tvalue = 1e10
			if dist < (rmax * rmax):
				tvalue = t0 + tdelta * (1 - exp(- dist * 1.0 / (rmax * rmax))) / (1 - exp(-1))

			#print("{} {} -> tvalue {}, lce value {}, dist {}, dist ratio {}".format(i - rmax, j - rmax, tvalue, img[point[0] - rmax + i, point[1] - rmax + j], dist, dist / (rmax * rmax)))	
			if img[x, y] > tvalue:
				mask[i, j] = 1


	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
	mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
	
	for i in range(diam):
		for j in range(diam):
			x = point[0] - rmax + i
			y = point[1] - rmax + j
			if x < 0 or y < 0 or x >= size[0] or y >= size[1]:
				continue

			if mask[i, j] == 1:
				cors[0][0] = min(cors[0][0], point[0] - rmax + i)
				cors[0][1] = min(cors[0][1], point[1] - rmax + j)
				cors[1][0] = max(cors[1][0], point[0] - rmax + i)
				cors[1][1] = max(cors[1][1], point[1] - rmax + j)

				if nod_mask != None:
					nod_mask[x, y] = 255
					mask[i, j] =  255

				area += 1
				arg += grad[x, y] * ( - cos_angle((rmax - i, rmax - j), (dx[x, y], dy[x, y])) )
	arg /= area
	'''
	if nod_mask != None:
		a, b, _, _ = cv2.minMaxLoc(mask)
		mask = (mask - a) / (b - a) 
		cv2.imshow('mask', mask)
		cv2.waitKey()
	'''
	return arg, cors[0], cors[1]

def adaptive_thold(img, point, dx, dy, grad, lung_mask, nod_mask):
	coorners = [[1e10, 1e10], [-1e10, -1e10]]
	best_arg = -1e10
	best_t = 0
	for t in np.arange(0.45,-0.25,-0.025):
		arg, _, _ = distance_thold(img, point, grad, lung_mask, t)
		#print("{} {}".format(t, arg))
		if best_arg < arg:
			best_arg = arg
			best_t = t
	#print("best {} {}".format(best_t, best_arg))
	_, tl, br = distance_thold(img, point, grad, lung_mask, best_t, nod_mask=nod_mask)

	return tl, br