import sys
from random import *
from math import *
from time import *

import numpy as np
import cv2

import jsrt
from util import *
from lnd import *

if __name__ == '__main__':
	out_file = sys.argv[1]
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
