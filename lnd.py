import numpy as np
from sklearn.cross_validation import StratifiedKFold

from data import DataProvider
import model
import eval
import util

import jsrt

def run_individual(): 
	model.pipeline(sys.argv[1], sys.argv[2], sys.argv[3])

def run_on_dataset():
	paths, locs, rads = jsrt.jsrt(set='jsrt140')
	left_masks = jsrt.left_lung(set='jsrt140')
	right_masks = jsrt.right_lung(set='jsrt140')

	blobs_ = []
	for i in range(len(paths)):
		img, blobs = model.pipeline(paths[i], left_masks[i], right_masks[i])
		#img, blobs = model.pipeline_features(paths[i], [(locs[i][0], locs[i][1], 25)], left_masks[i], right_masks[i])
		blobs_.append(blobs)
		print_detection(paths[i], blobs)
		sys.stdout.flush()

def protocol():
	paths, locs, rads = jsrt.jsrt(set='jsrt140')
	left_masks = jsrt.left_lung(set='jsrt140')
	right_masks = jsrt.right_lung(set='jsrt140')
	size = len(paths)

	blobs = []
	imgs = []
	for i in range(size):
		blobs.append([locs[i][0], locs[i][1], rads[i]])
	blobs = np.array(blobs)

	Y = (140 > np.array(range(size))).astype(np.uint8)
	skf = StratifiedKFold(Y, n_folds=10, shuffle=True, random_state=113)
	
	fold = 0

	sens = []
	fppi_mean = []
	fppi_std = []
	for tr_idx, te_idx in skf:
		fold += 1
		print "Fold {}".format(fold)

		xtr = DataProvider(paths[tr_idx], left_masks[tr_idx], right_masks[tr_idx])
		model.train(xtr, blobs[tr_idx])

		xte = DataProvider(paths[te_idx], left_masks[te_idx], right_masks[te_idx])
		blobs_te_pred = model.predict(xte)

		paths_te = paths[te_idx]
		for i in range(len(blobs_te_pred)):
			util.print_detection(paths_te[i], blobs_te_pred[i])

		blobs_te = []
		for bl in blobs[te_idx]:
			blobs_te.append([bl])
		blobs_te = np.array(blobs_te)

		s, fm, fs = eval.evaluate(blobs_te, blobs_te_pred, paths[te_idx])
		print "Result: sens {}, fppi mean {}, fppi std {}".format(s, fm, fs)

		sens.append(s)
		fppi_mean.append(fm)
		fppi_std.append(fs)

	sens = np.array(sens)
	fppi_mean = np.array(fppi_mean)
	fppi_std = np.array(fppi_std)

	print "Final: sens_mean {}, sens_std {}, fppi_mean {}, fppi_stds_mean {}".format(sens.mean(), sens.std(), fppi_mean.mean(), fppi_std.mean())


if __name__=="__main__":
	protocol()