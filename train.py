import sys
import numpy as np
from sklearn.lda import LDA
from sklearn import svm
from sklearn.metrics import classification_report
import sklearn.cross_validation as cross_val
from sklearn.externals import joblib
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix


def train(X, Y, clf, scaler):
	iters = 1
	tr_prop = 0.7
	te_prop = 1 - tr_prop
	seed = 113

	# hardcoded
	NX = []
	for xi in X:
		NX.append(np.array(xi))
	X = np.array(NX)

	folds = np.array(list(cross_val.StratifiedShuffleSplit(Y, iters, train_size=tr_prop, test_size=te_prop, random_state=seed)))

	tr = folds[0][0]
	te = folds[0][1]


	scaler.fit(X[tr])
	clf.fit(scaler.transform(X[tr]), Y[tr])

	# pred
	pred = clf.predict(scaler.transform(X[te]))

	print "Evaluate performance on patches: "
	print "classification report: "
	print classification_report(Y[te].astype(int), pred.astype(int))
	print "confusion matrix: "
	print confusion_matrix(Y[te].astype(int), pred.astype(int))
	
	return clf, scaler

if __name__ == "__main__":
	fname = sys.argv[1]
	data = np.load(fname)
	print "fname {}".format(fname)

	X = data[0]
	Y = data[1]
	clf = LDA()
	scaler = preprocessing.StandardScaler()
	train(X, Y, clf, scaler)

