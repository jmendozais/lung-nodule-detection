import sys
import numpy as np
from sklearn.lda import LDA
from sklearn import svm
from sklearn.metrics import classification_report
import sklearn.cross_validation as cross_val
from sklearn.externals import joblib
from sklearn import preprocessing


if __name__ == "__main__":
	fname = sys.argv[1]
	print "fname {}".format(fname)
	iters = 1
	tr_prop = 0.7
	te_prop = 1 - tr_prop
	seed = 113

	data = np.load(fname)
	X = data[0]


	Y = data[1]

	# hardcoded
	NX = []
	for xi in X:
		NX.append(np.array(xi))
	X = np.array(NX)

	folds = np.array(list(cross_val.StratifiedShuffleSplit(Y, iters, train_size=tr_prop, test_size=te_prop, random_state=seed)))

	tr = folds[0][0]
	te = folds[0][1]

	#clf = svm.SVC()
	clf = LDA()

	scaler = preprocessing.StandardScaler().fit(X[tr])
	#scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(X[tr])
	clf.fit(scaler.transform(X[tr]), Y[tr])

	# pred
	pred = clf.predict(scaler.transform(X[te]))
	print "acc(uniform): " + str(np.mean(pred == Y[te]))
	print classification_report(pred.astype(int), Y[te].astype(int))

	# check sets
	np.savetxt("data/xtr", X[tr])
	np.savetxt("data/xte", X[te])
	np.savetxt("data/ytr", Y[tr])
	np.savetxt("data/yte", Y[te])
	
	# save classifier
	joblib.dump(clf, 'data/clf.pkl') 
	joblib.dump(scaler, 'data/scaler.pkl')
