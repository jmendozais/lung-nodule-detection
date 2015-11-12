def classify(features):
	return 0

def classify_blobs(features_set):
	Y = []

	for features in features_set:
		Y.append(classify(features))	

	return np.array(Y)
