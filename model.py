import cv2
import sys
import skimage.io as io
from sklearn.lda import LDA
from sklearn import svm
from sklearn import preprocessing
from sklearn.externals import joblib
from sklearn.feature_selection import RFE
from time import *
from os import path

#from __future__ import print_function as print_f

from keras.models import Sequential
from keras.models import model_from_json
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from augment import ImageDataGenerator
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from six.moves import range

import classify
from detect import *
from extract import * 
from preprocess import *
from segment import *
from util import *

import jsrt

#TODO: up classifier
'''
def input(img_path, ll_path, lr_path):
	img = np.load(img_path)
	img = img.astype(np.float)
	ll_mask = cv2.imread(ll_path)
	lr_mask = cv2.imread(lr_path)
	lung_mask = ll_mask + lr_mask
	dsize = (512, 512)
	lung_mask = cv2.resize(lung_mask, dsize, interpolation=cv2.INTER_CUBIC)
	lung_mask = cv2.cvtColor(lung_mask, cv2.COLOR_BGR2GRAY)
 	lung_mask.astype(np.uint8)

 	return img, lung_mask

def detect_blobs(img, lung_mask):
	sampled, lce, norm = preprocess(img, lung_mask)
	blobs, ci = wmci(lce, lung_mask)
	#ci = lce
	#blobs = log_(lce, lung_mask)

	return blobs, norm, lce, ci

def segment(img, blobs):
	blobs, nod_masks = adaptive_distance_thold(img, blobs)

	return blobs, nod_masks

def extract(norm, lce, wmci, lung_mask, blobs, nod_masks):

	return hardie(norm, lce, wmci, lung_mask, blobs, nod_masks)

def extract_feature_set(data):
	feature_set = []
	blob_set = []
	print '[',
	for i in range(len(data)):
		if i % (len(data)/10) == 0:
			print ".",
			sys.stdout.flush()

		img, lung_mask = data.get(i)
		blobs, norm, lce, ci = detect_blobs(img, lung_mask)
		blobs, nod_masks = segment(lce, blobs)
		feats = extract(norm, lce, ci, lung_mask, blobs, nod_masks)
		feature_set.append(feats)
		blob_set.append(blobs)
	print ']'
	return np.array(feature_set), np.array(blob_set)

def _classify(blobs, feature_vectors, thold=0.012):
	clf = joblib.load('data/clf.pkl')
	scaler = joblib.load('data/scaler.pkl')
	selector = joblib.load('data/selector.pkl')

	feature_vectors = scaler.transform(feature_vectors)
	feature_vectors = selector.transform(feature_vectors)
	probs = clf.predict_proba(feature_vectors)
	probs = probs.T[1]

	blobs = np.array(blobs)
	blobs = blobs[probs>thold]

	return blobs, probs[probs>thold]


	Input: images & masks (data provider), blobs
	Returns: classifier, scaler

def deprecated_train(data, blobs):
	X, Y = classify.create_training_set(data, blobs)
	#clf = svm.SVC()
	clf = LDA()
	#scaler = preprocessing.MinMaxScaler()
	scaler = preprocessing.StandardScaler()
	
	clf, scaler = classify.train(X, Y, clf, scaler)

	joblib.dump(clf, 'data/clf.pkl')
	joblib.dump(scaler, 'data/scaler.pkl')

	return clf, scaler

def deprecated_train_with_feature_set(feature_set, pred_blobs, real_blobs):
	X, Y = classify.create_training_set_from_feature_set(feature_set, pred_blobs, real_blobs)

	#scaler = preprocessing.MinMaxScaler()
	scaler = preprocessing.StandardScaler()
	selector = RFE(estimator=LDA(), n_features_to_select=len(X[0]), step=1)
	#clf = svm.SVC()
	clf = LDA()
	clf, scaler, selector = classify.train(X, Y, clf, scaler, selector)

	joblib.dump(clf, 'data/clf.pkl')
	joblib.dump(scaler, 'data/scaler.pkl')
	joblib.dump(selector, 'data/selector.pkl')

	return clf, scaler, selector


	Input: data provider
	Returns: blobs

def predict(data):
	data_blobs = []
	for i in range(len(data)):
		img, lung_mask = data.get(i)
		blobs, norm, lce, ci = detect_blobs(img, lung_mask)
		blobs, nod_masks = segment(lce, blobs)
		feats = extract(norm, lce, ci, lung_mask, blobs, nod_masks)
		blobs, probs = _classify(blobs, feats)

		# candidate cue adjacency rule: 22 mm
		filtered_blobs = []
		for j in range(len(blobs)):
			valid = True
			for k in range(len(blobs)):
				dist2 = (blobs[j][0] - blobs[k][0]) ** 2 + (blobs[j][1] - blobs[k][1]) ** 2
				if dist2 < 988 and probs[j] + EPS < probs[k]:
					valid = False
					break

			if valid:
				filtered_blobs.append(blobs[j])

		#show_blobs("Predict result ...", lce, filtered_blob)
		data_blobs.append(np.array(filtered_blobs))		

	return np.array(data_blobs)

def predict_from_feature_set(feature_set, blob_set, thold=0.012):
	data_blobs = []
	for i in range(len(feature_set)):
		blobs, probs = _classify(blob_set[i], feature_set[i], thold)

		# candidate cue adjacency rule: 22 mm
		filtered_blobs = []
		for j in range(len(blobs)):
			valid = True
			for k in range(len(blobs)):
				dist2 = (blobs[j][0] - blobs[k][0]) ** 2 + (blobs[j][1] - blobs[k][1]) ** 2
				if dist2 < 988 and probs[j] + EPS < probs[k]:
					valid = False
					break

			if valid:
				filtered_blobs.append(blobs[j])

		#show_blobs("Predict result ...", lce, filtered_blob)
		data_blobs.append(np.array(filtered_blobs))		

	return np.array(data_blobs)

# PIPELINES 
def pipeline_features(img_path, blobs, ll_path, lr_path):
	img, lung_mask = input(img_path, ll_path, lr_path)
	_, norm, lce, ci = detect_blobs(img, lung_mask)
	blobs, nod_masks = segment(lce, blobs)
	feats = extract(norm, lce, ci, lung_mask, blobs, nod_masks)

	return lce, feats

def pipeline(img_path, ll_path, lr_path):
	img, lung_mask = input(img_path, ll_path, lr_path)
	blobs = predict(img, lung_mask)

	return img, blobs
'''

def create_rois(data, blob_set, dsize=(32, 32)):
	# Create image set
	img_set = []
	for i in range(len(data)):
		img, lung_mask = data.get(i)
		sampled, lce, norm = preprocess(img, lung_mask)
		img_set.append(lce)

	# Create roi set
	roi_set = []
	for i in range(len(img_set)):
		img = img_set[i]
		rois = []
		for j in range(len(blob_set[i])):
			x, y, r = blob_set[i][j]
			shift = 0 
			side = 2 * shift + 2 * r + 1

			tl = (x - shift - r, y - shift - r)
			ntl = (max(0, tl[0]), max(0, tl[1]))
			br = (x + shift + r + 1, y + shift + r + 1)
			nbr = (min(img.shape[0], br[0]), min(img.shape[1], br[1]))

			roi = img[ntl[0]:nbr[0], ntl[1]:nbr[1]]
			roi = cv2.resize(roi, dsize, interpolation=cv2.INTER_CUBIC)
			rois.append(roi)
		roi_set.append(np.array(rois))
	return np.array(roi_set)

import keras

class PlateauScheduler(keras.callbacks.Callback):
	def __init__(self):
		self.prev_loss = 0
		self.grads = np.full((4,), dtype=np.float32, fill_value=0.0)
		self.grads[0] = -1
	
	def on_batch_begin(self, batch, logs={}):
		self.losses = []

	def on_batch_end(self, batch, logs={}):
		self.losses.append(logs['loss'])

	def on_epoch_end(self, epoch, logs={}):
		loss = np.sum(self.losses)
		if epoch > 0:
			self.grads[epoch % len(self.grads)] = loss - self.prev_loss
		self.prev_loss = loss
		print loss
		print self.grads
		print np.mean(self.grads)
		assert hasattr(self.model.optimizer, 'lr'), \
		    'Optimizer must have a "lr" attribute.'
		lr = self.model.optimizer.lr.get_value()
		if np.mean(self.grads) > (0 - util.EPS) and lr > 1e-4:
			print 'Updating lr {} ...'.format(lr / 10.0)
			self.model.optimizer.lr.set_value(float(lr / 10.0))

class StageScheduler(keras.callbacks.Callback):
	def __init__(self, stages=[], decay=0.1):
		sorted(stages)
		self.stages = stages
		self.idx = 0
		self.decay = decay
	
	def on_epoch_end(self, epoch, logs={}):
		if self.idx < len(self.stages):
			if epoch + 1 == self.stages[self.idx]:
				lr = self.model.optimizer.lr.get_value()
				self.model.optimizer.lr.set_value(float(lr * self.decay))
				self.idx += 1
		print 'lr {}'.format(self.model.optimizer.lr.get_value())
				
#
# Baseline Model
class BaselineModel:
	def __init__(self, name='default'):
		self.name = name

		self.clf = LDA()
		self.scaler = preprocessing.StandardScaler()
		self.selector = None
		self.transform = None
		self.extractor = HardieExtractor()
		self.feature_set = None
		self.keras_model = None

	def load(self, name):
		# Model
		if path.isfile('{}_extractor.pkl'.format(name)):
			self.extractor = joblib.load('{}_extractor.pkl'.format(name))
		if path.isfile('{}_clf.pkl'.format(name)):
			self.clf = joblib.load('{}_clf.pkl'.format(name))
		if path.isfile('{}_scaler.pkl'.format(name)):
			self.scaler = joblib.load('{}_scaler.pkl'.format(name))
		if path.isfile('{}self.keras__selector.pkl'.format(name)):
			self.selector = joblib.load('{}_selector.pkl'.format(name))
		if path.isfile('{}_transform.pkl'.format(name)):
			self.transform = joblib.load('{}_transform.pkl'.format(name))
		if path.isfile('{}_arch.json'.format(name)):
			self.keras_model = model_from_json(open('{}_arch.json'.format(name)).read())
			self.keras_model.load_weights('{}_weights.h5'.format(name))


		# Data
		if path.isfile('{}_fs.npy'.format(name)):
			self.feature_set = np.load('{}_fs.npy'.format(name))

	def save(self, name):
		if self.extractor != None:
			joblib.dump(self.extractor, '{}_extractor.pkl'.format(name))
		if self.clf != None:
			joblib.dump(self.clf, '{}_clf.pkl'.format(name))
		if self.scaler != None:
			joblib.dump(self.scaler, '{}_scaler.pkl'.format(name))
		if self.selector != None:
			joblib.dump(self.selector, '{}_selector.pkl'.format(name))
		if self.transform != None:
			joblib.dump(self.transform, '{}_transform.pkl'.format(name))
		if self.keras_model != None:
			json_string = self.keras_model.to_json()
			open('{}_arch.json'.format(name), 'w').write(json_string)
			self.keras_model.save_weights('{}_weights.h5'.format(name), overwrite=True)

		if self.feature_set != None:
			np.save(self.feature_set, '{}_fs.npy'.format(name))

	def detect_blobs(self, img, lung_mask, threshold=0.5):
		sampled, lce, norm = preprocess(img, lung_mask)
		blobs, ci = wmci(lce, lung_mask, threshold)

		return blobs, norm, lce, ci

	def detect_blobs_proba(self, img, lung_mask, threshold=0.5):
		sampled, lce, norm = preprocess(img, lung_mask)
		blobs, ci, proba = wmci_proba(lce, lung_mask, threshold)

		return blobs, norm, lce, ci, proba

	def segment(self, img, blobs):
		blobs, nod_masks = adaptive_distance_thold(img, blobs)

		return blobs, nod_masks

	def extract(self, norm, lce, wmci, lung_mask, blobs, nod_masks):
		return self.extractor.extract(norm, lce, wmci, lung_mask, blobs, nod_masks)

	def extract_feature_set(self, data):
		feature_set = []
		blob_set = []
		print '[',
		for i in range(len(data)):
			if i % (len(data)/10) == 0:
				print ".",
				sys.stdout.flush()

			img, lung_mask = data.get(i)
			blobs, norm, lce, ci = self.detect_blobs(img, lung_mask)
			blobs, nod_masks = self.segment(lce, blobs)
			feats = self.extract(norm, lce, ci, lung_mask, blobs, nod_masks)
			feature_set.append(feats)
			blob_set.append(blobs)
		print ']'
		return np.array(feature_set), np.array(blob_set)

	def extract_feature_set_proba(self, data, detecting_proba=0.3):
		feature_set = []
		blob_set = []
		proba_set = []
		print '[',
		for i in range(len(data)):
			if i % (len(data)/10) == 0:
				print ".",
				sys.stdout.flush()

			img, lung_mask = data.get(i)
			blobs, norm, lce, ci, proba = self.detect_blobs_proba(img, lung_mask, detecting_proba)
			blobs, nod_masks = self.segment(lce, blobs)
			feats = self.extract(norm, lce, ci, lung_mask, blobs, nod_masks)
			feature_set.append(feats)
			blob_set.append(blobs)
			proba_set.append(proba)
		print ']'
		return np.array(feature_set), np.array(blob_set), proba_set

	def predict_proba_one(self, blobs, feature_vectors):
		feature_vectors = self.scaler.transform(feature_vectors)

		if self.selector != None:
			feature_vectors = self.selector.transform(feature_vectors)

		probs = self.clf.predict_proba(feature_vectors)
		probs = probs.T[1]
		blobs = np.array(blobs)

		return blobs, probs

	def predict_proba_one_keras(self, blobs, rois):
		img_rows, img_cols = rois[0].shape
		print 'img shape',img_rows, img_cols
		X = rois.reshape(rois.shape[0], 1, img_rows, img_cols)
		X = X.astype('float32')

		probs = self.keras_model.predict_proba(X)
		probs = probs.T[1]
		blobs = np.array(blobs)

		return blobs, probs

	def _classify(self, blobs, feature_vectors, thold=0.012):
		blobs, probs = self.predict_proba_one(blobs, feature_vectors)
		blobs = blobs[probs>thold]

		return blobs, probs[probs>thold]

	''' 	
		Input: images & masks (data provider), blobs
		Returns: classifier, scaler

	def train(self, data, blobs):
		X, Y = classify.create_training_set(data, blobs)
		
		self.clf, self.scaler = classify.train(X, Y, self.clf, self.scaler)

		self.save(self.name)

		return self.clf, self.scaler
	'''

	def train_with_feature_set(self, feature_set, pred_blobs, real_blobs):
		X, Y = classify.create_training_set_from_feature_set(feature_set, pred_blobs, real_blobs)
		clf, scaler, selector = classify.train(X, Y, self.clf, self.scaler, self.selector)
		self.save(self.name)

		return clf, scaler, selector

	def fit_mnist_model(self, X_train, Y_train, batch_size=128, nb_epoch=12):
		np.random.seed(1337)  # for reproducibility
		batch_size = 128
		nb_classes = 2#10
		nb_epoch = 50
		# input image dimensions
		img_rows, img_cols = 28, 28
		# number of convolutional filters to use
		nb_filters = 32
		# size of pooling area for max pooling
		nb_pool = 2
		# convolution kernel size
		nb_conv = 3

		#print('X_train shape:', X_train.shape)
		#print(X_train.shape[0], 'train samples')

		_init = 'orthogonal'
		self.keras_model = Sequential()

		self.keras_model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
		                        border_mode='valid',
		                        input_shape=(1, img_rows, img_cols), init=_init))
		self.keras_model.add(Activation('relu'))
		self.keras_model.add(Convolution2D(nb_filters, nb_conv, nb_conv, init=_init))
		self.keras_model.add(Activation('relu'))
		self.keras_model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
		self.keras_model.add(Dropout(0.25))

		self.keras_model.add(Flatten())
		self.keras_model.add(Dense(128))
		self.keras_model.add(Activation('relu'))
		self.keras_model.add(Dropout(0.5))
		self.keras_model.add(Dense(nb_classes))
		self.keras_model.add(Activation('softmax'))

		# TODO: validation
		self.keras_model.compile(loss='categorical_crossentropy', optimizer='adadelta')
		self.keras_model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=1) # , validation_data=(X_test, Y_test))

	def fit_cifar(self, X_train, Y_train):
		np.random.seed(1337)  # for reproducibility
		batch_size = 32
		nb_classes = 2
		nb_epoch = 20
		data_augmentation = True

		# input image dimensions
		img_rows, img_cols = 32, 32
		# the CIFAR10 images are RGB
		img_channels = 1

		#print('X_train shape:', X_train.shape)
		#print(X_train.shape[0], 'train samples')

		_init = 'he_normal'
		model = Sequential()

		model.add(Convolution2D(32, 3, 3, border_mode='same',
					input_shape=(img_channels, img_rows, img_cols), init=_init))
		model.add(Activation('relu'))
		model.add(Convolution2D(32, 3, 3, init=_init))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		model.add(Convolution2D(64, 3, 3, border_mode='same', init=_init))
		model.add(Activation('relu'))
		model.add(Convolution2D(64, 3, 3, init=_init))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		model.add(Flatten())
		model.add(Dense(512))
		model.add(Activation('relu'))
		model.add(Dropout(0.5))
		model.add(Dense(nb_classes))
		model.add(Activation('softmax'))

		# let's train the model using SGD + momentum (how original).
		sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
		model.compile(loss='categorical_crossentropy', optimizer=sgd)

		lr_scheduler = StageScheduler([50, 70])
		if not data_augmentation:
			print('Not using data augmentation or normalization')
			model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, callbacks=[lr_scheduler])
			#score = model.evaluate(X_test, Y_test, batch_size=batch_size)
			#print('Test score:', score)

		else:
			print('Using real time data augmentation')

			# this will do preprocessing and realtime data augmentation
			datagen = ImageDataGenerator(
				featurewise_center=True,  # set input mean to 0 over the dataset
				samplewise_center=False,  # set each sample mean to 0
				featurewise_std_normalization=True,  # divide inputs by std of the dataset
				samplewise_std_normalization=False,  # divide each input by its std
				zca_whitening=False,  # apply ZCA whitening
				rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
				width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
				height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
				horizontal_flip=True,  # randomly flip images
				vertical_flip=False)  # randomly flip images

			# clone positive samples in order to get a balanced dataset and ...
			# compute quantities required for featurewise normalization
			# (std, mean, and principal components if ZCA whitening is applied)
			X_train, Y_train = datagen.fit_transform(X_train, Y_train)
			#datagen.fit(X_train)

			for e in range(nb_epoch):
				print('-'*40)
				print('Epoch', e)
				print('-'*40)
				print('Training...')
				# batch train with realtime data augmentation
				progbar = generic_utils.Progbar(X_train.shape[0])
				for X_batch, Y_batch in datagen.flow(X_train, Y_train):
					loss = model.train_on_batch(X_batch, Y_batch)
					progbar.add(X_batch.shape[0], values=[('train loss', loss[0])])
		self.keras_model = model

	def fit_graham(self, X_train, Y_train):
		np.random.seed(1337)  # for reproducibility
		batch_size = 32
		nb_classes = 2
		nb_epoch = 100
		data_augmentation = False

		# input image dimensions
		img_rows, img_cols = 32, 32
		# the CIFAR10 images are RGB
		img_channels = 1

		#print('X_train shape:', X_train.shape)
		#print(X_train.shape[0], 'train samples')

		_init = 'orthogonal'
		_activation = 'linear' #LeakyReLU(alpha=.333)
		_filters = 32
		_dropout = 0.1

		model = Sequential()

		model.add(Convolution2D(_filters, 3, 3, border_mode='same',
					input_shape=(img_channels, img_rows, img_cols), init=_init))
		#model.add(Activation(_activation))
		model.add(LeakyReLU(alpha=.333))
		model.add(Convolution2D(_filters, 3, 3, init=_init))
		#model.add(Activation(_activation))
		model.add(LeakyReLU(alpha=.333))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(_dropout))

		model.add(Convolution2D(2 * _filters, 3, 3, border_mode='same', init=_init))
		#model.add(Activation(_activation))
		model.add(LeakyReLU(alpha=.333))
		model.add(Convolution2D(2 * _filters, 3, 3, init=_init))
		#model.add(Activation(_activation))
		model.add(LeakyReLU(alpha=.333))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(2 * _dropout))

		model.add(Convolution2D(3 * _filters, 3, 3, border_mode='same', init=_init))
		#model.add(Activation(_activation))
		model.add(LeakyReLU(alpha=.333))
		model.add(Convolution2D(3 * _filters, 3, 3, init=_init))
		#model.add(Activation(_activation))
		model.add(LeakyReLU(alpha=.333))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(3 * _dropout))

		model.add(Flatten())
		model.add(Dense(512, init=_init))
		model.add(Activation('relu'))
		model.add(Dropout(0.5))
		model.add(Dense(nb_classes))
		model.add(Activation('softmax'))

		# let's train the model using SGD + momentum (how original).
		sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
		model.compile(loss='categorical_crossentropy', optimizer=sgd)

		if not data_augmentation:
			print('Not using data augmentation or normalization')
			model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch)

		self.keras_model = model
	
	def train_with_feature_set_keras(self, feature_set, pred_blobs, real_blobs):
		X_train, y_train = classify.create_training_set_from_feature_set(feature_set, pred_blobs, real_blobs)
		
		img_rows, img_cols = feature_set[0][0].shape
		nb_classes = 2
		X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
		X_train = X_train.astype('float32')
		# convert class vectors to binary class matrices
		Y_train = np_utils.to_categorical(y_train, nb_classes)
		
		self.fit_cifar(X_train, Y_train)

		self.save(self.name)
		return self.keras_model

	def predict_proba(self, data):
		self.load(self.name)

		data_blobs = []
		data_probs = []
		for i in range(len(data)):
			img, lung_mask = data.get(i)
			blobs, norm, lce, ci = self.detect_blobs(img, lung_mask)
			blobs, nod_masks = self.segment(lce, blobs)
			feats = self.extract(norm, lce, ci, lung_mask, blobs, nod_masks)
			blobs, probs = self.predict_proba_one(blobs, feats)

			# candidate cue adjacency rule: 22 mm
			filtered_blobs = []
			filtered_probs = []
			for j in range(len(blobs)):
				valid = True
				for k in range(len(blobs)):
					dist2 = (blobs[j][0] - blobs[k][0]) ** 2 + (blobs[j][1] - blobs[k][1]) ** 2
					if dist2 < 988 and probs[j] + EPS < probs[k]:
						valid = False
						break

				if valid:
					filtered_blobs.append(blobs[j])
					filtered_probs.append(probs[j])

			#show_blobs("Predict result ...", lce, filtered_blob)
			data_blobs.append(np.array(filtered_blobs))	
			data_probs.append(np.array(filtered_probs))

		return np.array(data_blobs), np.array(data_probs)

	'''
		Input: data provider
		Returns: blobs
	'''
	def predict(self, data):
		data_blobs = []
		for i in range(len(data)):
			img, lung_mask = data.get(i)
			blobs, norm, lce, ci = self.detect_blobs(img, lung_mask)
			blobs, nod_masks = self.segment(lce, blobs)
			feats = self.extract(norm, lce, ci, lung_mask, blobs, nod_masks)
			blobs, probs = self._classify(blobs, feats)

			# candidate cue adjacency rule: 22 mm
			filtered_blobs = []
			for j in range(len(blobs)):
				valid = True
				for k in range(len(blobs)):
					dist2 = (blobs[j][0] - blobs[k][0]) ** 2 + (blobs[j][1] - blobs[k][1]) ** 2
					if dist2 < 988 and probs[j] + EPS < probs[k]:
						valid = False
						break

				if valid:
					filtered_blobs.append(blobs[j])

			#show_blobs("Predict result ...", lce, filtered_blob)
			data_blobs.append(np.array(filtered_blobs))		

		return np.array(data_blobs)

	def predict_from_feature_set(self, feature_set, blob_set, thold=0.012):
		data_blobs = []
		for i in range(len(feature_set)):
			blobs, probs = self._classify(blob_set[i], feature_set[i], thold)

			# candidate cue adjacency rule: 22 mm
			filtered_blobs = []
			for j in range(len(blobs)):
				valid = True
				for k in range(len(blobs)):
					dist2 = (blobs[j][0] - blobs[k][0]) ** 2 + (blobs[j][1] - blobs[k][1]) ** 2
					if dist2 < 988 and probs[j] + EPS < probs[k]:
						valid = False
						break

				if valid:
					filtered_blobs.append(blobs[j])

			#show_blobs("Predict result ...", lce, filtered_blob)
			data_blobs.append(np.array(filtered_blobs))		

		return np.array(data_blobs)

	def predict_proba_from_feature_set(self, feature_set, blob_set):
		self.load(self.name)
		
		data_blobs = []
		data_probs = []
		for i in range(len(feature_set)):
			blobs, probs = self.predict_proba_one(blob_set[i], feature_set[i])

			## candidate cue adjacency rule: 22 mm
			filtered_blobs = []
			filtered_probs = []
			for j in range(len(blobs)):
				valid = True
				for k in range(len(blobs)):
					dist2 = (blobs[j][0] - blobs[k][0]) ** 2 + (blobs[j][1] - blobs[k][1]) ** 2
					if dist2 < 988 and probs[j] + EPS < probs[k]:
						valid = False
						break

				if valid:
					filtered_blobs.append(blobs[j])
					filtered_probs.append(probs[j])

			#show_blobs("Predict result ...", lce, filtered_blob)
			data_blobs.append(np.array(filtered_blobs))	
			data_probs.append(np.array(filtered_probs))

		return np.array(data_blobs), np.array(data_probs)

	def predict_proba_from_feature_set_keras(self, feature_set, blob_set):
		self.load(self.name)
		
		data_blobs = []
		data_probs = []
		for i in range(len(feature_set)):
			blobs, probs = self.predict_proba_one_keras(blob_set[i], feature_set[i])

			## candidate cue adjacency rule: 22 mm
			filtered_blobs = []
			filtered_probs = []
			for j in range(len(blobs)):
				valid = True
				for k in range(len(blobs)):
					dist2 = (blobs[j][0] - blobs[k][0]) ** 2 + (blobs[j][1] - blobs[k][1]) ** 2
					if dist2 < 988 and probs[j] + EPS < probs[k]:
						valid = False
						break

				if valid:
					filtered_blobs.append(blobs[j])
					filtered_probs.append(probs[j])

			#show_blobs("Predict result ...", lce, filtered_blob)
			data_blobs.append(np.array(filtered_blobs))	
			data_probs.append(np.array(filtered_probs))

		return np.array(data_blobs), np.array(data_probs)

	def filter_by_proba(self, blob_set, prob_set, thold = 0.012):
		data_blobs = []
		data_probs = []
		for i in range(len(blob_set)):
			blobs = blob_set[i]
			probs = prob_set[i]

			## candidate cue adjacency rule: 22 mm
			filtered_blobs = []
			filtered_probs = []
			for j in range(len(blobs)):
				if probs[j] > thold:
					filtered_blobs.append(blobs[j])
					filtered_probs.append(probs[j])

			#show_blobs("Predict result ...", lce, filtered_blob)
			data_blobs.append(np.array(filtered_blobs))	
			data_probs.append(np.array(filtered_probs))

		return np.array(data_blobs), np.array(data_probs)





