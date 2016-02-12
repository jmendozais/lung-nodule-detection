import cv2
import sys

import skimage.io as io
from sklearn import lda
from sklearn import svm
from sklearn import decomposition
from sklearn import preprocessing
from sklearn import feature_selection as selection
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
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from six.moves import range

import classify
from detect import *
from extract import * 
from preprocess import *
from segment import *
from augment import *
from util import *

import jsrt

#TODO: up classifier

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

def create_rois(data, blob_set, dsize=(32, 32), mode=None):
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
		masks = []
		if mode == 'mask':
			_, masks = adaptive_distance_thold(img, blob_set[i])
		
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

			if mode == 'mask':
				mask = cv2.resize(masks[j], dsize, interpolation=cv2.INTER_CUBIC)
				roi *= mask.astype(np.float64)

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
				
# Baseline Model
class BaselineModel:
	def __init__(self, name='default'):
		self.name = name

		self.clf = lda.LDA()
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

	def fit_mnist(self, X_train, Y_train, batch_size=128, nb_epoch=12):
		np.random.seed(1337)  # for reproducibility
		batch_size = 128
		nb_classes = 2#10
		nb_epoch = 12
		# input image dimensions
		img_rows, img_cols = 28, 28
		# number of convolutional filters to use
		nb_filters = 32
		# size of pooling area for max pooling
		nb_pool = 2
		# convolution kernel size
		nb_conv = 3
		# augment
		data_augmentation = True

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

		#self.keras_model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=1) # , validation_data=(X_test, Y_test))

		lr_scheduler = StageScheduler([50, 70])
		if not data_augmentation:
			print('Not using data augmentation or normalization')
			self.keras_model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, callbacks=[lr_scheduler])
			#score = model.evaluate(X_test, Y_test, batch_size=batch_size)
			#print('Test score:', score)

		else:
			print('Using data augmentation')
			X_train, Y_train = offline_augment(X_train, Y_train, ratio=1, rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)

			self.keras_model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, callbacks=[lr_scheduler])


	def fit_cifar(self, X_train, Y_train):
		np.random.seed(1337)  # for reproducibility
		batch_size = 32
		nb_classes = 2
		nb_epoch = 60
		data_augmentation = True

		# input image dimensions
		img_rows, img_cols = X_train[0][0].shape
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

		lr_scheduler = StageScheduler([30, 50])
		if not data_augmentation:
			print('Not using data augmentation or normalization')
			model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, callbacks=[lr_scheduler])
			#score = model.evaluate(X_test, Y_test, batch_size=batch_size)
			#print('Test score
		else:
			print('Using data augmentation')
			X_train, Y_train = offline_augment(X_train, Y_train, ratio=1, rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
			#X_train, Y_train = bootstraping_augment(X_train, Y_train, ratio=1, batch_size=batch_size, rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
			print 'Negatives: {}'.format(np.sum(Y_train.T[0]))
			print 'Positives: {}'.format(np.sum(Y_train.T[1]))
			model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, callbacks=[lr_scheduler])

		self.keras_model = model

	def fit_vgg(self, X_train, Y_train):
		np.random.seed(1337)  # for reproducibility
		batch_size = 32
		nb_classes = 2
		nb_epoch = 40
		data_augmentation = True

		# input image dimensions
		img_rows, img_cols = 32, 32
		# the CIFAR10 images are RGB
		img_channels = 1

		#print('X_train shape:', X_train.shape)
		#print(X_train.shape[0], 'train samples')

		_init = 'orthogonal'
		_activation = 'linear' #LeakyReLU(alpha=.333)
		_filters = 64
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

		lr_scheduler = StageScheduler([20, 30])
		if not data_augmentation:
			print('Not using data augmentation or normalization')
			model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, callbacks=[lr_scheduler])
			#score = model.evaluate(X_test, Y_test, batch_size=batch_size)
			#print('Test score
		else:
			print('Using data augmentation')
			X_train, Y_train = offline_augment(X_train, Y_train, ratio=1, 
						rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)

			print 'Negatives: {}'.format(np.sum(Y_train.T[0]))
			print 'Positives: {}'.format(np.sum(Y_train.T[1]))
			model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, callbacks=[lr_scheduler])

		self.keras_model = model

	def fit_graham(self, X_train, Y_train):
		Y_train = np.array([Y_train.T[1]]).T
		print 'labels shape {}'.format(Y_train)
		np.random.seed(1337)  # for reproducibility
		batch_size = 32
		nb_classes = 2
		nb_epoch = 40
		data_augmentation = True

		# input image dimensions
		img_rows, img_cols = X_train[0][0].shape
		print img_rows, img_cols
		# the CIFAR10 images are RGB
		img_channels = 1

		#print('X_train shape:', X_train.shape)
		#print(X_train.shape[0], 'train samples')

		_init = 'orthogonal'
		_activation = 'linear' #LeakyReLU(alpha=.333)
		_filters = 64
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

		model.add(Convolution2D(2 * _filters, 2, 2, border_mode='same', init=_init))
		#model.add(Activation(_activation))
		model.add(LeakyReLU(alpha=.333))
		model.add(Dropout(_dropout))
		model.add(Convolution2D(2 * _filters, 2, 2, init=_init))
		#model.add(Activation(_activation))
		model.add(LeakyReLU(alpha=.333))
		model.add(Dropout(_dropout))
		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Convolution2D(3 * _filters, 2, 2, border_mode='same', init=_init))
		#model.add(Activation(_activation))
		model.add(LeakyReLU(alpha=.333))
		model.add(Dropout(2 * _dropout))
		model.add(Convolution2D(3 * _filters, 2, 2, init=_init))
		#model.add(Activation(_activation))
		model.add(LeakyReLU(alpha=.333))
		model.add(Dropout(2 * _dropout))
		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Convolution2D(4 * _filters, 2, 2, border_mode='same', init=_init))
		#model.add(Activation(_activation))
		model.add(LeakyReLU(alpha=.333))
		model.add(Dropout(3 * _dropout))
		model.add(Convolution2D(4 * _filters, 2, 2, init=_init))
		#model.add(Activation(_activation))
		model.add(LeakyReLU(alpha=.333))
		model.add(Dropout(3 * _dropout))
		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Convolution2D(5 * _filters, 2, 2, border_mode='same', init=_init))
		#model.add(Activation(_activation))
		model.add(LeakyReLU(alpha=.333))
		model.add(Dropout(4 * _dropout))
		model.add(Convolution2D(5 * _filters, 2, 2, init=_init))
		#model.add(Activation(_activation))
		model.add(LeakyReLU(alpha=.333))
		model.add(Dropout(4 * _dropout))
		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Flatten())
		model.add(Dense(512, init=_init))
		model.add(Activation('relu'))
		model.add(Dropout(0.5))
		model.add(Dense(1))
		model.add(Activation('softmax'))

		# let's train the model using SGD + momentum (how original).
		sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
		model.compile(loss='binary_crossentropy', optimizer=sgd)

		lr_scheduler = StageScheduler([15, 30])
		if not data_augmentation:
			print('Not using data augmentation or normalization')
			model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, callbacks=[lr_scheduler])
			#score = model.evaluate(X_test, Y_test, batch_size=batch_size)
			#print('Test score
		else:
			print('Using data augmentation')
			X_train, Y_train = offline_augment(X_train, Y_train, ratio=1, 
						rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)

			model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, callbacks=[lr_scheduler])

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
		DIST2 = 987.755
		
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
					if dist2 < DIST2 and probs[j] + EPS < probs[k]:
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
		DIST2 = 987.755

		data_blobs = []
		data_probs = []
		for i in range(len(feature_set)):
			print 'predict proba one keras'
			print feature_set[i].shape
			blobs, probs = self.predict_proba_one_keras(blob_set[i], feature_set[i])

			## candidate cue adjacency rule: 22 mm
			filtered_blobs = []
			filtered_probs = []
			for j in range(len(blobs)):
				valid = True
				for k in range(len(blobs)):
					dist2 = (blobs[j][0] - blobs[k][0]) ** 2 + (blobs[j][1] - blobs[k][1]) ** 2
					if dist2 < DIST2 and probs[j] + EPS < probs[k]:
						valid = False
						break

				if valid:
					filtered_blobs.append(blobs[j])
					filtered_probs.append(probs[j])

			#show_blobs("Predict result ...", lce, filtered_blob)
			data_blobs.append(np.array(filtered_blobs))	
			data_probs.append(np.array(filtered_probs))

		return np.array(data_blobs), np.array(data_probs)

	# Join filter and eval on the same function and vectorize
	def filter_by_proba(self, blob_set, prob_set, thold = 0.012):
		data_blobs = []
		data_probs = []
		for i in range(len(blob_set)):
			probs = prob_set[i]
			filtered_blobs = blob_set[i][probs > thold]
			filtered_probs = prob_set[i][probs > thold]
			'''
			probs = prob_set[i]
			blobs = blob_set[i]
			filtered_blobs = []
			filtered_probs = []
			for j in range(len(blobs)):
				if probs[j] > thold:
					filtered_blobs.append(blobs[j])
					filtered_probs.append(probs[j])
			'''

			#show_blobs("Predict result ...", lce, filtered_blob)
			data_blobs.append(np.array(filtered_blobs))	
			data_probs.append(np.array(filtered_probs))

		return np.array(data_blobs), np.array(data_probs)

# optimized
opt_classifiers = {'svm':svm.SVC(probability=True, C=0.0373, gamma=0.002), 'lda':lda.LDA()}
# default
#classifiers = {'svm':svm.SVC(probability=True, C=0.44668359215096315, gamma=0.0005623413251903491, max_iter=1000), 'lda':lda.LDA()}
classifiers = {'svm':svm.SVC(kernel='rbf', probability=True), 'lda':lda.LDA()}
#classifiers = {'svm':svm.SVC(probability=True, C=0.0373, gamma=0.002), 'lda':lda.LDA()}
#classifiers = {'svm':svm.SVC(probability=True), 'lda':lda.LDA()}
reductors = {'none':None, 'pca':decomposition.PCA(n_components=0.99999999999, whiten=True), 'lda':selection.SelectFromModel(lda.LDA())}

