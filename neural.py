import numpy as np
import theano

import keras
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers.core import Dense, Dropout, Activation, Flatten, MaxoutDense
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from six.moves import range

from augment import ImageDataGenerator, Preprocessor
from sklearn.externals import joblib

# Utils 

def get_activations(model, layer, X_batch):
    get_activations = theano.function([model.layers[0].input], model.layers[layer].get_output(train=False), allow_input_downcast=True)
    activations = get_activations(X_batch) # same result as above
    return activations

# Feature blocks 
def x_conv_pool(model, depth=2, nb_filters=64, nb_conv=3, nb_pool=2, init='orthogonal', input_shape=None, activation='relu', batch_norm=False, **junk):
    for i in range(depth):
        if input_shape != None:
            model.add(Convolution2D(nb_filters, nb_conv, nb_conv, border_mode='same', init=init, input_shape=input_shape))
        else:
            model.add(Convolution2D(nb_filters, nb_conv, nb_conv, border_mode='same', init=init ))

        if batch_norm:
            model.add(BatchNormalization())

        if activation == 'leaky_relu':
            model.add(LeakyReLU(alpha=.333))
        else:
            model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

def convpool_fs(model, nb_modules=1, module_depth=2,  nb_filters=[64], conv_size=[3], input_shape=(3, 64, 64),  init='orthogonal', batch_norm=False, activation='relu'):
    assert nb_modules == len(nb_filters)

    for i in range(nb_modules):
        if i == 0:
            x_conv_pool(model, module_depth, nb_filters[i], nb_conv=conv_size[i], input_shape=input_shape, init=init, activation=activation)
        else:
            x_conv_pool(model, module_depth, nb_filters[i], nb_conv=conv_size[i], activation=activation)

        if batch_norm is False:
            model.add(Dropout(0.25))

    return model

def convpool_fs_ldp(model, nb_modules=1, module_depth=2,  nb_filters=[64], conv_size=[3], input_shape=(3, 64, 64),  init='orthogonal', batch_norm=False, activation='relu', nb_dp=5, dp_init=0.15, dp_inc=0.05):
    assert nb_modules == len(nb_filters)
    
    for i in range(nb_modules):
        if i == 0:
            x_conv_pool(model, module_depth, nb_filters[i], nb_conv=conv_size[i], input_shape=input_shape, init=init, activation=activation)
        else:
            x_conv_pool(model, module_depth, nb_filters[i], nb_conv=conv_size[i], activation=activation)

        if batch_norm is False:
            if i >= nb_modules - nb_dp:
                model.add(Dropout(dp_init + (i + nb_dp - nb_modules) * dp_inc))

    return model


def allcnn_module(model, depth=2, nb_filters=64, nb_conv=3, subsample=2, init='orthogonal', input_shape=None, activation='relu', batch_norm=False, **junk):
    for i in range(depth):
        if input_shape != None:
            model.add(Convolution2D(nb_filters, nb_conv, nb_conv, border_mode='same', init=init, input_shape=input_shape))
        else:
            model.add(Convolution2D(nb_filters, nb_conv, nb_conv, border_mode='same', init=init ))

        if batch_norm:
            model.add(BatchNormalization())

        if activation == 'leaky_relu':
            model.add(LeakyReLU(alpha=.333))
        else:
            model.add(Activation(activation))

    model.add(Convolution2D(nb_filters, subsample, subsample, border_mode='valid', init=init, subsample=(subsample, subsample)))
    if activation == 'leaky_relu':
        model.add(LeakyReLU(alpha=.333))
    else:
        model.add(Activation(activation))

def allcnn_fs(model, nb_modules=1, module_depth=2,  nb_filters=[64], conv_size=[3], input_shape=(3, 64, 64),  init='orthogonal', batch_norm=False, activation='relu'):
    assert nb_modules == len(nb_filters)
    for i in range(nb_modules):
        if i == 0:
            allcnn_module(model, module_depth, nb_filters[i], nb_conv=conv_size[i], input_shape=input_shape, init=init, activation=activation, subsample=2)
        else:
            allcnn_module(model, module_depth, nb_filters[i], nb_conv=conv_size[i], activation=activation, subsample=2)

        if batch_norm is False:
            model.add(Dropout(0.25))

    return model

# Classification blocks

def mlp_softmax(model, nb_dense=1, dense_units=[512], nb_classes=2, init='orthogonal', batch_norm=False, activation='relu'):
    for i in range(nb_dense):
        model.add(Dense(dense_units[i], init=init))
        if activation == 'leaky_relu':
            model.add(LeakyReLU(alpha=.333))
        else:
            model.add(Activation(activation))
        model.add(Dropout(0.5))

    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

def nin_softmax(model, nb_dense=1, dense_units=[512], nb_classes=2, init='orthogonal', batch_norm=False, activation='relu'):
    for i in range(nb_dense):
        model.add(Dense(dense_units[i], init=init))
        if activation == 'leaky_relu':
            model.add(LeakyReLU(alpha=.333))
        else:
            model.add(Activation(activation))
        model.add(Dropout(0.5))

    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

def maxout_softmax(model, nb_dense=1, dense_units=[512], nb_classes=2, init='orthogonal', batch_norm=False, activation='relu', nb_feature=4):

    for i in range(nb_dense):
        model.add(MaxoutDense(dense_units[i], init=init, nb_feature=nb_feature))
        model.add(Activation('linear'))
        model.add(Dropout(0.5))

    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

# Models 

def standard_cnn(model, nb_modules=1, module_depth=2,  nb_filters=[64], conv_size=[3], input_shape=(3, 64, 64), nb_dense=1, dense_units=[512], nb_classes=2, init='orthogonal', batch_norm=False, activation='relu'):
    convpool_fs(model, nb_modules, module_depth,  nb_filters, conv_size, input_shape,  init, batch_norm, activation)
    model.add(Flatten())
    mlp_softmax(model, nb_dense, dense_units, nb_classes, init, batch_norm, activation)
    return model

def standard_cnn_ldp(model, nb_modules=1, module_depth=2,  nb_filters=[64], conv_size=[3], input_shape=(3, 64, 64), nb_dense=1, dense_units=[512], nb_classes=2, init='orthogonal', batch_norm=False, activation='relu', nb_dp=5, dp_init=0.15, dp_inc=0.05):
    convpool_fs_ldp(model, nb_modules, module_depth,  nb_filters, conv_size, input_shape,  init, batch_norm, activation, nb_dp=nb_dp, dp_init=dp_init, dp_inc=dp_inc)
    model.add(Flatten())
    mlp_softmax(model, nb_dense, dense_units, nb_classes, init, batch_norm, activation)
    return model

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

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
    
import pickle
def fit_model(model, X_train, Y_train, X_test=None, Y_test=None, batch_size=64, lr=0.01, schedule=[50], momentum=0.9, nesterov=True, nb_epoch=60, augment='bt'):

    sgd = SGD(lr=lr, decay=1e-6, momentum=momentum, nesterov=nesterov)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)

    lr_scheduler = StageScheduler(schedule)

    print 'Augment data: {}'.format(augment)
    if augment == 'rd':
        X_train, Y_train = offline_augment(X_train, Y_train, ratio=1, rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
    elif augment == 'bt':
        X_train, Y_train = bootstraping_augment(X_train, Y_train, ratio=1, batch_size=batch_size, rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
    elif augment == 'btzca':
        X_train, Y_train = bootstraping_augment(X_train, Y_train, ratio=1, batch_size=batch_size, rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True, zca_whitening=True)

    print 'Negatives: {}'.format(np.sum(Y_train.T[0]))
    print 'Positives: {}'.format(np.sum(Y_train.T[1]))

    loss_bw_history = LossHistory()
    history = None
    if X_test is None:
        print 'train & val 0.1'
        history = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, callbacks=[lr_scheduler, loss_bw_history], shuffle=False, validation_split=0.1, show_accuracy=True)
    else:
        print 'train & test'
        history = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, callbacks=[lr_scheduler, loss_bw_history], shuffle=False, validation_data=(X_test, Y_test), show_accuracy=True)

    history.history['loss_detail'] = loss_bw_history.losses
    return history.history

def shallow_1(input_shape):
    model = Sequential()
    return standard_cnn(model, nb_modules=1, nb_filters=[64], nb_dense=1, dense_units=[256], input_shape=input_shape)

def shallow_3(input_shape):
    model = Sequential()
    return standard_cnn(model, nb_modules=3, nb_filters=[64, 128, 192], nb_dense=1, dense_units=[512], input_shape=input_shape)

def shallow(input_shape):
    model = Sequential()
    return standard_cnn(model, module_depth=1, nb_modules=2, nb_filters=[32, 64], nb_dense=1, dense_units=[256], 
            input_shape=input_shape, init='he_normal')

def shallow_bn(input_shape):
    model = Sequential()
    return standard_cnn(model, module_depth=1, nb_modules=3, nb_filters=[32, 64, 96], nb_dense=1, dense_units=[256], input_shape=input_shape, batch_norm=True)


def fit(X_train, Y_train, X_val=None, Y_val=None, model='shallow_1'):
    keras_model = None
    input_shape = X_train[0].shape

    print 'Fit model: {}'.format(model)
    print 'X-train shape: {}'.format(X_train.shape)

    hist = None
    if model == 'shallow_1':
        keras_model = shallow_1(input_shape)
        schedule=[20,40]
        hist = fit_model(keras_model, X_train, Y_train, X_val, Y_val, schedule=schedule)
    elif model == 'shallow_2':
        keras_model = Sequential()
        keras_model = standard_cnn(keras_model, nb_modules=2, nb_filters=[64, 128], conv_size=[7, 7], nb_dense=2, dense_units=[512, 128], input_shape=input_shape)
        schedule=[15, 25, 30]
        hist = fit_model(keras_model, X_train, Y_train, X_val, Y_val, schedule=schedule, nb_epoch=35, lr=0.001)
    elif model == 'shallow_3':
        keras_model = shallow_3(input_shape)
        schedule=[20,40]
        hist = fit_model(keras_model, X_train, Y_train, X_val, Y_val, schedule=schedule)
    elif model == 'shallow':
        keras_model = shallow(input_shape)
        schedule=[20,40]
        hist = fit_model(keras_model, X_train, Y_train, X_val, Y_val, schedule=schedule, nb_epoch=60, batch_size=32, lr=0.001)
    elif model == 'cifar':
        keras_model = Sequential()
        keras_model = standard_cnn(keras_model, nb_modules=2, nb_filters=[32, 64], nb_dense=1, dense_units=[512], input_shape=input_shape)
        schedule=[30, 50]
        hist = fit_model(keras_model, X_train, Y_train, X_val, Y_val, schedule=schedule, nb_epoch=60, batch_size=32, lr=0.01)
    elif model == 'cifarc':
        keras_model = Sequential()
        keras_model = standard_cnn(keras_model, nb_modules=2, nb_filters=[32, 64], nb_dense=1, dense_units=[512], input_shape=input_shape, init='orthogonal')
        schedule=[10, 20, 25]
        hist = fit_model(keras_model, X_train, Y_train, X_val, Y_val, schedule=schedule, nb_epoch=30, batch_size=32, lr=0.001)
    elif model == 'cifard':
        keras_model = Sequential()
        keras_model = standard_cnn(keras_model, nb_modules=2, nb_filters=[32, 64], nb_dense=2, dense_units=[512, 256], input_shape=input_shape, init='orthogonal')
        schedule=[15, 25, 30]
        hist = fit_model(keras_model, X_train, Y_train, X_val, Y_val, schedule=schedule, nb_epoch=35, batch_size=32, lr=0.001)
    elif model == 'cifard2':
        keras_model = Sequential()
        keras_model = standard_cnn(keras_model, nb_modules=2, nb_filters=[32, 64], nb_dense=2, dense_units=[512, 256], input_shape=input_shape, init='orthogonal')
        schedule=[15, 25, 30]
        hist = fit_model(keras_model, X_train, Y_train, X_val, Y_val, schedule=schedule, nb_epoch=35, batch_size=32, lr=0.001)
    elif model == 'shallow_bn':
        keras_model = shallow_bn(input_shape)
        schedule=[20,40]
        hist = fit_model(keras_model, X_train, Y_train, X_val, Y_val, schedule=schedule)

    elif model == 'LND-A':
        keras_model = Sequential()
        keras_model = standard_cnn(keras_model, nb_modules=3, module_depth=1, nb_filters=[32, 64, 96], conv_size=[3, 3, 3], nb_dense=2, dense_units=[512, 128], input_shape=input_shape, init='orthogonal')
        schedule=[20, 30, 35]
        hist = fit_model(keras_model, X_train, Y_train, X_val, Y_val, schedule=schedule, nb_epoch=3, batch_size=32, lr=0.001)

    elif model == 'LND-A-2P':
        keras_model = Sequential()
        keras_model = standard_cnn(keras_model, nb_modules=2, module_depth=1, nb_filters=[32, 64], conv_size=[3, 3], nb_dense=2, dense_units=[512, 256], input_shape=input_shape, init='orthogonal')
        schedule=[20, 30, 35]
        hist = fit_model(keras_model, X_train, Y_train, X_val, Y_val, schedule=schedule, nb_epoch=40, batch_size=32, lr=0.001)

    elif model == 'LND-A-3P':
        keras_model = Sequential()
        keras_model = standard_cnn(keras_model, nb_modules=3, module_depth=1, nb_filters=[32, 64, 96], conv_size=[3, 3, 3], nb_dense=2, dense_units=[512, 256], input_shape=input_shape, init='orthogonal')
        schedule=[20, 30, 35]
        hist = fit_model(keras_model, X_train, Y_train, X_val, Y_val, schedule=schedule, nb_epoch=40, batch_size=32, lr=0.001)

    elif model == 'LND-A-4P':
        keras_model = Sequential()
        keras_model = standard_cnn(keras_model, nb_modules=4, module_depth=1, nb_filters=[32, 64, 96, 128], conv_size=[3, 3, 3, 3], nb_dense=2, dense_units=[512, 256], input_shape=input_shape, init='orthogonal')
        schedule=[20, 30, 35]
        hist = fit_model(keras_model, X_train, Y_train, X_val, Y_val, schedule=schedule, nb_epoch=40, batch_size=32, lr=0.001)

    elif model == 'LND-A-5P':
        keras_model = Sequential()
        keras_model = standard_cnn(keras_model, nb_modules=5, module_depth=1, nb_filters=[32, 64, 96, 128, 160], conv_size=[3, 3, 3, 3, 3], nb_dense=2, dense_units=[512, 256], input_shape=input_shape, init='orthogonal')
        schedule=[20, 30, 35]
        hist = fit_model(keras_model, X_train, Y_train, X_val, Y_val, schedule=schedule, nb_epoch=40, batch_size=32, lr=0.001)

    elif model == 'LND-A-5P-HEQ':
        keras_model = Sequential()
        keras_model = standard_cnn(keras_model, nb_modules=5, module_depth=1, nb_filters=[32, 64, 96, 128, 160], conv_size=[3, 3, 3, 3, 3], nb_dense=2, dense_units=[512, 256], input_shape=input_shape, init='orthogonal')
        schedule=[20, 30, 35]
        hist = fit_model(keras_model, X_train, Y_train, X_val, Y_val, schedule=schedule, nb_epoch=40, batch_size=32, lr=0.001)

    elif model == 'LND-A-5P-NLM':
        keras_model = Sequential()
        keras_model = standard_cnn(keras_model, nb_modules=5, module_depth=1, nb_filters=[32, 64, 96, 128, 160], conv_size=[3, 3, 3, 3, 3], nb_dense=2, dense_units=[512, 256], input_shape=input_shape, init='orthogonal')
        schedule=[20, 30, 35]
        hist = fit_model(keras_model, X_train, Y_train, X_val, Y_val, schedule=schedule, nb_epoch=40, batch_size=32, lr=0.001)

    elif model == 'LND-A-5P-LDP':
        keras_model = Sequential()
        keras_model = standard_cnn_ldp(keras_model, nb_modules=5, module_depth=1, nb_filters=[32, 64, 96, 128, 160], conv_size=[3, 3, 3, 3, 3], nb_dense=2, dense_units=[512, 256], input_shape=input_shape, init='orthogonal')
        schedule=[30, 40, 45]
        hist = fit_model(keras_model, X_train, Y_train, X_val, Y_val, schedule=schedule, nb_epoch=50, batch_size=32, lr=0.001)

    elif model == 'LND-A-5P-LDP2':
        keras_model = Sequential()
        keras_model = standard_cnn_ldp(keras_model, nb_modules=5, module_depth=1, nb_filters=[32, 64, 96, 128, 160], conv_size=[3, 3, 3, 3, 3], nb_dense=2, dense_units=[512, 256], input_shape=input_shape, init='orthogonal', nb_dp=5, dp_init=0.1, dp_inc=0.1)
        schedule=[30, 40, 45]
        hist = fit_model(keras_model, X_train, Y_train, X_val, Y_val, schedule=schedule, nb_epoch=50, batch_size=32, lr=0.001)

    elif model == 'LND-A-ALLCNN-5P':
        keras_model = Sequential()
        allcnn_fs(keras_model, nb_modules=5, module_depth=1,  nb_filters=[32, 64, 96, 128, 160], conv_size=[3, 3, 3, 3, 3], input_shape=input_shape,  init='orthogonal')
        keras_model.add(Flatten())
        mlp_softmax(keras_model, nb_dense=2, dense_units=[512, 256], nb_classes=2, init='orthogonal')
        schedule=[20, 30, 35]
        hist = fit_model(keras_model, X_train, Y_train, X_val, Y_val, schedule=schedule, nb_epoch=40, batch_size=32, lr=0.001)

    elif model == 'LND-A-5P-MAXOUT':
        keras_model = Sequential()
        convpool_fs(keras_model, nb_modules=5, module_depth=1,  nb_filters=[32, 64, 96, 128, 160], conv_size=[3, 3, 3, 3, 3], input_shape=input_shape,  init='orthogonal')
        keras_model.add(Flatten())
        maxout_softmax(keras_model, nb_dense=2, dense_units=[512, 256], nb_classes=2, init='orthogonal')
        schedule=[20, 30, 35]
        hist = fit_model(keras_model, X_train, Y_train, X_val, Y_val, schedule=schedule, nb_epoch=40, batch_size=32, lr=0.001)

    elif model == 'LND-A-5P-64':
        keras_model = Sequential()
        keras_model = standard_cnn(keras_model, nb_modules=5, module_depth=1, nb_filters=[64, 64*2, 64*3, 64*4, 64*5], conv_size=[3, 3, 3, 3, 3], nb_dense=2, dense_units=[512, 256], input_shape=input_shape, init='orthogonal')
        schedule=[20, 30, 35]
        hist = fit_model(keras_model, X_train, Y_train, X_val, Y_val, schedule=schedule, nb_epoch=40, batch_size=32, lr=0.001)

    elif model == 'LND-A-5P-96':
        keras_model = Sequential()
        keras_model = standard_cnn(keras_model, nb_modules=5, module_depth=1, nb_filters=[96, 96*2, 96*3, 96*4, 96*5], conv_size=[3, 3, 3, 3, 3], nb_dense=2, dense_units=[512, 256], input_shape=input_shape, init='orthogonal')
        schedule=[35, 45, 50]
        hist = fit_model(keras_model, X_train, Y_train, X_val, Y_val, schedule=schedule, nb_epoch=55, batch_size=64, lr=0.001)

    elif model == 'LND-A-5P-MLP512':
        keras_model = Sequential()
        keras_model = standard_cnn(keras_model, nb_modules=5, module_depth=1, nb_filters=[32, 64, 96, 128, 160], conv_size=[3, 3, 3, 3, 3], nb_dense=2, dense_units=[512, 512], input_shape=input_shape, init='orthogonal')
        schedule=[30, 40, 45]
        hist = fit_model(keras_model, X_train, Y_train, X_val, Y_val, schedule=schedule, nb_epoch=50, batch_size=32, lr=0.001)

    elif model == 'LND-A-5P-MLP1024':
        keras_model = Sequential()
        keras_model = standard_cnn(keras_model, nb_modules=5, module_depth=1, nb_filters=[32, 64, 96, 128, 160], conv_size=[3, 3, 3, 3, 3], nb_dense=2, dense_units=[1024, 1024], input_shape=input_shape, init='orthogonal')
        schedule=[30, 40, 45]
        hist = fit_model(keras_model, X_train, Y_train, X_val, Y_val, schedule=schedule, nb_epoch=50, batch_size=32, lr=0.001)

    elif model == 'LND-A-5P-ZCA':
        keras_model = Sequential()
        keras_model = standard_cnn(keras_model, nb_modules=5, module_depth=1, nb_filters=[32, 64, 96, 128, 160], conv_size=[3, 3, 3, 3, 3], nb_dense=2, dense_units=[512, 256], input_shape=input_shape, init='orthogonal')
        schedule=[5, 10, 15]
        hist = fit_model(keras_model, X_train, Y_train, X_val, Y_val, schedule=schedule, nb_epoch=20, batch_size=32, lr=0.001, augment='btzca')

    elif model == 'LND-A-5P-C2':
        init = 'orthogonal'
        activation = 'relu'
        keras_model = Sequential()

        convpool_fs(keras_model, nb_modules=5, module_depth=1, nb_filters=[32, 64, 96, 128, 160], conv_size=[3, 3, 3, 3, 3], input_shape=input_shape, init=init, activation=activation)
        keras_model.add(Convolution2D(196, 2, 2, border_mode='valid', init=init))
        if activation == 'leaky_relu':
            keras_model.add(LeakyReLU(alpha=.333))
        else:
            keras_model.add(Activation(activation))
        keras_model.add(Dropout(0.25))
        keras_model.add(Flatten())

        mlp_softmax(keras_model, nb_dense=2, dense_units=[512, 256], init=init)

        schedule=[20, 30, 35]
        hist = fit_model(keras_model, X_train, Y_train, X_val, Y_val, schedule=schedule, nb_epoch=40, batch_size=32, lr=0.001)
    elif model == 'LND-B':
        keras_model = Sequential()
        keras_model = standard_cnn(keras_model, nb_modules=3, module_depth=2, nb_filters=[32, 64, 96], conv_size=[3, 3, 3], nb_dense=2, dense_units=[512, 128], input_shape=input_shape, init='orthogonal')
        schedule=[20, 30, 35]
        hist = fit_model(keras_model, X_train, Y_train, X_val, Y_val, schedule=schedule, nb_epoch=40, batch_size=32, lr=0.001)

    elif model == 'LND-C':
        keras_model = Sequential()
        keras_model = standard_cnn(keras_model, nb_modules=3, module_depth=3, nb_filters=[32, 64, 96], conv_size=[3, 3, 3], nb_dense=2, dense_units=[512, 128], input_shape=input_shape, init='orthogonal')
        schedule=[20, 30, 35]
        hist = fit_model(keras_model, X_train, Y_train, X_val, Y_val, schedule=schedule, nb_epoch=40, batch_size=32, lr=0.001)

    elif model == 'LND-C-5P':
        keras_model = Sequential()
        keras_model = standard_cnn(keras_model, nb_modules=5, module_depth=3, nb_filters=[32, 64, 96, 128, 160], conv_size=[3, 3, 3, 3, 3], nb_dense=2, dense_units=[512, 256], input_shape=input_shape, init='orthogonal')
        schedule=[20, 30, 35]
        hist = fit_model(keras_model, X_train, Y_train, X_val, Y_val, schedule=schedule, nb_epoch=40, batch_size=32, lr=0.001)

    elif model == 'LND-D':
        keras_model = Sequential()
        keras_model = standard_cnn(keras_model, nb_modules=3, module_depth=4, nb_filters=[32, 64, 96], conv_size=[3, 3, 3], nb_dense=2, dense_units=[512, 128], input_shape=input_shape, init='orthogonal')
        schedule=[20, 30, 35]
        hist = fit_model(keras_model, X_train, Y_train, X_val, Y_val, schedule=schedule, nb_epoch=40, batch_size=32, lr=0.001)
    elif model == 'LND-C-256':
        keras_model = Sequential()
        keras_model = standard_cnn(keras_model, nb_modules=3, module_depth=3, nb_filters=[32, 64, 96], conv_size=[3, 3, 3], nb_dense=2, dense_units=[512, 256], input_shape=input_shape, init='orthogonal')
        schedule=[20, 30, 35]
        hist = fit_model(keras_model, X_train, Y_train, X_val, Y_val, schedule=schedule, nb_epoch=40, batch_size=32, lr=0.001)
    elif model == 'LND-C-512':
        keras_model = Sequential()
        keras_model = standard_cnn(keras_model, nb_modules=3, module_depth=3, nb_filters=[32, 64, 96], conv_size=[3, 3, 3], nb_dense=2, dense_units=[512, 512], input_shape=input_shape, init='orthogonal')
        schedule=[20, 30, 35]
        hist = fit_model(keras_model, X_train, Y_train, X_val, Y_val, schedule=schedule, nb_epoch=40, batch_size=32, lr=0.001)
    
    return keras_model, hist

'''
config.py
'''

class NetModel: 
    def __init__(self, network, training_params, augment_params, preproc_params):
        self.network = network
        self.training_params = training_params
        self.augment_params = augment_params
        self.preproc_params = preproc_params
        
        self.generator = ImageDataGenerator(**self.augment_params)
        self.preprocessor = Preprocessor(**self.preproc_params)

    def load(self, name):
        if path.isfile('{}_arch.json'.format(name)):
            self.network = model_from_json(open('{}_arch.json'.format(name)).read())
            self.network.load_weights('{}_weights.h5'.format(name))
        if path.isfile('{}_tra.pkl'.format(name)):
            self.extractor = joblib.load('{}_extractor.pkl'.format(name))
        if path.isfile('{}_aug.pkl'.format(name)):
            self.clf = joblib.load('{}_clf.pkl'.format(name))
        if path.isfile('{}_pre.pkl'.format(name)):
            self.scaler = joblib.load('{}_scaler.pkl'.format(name))

    def save(self, name):
        json_string = self.network.to_json()
        open('{}_arch.json'.format(name), 'w').write(json_string)
        self.network.save_weights('{}_weights.h5'.format(name), overwrite=True)

        joblib.dump(self.training_params, '{}_tra.pkl'.format(name))
        joblib.dump(self.augment_params, '{}_aug.pkl'.format(name))
        joblib.dump(self.preproc_params, '{}_pre.pkl'.format(name))

    def fit(self, X_train, Y_train, X_test=None, Y_test=None):
        print 'Augment ...'
        X_train, Y_train = self.generator.augment(X_train, Y_train)
        print 'Preprocess ...'
        X_train = self.preprocessor.fit_transform(X_train, Y_train)

        print 'Fit network  ...'
        print 'input shape: {}'.format(X_train.shape)
        sgd = SGD(lr=self.training_params['lr'], 
                decay=self.training_params['decay'],
                momentum=self.training_params['momentum'], 
                nesterov=self.training_params['nesterov'])
        self.network.compile(loss='categorical_crossentropy', optimizer=sgd)
        lr_scheduler = StageScheduler(self.training_params['schedule'])

        print 'Negatives: {}'.format(np.sum(Y_train.T[0]))
        print 'Positives: {}'.format(np.sum(Y_train.T[1]))

        batch_size = self.training_params['batch_size']
        nb_epoch = self.training_params['nb_epoch']
        loss_bw_history = LossHistory()

        print 'predict input shape:{}'.format(X_train.shape) 
        history = None
        if X_test is None:
            print 'train & val 0.1'
            history = self.network.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, callbacks=[lr_scheduler, loss_bw_history], shuffle=False, validation_split=0.1, show_accuracy=True)
        else:
            print 'train & test'
            history = self.network.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, callbacks=[lr_scheduler, loss_bw_history], shuffle=False, validation_data=(X_test, Y_test), show_accuracy=True)

        history.history['loss_detail'] = loss_bw_history.losses
        return history.history

    def predict_proba(self, X):
        X = self.generator.centering_crop(X)
        X = self.preprocessor.transform(X)
        print 'predict input shape:{}'.format(X.shape)
        return self.network.predict_proba(X)

def fit2(X_train, Y_train, X_val=None, Y_val=None, model='shallow_1'):
    net_model = None
    input_shape = X_train[0].shape

    print 'Fit model: {}'.format(model)
    print 'X-train shape: {}'.format(X_train.shape)

    hist = None

    if model == 'LND-A':
        network = standard_cnn(Sequential(), nb_modules=3, module_depth=1, nb_filters=[32, 64, 96], conv_size=[3, 3, 3], nb_dense=2, dense_units=[512, 128], input_shape=input_shape, init='orthogonal')
        schedule=[20, 30, 35]
        train_params = {'schedule':schedule, 'nb_epoch':3, 'batch_size':32, 'lr':0.001, 'momentum':0.9, 'nesterov':True, 'decay':1e-6}
        augment_params = {'output_shape':(64, 64), 'ratio':1, 'batch_size':32, 'rotation_range':(-15,15), 'translation_range':(-0.1, 0.1), 'flip':True, 'mode':'balance_batch'}
        preproc_params = {}
        net_model = NetModel(network, train_params, augment_params, preproc_params)

    elif model == 'LND-A-5P':   
        network = standard_cnn(Sequential(), nb_modules=5, module_depth=1, nb_filters=[32, 64, 96, 128, 160], conv_size=[3, 3, 3, 3, 3], nb_dense=2, dense_units=[512, 256], input_shape=input_shape, init='orthogonal')
        schedule=[25, 35, 40]
        train_params = {'schedule':schedule, 'nb_epoch':45, 'batch_size':32, 'lr':0.001, 'momentum':0.9, 'nesterov':True, 'decay':1e-6}
        augment_params = {'output_shape':(64, 64), 'ratio':1, 'batch_size':32, 'rotation_range':(-5, 5), 'translation_range':(-0.05, 0.05), 'flip':True, 'mode':'balance_batch'}
        preproc_params = {}
        net_model = NetModel(network, train_params, augment_params, preproc_params)

    elif model == 'LND-A-6P':   
        network = standard_cnn(Sequential(), nb_modules=6, module_depth=1, nb_filters=[32, 64, 96, 128, 160, 192], conv_size=[3, 3, 3, 3, 3,3], nb_dense=2, dense_units=[512, 256], input_shape=input_shape, init='orthogonal')
        schedule=[25, 35, 40]
        train_params = {'schedule':schedule, 'nb_epoch':45, 'batch_size':32, 'lr':0.001, 'momentum':0.9, 'nesterov':True, 'decay':1e-6}
        augment_params = {'output_shape':(128, 128), 'ratio':1, 'batch_size':32, 'rotation_range':(-5, 5), 'translation_range':(-0.05, 0.05), 'flip':True, 'mode':'balance_batch'}
        preproc_params = {}
        net_model = NetModel(network, train_params, augment_params, preproc_params)

    else:
        print "Model config not found."
    
    hist = net_model.fit(X_train, Y_train, X_val, Y_val)
    return net_model, hist
 
''' .... '''
