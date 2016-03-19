import numpy as np
import theano

import keras
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from six.moves import range

from augment import bootstraping_augment

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
    
import pickle
def fit_model(model, X_train, Y_train, X_test=None, Y_test=None, batch_size=64, lr=0.01, schedule=[50], momentum=0.9, nesterov=True, nb_epoch=60): 

    sgd = SGD(lr=lr, decay=1e-6, momentum=momentum, nesterov=nesterov)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)

    lr_scheduler = StageScheduler(schedule)

    #X_train, Y_train = offline_augment(X_train, Y_train, ratio=1, rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
    X_train, Y_train = bootstraping_augment(X_train, Y_train, ratio=1, batch_size=batch_size, rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)

    print 'Negatives: {}'.format(np.sum(Y_train.T[0]))
    print 'Positives: {}'.format(np.sum(Y_train.T[1]))

    history = None
    if X_test is None:
        print 'train & val 0.1'
        history = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, callbacks=[lr_scheduler], shuffle=False, validation_split=0.1, show_accuracy=True)
    else:
        print 'train & test'
        history = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, callbacks=[lr_scheduler], shuffle=False, validation_data=(X_test, Y_test), show_accuracy=True)
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
    print 'Input shape: {}'.format(input_shape)

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

    elif model == 'LND-A-5P-LDP1':
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
