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

# Blocks 

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

def standard_cnn(model, nb_modules=1, module_depth=2,  nb_filters=[64], input_shape=(3, 64, 64), nb_dense=1, dense_units=[512], nb_classes=2, init='orthogonal', batch_norm=False, activation='relu'):
    assert nb_modules == len(nb_filters)
    assert nb_dense  == len(dense_units)

    for i in range(nb_modules):
        if i == 0:
            x_conv_pool(model, module_depth, nb_filters[i], input_shape=input_shape, init=init, activation=activation)
        else:
            x_conv_pool(model, module_depth, nb_filters[i], activation=activation)

        if batch_norm is False:
            model.add(Dropout(0.25))

    model.add(Flatten())
    for i in range(nb_dense):
        model.add(Dense(dense_units[i], init=init))
        if activation == 'leaky_relu':
            model.add(LeakyReLU(alpha=.333))
        else:
            model.add(Activation(activation))
        model.add(Dropout(0.5))

    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
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

def shallow_2(input_shape):
    model = Sequential()
    return standard_cnn(model, nb_modules=2, nb_filters=[64, 128], nb_dense=1, dense_units=[512], input_shape=input_shape)

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
        keras_model = shallow_2(input_shape)
        schedule=[20,40]
        hist = fit_model(keras_model, X_train, Y_train, X_val, Y_val, schedule=schedule)
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

    return keras_model, hist
