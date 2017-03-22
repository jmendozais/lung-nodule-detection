from os import path

import pickle

import numpy as np
np.random.seed(1000000007) # for reproducibility

import theano
import keras
import gc

from keras.layers import Input, merge, Dropout
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers import BatchNormalization, Flatten, Dense, Dropout, Activation, MaxoutDense
from keras.layers import Input, merge
from keras.models import Model, Sequential
from keras.models import model_from_json
from keras.layers.advanced_activations import LeakyReLU, PReLU

from keras.optimizers import SGD, Adadelta, Adagrad, Adam
from keras.utils import np_utils, generic_utils
from keras.regularizers import WeightRegularizer
from six.moves import range
from sklearn.externals import joblib
import util
import classify
from augment import *
from resnet import resnet_cifar10 as resnet

# Layers
from keras.engine import Layer
from keras import backend as K

class Softmax4D(Layer):
    def __init__(self, axis=-1,**kwargs):
        self.axis=axis
        super(Softmax4D, self).__init__(**kwargs)

    def build(self,input_shape):
        pass

    def call(self, x,mask=None):
        e = K.exp(x - K.max(x, axis=self.axis, keepdims=True))
        s = K.sum(e, axis=self.axis, keepdims=True)
        return e / s

    def get_output_shape_for(self, input_shape):
        axis_index = self.axis % len(input_shape)
        return tuple([input_shape[i] for i in range(len(input_shape)) \
                      if i != axis_index ])

# Utils 
def print_trainable_state(layers):
    for k in range(len(layers)):
        if isinstance(layers[k], Dense):
            print("Layer {} - {} trainable {} weight sample {}".format(k, layers[k].name, layers[k].trainable, layers[k].get_weights()[0][0][0]))
        elif isinstance(layers[k], Convolution2D):
            print("Layer {} - {} trainable {} weight sample {}".format(k, layers[k].name, layers[k].trainable, layers[k].get_weights()[0][0][0][0][0]))
    print("")

def get_activations(model, layer, X_batch):
    get_activations = theano.function([model.layers[0].input], model.layers[layer].get_output(train=False), allow_input_downcast=True)
    activations = get_activations(X_batch) # same result as above
    return activations

def get_optimizer(config):
    if 'opt' in config:
        optimizer = config['opt']
        config.pop('opt', None)
        if optimizer == 'sgd-nesterov':
            return SGD(lr=config['lr'], momentum=config['momentum'], decay=config['decay'], nesterov=config['nesterov'])
        elif optimizer == 'adagrad':
            return Adagrad(lr=config['lr'], epsilon=config['epsilon'], decay=config['decay']) 
        elif optimizer == 'adadelta':
            return Adadelta(lr=config['lr'], rho=config['rho'], epsilon=config['epsilon'], decay=config['decay'])
        elif optimizer == 'adam':
            return Adam(lr=config['lr'], beta_1=config['beta_1'], beta_2=config['beta_2'], epsilon=config['epsilon'], decay=config['decay'])

    print('Undefined optimizer. Using SGD')
    return SGD(lr=config['lr'], momentum=config['momentum'], decay=config['decay'], nesterov=config['nesterov'])

# Feature blocks 
def convpool_block(inp, depth=2, nb_filters=64, nb_conv=3, nb_pool=2, init='orthogonal', activation='relu', batch_norm=False, regularizer=None, **junk):
    out = inp
    for i in range(depth):
        out = Convolution2D(nb_filters, nb_conv, nb_conv, border_mode='same', init=init, W_regularizer=regularizer)(out)
        if batch_norm:
            out = BatchNormalization()(out)
        if activation == 'leaky_relu':
            out = LeakyReLU(alpha=.333)(out)
        else:
            out = Activation(activation)(out)
    out = MaxPooling2D(pool_size=(nb_pool, nb_pool))(out)
    return out

def convpool_fs(inp, nb_modules=1, module_depth=2,  nb_filters=[64], conv_size=[3], init='orthogonal', batch_norm=False, activation='relu', regularizer=None, dropout=0.25):
    assert nb_modules == len(nb_filters)
    layewise_dropout = type(dropout) == type([])

    out = inp
    prevent_coadapt = dropout == 0.5
    if prevent_coadapt:
        print "prevent co-adaptation with dropout ..."
        out = Dropout(0.2)(out)

    for i in range(nb_modules):
        if i == 0:
            if prevent_coadapt:
                out = convpool_block(out, module_depth, nb_filters[i], nb_conv=conv_size[i], init=init, activation=activation, regularizer=regularizer)
            else:
                out = convpool_block(out, module_depth, nb_filters[i], nb_conv=conv_size[i], init=init, activation=activation, regularizer=regularizer)
        else:
            out = convpool_block(out, module_depth, nb_filters[i], nb_conv=conv_size[i], activation=activation, regularizer=regularizer)

        if batch_norm is False:
            if layewise_dropout:
                out = Dropout(dropout[i])(out)
            else:
                out = Dropout(dropout)(out)

    return out

def convpool_fs_ldp(model, nb_modules=1, module_depth=2,  nb_filters=[64], conv_size=[3], input_shape=(3, 64, 64),  init='orthogonal', batch_norm=False, activation='relu', nb_dp=5, dp_init=0.15, dp_inc=0.05, regularizer=None):
    assert nb_modules == len(nb_filters)
    for i in range(nb_modules):
        if i == 0:
            convpool_block(model, module_depth, nb_filters[i], nb_conv=conv_size[i], input_shape=input_shape, init=init, activation=activation, regularizer=regularizer)
        else:
            convpool_block(model, module_depth, nb_filters[i], nb_conv=conv_size[i], activation=activation, regularizer=regularizer)

        if batch_norm is False:
            if i >= nb_modules - nb_dp:
                model.add(Dropout(dp_init + (i + nb_dp - nb_modules) * dp_inc))

    return model


def allcnn_block(model, depth=2, nb_filters=64, nb_conv=3, subsample=2, init='orthogonal', input_shape=None, activation='relu', batch_norm=False, regularizer=None, **junk):
    for i in range(depth):
        if input_shape != None:
            model.add(Convolution2D(nb_filters, nb_conv, nb_conv, border_mode='same', init=init, input_shape=input_shape, W_regularizer=regularizer))
        else:
            model.add(Convolution2D(nb_filters, nb_conv, nb_conv, border_mode='same', init=init, W_regularizer=regularizer))

        if batch_norm:
            model.add(BatchNormalization())

        if activation == 'leaky_relu':
            model.add(LeakyReLU(alpha=.333))
        else:
            model.add(Activation(activation))

    model.add(Convolution2D(nb_filters, subsample, subsample, border_mode='valid', init=init, subsample=(subsample, subsample), W_regularizer=regularizer))
    if activation == 'leaky_relu':
        model.add(LeakyReLU(alpha=.333))
    else:
        model.add(Activation(activation))

def allcnn_fs(model, nb_modules=1, module_depth=2,  nb_filters=[64], conv_size=[3], input_shape=(3, 64, 64),  init='orthogonal', batch_norm=False, activation='relu', regularizer=None):
    assert nb_modules == len(nb_filters)
    for i in range(nb_modules):
        if i == 0:
            allcnn_block(model, module_depth, nb_filters[i], nb_conv=conv_size[i], input_shape=input_shape, init=init, activation=activation, subsample=2, regularizer=regularizer)
        else:
            allcnn_block(model, module_depth, nb_filters[i], nb_conv=conv_size[i], activation=activation, subsample=2, regularizer=regularizer)

        if batch_norm is False:
            model.add(Dropout(0.25))

    return model

# Classification blocks

def mlp_softmax(inp, nb_dense=1, dense_units=[512], nb_classes=2, init='orthogonal', batch_norm=False, activation='relu', regularizer=None):
    out = inp
    for i in range(nb_dense):
        out = Dense(dense_units[i], init=init, W_regularizer=regularizer)(out)
        if activation == 'leaky_relu':
            out = LeakyReLU(alpha=.333)(out)
        else:
            out = Activation(activation)(out)
        out = Dropout(0.5)(out)

    out = Dense(nb_classes, W_regularizer=regularizer)(out)
    out = Activation('softmax')(out)
    return out

def nin_softmax(model, nb_dense=1, dense_units=[512], nb_classes=2, init='orthogonal', batch_norm=False, activation='relu', regularizer=None):
    for i in range(nb_dense):
        model.add(Dense(dense_units[i], init=init, W_regularizer=regularizer))
        if activation == 'leaky_relu':
            model.add(LeakyReLU(alpha=.333))
        else:
            model.add(Activation(activation))
        model.add(Dropout(0.5))

    model.add(Dense(nb_classes, W_regularizer=regularizer))
    model.add(Activation('softmax'))

def maxout_softmax(inp, nb_dense=1, dense_units=[512], nb_classes=2, init='orthogonal', batch_norm=False, activation='relu', nb_feature=2, regularizer=None):
    out = inp
    for i in range(nb_dense):
        out = MaxoutDense(dense_units[i], init=init, nb_feature=nb_feature, W_regularizer=regularizer)(out)
        out = Activation('linear')(out)
        out = Dropout(0.5)(out)

    out = Dense(nb_classes, W_regularizer=regularizer)(out)
    out = Activation('softmax')(out)

# Models 

def standard_cnn(nb_modules=1, module_depth=2,  nb_filters=[64], conv_size=[3], input_shape=(3, 64, 64), nb_dense=1, dense_units=[512], nb_classes=2, init='orthogonal', batch_norm=False, activation='relu', regularizer=None):
    print "Input Shape {}".format(input_shape)
    inp = Input(shape=input_shape, dtype='float32', name='input_layer')   
    out = convpool_fs(inp, nb_modules, module_depth,  nb_filters, conv_size, init, batch_norm, activation, regularizer=regularizer)
    out = Flatten()(out)
    out = mlp_softmax(out, nb_dense, dense_units, nb_classes, init, batch_norm, activation, regularizer=regularizer)
    return Model(input=inp, output=out)

def standard_cnn_ldp(nb_modules=1, module_depth=2,  nb_filters=[64], conv_size=[3], input_shape=(3, 64, 64), nb_dense=1, dense_units=[512], nb_classes=2, init='orthogonal', batch_norm=False, activation='relu', nb_dp=5, dp_init=0.15, dp_inc=0.05, regularizer=None):
    convpool_fs_ldp(model, nb_modules, module_depth,  nb_filters, conv_size, input_shape,  init, batch_norm, activation, nb_dp=nb_dp, dp_init=dp_init, dp_inc=dp_inc, regularizer=regularizer)
    model.add(Flatten())
    mlp_softmax(model, nb_dense, dense_units, nb_classes, init, batch_norm, activation, regularizer=regularizer)
    return model

class StageScheduler(keras.callbacks.Callback):
    def __init__(self, stages=[], decay=0.1):
        sorted(stages)
        self.stages = stages
        self.idx = 0
        self.decay = decay
    
    def on_epoch_begin(self, epoch, logs={}):
        print 'lr: {}'.format(self.model.optimizer.lr.get_value())

    def on_epoch_end(self, epoch, logs={}):
        if self.idx < len(self.stages):
            if epoch + 1 == self.stages[self.idx]:
                lr = self.model.optimizer.lr.get_value()
                self.model.optimizer.lr.set_value(float(lr * self.decay))
                self.idx += 1

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

class ModelCheckpoint(keras.callbacks.Callback):
    def __init__(self, filepath, verbose=0, epoch_interval=5):
        self.verbose = verbose
        self.filepath = filepath
        self.epoch_interval = epoch_interval
        self.epoch = -1

    def on_epoch_end(self, epoch, logs={}):
        self.epoch = epoch
        if (epoch + 1) % self.epoch_interval == 0:
            if self.verbose > 0:
                print('Epoch {}: saving model to {}'.format(epoch, self.filepath))
            self.model.save_weights((self.filepath + ".epoch_{}_weights.h5").format(epoch + 1), overwrite=True)
            
    def on_train_end(self, logs={}):
        self.epoch += self.epoch_interval
        if self.verbose > 0:
            print('Epoch %05d: saving model to %s' % (self.epoch, self.filepath))
        self.model.save_weights((self.filepath + ".epoch_{}_weights.h5").format(self.epoch + 1), overwrite=True)

class NetModel: 
    def __init__(self, network=None, training_params=None, augment_params=None, preproc_params=None):
        self.network = network
        self.training_params = training_params
        
        if augment_params != None:
            self.generator = ImageDataGenerator(**augment_params)
        else:
            self.generator = ImageDataGenerator()

        if preproc_params != None:
            self.preprocessor = Preprocessor(**preproc_params)
        else:
            self.preprocessor = Preprocessor()

    def load(self, name):
        # Network
        if path.isfile('{}_arch.json'.format(name)):
            self.network = model_from_json(open('{}_arch.json'.format(name)).read())
            self.network.load_weights('{}_weights.h5'.format(name))

        # Preprocessor
        if path.isfile('{}_pre.pkl'.format(name)):
            self.preprocessor = joblib.load('{}_pre.pkl'.format(name))
        else:
            self.preprocessor = Preprocessor()

        # Generator
        if path.isfile('{}_gen.pkl'.format(name)):
            self.generator = joblib.load('{}_gen.pkl'.format(name))
        else:
            self.generator = ImageDataGenerator()

        # Training params
        if path.isfile('{}_tra.pkl'.format(name)):
            self.training_params = joblib.load('{}_tra.pkl'.format(name))

    def save(self, name):
        json_string = self.network.to_json()
        open('{}_arch.json'.format(name), 'w').write(json_string)
        self.network.save_weights('{}_weights.h5'.format(name), overwrite=True)

        joblib.dump(self.training_params, '{}_tra.pkl'.format(name))
        joblib.dump(self.preprocessor, '{}_pre.pkl'.format(name))
        joblib.dump(self.generator, '{}_gen.pkl'.format(name))

    # DEPRECATED
    def preprocess_augment(self, X_train, Y_train, X_test=None, Y_test=None, streams=False, cropped_shape=None, disable_perturb=False):
        gc.collect()
        if streams:
            num_streams = len(X_train)
            num_channels = len(X_train[0][0])

            print "Augment & Preprocess train set ..."
            print 'Augment ...'
            print self.generator.__dict__
            tmp = Y_train
            rng_state = self.generator.rng.get_state()
            for k in range(num_streams):
                self.generator.rng.set_state(rng_state)
                X_train[k], Y_train = self.generator.augment(X_train[k], tmp, cropped_shape, disable_perturb=disable_perturb)
            gc.collect()

            print 'Preprocess ...'
            print self.preprocessor.__dict__
            X_train[0] = self.preprocessor.fit_transform(X_train[0], Y_train)
            for i in range(1, num_streams):
                X_train[i] = self.preprocessor.transform(X_train[i])
            gc.collect()

            print "Augment & Preprocess test set ..."
            if X_test != None:
                num_streams = len(X_test)
                tmp = Y_test
                rng_state = self.generator.rng.get_state()
                for k in range(num_streams):
                    self.generator.rng.set_state(rng_state)
                    X_test[k], Y_test = self.generator.augment(X_test[k], tmp, cropped_shape)
                gc.collect()

                X_test[0] = self.preprocessor.fit_transform(X_test[0], Y_test)
                for k in range(1, num_streams):
                    X_test[k] = self.preprocessor.transform(X_test[k])
                gc.collect()

        else:
            num_channels = len(X_train[0])
            print "Augment & Preprocess train set ..."
            print 'Augment ...'
            print self.generator.__dict__
            X_train, Y_train = self.generator.augment(X_train, Y_train, cropped_shape, disable_perturb=disable_perturb)
            gc.collect()

            print 'Preprocess ...'
            print self.preprocessor.__dict__
            X_train = self.preprocessor.fit_transform(X_train, Y_train)
            gc.collect()

            print "Augment & Preprocess test set ..."
            if X_test != None:
                X_test, Y_test = self.generator.augment(X_test, Y_test, cropped_shape)
                gc.collect()
                X_test = self.preprocessor.transform(X_test)
                gc.collect()
 
        print 'negatives: {}'.format(np.sum(Y_train.T[0]))
        print 'positives: {}'.format(np.sum(Y_train.T[1]))

        return X_train, Y_train, X_test, Y_test


    def perturb_dataset(self, X, new_X):
        print "Begin perturb ..."
        for i in range(len(X)):
            new_X.append(self.generator.perturb(X[i]))
        print "End perturb ..."


    def fit(self, X_train, Y_train, X_test=None, Y_test=None, streams=False, cropped_shape=None, checkpoint_prefix=None, checkpoint_interval=5, loss='categorical_crossentropy'):

        batch_size = self.training_params['batch_size']
        nb_epoch = self.training_params['nb_epoch']
        data_shape = (len(X_train[0]),) + self.generator.output_shape

        X_train, Y_train, X_test, Y_test = preprocess_dataset(self.preprocessor, X_train, Y_train, X_test, Y_test, streams)
        self.generator.fit(X_train)

        X_train, Y_train = util.split_data_pos_neg(X_train, Y_train)
        X_test, Y_test = util.split_data_pos_neg(X_test, Y_test)

        input_name = self.network.layers[0].name
        output_name = self.network.layers[-1].name
        train_gen = DataGenerator({input_name:[X_train[0], X_train[1]]}, {output_name:[Y_train[0], Y_train[1]]}, batch_size, self.generator.perturb, data_shape)
        test_gen = DataGenerator({input_name:[X_test[0], X_test[1]]}, {output_name:[Y_test[0], Y_test[1]]}, batch_size, self.generator.perturb, data_shape)

        print 'Fit network ...'
        opt = get_optimizer(self.training_params)

        self.network.compile(loss=loss, optimizer=opt, metrics=['accuracy'])

        loss_bw_history = LossHistory()
        callbacks = [loss_bw_history]
        if checkpoint_prefix != None:
            checkpoint_cb = ModelCheckpoint(verbose=True, filepath=checkpoint_prefix, epoch_interval=checkpoint_interval)
            callbacks.append(checkpoint_cb)
        if 'schedule' in self.training_params:
            lr_scheduler = StageScheduler(self.training_params['schedule'])
            callbacks.append(lr_scheduler)

        history = None
        if X_test is None:
            history = self.network.fit_generator(train_gen, 2*len(X_train[1]), nb_epoch, verbose=1, callbacks=callbacks)
        else:
            history = self.network.fit_generator(train_gen, 2*len(X_train[1]), nb_epoch, validation_data=test_gen, nb_val_samples=2*len(X_test[1]), verbose=1, callbacks=callbacks)
        gc.collect()

        print_trainable_state(self.network.layers)

        print history.history
        history.history['loss_detail'] = loss_bw_history.losses
        return history.history

    def predict_proba(self, X, streams=False):
        gc.collect()
        if streams:
            num_streams = len(X)
            for k in range(num_streams):
                X[k] = X[k].astype('float32')
                X[k] = self.generator.centering_crop(X[k])
                X[k] = self.preprocessor.transform(X[k])
            return self.network.predict(list(X))
        else:
            X = X.astype('float32')
            X = self.generator.centering_crop(X)
            X = self.preprocessor.transform(X)
            return self.network.predict(X)


''' 
Configurations
'''

default_augment_params = {'output_shape':(64, 64), 'ratio':1, 'batch_size':32, 'rotation_range':(-5, 5), 'translation_range':(-0.05, 0.05), 'flip':True, 'intensity_shift_std':0.5, 'mode':'balance_batch', 'zoom_range':(1.0, 1.2)}
default_preproc_params = {'zmuv':True}

def lnd_a_3p(input_shape, repeats=1, nb_classes=2, base_filters=32):
    network = standard_cnn(nb_modules=3, module_depth=repeats, nb_filters=[base_filters, 2*base_filters, 3*base_filters], conv_size=[3, 3, 3], nb_dense=2, dense_units=[512, 512], input_shape=input_shape, init='orthogonal', nb_classes=2)
    return network

def lnd_a_4p(input_shape, repeats):
    network = standard_cnn(nb_modules=4, module_depth=repeats, nb_filters=[32, 64, 96, 128], conv_size=[3, 3, 3, 3], nb_dense=2, dense_units=[512, 512], input_shape=input_shape, init='orthogonal')
    return network

def lnd_a_5p(input_shape, repeats, base_filters=32):
    nb_filters = (np.array(range(5)) + 1) * base_filters
    network = standard_cnn(nb_modules=5, module_depth=repeats, nb_filters=nb_filters, conv_size=[3, 3, 3, 3, 3], nb_dense=2, dense_units=[512, 512], input_shape=input_shape, init='orthogonal', activation='leaky_relu')
    return network

def lnd_a_5p_do(input_shape):
    network = convpool_fs(nb_modules=5, module_depth=1, nb_filters=[32, 64, 96, 128, 160], conv_size=[3, 3, 3, 3, 3], input_shape=input_shape, init='orthogonal', activation='relu', dropout=0.5)
    network.add(Flatten())
    mlp_softmax(network, nb_dense=2, dense_units=[512, 512], nb_classes=2, init='orthogonal', activation='relu')
    return network

def lnd_a_5p_reg(input_shape, l1=0., l2=0.):
    regularizer = WeightRegularizer(l1=l1, l2=l2) 
    network = convpool_fs(nb_modules=5, module_depth=1, nb_filters=[32, 64, 96, 128, 160], conv_size=[3, 3, 3, 3, 3], input_shape=input_shape, init='orthogonal', activation='relu', regularizer=regularizer)
    network.add(Flatten())
    mlp_softmax(network, nb_dense=2, dense_units=[512, 512], nb_classes=2, init='orthogonal', activation='relu', regularizer=regularizer)
    return network

def lnd_a_6p(input_shape, repeats=1):
    network = standard_cnn(nb_modules=6, module_depth=repeats, nb_filters=[32, 64, 96, 128, 160, 192], conv_size=[3, 3, 3, 3, 3, 3], nb_dense=2, dense_units=[512, 512], input_shape=input_shape, init='orthogonal', activation='leaky_relu')
    return network

    ''' 
    inp = Input(shape=input_shape, dtype='float32', name='input_layer')   
    out = convpool_fs(inp, nb_modules=6, module_depth=1, nb_filters=[32, 64, 96, 128, 160, 192], conv_size=[3, 3, 3, 3, 3,3], init='orthogonal', activation='leaky_relu')
    out = Flatten()(out)
    out = mlp_softmax(out, nb_dense=2, dense_units=[512, 512], nb_classes=2, init='orthogonal', activation='leaky_relu')
    return Model(input=inp, output=out)
    '''

def lnd_a_6p_thin(input_shape):
    network = convpool_fs(nb_modules=6, module_depth=1, nb_filters=[32, 32, 32, 64, 96, 128], conv_size=[3, 3, 3, 3, 3,3], input_shape=input_shape, init='orthogonal', activation='leaky_relu')
    network.add(Flatten())
    mlp_softmax(network, nb_dense=2, dense_units=[512, 512], nb_classes=2, init='orthogonal', activation='leaky_relu')
    return network

def lnd_a_6p_2(input_shape):
    network = convpool_fs(nb_modules=6, module_depth=1, nb_filters=[64, 128, 192, 256, 320, 384], conv_size=[3, 3, 3, 3, 3,3], input_shape=input_shape, init='orthogonal', activation='leaky_relu')
    network.add(Flatten())
    mlp_softmax(network, nb_dense=2, dense_units=[512, 512], nb_classes=2, init='orthogonal', activation='leaky_relu')
    return network

def lnd_a_6p_3(input_shape):
    network = convpool_fs(nb_modules=6, module_depth=1, nb_filters=[32, 64, 96, 128, 160, 192], conv_size=[3, 3, 3, 3, 3,3], input_shape=input_shape, init='orthogonal', activation='leaky_relu', dropout=[0.0, 0.0, 0.0, 0.1, 0.2, 0.3])
    network.add(Flatten())
    mlp_softmax(network, nb_dense=2, dense_units=[512, 512], nb_classes=2, init='orthogonal', activation='leaky_relu')
    return network

def lnd_a_6p_4(input_shape):
    network = convpool_fs(nb_modules=6, module_depth=1, nb_filters=[32, 64, 96, 128, 160, 192], conv_size=[3, 3, 3, 3, 3,3], input_shape=input_shape, init='orthogonal', activation='leaky_relu', dropout=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
    network.add(Flatten())
    mlp_softmax(network, nb_dense=2, dense_units=[512, 512], nb_classes=2, init='orthogonal', activation='leaky_relu')
    return network

def lnd_a_5p_1rc7(input_shape):
    init = 'orthogonal'
    activation = 'relu'

    network = Sequential()
    convpool_block(network, 3, 32, nb_conv=3, input_shape=input_shape, init=init, activation=activation)
    network.add(Dropout(0.25))
    convpool_fs(network, nb_modules=4, module_depth=1, nb_filters=[64, 96, 128, 160], conv_size=[3, 3, 3, 3], init=init, activation=activation)
    network.add(Flatten())
    mlp_softmax(network, nb_dense=2, dense_units=[512, 512], nb_classes=2, init=init, activation=activation)
    return network

# Multi-stream convnet implemented with functional API

def lnd_3p(input_layer, activation, init, dropout):
    kernels = [32, 64, 96]
    nb_pool = 2

    out = Convolution2D(kernels[0], 3, 3, border_mode='same', init=init)(input_layer)
    out = Activation(activation)(out)
    out = MaxPooling2D((nb_pool, nb_pool), border_mode='same')(out)
    out = Dropout(dropout)(out)
    out = Convolution2D(kernels[1], 3, 3, border_mode='same', init=init)(out)
    out = Activation(activation)(out)
    out = MaxPooling2D((nb_pool, nb_pool), border_mode='same')(out)
    out = Dropout(dropout)(out)
    out = Convolution2D(kernels[2], 3, 3, border_mode='same', init=init)(out)
    out = Activation(activation)(out)
    out = MaxPooling2D((nb_pool, nb_pool), border_mode='same')(out)
    out = Dropout(dropout)(out)
    out = Flatten()(out)

    return out

def lnd_a_3p_streams(input_shape, num_streams=3, late_fusion=False):
    activation = 'relu'
    init = 'orthogonal'
    dropout = 0.25
    nb_classes = 2
    dense_units = 512

    ins = []
    outs = []
    for i in xrange(num_streams):
        in_ = Input(shape=input_shape, dtype='float32', name='input{}'.format(i))   
        out_ = lnd_3p(in_, activation, init, dropout)
        ins.append(in_)
        outs.append(out_)
        
    if late_fusion:
        outs2 = []
        outs3 = []
        for i in xrange(num_streams):
            out = Dense(dense_units, init=init)(outs[i])
            out = Activation(activation)(out)
            out = Dropout(0.5)(out)
            outs2.append(out)

        for i in xrange(num_streams):
            out = Dense(dense_units, init=init)(outs2[i])
            out = Activation(activation)(out)
            out = Dropout(0.5)(out)
            outs3.append(out)
 
        out = merge(outs3, 'concat')
    else:
        out = merge(outs, 'concat')
        out = Dense(dense_units, init=init)(out)
        out = Activation(activation)(out)
        out = Dropout(0.5)(out)

    out = Dense(dense_units, init=init)(out)
    out = Activation(activation)(out)
    out = Dropout(0.5)(out)

    out = Dense(nb_classes)(out)
    out = Activation('softmax')(out)
    return Model(input=ins, output=out)

# Detector 

def dxp(input_shape, detector=False, blocks=2, kernels=[32, 32]):
    activation = 'relu'
    init = 'orthogonal'
    dropout = 0.25
    nb_classes = 2
    dense_units =128
    nb_pool = 2
    ksize = 3

    # Feature extraction stage
    inp = Input(shape=input_shape, dtype='float32', name='input')   
    for i in range(blocks):
        if i == 0:
            out = Convolution2D(kernels[0], ksize, ksize, border_mode='same', init=init)(inp)
        else:
            out = Convolution2D(kernels[0], ksize, ksize, border_mode='same', init=init)(out)
        out = Activation(activation)(out)
        out = Convolution2D(kernels[0], ksize, ksize, border_mode='same', init=init)(out)
        out = Activation(activation)(out)
        out = Convolution2D(kernels[0], ksize, ksize, border_mode='same', init=init)(out)
        out = Activation(activation)(out)
        out = MaxPooling2D((nb_pool, nb_pool), border_mode='same')(out)
        out = Dropout(dropout)(out)

    # Classification stage
    size = 32 / (2**blocks)
    if detector:
        out = Convolution2D(dense_units, size, size, border_mode='same', init=init)(out)
        out = Activation(activation)(out)
        out = Convolution2D(dense_units, 1, 1, border_mode='same', init=init)(out)
        out = Activation(activation)(out)
        out = Convolution2D(nb_classes, 1, 1, border_mode='same', init=init)(out)
        out = Softmax4D(axis=1,name="softmax")(out)
 
    else:
        out = Flatten()(out)
        out = Dense(dense_units, init=init)(out)
        out = Activation(activation)(out)
        out = Dropout(0.5)(out)
        out = Dense(dense_units, init=init)(out)
        out = Activation(activation)(out)
        out = Dropout(0.5)(out)
        out = Dense(nb_classes)(out)
        out = Activation('softmax')(out)

    return Model(input=inp, output=out)

def _3pnd(input_shape, activation='relu', init='orthogonal'):
    kernels = [32, 64, 96]
    nb_pool = 2
    dense_units = 512
    nb_classes = 2

    inp = Input(shape=input_shape, dtype='float32', name='input')   
    out = Convolution2D(kernels[0], 3, 3, border_mode='same', init=init)(inp)
    out = Activation(activation)(out)
    out = MaxPooling2D((nb_pool, nb_pool), border_mode='same')(out)
    out = Convolution2D(kernels[1], 3, 3, border_mode='same', init=init)(out)
    out = Activation(activation)(out)
    out = MaxPooling2D((nb_pool, nb_pool), border_mode='same')(out)
    out = Convolution2D(kernels[2], 3, 3, border_mode='same', init=init)(out)
    out = Activation(activation)(out)
    out = MaxPooling2D((nb_pool, nb_pool), border_mode='same')(out)
    out = Flatten()(out)

    out = Dense(dense_units, init=init)(out)
    out = Activation(activation)(out)
    out = Dropout(0.5)(out)
    out = Dense(dense_units, init=init)(out)
    out = Activation(activation)(out)
    out = Dropout(0.5)(out)
    out = Dense(nb_classes)(out)
    out = Activation('softmax')(out)

    return Model(input=inp, output=out)

# https://arxiv.org/pdf/1611.06651.pdf
def ct_a(input_shape, init='orthogonal'):
    kernels = [20, 50, 500, 2]
    nb_pool = 2
    dense_units = 512
    nb_classes = 2
    activation = 'relu'

    inp = Input(shape=input_shape, dtype='float32', name='input')   
    out = Convolution2D(kernels[0], 7, 7, border_mode='valid', init=init)(inp)
    out = MaxPooling2D((nb_pool, nb_pool), border_mode='same')(out)
    out = Convolution2D(kernels[1], 7, 7, border_mode='valid', init=init)(out)
    out = MaxPooling2D((nb_pool, nb_pool), border_mode='same')(out)
    out = Convolution2D(kernels[2], 7, 7, border_mode='valid', init=init)(out)
    out = Activation(activation)(out)
    out = MaxPooling2D((nb_pool, nb_pool), border_mode='same')(out)
    out = Convolution2D(kernels[3], 1, 1, border_mode='valid', init=init)(out)
    out = Flatten()(out)
    out = Activation('softmax')(out)

    return Model(input=inp, output=out)

def vgg16(mode='fc'):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    weights_path = 'data/vgg16_weights.h5'
    if weights_path:
        model.load_weights(weights_path)
    
    model.pop()
    model.add(Dense(2, activation='softmax'))

    if mode == 'fc':
        for i in range(len(model.layers) - 5):
            model.layers[i].trainable = False

    return model

# Convnets
    
def to_two_class_probs(y):
    Y = np.zeros((len(y), 2), dtype=np.float32)
    for i in range(len(y)):
        Y[i][1] = y[i]
        Y[i][0] = 1.0 - y[i]
    return Y

# Utils

def create_train_test_sets(feats_tr, pred_blobs_tr, real_blobs_tr,
                            feats_test=None, pred_blobs_test=None, real_blobs_test=None,
                            streams='none', detector=False, dataset_type='numpy', container=None):
    nb_classes = 2
    X_train, tmp, y_train = [], [], []
    
    if streams != 'none':
        num_streams = len(feats_tr)
        for i in range(num_streams):
            if detector:
                tmp, y_train = classify.create_training_set_for_detector(feats_tr[i], pred_blobs_tr, real_blobs_tr)
            else:
                tmp, y_train = classify.create_training_set_from_feature_set(feats_tr[i], pred_blobs_tr, real_blobs_tr, dataset_type, container, 'train')
            X_train.append(tmp.astype('float32'))
    else:
        if detector: 
            X_train, y_train = classify.create_training_set_for_detector(feats_tr, pred_blobs_tr, real_blobs_tr)
        else:
            X_train, y_train = classify.create_training_set_from_feature_set(feats_tr, pred_blobs_tr, real_blobs_tr, dataset_type, container=container, suffix='train')
        #X_train = X_train.astype('float32')

    X_test, Y_test = None, None
    if feats_test != None:
        X_test, y_test = [], []
        if streams != 'none':
            num_streams = len(feats_test)
            for i in range(num_streams):
                if detector:
                    tmp, y_test = classify.create_training_set_for_detector(feats_test[i], pred_blobs_test, real_blobs_test)
                else:
                    tmp, y_test = classify.create_training_set_from_feature_set(feats_test[i], pred_blobs_test, real_blobs_test, dataset_type, container, 'test')
                X_test.append(tmp.astype('float32'))
        else:
            if detector:
                X_test, y_test = classify.create_training_set_for_detector(feats_test, pred_blobs_test, real_blobs_test)
            else:
                X_test, y_test = classify.create_training_set_from_feature_set(feats_test, pred_blobs_test, real_blobs_test, dataset_type, container, 'test')
            #X_test = X_test.astype('float32')

    if detector:
        Y_train = to_two_class_probs(y_train)
        Y_test = to_two_class_probs(y_test)
    else:
        Y_train = np_utils.to_categorical(y_train, nb_classes)
        Y_test = np_utils.to_categorical(y_test, nb_classes)

    return X_train, Y_train, X_test, Y_test

def create_network(model, input_shape=(1, 32, 32), fold=-1, streams=-1, detector=False):
    print 'Create model: {}'.format(model)
    if streams:
        print 'X-train streams: {} shape: {}'.format(streams, input_shape)
    else:
        print 'X-train shape: {}'.format(input_shape)

    net_model = None
    print 'cropped shape {}'.format(input_shape)
    default_augment_params['output_shape'] = input_shape[1:]

    if detector:
        input_shape = (1, 512, 512)
 
    if model[-3:] == '_ua':
        model = model[:-3]

    if model == 'resnet_56':
        network = resnet(input_shape, nb_classes=2, depth=56)
        train_params = {'schedule':schedule, 'nb_epoch':30, 'batch_size':128, 
                        'lr':0.1, 'momentum':0.9, 'nesterov':True, 'decay':0}
        augment_params = default_augment_params
        augment_params['batch_size'] = 128
        net_model = NetModel(network, train_params, augment_params, default_preproc_params)
    elif model == 'd1p_a':
        network = dxp(input_shape, detector=detector, kernels=[16], blocks=1)
        net_model = NetModel(network, train_params, augment_params, default_preproc_params)
    elif model == 'd1p_b':
        network = dxp(input_shape, detector=detector, kernels=[32], blocks=1)
        net_model = NetModel(network, train_params, augment_params, default_preproc_params)
    elif model == 'd1p_c':
        network = dxp(input_shape, detector=detector, kernels=[64], blocks=1)
        net_model = NetModel(network, train_params, augment_params, default_preproc_params)
    elif model == 'd2p_a':
        train_params = {'schedule':schedule, 'nb_epoch':30, 'batch_size':32, 
                            'lr':0.01, 'momentum':0.9, 'nesterov':True, 'decay':0}
        network = dxp(input_shape, detector=detector, kernels=[16, 16], blocks=2)
        net_model = NetModel(network, train_params, augment_params, default_preproc_params)
    elif model == 'd2p_b':
        network = dxp(input_shape, detector=detector, kernels=[32, 32], blocks=2)
        net_model = NetModel(network, train_params, augment_params, default_preproc_params)
    elif model == 'd2p_c':
        network = dxp(input_shape, detector=detector, kernels=[32, 64], blocks=2)
        net_model = NetModel(network, train_params, augment_params, default_preproc_params)
    elif model == 'd3p_a':
        network = dxp(input_shape, detector=detector, kernels=[16, 16, 32], blocks=3)
        net_model = NetModel(network, train_params, augment_params, default_preproc_params)
    elif model == 'd3p_b':
        network = dxp(input_shape, detector=detector, kernels=[32, 32, 32], blocks=3)
        net_model = NetModel(network, train_params, augment_params, default_preproc_params)
    elif model == 'd3p_c':
        network = dxp(input_shape, detector=detector, kernels=[32, 64, 64], blocks=3)
        net_model = NetModel(network, train_params, augment_params, default_preproc_params)

    elif model == '3P':
        network = lnd_a_3p(input_shape)
        schedule=[15, 36, 36]
        train_params = {'schedule':schedule, 'nb_epoch':35, 'batch_size':32, 
                        'lr':0.001, 'momentum':0.9, 'nesterov':True, 'decay':0}
        net_model = NetModel(network, train_params, default_augment_params, default_preproc_params)

    elif model == '3PND':
        network = _3pnd(input_shape)
        schedule=[50, 50, 50]
        train_params = {'schedule':schedule, 'nb_epoch':50, 'batch_size':32, 
                        'lr':0.001, 'momentum':0.9, 'nesterov':True, 'decay':0}
        net_model = NetModel(network, train_params, default_augment_params, default_preproc_params)

    elif model == '3P-imagenet':
        network = lnd_a_3p((1, 32, 32), nb_classes=2)
        schedule=[15, 36, 36]
        train_params = {'schedule':schedule, 'nb_epoch':35, 'batch_size':32, 
                        'lr':0.001, 'momentum':0.9, 'nesterov':True, 'decay':0}
        net_model = NetModel(network, train_params, default_augment_params, default_preproc_params)

    elif model == '3P-B':
        network = lnd_a_3p((1, 32, 32), 2)
        schedule=[30, 40, 40]
        train_params = {'schedule':schedule, 'nb_epoch':40, 'batch_size':32, 
                        'lr':0.001, 'momentum':0.9, 'nesterov':True, 'decay':0}
        net_model = NetModel(network, train_params, default_augment_params, default_preproc_params)

    elif model == '3P-C':
        network = lnd_a_3p((1, 32, 32), 3)
        schedule=[30, 50, 50]
        train_params = {'schedule':schedule, 'nb_epoch':40, 'batch_size':32, 
                        'lr':0.001, 'momentum':0.9, 'nesterov':True, 'decay':0}
        net_model = NetModel(network, train_params, default_augment_params, default_preproc_params)

    elif model == '4P':
        network = lnd_a_4p((1, 32, 32), 1)
        schedule=[35, 50, 50]
        train_params = {'schedule':schedule, 'nb_epoch':50, 'batch_size':32, 
                        'lr':0.001, 'momentum':0.9, 'nesterov':True, 'decay':0}
        net_model = NetModel(network, train_params, default_augment_params, default_preproc_params)

    elif model == '4P-64':
        network = lnd_a_4p(input_shape, 1)
        schedule=[35, 50, 50]
        train_params = {'schedule':schedule, 'nb_epoch':50, 'batch_size':32, 
                        'lr':0.001, 'momentum':0.9, 'nesterov':True, 'decay':0}
        net_model = NetModel(network, train_params, default_augment_params, default_preproc_params)

    elif model == '4P-B':
        network = lnd_a_4p(input_shape, 2)
        schedule=[50, 50, 50]
        train_params = {'schedule':schedule, 'nb_epoch':46, 'batch_size':32, 
                        'lr':0.001, 'momentum':0.9, 'nesterov':True, 'decay':0}
        net_model = NetModel(network, train_params, default_augment_params, default_preproc_params)

    elif model == '4P-C':
        network = lnd_a_4p(input_shape, 3)
        schedule=[35, 56, 56]
        train_params = {'schedule':schedule, 'nb_epoch':56, 'batch_size':32, 
                        'lr':0.001, 'momentum':0.9, 'nesterov':True, 'decay':0}
        net_model = NetModel(network, train_params, default_augment_params, default_preproc_params)

    elif model == '5P-A':
        network = lnd_a_5p(input_shape, 1)
        schedule=[58, 75, 75]
        train_params = {'schedule':schedule, 'nb_epoch':75, 'batch_size':32, 
                        'lr':0.001, 'momentum':0.9, 'nesterov':True, 'decay':0}
        net_model = NetModel(network, train_params, default_augment_params, default_preproc_params)

    elif model == '5P-A-wide':
        input_shape = (128, 128)
        network = lnd_a_5p(input_shape, 1, base_filters=64)
        schedule=[65, 65, 65]
        train_params = {'schedule':schedule, 'nb_epoch':65, 'batch_size':32, 
                        'lr':0.001, 'momentum':0.9, 'nesterov':True, 'decay':0}
        net_model = NetModel(network, train_params, default_augment_params, default_preproc_params)

    elif model == '5P-B':
        input_shape = (128, 128)
        network = lnd_a_5p(input_shape, 2)
        schedule=[35, 55, 55]
        train_params = {'schedule':schedule, 'nb_epoch':55, 'batch_size':32, 
                        'lr':0.001, 'momentum':0.9, 'nesterov':True, 'decay':0}
        net_model = NetModel(network, train_params, default_augment_params, default_preproc_params)

    elif model == '5P-C':
        input_shape = (128, 128)
        network = lnd_a_5p(input_shape, 2)
        schedule=[50, 50, 50]
        train_params = {'schedule':schedule, 'nb_epoch':50, 'batch_size':32, 
                        'lr':0.001, 'momentum':0.9, 'nesterov':True, 'decay':0}
        net_model = NetModel(network, train_params, augment_params, default_preproc_params)

    elif model == '3P-stretch':
        network = lnd_a_3p(input_shape)
        schedule=[30, 45, 45]
        train_params = {'schedule':schedule, 'nb_epoch':45, 'batch_size':32, 
                        'lr':0.001, 'momentum':0.9, 'nesterov':True, 'decay':0}
        augment_params = default_augment_params
        augment_params['output_shape'] = input_shape[1:]
        net_model = NetModel(network, train_params, augment_params, default_preproc_params)

    elif model == '3P-max-white':
        network = lnd_a_3p(input_shape)
        schedule=[25, 40, 40]
        train_params = {'schedule':schedule, 'nb_epoch':40, 'batch_size':32, 
                        'lr':0.001, 'momentum':0.9, 'nesterov':True, 'decay':0}
        augment_params = default_augment_params
        augment_params['output_shape'] = input_shape[1:]
        net_model = NetModel(network, train_params, augment_params, default_preproc_params)

    elif model == '3P-norm':
        network = lnd_a_3p(input_shape)
        schedule=[30, 45, 45]
        train_params = {'schedule':schedule, 'nb_epoch':45, 'batch_size':32, 
                        'lr':0.001, 'momentum':0.9, 'nesterov':True, 'decay':0}
        augment_params = default_augment_params
        augment_params['output_shape'] = input_shape[1:]
        net_model = NetModel(network, train_params, augment_params, default_preproc_params)

    elif model == '3P-norm-heq':
        network = lnd_a_3p(input_shape)
        schedule=[25, 45, 45]
        train_params = {'schedule':schedule, 'nb_epoch':45, 'batch_size':32, 
                        'lr':0.001, 'momentum':0.9, 'nesterov':True, 'decay':0}
        augment_params = default_augment_params
        augment_params['output_shape'] = input_shape[1:]
        net_model = NetModel(network, train_params, augment_params, default_preproc_params)

    elif model in {'3P-norm-norm', '3P-us-1', '3P-us-2', '3P-us-4', '3P-us-8', '3P-us-16', '3P-us-12', '3P-us-32', '3P-us-init'}:
        network = lnd_a_3p(input_shape)
        schedule=[30, 50, 50]
        train_params = {'schedule':schedule, 'nb_epoch':50, 'batch_size':32, 
                        'lr':0.001, 'momentum':0.9, 'nesterov':True, 'decay':0}
        augment_params = default_augment_params
        augment_params['output_shape'] = input_shape[1:]
        net_model = NetModel(network, train_params, augment_params, default_preproc_params)

    elif model == '3P-GN':
        network = lnd_a_3p(input_shape, base_filters=64)
        schedule=[35, 60, 60]
        train_params = {'schedule':schedule, 'nb_epoch':60, 'batch_size':32, 
                        'lr':0.001, 'momentum':0.9, 'nesterov':True, 'decay':0}
        augment_params = default_augment_params
        augment_params['output_shape'] = input_shape[1:]
        augment_params['gn_mean'] = .0
        augment_params['gn_std'] = .1
        net_model = NetModel(network, train_params, augment_params, default_preproc_params)

    elif model == '3P-GS':
        network = lnd_a_3p(input_shape)
        schedule=[35, 60, 60]
        train_params = {'schedule':schedule, 'nb_epoch':60, 'batch_size':32, 
                        'lr':0.001, 'momentum':0.9, 'nesterov':True, 'decay':0}
        augment_params = default_augment_params
        augment_params['output_shape'] = input_shape[1:]
        augment_params['gs_size'] = 5
        augment_params['gs_sigma'] = .5
        net_model = NetModel(network, train_params, augment_params, default_preproc_params)


    elif model in {'3P-wide', '3P-us-24', '3P-us-24-b', '3P-us-24-ar'}:
        network = lnd_a_3p(input_shape, base_filters=64)
        schedule=[45, 65, 65]
        train_params = {'schedule':schedule, 'nb_epoch':60, 'batch_size':32, 
                        'lr':0.001, 'momentum':0.9, 'nesterov':True, 'decay':0}
        net_model = NetModel(network, train_params, default_augment_params, default_preproc_params)

    elif model in {'3P-br16'}:
        network = lnd_a_3p(input_shape, base_filters=64)
        schedule=[45, 70, 70]
        train_params = {'schedule':schedule, 'nb_epoch':70, 'batch_size':32, 
                        'lr':0.001, 'momentum':0.9, 'nesterov':True, 'decay':0}
        net_model = NetModel(network, train_params, default_augment_params, default_preproc_params)

    elif model in {'3P-br24'}:
        network = lnd_a_3p(input_shape, base_filters=64)
        schedule=[45, 65, 65]
        train_params = {'schedule':schedule, 'nb_epoch':65, 'batch_size':32, 
                        'lr':0.001, 'momentum':0.9, 'nesterov':True, 'decay':0}
        net_model = NetModel(network, train_params, default_augment_params, default_preproc_params)

    elif model in {'3P-br32-meanshape', '3P-br32', '3P-br32-is0.5', '3P-br32-is0.3', '3P-br32-is0.2', '3P-br32-is0.1', '3P-br32-is0.0', '3P-br32-uar24-rpi500', '3P-br32-uar24-rpi1000', '3p-br32-uar24-rpi1000-bal'}:
        network = lnd_a_3p(input_shape, base_filters=64)
        #schedule=[50, 70, 70]
        schedule=[70, 70, 70]
        train_params = {'schedule':schedule, 'nb_epoch':70, 'batch_size':32, 
                        'lr':0.001, 'momentum':0.9, 'nesterov':True, 'decay':0}
        augment_params = default_augment_params
        augment_params['intensity_shift_std'] = 0.5
        net_model = NetModel(network, train_params, augment_params, default_preproc_params)

    elif model in {'3P-br40'}:
        network = lnd_a_3p(input_shape, base_filters=64)
        schedule=[65, 80, 80]
        train_params = {'schedule':schedule, 'nb_epoch':80, 'batch_size':32, 
                        'lr':0.001, 'momentum':0.9, 'nesterov':True, 'decay':0}
        net_model = NetModel(network, train_params, default_augment_params, default_preproc_params)
        
    elif model in {'3P-br48'}:
        network = lnd_a_3p(input_shape, base_filters=64)
        schedule=[65, 65, 65]
        train_params = {'schedule':schedule, 'nb_epoch':65, 'batch_size':32, 
                        'lr':0.001, 'momentum':0.9, 'nesterov':True, 'decay':0}
        net_model = NetModel(network, train_params, default_augment_params, default_preproc_params)

    elif model == '3P-norm-nlm':
        network = lnd_a_3p((1, 32, 32))
        schedule=[32, 50, 50]
        train_params = {'schedule':schedule, 'nb_epoch':50, 'batch_size':32, 
                        'lr':0.001, 'momentum':0.9, 'nesterov':True, 'decay':0}
        augment_params = default_augment_params
        augment_params['output_shape'] = (32, 32)
        net_model = NetModel(network, train_params, augment_params, default_preproc_params)

    elif model == '3P-norm-stretch':
        network = lnd_a_3p((1, 32, 32))
        schedule=[30, 50, 50]
        train_params = {'schedule':schedule, 'nb_epoch':50, 'batch_size':32, 
                        'lr':0.001, 'momentum':0.9, 'nesterov':True, 'decay':0}
        augment_params = default_augment_params
        augment_params['output_shape'] = (32, 32)
        net_model = NetModel(network, train_params, augment_params, default_preproc_params)

    elif model == '3P-norm-max-white':
        network = lnd_a_3p((1, 32, 32))
        schedule=[30, 50, 50]
        train_params = {'schedule':schedule, 'nb_epoch':50, 'batch_size':32, 
                        'lr':0.001, 'momentum':0.9, 'nesterov':True, 'decay':0}
        augment_params = default_augment_params
        augment_params['output_shape'] = (32, 32)
        net_model = NetModel(network, train_params, augment_params, default_preproc_params)

    elif model == '3P-original':
        network = lnd_a_3p((1, 32, 32))
        schedule=[35, 50, 50]
        train_params = {'schedule':schedule, 'nb_epoch':50, 'batch_size':32, 
                        'lr':0.001, 'momentum':0.9, 'nesterov':True, 'decay':0}
        augment_params = default_augment_params
        augment_params['output_shape'] = (32, 32)
        net_model = NetModel(network, train_params, augment_params, default_preproc_params)

    elif model == '3P-sub':
        network = lnd_a_3p((1, 32, 32))
        schedule=[20, 30, 35]
        train_params = {'schedule':schedule, 'nb_epoch':40, 'batch_size':32, 
                        'lr':0.001, 'momentum':0.9, 'nesterov':True, 'decay':0}
        augment_params = default_augment_params
        augment_params['output_shape'] = (32, 32)
        net_model = NetModel(network, train_params, augment_params, default_preproc_params)

    elif model == '3PE40':
        network = lnd_a_3p((1, 32, 32))
        schedule=[20, 30, 35]
        train_params = {'schedule':schedule, 'nb_epoch':40, 'batch_size':32, 
                        'lr':0.001, 'momentum':0.9, 'nesterov':True, 'decay':0}
        augment_params = default_augment_params
        augment_params['output_shape'] = (32, 32)
        net_model = NetModel(network, train_params, augment_params, default_preproc_params)


    elif model == '3P-CAE':
        network = lnd_a_3p((1, 32, 32))
        schedule=[30, 50, 50]
        train_params = {'schedule':schedule, 'nb_epoch':50, 'batch_size':32, 
                        'lr':0.001, 'momentum':0.9, 'nesterov':True, 'decay':0}
        augment_params = default_augment_params
        augment_params['output_shape'] = (32, 32)
        net_model = NetModel(network, train_params, augment_params, default_preproc_params)

    elif model == '3P-CAE1':
        input_shape = (32, 32)
        network = lnd_a_3p((1, ) + input_shape)
        schedule=[30, 50, 50]
        train_params = {'schedule':schedule, 'nb_epoch':50, 'batch_size':32, 
                        'lr':0.001, 'momentum':0.9, 'nesterov':True, 'decay':0}
        augment_params = default_augment_params
        augment_params['output_shape'] = input_shape
        net_model = NetModel(network, train_params, augment_params, default_preproc_params)

    elif model == '3P-CAE5':
        input_shape = (32, 32)
        network = lnd_a_3p((1, ) + input_shape)
        schedule=[30, 50, 50]
        train_params = {'schedule':schedule, 'nb_epoch':50, 'batch_size':32, 
                        'lr':0.001, 'momentum':0.9, 'nesterov':True, 'decay':0}
        augment_params = default_augment_params
        augment_params['output_shape'] = input_shape
        net_model = NetModel(network, train_params, augment_params, default_preproc_params)

    elif model == '3P-CAE10':
        input_shape = (32, 32)
        network = lnd_a_3p((1, ) + input_shape)
        schedule=[30, 50, 50]
        train_params = {'schedule':schedule, 'nb_epoch':50, 'batch_size':32, 
                        'lr':0.001, 'momentum':0.9, 'nesterov':True, 'decay':0}
        augment_params = default_augment_params
        augment_params['output_shape'] = input_shape
        net_model = NetModel(network, train_params, augment_params, default_preproc_params)

    elif model == '3P-CAE15':
        network = lnd_a_3p((1, 32, 32))
        schedule=[30, 50, 50]
        train_params = {'schedule':schedule, 'nb_epoch':50, 'batch_size':32, 
                        'lr':0.001, 'momentum':0.9, 'nesterov':True, 'decay':0}
        augment_params = default_augment_params
        augment_params['output_shape'] = (32, 32)
        net_model = NetModel(network, train_params, augment_params, default_preproc_params)

    elif model == '3P-CAE20':
        network = lnd_a_3p((1, 32, 32))
        schedule=[20, 30, 35]
        train_params = {'schedule':schedule, 'nb_epoch':10, 'batch_size':32, 
                        'lr':0.001, 'momentum':0.9, 'nesterov':True, 'decay':0}
        augment_params = default_augment_params
        augment_params['output_shape'] = (32, 32)
        net_model = NetModel(network, train_params, augment_params, default_preproc_params)

    elif model == '3P-CAE30':
        network = lnd_a_3p((1, 32, 32))
        schedule=[20, 30, 35]
        train_params = {'schedule':schedule, 'nb_epoch':10, 'batch_size':32, 
                        'lr':0.001, 'momentum':0.9, 'nesterov':True, 'decay':0}
        augment_params = default_augment_params
        augment_params['output_shape'] = (32, 32)
        net_model = NetModel(network, train_params, augment_params, default_preproc_params)

    elif model == 'LND-A-3P-TRFS':
        network = lnd_a_3p(input_shape)
        schedule=[20, 30, 35]
        train_params = {'schedule':schedule, 'nb_epoch':40, 'batch_size':32, 
                        'lr':0.001, 'momentum':0.9, 'nesterov':True, 'decay':0}
        augment_params = default_augment_params
        augment_params['output_shape'] = (32, 32)
        net_model = NetModel(network, train_params, augment_params)

    elif model == 'LND-A-3P-STR-SEG':
        network = lnd_a_3p_streams(input_shape, 2)
        schedule=[20, 30, 35]
        train_params = {'schedule':schedule, 'nb_epoch':40, 'batch_size':32, 
                        'lr':0.001, 'momentum':0.9, 'nesterov':True, 'decay':0}
        augment_params = default_augment_params
        augment_params['output_shape'] = (32, 32)
        net_model = NetModel(network, train_params, augment_params, default_preproc_params)

    elif model == 'LND-A-3P-STR-FOVEA':
        network = lnd_a_3p_streams(input_shape, 2)
        schedule=[60, 60, 50]
        train_params = {'schedule':schedule, 'nb_epoch':60, 'batch_size':32, 
                        'lr':0.001, 'momentum':0.9, 'nesterov':True, 'decay':0}
        augment_params = default_augment_params
        augment_params['output_shape'] = (32, 32)
        net_model = NetModel(network, train_params, augment_params, {})

    elif model == 'LND-3P-STR-TRF':
        network = lnd_a_3p_streams(input_shape, 3)
        schedule=[20, 30, 35]
        train_params = {'schedule':schedule, 'nb_epoch':40, 'batch_size':32, 
                        'lr':0.001, 'momentum':0.9, 'nesterov':True, 'decay':0}
        augment_params = default_augment_params
        augment_params['output_shape'] = (32, 32)
        net_model = NetModel(network, train_params, augment_params, default_preproc_params)

    elif model == '3P-LATE':
        network = lnd_a_3p_streams(input_shape, 3, late_fusion=True)
        schedule=[50, 50, 50]
        train_params = {'schedule':schedule, 'nb_epoch':50, 'batch_size':32, 
                        'lr':0.001, 'momentum':0.9, 'nesterov':True, 'decay':0}
        augment_params = default_augment_params
        augment_params['output_shape'] = (32, 32)
        net_model = NetModel(network, train_params, augment_params)

    elif model == '3P-LATE2':
        network = lnd_a_3p_streams(input_shape, 3, late_fusion=True)
        schedule=[50, 50, 50]
        train_params = {'schedule':schedule, 'nb_epoch':50, 'batch_size':32, 
                        'lr':0.001, 'momentum':0.9, 'nesterov':True, 'decay':0}
        augment_params = default_augment_params
        augment_params['output_shape'] = (32, 32)
        net_model = NetModel(network, train_params, augment_params)

    elif model == 'LND-A-4P':
        network = lnd_a_4p(input_shape)
        schedule=[20, 30, 35]
        train_params = {'schedule':schedule, 'nb_epoch':40, 'batch_size':32, 
                        'lr':0.001, 'momentum':0.9, 'nesterov':True, 'decay':0}
        augment_params = default_augment_params
        augment_params['output_shape'] = (32, 32)
        net_model = NetModel(network, train_params, augment_params)

    elif model == '5P-TRF':   
        network = lnd_a_5p(input_shape)
        schedule=[50, 50, 50]
        train_params = {'schedule':schedule, 'nb_epoch':50, 'batch_size':32, 'lr':0.001, 'momentum':0.9, 'nesterov':True, 'decay':0}
        net_model = NetModel(network, train_params, default_augment_params)

    elif model == 'LND-A-5P':   
        network = lnd_a_5p(input_shape)
        schedule=[25, 35, 40]
        train_params = {'schedule':schedule, 'nb_epoch':45, 'batch_size':32, 'lr':0.001, 'momentum':0.9, 'nesterov':True, 'decay':0}
        net_model = NetModel(network, train_params, default_augment_params)

    elif model == 'LND-A-5P-1RC7':   
        network = lnd_a_5p_1rc7(input_shape)
        schedule=[25, 35, 40]
        train_params = {'schedule':schedule, 'nb_epoch':45, 'batch_size':32, 'lr':0.001, 'momentum':0.9, 'nesterov':True, 'decay':0}
        net_model = NetModel(network, train_params, default_augment_params, default_preproc_params)

    elif model == 'LND-A-5P-DO':   
        network = lnd_a_5p_do(input_shape)
        schedule=[25, 35, 40]
        train_params = {'schedule':schedule, 'nb_epoch':45, 'batch_size':32, 'lr':0.003, 'momentum':0.9, 'nesterov':True, 'decay':0}
        net_model = NetModel(network, train_params, default_augment_params, default_preproc_params)

    elif model == 'LND-A-5P-FIX':   
        network = lnd_a_5p(input_shape)
        schedule=[10, 50, 55]
        train_params = {'schedule':schedule, 'nb_epoch':60, 'batch_size':32, 'lr':0.001, 'momentum':0.9, 'nesterov':True, 'decay':0}
        net_model = NetModel(network, train_params, default_augment_params, default_preproc_params)

    elif model == 'LND-A-5P-LP':   
        network = lnd_a_5p_reg(input_shape, l2=0.0005)
        schedule=[45, 55, 60]
        train_params = {'schedule':schedule, 'nb_epoch':65, 'batch_size':64, 'lr':0.0001, 'momentum':0.9, 'nesterov':True, 'decay':0}
        net_model = NetModel(network, train_params, default_augment_params)

    elif model == 'LND-A-5P-FWC':   
        network = lnd_a_5p(input_shape)
        schedule=[25, 35, 40]
        train_params = {'schedule':schedule, 'nb_epoch':45, 'batch_size':32, 'lr':0.001, 'momentum':0.9, 'nesterov':True, 'decay':0}
        preproc_params = {'featurewise_center':True}
        net_model = NetModel(network, train_params, default_augment_params, preproc_params)

    elif model == 'LND-A-5P-RS-01':   
        network = lnd_a_5p(input_shape)
        schedule=[25, 35, 40]
        train_params = {'schedule':schedule, 'nb_epoch':45, 'batch_size':32, 'lr':0.001, 'momentum':0.9, 'nesterov':True, 'decay':0}
        preproc_params = {'featurewise_rescaling':True, 'featurewise_rescaling_range':[0, 1]}
        net_model = NetModel(network, train_params, default_augment_params, preproc_params)

    elif model == 'LND-A-5P-ZCA':   
        network = lnd_a_5p(input_shape)
        schedule=[25, 35, 40]
        train_params = {'schedule':schedule, 'nb_epoch':45, 'batch_size':32, 'lr':0.001, 'momentum':0.9, 'nesterov':True, 'decay':0}
        preproc_params = {'zca_whitening':True}
        net_model = NetModel(network, train_params, default_augment_params, preproc_params)

    elif model == 'LND-A-5P-ZMUV':   
        network = lnd_a_5p(input_shape)
        schedule=[25, 35, 40]
        train_params = {'schedule':schedule, 'nb_epoch':45, 'batch_size':32, 'lr':0.001, 'momentum':0.9, 'nesterov':True, 'decay':0}
        preproc_params = {'zmuv':True}
        net_model = NetModel(network, train_params, default_augment_params, preproc_params)

    elif model == 'NORM-LND-A-5P-ZMUV':   
        network = lnd_a_5p(input_shape)
        schedule=[20, 35, 40]
        train_params = {'schedule':schedule, 'nb_epoch':45, 'batch_size':32, 'lr':0.001, 'momentum':0.9, 'nesterov':True, 'decay':0}
        preproc_params = {'zmuv':True, 'samplewise_std_normalization':True}
        net_model = NetModel(network, train_params, default_augment_params, preproc_params)

    elif model == 'LND-A-5P-GCN':   
        network = lnd_a_5p(input_shape)
        schedule=[25, 35, 40]
        train_params = {'schedule':schedule, 'nb_epoch':45, 'batch_size':32, 'lr':0.001, 'momentum':0.9, 'nesterov':True, 'decay':0}
        preproc_params = {'samplewise_std_normalization':True}
        net_model = NetModel(network, train_params, default_augment_params, preproc_params)

    elif model == 'LND-A-5P-L2':   
        network = lnd_a_5p_reg(input_shape, l2=0.0005)
        schedule=[35, 45, 50]
        train_params = {'schedule':schedule, 'nb_epoch':55, 'batch_size':32, 'lr':0.001, 'momentum':0.9, 'nesterov':True, 'decay':0}
        preproc_params = {}
        net_model = NetModel(network, train_params, default_augment_params, preproc_params)

    elif model == '5P':   
        network = lnd_a_5p(input_shape)
        schedule=[50, 50, 50]
        train_params = {'schedule':schedule, 'nb_epoch':45, 'batch_size':32, 'lr':0.001, 'momentum':0.9, 'nesterov':True, 'decay':0}

        augment_params = default_augment_params
        augment_params['output_shape'] = (128, 128)
        net_model = NetModel(network, train_params, augment_params)

    elif model == '6P':   
        network = lnd_a_6p(input_shape)
        schedule=[25, 60, 60]
        train_params = {'opt':'sgd', 'schedule':schedule, 'nb_epoch':60, 'batch_size':32, 'lr':0.001, 'momentum':0.9, 'nesterov':True, 'decay':0}
        augment_params = default_augment_params
        augment_params['output_shape'] = input_shape[1:]
        net_model = NetModel(network, train_params, augment_params, {})

    elif model == '6P-B':   
        input_shape = (128, 128)
        network = lnd_a_6p((1,) + input_shape, 2)
        schedule=[50, 50, 50]
        train_params = {'schedule':schedule, 'nb_epoch':50, 'batch_size':32, 'lr':0.001, 'momentum':0.9, 'nesterov':True, 'decay':0}

        augment_params = default_augment_params
        augment_params['output_shape'] = input_shape
        net_model = NetModel(network, train_params, augment_params)

    elif model == '6P-C':   
        input_shape = (128, 128)
        network = lnd_a_6p((1,) + input_shape, 3)
        schedule=[50, 50, 50]
        train_params = {'schedule':schedule, 'nb_epoch':50, 'batch_size':32, 'lr':0.001, 'momentum':0.9, 'nesterov':True, 'decay':0}

        augment_params = default_augment_params
        augment_params['output_shape'] = input_shape
        net_model = NetModel(network, train_params, augment_params)

    elif model == '6P-norm-norm':   
        network = lnd_a_6p((1,128, 128))
        schedule=[25, 60, 60]
        train_params = {'opt':'sgd', 'schedule':schedule, 'nb_epoch':50, 'batch_size':32, 'lr':0.001, 'momentum':0.9, 'nesterov':True, 'decay':0}
        augment_params = default_augment_params
        augment_params['output_shape'] = (128, 128)
        net_model = NetModel(network, train_params, augment_params, default_preproc_params)

    elif model == '6P-br32':   
        network = lnd_a_6p((1,128, 128))
        schedule=[60, 60, 60]
        train_params = {'opt':'sgd', 'schedule':schedule, 'nb_epoch':60, 'batch_size':32, 'lr':0.001, 'momentum':0.9, 'nesterov':True, 'decay':0}
        augment_params = default_augment_params
        augment_params['output_shape'] = (128, 128)
        net_model = NetModel(network, train_params, augment_params, default_preproc_params)

    elif model == '6P-sub':   
        network = lnd_a_6p((1,128, 128))
        schedule=[25, 60, 60]
        train_params = {'opt':'sgd', 'schedule':schedule, 'nb_epoch':60, 'batch_size':32, 'lr':0.001, 'momentum':0.9, 'nesterov':True, 'decay':0}
        augment_params = default_augment_params
        augment_params['output_shape'] = (128, 128)
        net_model = NetModel(network, train_params, augment_params, {})

    elif model == '6P-adagrad':   
        network = lnd_a_6p((1,128, 128))
        train_params = {'opt':'adagrad', 'nb_epoch':60, 'batch_size':32, 'lr':0.001, 'epsilon':1e-08, 'decay':0.0}
        augment_params = default_augment_params
        augment_params['output_shape'] = (128, 128)
        net_model = NetModel(network, train_params, augment_params, {})

    elif model == '6P-adadelta':   
        network = lnd_a_6p((1,128, 128))
        train_params = {'opt':'adadelta', 'nb_epoch':60, 'batch_size':32, 'lr':1.0, 'rho':0.95, 'epsilon':1e-08, 'decay':0.0}
        augment_params = default_augment_params
        augment_params['output_shape'] = (128, 128)
        net_model = NetModel(network, train_params, augment_params, {})

    elif model == '6P-adam':   
        network = lnd_a_6p((1,128, 128))
        train_params = {'opt':'adam', 'nb_epoch':60, 'batch_size':32, 'lr':0.0005, 'beta_1':0.9, 'beta_2':0.999, 'epsilon':1e-08, 'decay':0.0}
        augment_params = default_augment_params
        augment_params['output_shape'] = (128, 128)
        net_model = NetModel(network, train_params, augment_params, {})

    elif model == '6P-ZMUV':   
        network = lnd_a_6p(input_shape)
        schedule=[45, 80, 80]
        train_params = {'schedule':schedule, 'nb_epoch':80, 'batch_size':32, 'lr':0.001, 'momentum':0.9, 'nesterov':True, 'decay':0}
        augment_params = default_augment_params
        augment_params['output_shape'] = (128, 128)
        net_model = NetModel(network, train_params, augment_params, default_preproc_params)

    elif model == 'LND-A-6P':   
        network = lnd_a_6p(input_shape)
        schedule=[35, 45, 50]
        train_params = {'schedule':schedule, 'nb_epoch':55, 'batch_size':32, 'lr':0.001, 'momentum':0.9, 'nesterov':True, 'decay':0}
        augment_params = default_augment_params
        augment_params['output_shape'] = (128, 128)
        net_model = NetModel(network, train_params, augment_params, {})

    elif model == 'LND-A-6P-TRF':
        network = lnd_a_6p(input_shape)
        schedule=[35, 75, 75]
        train_params = {'schedule':schedule, 'nb_epoch':75, 'batch_size':32, 'lr':0.001, 'momentum':0.9, 'nesterov':True, 'decay':0}
        augment_params = default_augment_params
        augment_params['output_shape'] = (128, 128)
        net_model = NetModel(network, train_params, augment_params, {})

    elif model == 'LND-A-6P-TRF2':
        network = lnd_a_6p_2(input_shape)
        schedule=[50, 50, 50]
        train_params = {'schedule':schedule, 'nb_epoch':50, 'batch_size':32, 'lr':0.001, 'momentum':0.9, 'nesterov':True, 'decay':0}
        augment_params = default_augment_params
        augment_params['output_shape'] = (128, 128)
        net_model = NetModel(network, train_params, augment_params, {})

    # Discarted
    elif model == 'LND-A-6P-TRF3':
        network = lnd_a_6p_3(input_shape)
        schedule=[50, 50, 50]
        train_params = {'schedule':schedule, 'nb_epoch':50, 'batch_size':32, 'lr':0.001, 'momentum':0.9, 'nesterov':True, 'decay':0}
        augment_params = default_augment_params
        augment_params['output_shape'] = (128, 128)
        net_model = NetModel(network, train_params, augment_params, {})

    # Discarted
    elif model == 'LND-A-6P-TRF4':
        network = lnd_a_6p_4(input_shape)
        schedule=[40, 70, 70]
        train_params = {'schedule':schedule, 'nb_epoch':70, 'batch_size':32, 'lr':0.001, 'momentum':0.9, 'nesterov':True, 'decay':0}
        augment_params = default_augment_params
        augment_params['output_shape'] = (128, 128)
        net_model = NetModel(network, train_params, augment_params, {})

    # LCE + WMCI
    elif model == 'LND-A-6P-TRF5':
        network = lnd_a_6p(input_shape)
        schedule=[35, 70, 70]
        train_params = {'schedule':schedule, 'nb_epoch':70, 'batch_size':32, 'lr':0.001, 'momentum':0.9, 'nesterov':True, 'decay':0}
        augment_params = default_augment_params
        augment_params['output_shape'] = (128, 128)
        net_model = NetModel(network, train_params, augment_params, {})

    # LCE + NORM
    elif model == 'LND-A-6P-TRF6':
        network = lnd_a_6p(input_shape)
        schedule=[30, 60, 60]
        train_params = {'schedule':schedule, 'nb_epoch':60, 'batch_size':32, 'lr':0.001, 'momentum':0.9, 'nesterov':True, 'decay':0}
        augment_params = default_augment_params
        augment_params['output_shape'] = (128, 128)
        net_model = NetModel(network, train_params, augment_params, {})

    elif model == 'LND-A-6P-TRF-THIN':
        network = lnd_a_6p_thin(input_shape)
        schedule=[30, 65, 65]
        train_params = {'schedule':schedule, 'nb_epoch':65, 'batch_size':32, 'lr':0.001, 'momentum':0.9, 'nesterov':True, 'decay':0}
        augment_params = default_augment_params
        augment_params['output_shape'] = (128, 128)
        net_model = NetModel(network, train_params, augment_params, {})

    elif model == 'LND-A-6P-ALLJSRT':   
        network = lnd_a_6p(input_shape)
        schedule=[35, 45, 50]
        train_params = {'schedule':schedule, 'nb_epoch':55, 'batch_size':32, 'lr':0.001, 'momentum':0.9, 'nesterov':True, 'decay':0}
        augment_params = default_augment_params
        augment_params['output_shape'] = (128, 128)
        net_model = NetModel(network, train_params, augment_params, default_preproc_params)

    elif model == 'vgg16-fc':
        network = vgg16(mode='fc')
        schedule=[40, 40, 40]
        train_params = {'schedule':schedule, 'nb_epoch':40, 'batch_size':32, 'lr':0.0001, 'momentum':0.9, 'nesterov':True, 'decay':0}
        augment_params = default_augment_params
        augment_params['output_shape'] = (224, 224)
        net_model = NetModel(network, train_params, augment_params, default_preproc_params)

    elif model in {'ct_a'}:
        network = ct_a(input_shape)
        schedule=[50, 70, 70]
        train_params = {'schedule':schedule, 'nb_epoch':50, 'batch_size':32, 
                        'lr':0.001, 'momentum':0.9, 'nesterov':True, 'decay':0}
        net_model = NetModel(network, train_params, default_augment_params, default_preproc_params)

    else:
        raise Exception("Model config not found.")
    
    return net_model
 
''' .... '''


