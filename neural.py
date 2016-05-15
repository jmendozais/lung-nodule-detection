from os import path

import pickle

import numpy as np
np.random.seed(1000000007) # for reproducibility

import theano
import keras
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers.core import Dense, Dropout, Activation, Flatten, MaxoutDense
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from keras.regularizers import WeightRegularizer
from six.moves import range
from augment import ImageDataGenerator, Preprocessor
from sklearn.externals import joblib
# Utils 

def get_activations(model, layer, X_batch):
    get_activations = theano.function([model.layers[0].input], model.layers[layer].get_output(train=False), allow_input_downcast=True)
    activations = get_activations(X_batch) # same result as above
    return activations

# Feature blocks 

def convpool_block(model, depth=2, nb_filters=64, nb_conv=3, nb_pool=2, init='orthogonal', input_shape=None, activation='relu', batch_norm=False, regularizer=None, **junk):
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
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    return model

def convpool_fs(model, nb_modules=1, module_depth=2,  nb_filters=[64], conv_size=[3], input_shape=(3, 64, 64),  init='orthogonal', batch_norm=False, activation='relu', regularizer=None, dropout=0.25):
    assert nb_modules == len(nb_filters)
    prevent_coadapt = dropout == 0.5
    if prevent_coadapt:
        print "prevent co-adaptation with dropout ..."
        model.add(Dropout(0.2, input_shape=input_shape))

    for i in range(nb_modules):
        if i == 0:
            if prevent_coadapt:
                convpool_block(model, module_depth, nb_filters[i], nb_conv=conv_size[i], init=init, activation=activation, regularizer=regularizer)
            else:
                convpool_block(model, module_depth, nb_filters[i], nb_conv=conv_size[i], input_shape=input_shape, init=init, activation=activation, regularizer=regularizer)
        else:
            convpool_block(model, module_depth, nb_filters[i], nb_conv=conv_size[i], activation=activation, regularizer=regularizer)

        if batch_norm is False:
            model.add(Dropout(dropout))

    return model

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

def mlp_softmax(model, nb_dense=1, dense_units=[512], nb_classes=2, init='orthogonal', batch_norm=False, activation='relu', regularizer=None):
    for i in range(nb_dense):
        model.add(Dense(dense_units[i], init=init, W_regularizer=regularizer))
        if activation == 'leaky_relu':
            model.add(LeakyReLU(alpha=.333))
        else:
            model.add(Activation(activation))
        model.add(Dropout(0.5))

    model.add(Dense(nb_classes, W_regularizer=regularizer))
    model.add(Activation('softmax'))

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

def maxout_softmax(model, nb_dense=1, dense_units=[512], nb_classes=2, init='orthogonal', batch_norm=False, activation='relu', nb_feature=2, regularizer=None):
    for i in range(nb_dense):
        model.add(MaxoutDense(dense_units[i], init=init, nb_feature=nb_feature, W_regularizer=regularizer))
        model.add(Activation('linear'))
        model.add(Dropout(0.5))

    model.add(Dense(nb_classes, W_regularizer=regularizer))
    model.add(Activation('softmax'))

# Models 

def standard_cnn(model, nb_modules=1, module_depth=2,  nb_filters=[64], conv_size=[3], input_shape=(3, 64, 64), nb_dense=1, dense_units=[512], nb_classes=2, init='orthogonal', batch_norm=False, activation='relu', regularizer=None):
    convpool_fs(model, nb_modules, module_depth,  nb_filters, conv_size, input_shape,  init, batch_norm, activation, regularizer=regularizer)
    model.add(Flatten())
    mlp_softmax(model, nb_dense, dense_units, nb_classes, init, batch_norm, activation, regularizer=regularizer)
    return model

def standard_cnn_ldp(model, nb_modules=1, module_depth=2,  nb_filters=[64], conv_size=[3], input_shape=(3, 64, 64), nb_dense=1, dense_units=[512], nb_classes=2, init='orthogonal', batch_norm=False, activation='relu', nb_dp=5, dp_init=0.15, dp_inc=0.05, regularizer=None):
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

    def fit(self, X_train, Y_train, X_test=None, Y_test=None):
        print 'fit ...'

        print 'augment ...'
        print self.generator.__dict__
        X_train, Y_train = self.generator.augment(X_train, Y_train)

        print 'preprocess ...'
        print self.preprocessor.__dict__
        X_train = self.preprocessor.fit_transform(X_train, Y_train)

        print 'input shape: {}'.format(X_train.shape)
        print 'negatives: {}'.format(np.sum(Y_train.T[0]))
        print 'positives: {}'.format(np.sum(Y_train.T[1]))

        sgd = SGD(lr=self.training_params['lr'], 
                decay=self.training_params['decay'],
                momentum=self.training_params['momentum'], 
                nesterov=self.training_params['nesterov'])
        self.network.compile(loss='categorical_crossentropy', optimizer=sgd)
        lr_scheduler = StageScheduler(self.training_params['schedule'])
        batch_size = self.training_params['batch_size']
        nb_epoch = self.training_params['nb_epoch']
        loss_bw_history = LossHistory()

        print 'fit network  ...'
        history = None
        if X_test is None:
            history = self.network.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, callbacks=[lr_scheduler, loss_bw_history], shuffle=False, validation_split=0.1, show_accuracy=True)

        else:
            X_test, Y_test = self.generator.augment(X_test, Y_test)
            X_test = self.preprocessor.transform(X_test)
            history = self.network.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, callbacks=[lr_scheduler, loss_bw_history], shuffle=False, validation_data=(X_test, Y_test), show_accuracy=True)

        print 'history obj'
        print history.history
        history.history['loss_detail'] = loss_bw_history.losses
        return history.history

    def predict_proba(self, X):
        X = self.generator.centering_crop(X)
        X = self.preprocessor.transform(X)
        return self.network.predict_proba(X)

''' 
Configurations
'''

default_augment_params = {'output_shape':(64, 64), 'ratio':1, 'batch_size':32, 'rotation_range':(-5, 5), 'translation_range':(-0.05, 0.05), 'flip':True, 'mode': 'balance_batch', 'zoom_range':(1.0, 1.2)}
default_preproc_params = {'zmuv':True}

def lnd_a(input_shape):
    network = standard_cnn(Sequential(), nb_modules=3, module_depth=1, nb_filters=[32, 64, 96], conv_size=[3, 3, 3], nb_dense=2, dense_units=[512, 128], input_shape=input_shape, init='orthogonal')

def lnd_a_3p(input_shape):
    network = standard_cnn(Sequential(), nb_modules=3, module_depth=1, nb_filters=[32, 64, 96], conv_size=[3, 3, 3], nb_dense=2, dense_units=[512, 512], input_shape=input_shape, init='orthogonal')
    return network

def lnd_a_4p(input_shape):
    network = standard_cnn(Sequential(), nb_modules=4, module_depth=1, nb_filters=[32, 64, 96, 128], conv_size=[3, 3, 3, 3], nb_dense=2, dense_units=[512, 512], input_shape=input_shape, init='orthogonal')
    return network

def lnd_a_5p(input_shape):
    network = convpool_fs(Sequential(), nb_modules=5, module_depth=1, nb_filters=[32, 64, 96, 128, 160], conv_size=[3, 3, 3, 3, 3], input_shape=input_shape, init='orthogonal', activation='relu')
    network.add(Flatten())
    mlp_softmax(network, nb_dense=2, dense_units=[512, 512], nb_classes=2, init='orthogonal', activation='relu')
    return network

def lnd_a_5p_do(input_shape):
    network = convpool_fs(Sequential(), nb_modules=5, module_depth=1, nb_filters=[32, 64, 96, 128, 160], conv_size=[3, 3, 3, 3, 3], input_shape=input_shape, init='orthogonal', activation='relu', dropout=0.5)
    network.add(Flatten())
    mlp_softmax(network, nb_dense=2, dense_units=[512, 512], nb_classes=2, init='orthogonal', activation='relu')
    return network

def lnd_a_5p_reg(input_shape, l1=0., l2=0.):
    regularizer = WeightRegularizer(l1=l1, l2=l2) 
    network = convpool_fs(Sequential(), nb_modules=5, module_depth=1, nb_filters=[32, 64, 96, 128, 160], conv_size=[3, 3, 3, 3, 3], input_shape=input_shape, init='orthogonal', activation='relu', regularizer=regularizer)
    network.add(Flatten())
    mlp_softmax(network, nb_dense=2, dense_units=[512, 512], nb_classes=2, init='orthogonal', activation='relu', regularizer=regularizer)
    return network

def lnd_a_6p(input_shape):
    network = convpool_fs(Sequential(), nb_modules=6, module_depth=1, nb_filters=[32, 64, 96, 128, 160, 192], conv_size=[3, 3, 3, 3, 3,3], input_shape=input_shape, init='orthogonal', activation='leaky_relu')
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

def fit(X_train, Y_train, X_val=None, Y_val=None, model='shallow_1'):
    net_model = None
    input_shape = X_train[0].shape

    print 'Fit model: {}'.format(model)
    print 'X-train shape: {}'.format(X_train.shape)

    hist = None
    if model == 'LND-A':
        network = lnd_a(input_shape)
        schedule=[20, 30, 35]
        train_params = {'schedule':schedule, 'nb_epoch':3, 'batch_size':32, 'lr':0.001, 'momentum':0.9, 'nesterov':True, 'decay':0}
        net_model = NetModel(network, train_params, default_augment_params)

    elif model == 'LND-A-3P':
        network = lnd_a_3p(input_shape)
        schedule=[20, 30, 35]
        train_params = {'schedule':schedule, 'nb_epoch':40, 'batch_size':32, 
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

    elif model == 'LND-A-6P':   
        network = lnd_a_6p(input_shape)
        schedule=[35, 45, 50]
        train_params = {'schedule':schedule, 'nb_epoch':55, 'batch_size':32, 'lr':0.001, 'momentum':0.9, 'nesterov':True, 'decay':0}
        augment_params = default_augment_params
        augment_params['output_shape'] = (128, 128)
        net_model = NetModel(network, train_params, augment_params, default_preproc_params)

    elif model == 'LND-A-6P-ALLJSRT':   
        network = lnd_a_6p(input_shape)
        schedule=[35, 45, 50]
        train_params = {'schedule':schedule, 'nb_epoch':55, 'batch_size':32, 'lr':0.001, 'momentum':0.9, 'nesterov':True, 'decay':0}
        augment_params = default_augment_params
        augment_params['output_shape'] = (128, 128)
        net_model = NetModel(network, train_params, augment_params, default_preproc_params)

    else:
        print "Model config not found."
    
    hist = net_model.fit(X_train, Y_train, X_val, Y_val)
    return net_model, hist
 
''' .... '''
