from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.optimizers import *
from keras.objectives import *
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.engine import Layer
from keras.layers.core import Lambda
from keras.layers.core import Reshape 
from keras.layers import Dense, Dropout, Activation, Flatten, Input, InputLayer
from keras.layers import Convolution2D, UpSampling2D
from keras.layers import MaxPooling2D
from keras.layers import merge
from keras.utils import np_utils, layer_utils
from keras import backend as K
from keras.engine import InputSpec
from scipy.misc import imresize
from theano import tensor as T
import gc

from itertools import product

import neural
import util

def get_conv_outsize(size, k, s, p, cover_all=False):
    if cover_all:
        return (size + p * 2 - k + s - 1) // s + 1
    else:
        return (size + p * 2 - k) // s + 1

def get_deconv_outsize(size, k, s, p, cover_all=False):
    if cover_all:
        return s * (size - 1) + k - s + 1 - 2 * p
    else:
        return s * (size - 1) + k - 2 * p

def conv_output_length(input_length, filter_size, border_mode, stride):
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - filter_size + 1
    return (output_length + stride - 1) // stride

def getwhere(x):
    ''' Calculate the "where" mask that contains switches indicating which
    index contained the max value when MaxPool2D was applied.  Using the
    gradient of the sum is a nice trick to keep everything high level.'''
    y_prepool, y_postpool = x
    return K.gradients(K.sum(y_postpool), y_prepool)

class _Pooling2D(Layer):
    def __init__(self, pool_size=(2, 2), strides=None, border_mode='valid',
                 dim_ordering='th', **kwargs):
        super(_Pooling2D, self).__init__(**kwargs)
        self.pool_size = tuple(pool_size)
        if strides is None:
            strides = self.pool_size
        self.strides = tuple(strides)
        assert border_mode in {'valid', 'same'}, 'border_mode must be in {valid, same}'
        self.border_mode = border_mode
        assert dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
        self.dim_ordering = dim_ordering
        self.input_spec = [InputSpec(ndim=4)]

    def get_output_shape_for(self, input_shape):
        if self.dim_ordering == 'th':
            rows = input_shape[2]
            cols = input_shape[3]
        elif self.dim_ordering == 'tf':
            rows = input_shape[1]
            cols = input_shape[2]
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

        rows = conv_output_length(rows, self.pool_size[0],
                                  self.border_mode, self.strides[0])
        cols = conv_output_length(cols, self.pool_size[1],
                                  self.border_mode, self.strides[1])

        if self.dim_ordering == 'th':
            return (input_shape[0], input_shape[1], rows, cols)
        elif self.dim_ordering == 'tf':
            return (input_shape[0], rows, cols, input_shape[3])
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

    def _pooling_function(self, inputs, pool_size, strides,
                          border_mode, dim_ordering):
        raise NotImplementedError

    def call(self, x, mask=None):
        output = self._pooling_function(inputs=x, pool_size=self.pool_size,
                                        strides=self.strides,
                                        border_mode=self.border_mode,
                                        dim_ordering=self.dim_ordering)
        return output

    def get_config(self):
        config = {'pool_size': self.pool_size,
                  'border_mode': self.border_mode,
                  'strides': self.strides,
                  'dim_ordering': self.dim_ordering}
        base_config = super(_Pooling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class WhatWhereMaxPooling2D(_Pooling2D):
    def __init__(self, pool_size=(2, 2), strides=None, border_mode='valid',
                 dim_ordering='th', **kwargs):
        super(WhatWhereMaxPooling2D, self).__init__(pool_size, strides, border_mode,
                                           dim_ordering, **kwargs)
    def _pooling_function(self, inputs, pool_size, strides,
                          border_mode, dim_ordering):
        output = K.pool2d(inputs, pool_size, strides, border_mode, dim_ordering, pool_mode='max')
        self.where = merge([inputs, output], mode=getwhere, output_shape=lambda x: x[0])
        return output

class WhatWhereUnPooling2D(Layer):
    def __init__(self, pooling_layer, size=(2, 2), dim_ordering='th', **kwargs):
        self.size = tuple(size)
        assert dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
        self.dim_ordering = dim_ordering
        self.input_spec = [InputSpec(ndim=4)]
        self.pooling_layer = pooling_layer
        super(WhatWhereUnPooling2D, self).__init__(**kwargs)

    def get_output_shape_for(self, input_shape):
        if self.dim_ordering == 'th':
            return (input_shape[0],
                    input_shape[1],
                    self.size[0] * input_shape[2],
                    self.size[1] * input_shape[3])
        elif self.dim_ordering == 'tf':
            return (input_shape[0],
                    self.size[0] * input_shape[1],
                    self.size[1] * input_shape[2],
                    input_shape[3])
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

    def call(self, x, mask=None):
        y = K.resize_images(x, self.size[0], self.size[1], self.dim_ordering)
        #y = T.nnet.abstract_conv.bilinear_upsampling(x, self.size[0])
        return merge([y, self.pooling_layer.where], mode='mul')

    def get_config(self):
        config = {'size': self.size}
        base_config = super(WhatWhereUnPooling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

'''
input: Network is a cnn built from conv-mp and fc-mp layers
output: out
'''

def get_trainable_layer(layers):
    for L in layers:
        if isinstance(L, Convolution2D) or isinstance(L, Dense):
            return L
    return None

def get_trainable_layers(layers):
    trainable_layers = []
    for L in layers:
        if isinstance(L, Convolution2D) or isinstance(L, Dense):
            trainable_layers.append(L)
    return trainable_layers

def get_trainable_layers_idx(layers):
    trainable_layers_idx = []
    for i in range(len(layers)):
        if isinstance(layers[i], Convolution2D) or isinstance(layers[i], Dense):
            trainable_layers_idx.append(i)
    return trainable_layers_idx

def invert_kernel(kernel, dim_ordering='th'):
    new_kernel = np.copy(kernel)
    s = new_kernel.shape

    new_kernel = new_kernel.reshape((s[1], s[0], s[2], s[3]))
    tmp = np.copy(new_kernel)

    a = kernel.shape[0]
    b = kernel.shape[1]
    w = kernel.shape[2]
    h = kernel.shape[3]

    for i in range(a):
        for j in range(b):
            tmp[j, i, :, :] = kernel[i, j, :, :]
    
    for i in range(w):
        for j in range(h):
            new_kernel[:, :, i, j] = tmp[:, :, w - i - 1, h - j - 1]
    
    '''
    if dim_ordering == 'th':
        w = kernel.shape[2]
        h = kernel.shape[3]
        for i in range(w):
            for j in range(h):
                new_kernel[:, :, i, j] = kernel[:, :, w - i - 1, h - j - 1]
    elif dim_ordering == 'tf':
        w = kernel.shape[0]
        h = kernel.shape[1]
        for i in range(w):
            for j in range(h):
                new_kernel[i, j, :, :] = kernel[w - i - 1, h - j - 1, :, :]
    else:
        raise Exception('Invalid dim_ordering: ' + str(dim_ordering))
    '''

    return new_kernel

def pretrain(network, mode='all', nb_modules=-1):
    nb_layers = len(network.layers)
    assert nb_layers > 0
    assert isinstance(network.layers[0], Input)

    ctn = 0
    rbegin = -1
    flatten_input_shape = None
    inp = Input(network.layers[0].get_config())
    out = inp

    i = 0
    for i in range(1, len(network.layers)):
        layer = network.layers[i]

        if isinstance(layer, Dense) or isinstance(layer, Convolution2D):
            if ctn == nb_layers:
                rbegin = i
                break
            ctn += 1

        if isinstance(layer, Flatten):
            flatten_input_shape = layer.input_shape

        new_layer = layer_utils.layer_from_config(layer.get_config())

        if isinstance(new_layer, Dropout):
            new_layer.p = 0.0
        out = new_layer(out)

    first_conv = True

    for j in reversed(range(rbegin + 1)):
        layer = network.layers[j]
        
        if isinstance(layer, Dense):
            out = Dense(layer.get_config())(out)
            out = Activation('linear')(out)

        if isinstance(layer, Convolution2D):
            if first_conv == True: 
                out = Reshape(flatten_input_shape)(out)

def pretrain_swwae(network, X, nb_modules=-1, batch_size=32, nb_epoch=12, nb_classes=2, bias=True, fixed_conv=False):
    nb_layers = len(network.layers)
    assert nb_layers > 2
    assert isinstance(network.layers[0], InputLayer)
    assert isinstance(network.layers[1], Convolution2D) or isinstance(network.layers[1], Dense)

    # Split network in modules
    modules = []
    i = 1
    nb_dense = 0
    while True:
        module = []
        while True:
            module.append(network.layers[i])
            if isinstance(network.layers[i], Dense):
                nb_dense += 1
            i += 1
            if  i >= nb_layers or isinstance(network.layers[i], Convolution2D) or isinstance(network.layers[i], Dense):
                break

        modules.append(module)
        if len(modules) == nb_modules or i >= nb_layers:
            break

    input_config = network.layers[0].get_config()
    encoder_input = InputLayer(**input_config)
    encoder_input = encoder_input.inbound_nodes[0].output_tensors[0]
    encoder_output = encoder_input

    if nb_modules > len(modules) or nb_modules < 0:
        print('nb_modules fixed to the existing number of convolutional modules')
        nb_modules = len(modules) - nb_dense
    print("modules {}, ae modules {}".format(modules, nb_modules))

    encoder_trainable_layers = []
    mp_layers = []
    input_shapes = [(None,) + X[0].shape]
    print('Input shape {} ...'.format(input_shapes[-1]))
    for i in range(nb_modules):
        print('Pretraining module {} ...'.format(i))
        mp_layers.append(None)
        input_shapes.append((None,))
        has_flatten = False
        convlike_input_shape = None
        conv_config = None
        
        # Create encoder
        for j in range(len(modules[i])):
            print('Layer {}, {} ...'.format(i, j))
            print(modules[i][j].__class__)

            # Use same activations objects because there is not a straightforward way to build custom activation from config
            if isinstance(modules[i][j], Activation):
                new_layer = modules[i][j]
            else:
                new_layer = layer_utils.layer_from_config({'class_name':modules[i][j].__class__, 'config':modules[i][j].get_config()})

            if isinstance(new_layer, Dropout):
                new_layer.p = 0.0
            
            if isinstance(new_layer, MaxPooling2D):
                new_layer = WhatWhereMaxPooling2D(**modules[i][j].get_config())
                new_layer.batch_size = batch_size
                mp_layers[-1] = new_layer

            tmp = encoder_output
            encoder_output = new_layer(encoder_output)
            new_layer.call(tmp)

            if isinstance(new_layer, Flatten):
                has_flatten = True
            if isinstance(new_layer, Convolution2D):
                conv_config = modules[i][j].get_config()
            if isinstance(new_layer, Convolution2D) or isinstance(new_layer, Dense):
                encoder_trainable_layers.append(new_layer) 
            if isinstance(modules[i][j], Convolution2D) or isinstance(modules[i][j], MaxPooling2D):
                convlike_input_shape = modules[i][j].output_shape
                input_shapes[-1] = modules[i][j].output_shape

        print('Input shape {} ...'.format(input_shapes[-1]))
        # Create decoder
        decoder_output = encoder_output
        decoder_layers = []
        for k in reversed(range(i + 1)):
            if has_flatten:
                print('Shape {} ...'.format((convlike_input_shape[1], convlike_input_shape[2], convlike_input_shape[3])))
                decoder_output = Reshape((convlike_input_shape[1], convlike_input_shape[2], convlike_input_shape[3]))(decoder_output)
                has_flatten = False
            if mp_layers[k] != None:
                #decoder_output = UpSampling2D(size=mp_layers[k].pool_size)(decoder_output)
                decoder_output = WhatWhereUnPooling2D(mp_layers[k], size=mp_layers[k].pool_size)(decoder_output)
            conv_config = encoder_trainable_layers[k].get_config()
            conv_config['name'] = 'dec_' + conv_config['name']
            conv_config['nb_filter'] = input_shapes[k][1]
            decoder_layer = Convolution2D(**conv_config)
            decoder_layers.append(decoder_layer)
            print('Conv input decoder_output shape {}'.format(decoder_output.shape))
            decoder_output = decoder_layer(decoder_output)
            decoder_output = Activation('linear')(decoder_output)

        input_maps = conv_config['nb_filter']

        # Train autoencoder
        model = Model(encoder_input, decoder_output)
        print("Compiling ...")
        model.compile(loss='mse', optimizer='adadelta')
        model.summary()

        for k in range(i):
            print("enc config: {}".format(encoder_trainable_layers[k].get_config()))
            print("dec config: {}".format(decoder_layers[i-k].get_config()))

            w_inverted = encoder_trainable_layers[k].get_weights()
            w_inverted[0] = invert_kernel(w_inverted[0])

            if bias:
                bias_mean = np.mean(w_inverted[1])
                w_inverted[1] = np.full(shape=(len(w_inverted[0]),), dtype='float32', fill_value=bias_mean)

            decoder_layers[i-k].set_weights(w_inverted)
            decoder_layers[i-k].trainable = False
         
        for k in range(len(model.layers)):
            if isinstance(model.layers[k], Dense) or isinstance(model.layers[k], Convolution2D):
                print("Layer {} trainable {}.".format(k, model.layers[k].trainable))

        model.fit(X, X, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1)
        encoder_trainable_layers[i].trainable = False
        print('Layer weights 0: {}'.format(encoder_trainable_layers[0].get_weights()[0][0]))

    # Copy weigths
    assert len(encoder_trainable_layers) == nb_modules
    for i in range(nb_modules):
        layer = get_trainable_layer(modules[i])
        layer.set_weights(encoder_trainable_layers[i].get_weights())
        if fixed_conv:
            layer.trainable = False
        print("fixed conv {} cnn layer {} trainable {}".format(fixed_conv, i,layer.trainable))

    return model

def clone_layer(layer):
    if isinstance(layer, WhatWhereUnPooling2D):
        new_layer = WhatWhereUnPooling2D(layer.pooling_layer, size=layer.size)
    elif isinstance(layer, Activation):
        new_layer = layer_utils.layer_from_config({'class_name':layer.__class__, 'config':layer.get_config()})
    else:
        new_layer = layer_utils.layer_from_config({'class_name':layer.__class__, 'config':layer.get_config()})
    return new_layer

def print_trainable_state(layers):
    for k in range(len(layers)):
        if isinstance(layers[k], Dense):
            print("Layer {} - {} trainable {} weight sample {}".format(k, layers[k].name, layers[k].trainable, layers[k].get_weights()[0][0][0]))
        elif isinstance(layers[k], Convolution2D):
            print("Layer {} - {} trainable {} weight sample {}".format(k, layers[k].name, layers[k].trainable, layers[k].get_weights()[0][0][0][0][0]))
    print("")

def get_unpooled_input(layers, layer_idx):
    len_layers = len(layers)
    print("Get unpooled input for {}".format(layers[layer_idx].name))
    while True:
        layer_idx -= 1
        if layer_idx < 0:
            break
        if isinstance(layers[layer_idx], WhatWhereUnPooling2D):
            print("unpooled input shape {}".format(layers[layer_idx].input_shape))
            return layers[layer_idx].get_input_at(0)
    raise Exception("There is no unpooled input")

def get_pooled_output(layers, layer_idx):
    len_layers = len(layers)
    print("Get pooled output for {}".format(layers[layer_idx].name))
    while True:
        layer_idx += 1
        if layer_idx >= len_layers:
            break
        print("Check {}".format(layers[layer_idx].name))
        if isinstance(layers[layer_idx], WhatWhereMaxPooling2D):
            print("pooled output for {}".format(layers[layer_idx].output_shape))
            return layers[layer_idx].get_output_at(0)
    raise Exception("There is no pooled output")
    
# TODO Consider macrolayer
def get_conv_input(layers, layer_idx):  
    len_layers = len(layers)
    assert layer_idx < len_layers
    return layers[layer_idx].get_input_at(0)

def get_conv_input_without_dropout(layers, layer_idx):  
    len_layers = len(layers)
    assert layer_idx < len_layers
    if layer_idx > 0 and isinstance(layers[layer_idx-1], Dropout):
        print('happens')
        return layers[layer_idx-1].get_input_at(0)
    return layers[layer_idx].get_input_at(0)

def get_conv_output(layers, layer_idx):
    len_layers = len(layers)
    assert layer_idx < len_layers
    while True:
        layer_idx += 1
        if layer_idx == len_layers:
            return layers[layer_idx-1].get_output_at(0)
        else:
            if isinstance(layers[layer_idx], Activation):
                return layers[layer_idx].get_output_at(0)
 
def get_encoder_acts(layers):
    weighted_layer_idx = []
    for i in range(len(layers)):
        layer = layers[i]
        if isinstance(layer, Convolution2D) or isinstance(layer, Dense):
            weighted_layer_idx.append(i)

    encoder_len = len(weighted_layer_idx)
    acts = []
    for i in range(encoder_len):
        #Clean input: acts.append(get_conv_input_without_dropout(layers, weighted_layer_idx[i]))
        acts.append(get_conv_input(layers, weighted_layer_idx[i]))
    return acts

def get_decoder_acts(layers):
    weighted_layer_idx = []
    for i in range(len(layers)):
        layer = layers[i]
        if isinstance(layer, Convolution2D) or isinstance(layer, Dense):
            weighted_layer_idx.append(i)

    decoder_len = len(weighted_layer_idx)
    acts = []
    for i in range(decoder_len):
        acts.append(get_conv_output(layers, weighted_layer_idx[i]))
    return list(reversed(acts))

def get_encoder_decoder_acts(model):
    weighted_layer_idx = []
    for i in range(len(model.layers)):
        layer = model.layers[i]
        if isinstance(layer, Convolution2D) or isinstance(layer, Dense):
            weighted_layer_idx.append(i)

    encoder_len = len(weighted_layers_idx) / 2
    encoder_acts = []
    decoder_acts = []
    
    for i in range(encoder_len):
        encoder_idx = i  
        decoder_idx = 2 * encoder_len - 1 - i
        encoder_acts.append(get_conv_input(model.layers, weighted_layer_idx[encoder_idx]))
        decoder_acts.append(get_conv_output(model.layers, weighted_layer_idx[decoder_idx]))
    return encoder_acts, decoder_acts
    

# Loss functions for 4D tensors
def l2_loss(y_true, y_pred):
    return T.mean(K.sqrt(K.sum(K.sum(K.sum(K.square(y_true - y_pred), axis=1, keepdims=True), axis=2, keepdims=True), axis=3, keepdims=True)))

def l22_loss(y_true, y_pred):
    return T.mean(K.sum(K.sum(K.sum(K.square(y_true - y_pred), axis=1, keepdims=True), axis=2, keepdims=True), axis=3, keepdims=True))

class IntermediaryLoss:
    def __init__(self, loss, multipliers=None):
        self.loss = loss
        self.multipliers = multipliers
    def call(self, tensors):
        assert len(tensors)%2 == 0
        size = len(tensors)/2
        if self.multipliers == None:
            self.multipliers = np.fill(shape=(size,), fill_value=1, dtype=np.float)
        else:
            assert len(self.multipliers) == size
        total_loss = self.multipliers[0] * self.loss(tensors[0], tensors[size])
        for i in range(1, size):
            total_loss += self.multipliers[i] * self.loss(tensors[i], tensors[i + size])
        print("total_loss {}".format(total_loss))
        return total_loss

def loss_output_shape(shapes):
    return shapes[0]

def average_loss(y_true,y_pred):    
    return T.mean(y_pred)

def create_and_fit(encoder_layers, decoder_layers, X, Y, supervised_output, encoder_input, optimizer, batch_size, nb_epoch, mode='all', loss=l22_loss, multipliers=None):
    print("Augmented network")
    encoder_tensors = get_encoder_acts(encoder_layers)
    decoder_tensors = get_decoder_acts(decoder_layers) 
    intermediary_loss = IntermediaryLoss(loss, multipliers).call

    if mode == 'first':
        intermediary_output = Lambda(intermediary_loss, output_shape=loss_output_shape, name='intermediary_proxy')([encoder_tensors[0], decoder_tensors[0]])
    elif mode == 'all':
        intermediary_output = Lambda(intermediary_loss, output_shape=loss_output_shape, name='intermediary_proxy')(encoder_tensors + decoder_tensors)

    new_model = Model(input=[encoder_input], output=[intermediary_output, supervised_output])
    new_model.compile(optimizer,{'intermediary_proxy':average_loss, 'supervised':'categorical_crossentropy'})
    new_model.summary()

    shape = encoder_tensors[0]._keras_shape
    shape = (X.shape[0],) + shape[1:]

    intermediary_holder = np.zeros(shape, dtype='float32')
    new_model.fit({'input_1':X},{'intermediary_proxy':intermediary_holder, 'supervised': Y}, batch_size=batch_size, nb_epoch=nb_epoch)
    return new_model, intermediary_holder

def fit_intermediary(model, X_train, Y_train, X_test, Y_test, intermediary_holder, supervised_output, encoder_input, optimizer, batch_size, nb_epoch, mode='all'):
    model.compile(optimizer,{'intermediary_proxy':average_loss, 'supervised':'categorical_crossentropy'})
    print_trainable_state(model.layers)
    tmp = np.zeros(shape=(len(Y_test),) + intermediary_holder.shape[1:])
    model.fit({'input_1':X_train},{'intermediary_proxy':intermediary_holder, 'supervised': Y_train}, batch_size=batch_size, nb_epoch=nb_epoch, 
        validation_data=({'input_1':X_test}, {'intermediary_proxy':tmp, 'supervised':Y_test}))

def swwae_augment(network, X_train, Y_train, X_test, Y_test, mode='all', nb_modules=-1, batch_size=32, layerwise_epochs=6, decoder_epochs=12, finetune_epochs=24, nb_classes=2, bias=True, fixed_encoder=True, unsupervised_loss=l22_loss, multipliers=[3e-4, 1e-5, 1e-5], lr=0.01, model_name='swwae'):
    nb_layers = len(network.layers)
    assert nb_layers > 2
    assert isinstance(network.layers[0], InputLayer)
    assert isinstance(network.layers[1], Convolution2D) or isinstance(network.layers[1], Dense)

    print("SWWAE-{} augmentation".format(mode))

    # Fix layerwise pretraining
    # optimizer = SGD(lr=lr, nesterov=True, momentum=0.9)
    optimizer = SGD(lr=lr*multipliers[0], nesterov=True, momentum=0.9)
    ''' Worst case loss contrib by the number of units VGG16
    1 input layer: 150k: 0.09 * 1e-4 : 1e-5
    2 layer : 800k: 0.5 : 1e-12 WTF! Maybe these terms are only to control that difference between layers is not absurd
    3 layer : 400k: 0.25 : 1e-12
    4 layer : 200k: 0.12 : 1e-13
    5 layer : 100k: 0.06 : 1e-11
    '''

    ''' Worst case loss contrib by the number of units in 3P
    Option 1
    1 input: 32 * 32: 0.04 * 3e-4       0.0000120
    2 layer: 16 * 16 * 64: 0.64 * 1e-5  0.0000064
    3 layer: 8 * 8 * 128: 0.32 * 1e-5   0.0000032

    Option 2
    1 input: 32 * 32: 0.04 * 1e-3       0.0000400
    2 layer: 16 * 16 * 64: 0.64 * 1e-5  0.0000064
    3 layer: 8 * 8 * 128: 0.32 * 1e-5   0.0000032
    '''

    #multipliers = [3e-4, 1e-5, 1e-5] last used
    #multipliers = [1e-3, 1e-5, 1e-5]

    supervised_output = network.layers[-1].get_output_at(0)
    #network.layers[-1].name = 'supervised'

    # Set input
    input_config = network.layers[0].get_config()
    input_layer = InputLayer(**input_config)
    input_layer.name = 'input_1'

    # Group layers in modules
    layers_idx_by_module = []
    i = 1
    while True:
        print("i {}".format(i))
        layers_idx = []
        while i < nb_layers:
            layers_idx.append(i)
            next_is_dense = i + 1 < nb_layers and isinstance(network.layers[i + 1], Dense)
            if  (i + 1 < nb_layers and network.layers[i+1].__class__ in {Flatten, Convolution2D}) or (not isinstance(network.layers[i], Flatten) and next_is_dense):
                i += 1
                break
            i += 1 
        layers_idx_by_module.append(layers_idx)
        if not i < nb_layers:
            break

    print('modules {}'.format(layers_idx_by_module))
    for i in range(len(network.layers)):
        print('{} -> {}'.format(i, network.layers[i].name))
        
    # Clone supervised pathway
    nb_dense = 0
    out = input_layer.inbound_nodes[0].output_tensors[0]
    classification_layers = [input_layer]
    for i in range(1, len(network.layers)):  
        if isinstance(network.layers[i], MaxPooling2D):
            new_layer = WhatWhereMaxPooling2D(**network.layers[i].get_config())
            new_layer.batch_size = batch_size
        else:
            new_layer = clone_layer(network.layers[i])

        out = new_layer(out)

        if isinstance(new_layer, Convolution2D) or isinstance(new_layer, Dense):
            new_layer.set_weights(network.layers[i].get_weights())
            new_layer.trainable = False
        if isinstance(new_layer, Dense):
            nb_dense += 1
        classification_layers.append(new_layer)

    supervised_output = classification_layers[-1].get_output_at(0)
    classification_layers[-1].name = 'supervised'

    print_trainable_state(classification_layers)

    if nb_modules > len(layers_idx_by_module) or nb_modules < 0:
        print('nb_modules fixed to the existing number of convolutional modules')
        nb_modules = len(layers_idx_by_module) - nb_dense
    print("modules {}, ae modules {}".format(len(layers_idx_by_module), nb_modules))


    # Build reconstruction pathways
    encoder_input = input_layer.inbound_nodes[0].output_tensors[0]
    encoder_output = encoder_input
    encoder_layers_idx = []

    decoder_layers = []
    decoder_output = None

    input_shapes = []
    input_shapes.append((None,) + X_train[0].shape)
    encoder_outputs = []

    encoder_inputs = get_encoder_acts(classification_layers)
    for i in range(nb_modules):
        print('Pretraining module {} ...'.format(i))
        has_flatten = False
        mp_layer_idx = -1
        convlike_input_shape = None
        conv_config = None

        module_layers_idx = layers_idx_by_module[i] 
        encoder_layers_idx += module_layers_idx
        print("Module layers idx {}".format(module_layers_idx))
        # Layerwise initialization
        for j in range(len(module_layers_idx)):
            layer_idx = layers_idx_by_module[i][j]
            layer = classification_layers[layer_idx]
            print('Layer {}, {} class {} ...'.format(i, j, layer.__class__))

            if isinstance(layer, Flatten):
                has_flatten = True

            if isinstance(layer, Convolution2D):
                conv_config = layer.get_config()
                input_shapes.append(layer.output_shape)
                convlike_input_shape = layer.output_shape

            if isinstance(layer, Dense):
                input_shapes.append(layer.output_shape)

            if isinstance(layer, WhatWhereMaxPooling2D):
                convlike_input_shape = layer.output_shape
                mp_layer_idx = module_layers_idx[j]

        # Augment decoder
        print('Decoder input shape (no mp): {} ...'.format(input_shapes[-1]))
        decoder_output = classification_layers[layers_idx_by_module[i][-1]].get_output_at(0)

        decoder_layer = None
        decoder_module_layers = []

        if has_flatten:
            decoder_layer = Reshape((convlike_input_shape[1], convlike_input_shape[2], convlike_input_shape[3]))
            decoder_output = decoder_layer(decoder_output)
            decoder_module_layers.append(decoder_layer)

        if mp_layer_idx != -1:
            pooling_layer = classification_layers[mp_layer_idx]
            decoder_layer = WhatWhereUnPooling2D(pooling_layer, size=pooling_layer.pool_size)
            decoder_output = decoder_layer(decoder_output)
            decoder_module_layers.append(decoder_layer)

        for j in reversed(range(len(module_layers_idx))):
            trainable_layer = classification_layers[module_layers_idx[j]]
            if not isinstance(trainable_layer, Convolution2D):
                continue

            conv_config = trainable_layer.get_config()
            conv_config['name'] = 'dec_' + conv_config['name']
            input_shape = network.layers[module_layers_idx[j]].input_shape
            conv_config['nb_filter'] = input_shape[1]

            decoder_layer = Convolution2D(**conv_config)
            decoder_layer.trainable = True
            decoder_output = decoder_layer(decoder_output)
            decoder_module_layers.append(decoder_layer)

            # TODO LeakyReLu
            if not (i == 0 and j + 1 == len(module_layers_idx)):
                decoder_layer = Activation('relu')
                decoder_output = decoder_layer(decoder_output)
                decoder_module_layers.append(decoder_layer)

        # Clone previous decoder layers, TODO: free memory
        cloned_decoder_layers = []
        for layer in decoder_layers:
            decoder_layer = clone_layer(layer)    
            print('Trainable {}'.format(decoder_layer.trainable))
            decoder_layer.trainable = True
            decoder_output = decoder_layer(decoder_output)
            decoder_layer.set_weights(layer.get_weights())
            decoder_layer.trainable = False
            cloned_decoder_layers.append(decoder_layer)

        decoder_layers = decoder_module_layers + cloned_decoder_layers
        decoder_layers[-1].name = 'unsupervised'
        '''
        # Fixed layerwise pretraining + Induce denoising on decoder pathway
        decoder_outputs = get_decoder_acts(decoder_layers)
        layerwise_loss = IntermediaryLoss(unsupervised_loss, [multipliers[0]]).call
        print('{} {} '.format(encoder_inputs[i]._keras_shape, decoder_outputs[i]._keras_shape))
        layerwise_output = Lambda(layerwise_loss, output_shape=loss_output_shape, name='layerwise_proxy')([encoder_inputs[i], decoder_outputs[i]])
        model = Model(input=[encoder_input], output=[layerwise_output])
        model.compile(optimizer,{'layerwise_proxy':average_loss})
        model.summary()
        shape = encoder_inputs[i]._keras_shape
        shape = (X_train.shape[0],) + shape[1:]
        layerwise_holder = np.zeros(shape, dtype='float32')
        model.fit({'input_1':X_train},{'layerwise_proxy':layerwise_holder}, batch_size=batch_size, nb_epoch=layerwise_epochs)
        '''
        model = Model(input=encoder_input, output=[supervised_output, decoder_output])
        model.compile(loss={'supervised':'categorical_crossentropy', 'unsupervised':unsupervised_loss}, optimizer=optimizer)
        model.summary()
        model.fit(X_train, {'unsupervised':X_train, 'supervised':Y_train}, batch_size=batch_size, nb_epoch=layerwise_epochs, verbose=1)
        print_trainable_state(model.layers)

        # Re-init module input shapes
        tmp = input_shapes[-1]
        input_shapes = [] 
        input_shapes.append(tmp)


    optimizer = SGD(lr=lr, nesterov=True, momentum=0.9)
    # For all/first modes
    print("Finetune decoder with mode {}".format(mode))
    helper_model = None
    dummy_output = None

    for layer in decoder_layers:
        if isinstance(layer, Dense) or isinstance(layer, Convolution2D):
            layer.trainable = True

    encoder_layers = np.array(model.layers)[encoder_layers_idx]

    if unsupervised_loss == l22_loss:
        #model.compile(loss=average_loss, optimizer=optimizer)
        print_trainable_state(model.layers)
        helper_model, intermediary_output = create_and_fit(encoder_layers, decoder_layers, X_train, Y_train, supervised_output, encoder_input, optimizer, batch_size, decoder_epochs, mode=mode, loss=unsupervised_loss, multipliers=multipliers)
        print_trainable_state(helper_model.layers)
        #model.fit(X, X, batch_size=batch_size, nb_epoch=decoder_epochs, verbose=1)
    else:
        model.compile(loss={'supervised':'categorical_crossentropy', 'unsupervised':unsupervised_loss}, optimizer=optimizer)
        print_trainable_state(model.layers)
        model.fit(X_train, {'unsupervised':X_train, 'supervised':Y_train}, batch_size=batch_size, nb_epoch=decoder_epochs, verbose=1)
        print_trainable_state(model.layers)

    # Finetune encoder/decoder with reduced learning rate
    print("Finetune encoder/decoder")
    optimizer.lr = lr/10.0
    for layer in model.layers:
        if isinstance(layer, Dense) or isinstance(layer, Convolution2D):
            layer.trainable = True

    if helper_model != None:
        print("Helper model")
        fit_intermediary(helper_model, X_train, Y_train, X_test, Y_test, intermediary_output, supervised_output, encoder_input, optimizer, batch_size, finetune_epochs, mode=mode)
         
        ae = Model(encoder_input, decoder_output)
        ae.compile(loss='mse', optimizer=optimizer)
        X_test_rec = ae.predict(X_test)
        print("X test rec {} {}".format(len(X_test_rec), X_test_rec[0].shape))
        for i in range(10):
            util.imwrite("data/{}_{}.jpg".format(model_name, i), X_test[i][0])
            util.imwrite("data/{}_{}_rec.jpg".format(model_name, i), X_test_rec[i][0])
            
        print_trainable_state(helper_model.layers)
    else:
        print("Normal model")
        model.compile(loss={'supervised':'categorical_crossentropy', 'unsupervised':'mse'}, optimizer=optimizer)
        model.fit(X_train, {'unsupervised':X_train, 'supervised':Y_train}, batch_size=batch_size, nb_epoch=finetune_epochs, verbose=1)

    # Copy weigths
    idx = 0
    print("Copy weights")
    assert len(network.layers) == len(classification_layers)
    print_trainable_state(network.layers)
    for i in range(len(network.layers)):
        if network.layers[i].trainable:
            network.layers[i].set_weights(classification_layers[i].get_weights())
    print_trainable_state(network.layers)

    return model, helper_model

def pretrain_layerwise(network, X, nb_modules=-1, batch_size=32, nb_epoch=12, nb_classes=2, bias=True):
    nb_layers = len(network.layers)
    assert nb_layers > 1
    print(network.layers[0].__class__)
    print("nb layers {}".format(len(network.layers)))
    assert isinstance(network.layers[0], InputLayer)
    assert isinstance(network.layers[1], Convolution2D) or isinstance(network.layers[1], Dense)

    # Split network in modules
    modules = []
    i = 1
    nb_dense = 0
    while True:
        module = []
        while True:
            module.append(network.layers[i])
            if isinstance(network.layers[i], Dense):
                nb_dense += 1
            i += 1
            if  i >= nb_layers or isinstance(network.layers[i], Convolution2D) or isinstance(network.layers[i], Dense):
                break

        modules.append(module)
        if len(modules) == nb_modules or i >= nb_layers:
            break

    input_config = network.layers[0].get_config()
    print(input_config)
    encoder_input = InputLayer(**input_config)
    print('Input layer {}'.format(type(encoder_input)))
    encoder_input = encoder_input.inbound_nodes[0].output_tensors[0]
    encoder_output = encoder_input

    if nb_modules > len(modules) or nb_modules < 0:
        print('nb_modules fixed to the existing number of convolutional modules')
        nb_modules = len(modules) - nb_dense
    print("modules {}, ae modules {}".format(modules, nb_modules))

    encoder_trainable_layers = []
    has_mp = []
    input_shapes = [(None,) + X[0].shape]
    print('Input shape {} ...'.format(input_shapes[-1]))
    for i in range(nb_modules):
        print('Pretraining module {} ...'.format(i))
        has_mp.append(False)
        input_shapes.append((None,))
        has_flatten = False
        convlike_input_shape = None
        conv_config = None
        
        # Create encoder
        for j in range(len(modules[i])):
            print('Layer {},{} ...'.format(i, j))
            print(modules[i][j].__class__)
            print(modules[i][j].get_config())

            # Use same activations objects because there is not a straightforward way to build custom activation from config
            if isinstance(modules[i][j], Activation):
                new_layer = modules[i][j]
            else:
                new_layer = layer_utils.layer_from_config({'class_name':modules[i][j].__class__, 'config':modules[i][j].get_config()})

            if isinstance(new_layer, Dropout):
                new_layer.p = 0.0

            encoder_output = new_layer(encoder_output)
            if isinstance(new_layer, MaxPooling2D):
                has_mp[-1] = True
            if isinstance(new_layer, Flatten):
                has_flatten = True
            if isinstance(new_layer, Convolution2D):
                conv_config = modules[i][j].get_config()
            if isinstance(new_layer, Convolution2D) or isinstance(new_layer, Dense):
                encoder_trainable_layers.append(new_layer) 
            if isinstance(modules[i][j], Convolution2D) or isinstance(modules[i][j], MaxPooling2D):
                convlike_input_shape = modules[i][j].output_shape
                input_shapes[-1] = modules[i][j].output_shape

        print('Input shape {} ...'.format(input_shapes[-1]))

        # Create decoder
        decoder_output = encoder_output
        decoder_layers = []
        for k in reversed(range(i + 1)):
            if has_flatten:
                print('Shape {} ...'.format((convlike_input_shape[1], convlike_input_shape[2], convlike_input_shape[3])))
                decoder_output = Reshape((convlike_input_shape[1], convlike_input_shape[2], convlike_input_shape[3]))(decoder_output)
                has_flatten = False
            if has_mp[k]:
                decoder_output = UpSampling2D()(decoder_output)
            conv_config = encoder_trainable_layers[k].get_config()
            conv_config['name'] = 'dec_' + conv_config['name']
            conv_config['nb_filter'] = input_shapes[k][1]
            decoder_layer = Convolution2D(**conv_config)
            decoder_layers.append(decoder_layer)
            decoder_output = decoder_layer(decoder_output)
            decoder_output = Activation('linear')(decoder_output)

        input_maps = conv_config['nb_filter']

        # Train autoencoder
        model = Model(encoder_input, decoder_output)
        model.summary()
        model.compile(loss='mse', optimizer='adadelta')

        for k in range(i):
            print("enc config: {}".format(encoder_trainable_layers[k].get_config()))
            print("dec config: {}".format(decoder_layers[i-k].get_config()))

            w_inverted = encoder_trainable_layers[k].get_weights()
            w_inverted[0] = invert_kernel(w_inverted[0])

            if bias:
                bias_mean = np.mean(w_inverted[1])
                w_inverted[1] = np.full(shape=(len(w_inverted[0]),), dtype='float32', fill_value=bias_mean)

            decoder_layers[i-k].set_weights(w_inverted)
            decoder_layers[i-k].trainable = False
         
        for k in range(len(model.layers)):
            if isinstance(model.layers[k], Dense) or isinstance(model.layers[k], Convolution2D):
                print("Layer {} trainable {}.".format(k, model.layers[k].trainable))

        model.fit(X, X, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1)
        encoder_trainable_layers[i].trainable = False
        print('Layer weights 0: {}'.format(encoder_trainable_layers[0].get_weights()[0][0]))

    # Copy weigths
    assert len(encoder_trainable_layers) == nb_modules
    for i in range(nb_modules):
        layer = get_trainable_layer(modules[i])
        layer.set_weights(encoder_trainable_layers[i].get_weights())
        print("cnn layer {} trainable {}".format(i,layer.trainable))

    return model


from keras import backend as K

# Models 
def base(img_rows, img_cols, nb_classes, bias=True):
    nb_filters = 32
    nb_pool = 2
    nb_conv = 3
    inp = Input(shape=(1, img_rows, img_cols), dtype='float32')   
    out = Convolution2D(nb_filters, nb_conv, nb_conv, border_mode='same', bias=bias)(inp)
    out = Activation('relu')(out)
    out = Convolution2D(nb_filters, nb_conv, nb_conv, border_mode='same', bias=bias)(out)
    out = Activation('relu')(out)
    out = MaxPooling2D(pool_size=(nb_pool, nb_pool))(out)
    out = Dropout(0.25)(out)

    out = Flatten()(out)
    out = Dense(128, bias=bias)(out)
    out = Activation('relu')(out)
    out = Dropout(0.5)(out)
    out = Dense(nb_classes)(out)
    out = Activation('softmax')(out)

    return Model(input=inp, output=out)
    
# See: Stacked Convolutional Autoencoders for Hierachical Feature Extraction
def scaled_tanh(x):
    return 1.7159 * K.tanh( 0.6666 *x)

def masci(img_rows, img_cols, nb_classes, bias=True):
    nb_pool = 2
    nb_classes = 10
    activation = "relu"#scaled_tanh
    inp = Input(shape=(1, img_rows, img_cols), dtype='float32')   
    out = Convolution2D(100, 5, 5, border_mode='same', bias=bias)(inp)
    out = Activation(activation)(out)
    out = MaxPooling2D(pool_size=(nb_pool, nb_pool))(out)
    out = Convolution2D(150, 5, 5, border_mode='same', bias=bias)(out)
    out = Activation(activation)(out)
    out = MaxPooling2D(pool_size=(nb_pool, nb_pool))(out)
    out = Convolution2D(200, 3, 3, border_mode='same', bias=bias)(out)
    out = Activation(activation)(out)
    out = MaxPooling2D(pool_size=(nb_pool, nb_pool))(out)
    out = Flatten()(out)
    out = Dense(300, bias=bias)(out)
    out = Activation(activation)(out)
    out = Dense(nb_classes)(out)
    out = Activation('softmax')(out)

    return Model(input=inp, output=out)

# SWWAE network: 
def swwae_mnist(img_rows, img_cols, nb_classes, bias=True):
    nb_pool = 2
    nb_classes = 10
    activation = "relu"#scaled_tanh
    inp = Input(shape=(1, img_rows, img_cols), dtype='float32')   
    out = Convolution2D(64, 5, 5, border_mode='same', bias=bias)(inp)
    out = Activation(activation)(out)
    out = MaxPooling2D(pool_size=(nb_pool, nb_pool))(out)
    out = Convolution2D(64, 3, 3, border_mode='same', bias=bias)(out)
    out = Activation(activation)(out)
    out = MaxPooling2D(pool_size=(nb_pool, nb_pool))(out)
    out = Convolution2D(64, 3, 3, border_mode='same', bias=bias)(out)
    out = Activation(activation)(out)
    out = MaxPooling2D(pool_size=(nb_pool, nb_pool))(out)
    out = Flatten()(out)
    out = Dense(nb_classes)(out)
    out = Activation('softmax')(out)

    return Model(input=inp, output=out)

def swwae_reconstruction(img_rows, img_cols, nb_classes, bias=True):
    nb_pool = 2
    nb_classes = 10
    activation = "relu"#scaled_tanh
    inp = Input(shape=(1, img_rows, img_cols), dtype='float32')   
    out = Convolution2D(16, 5, 5, border_mode='same', bias=bias)(inp)
    out = Activation(activation)(out)
    out = Convolution2D(32, 3, 3, border_mode='same', bias=bias)(out)
    out = Activation(activation)(out)
    out = MaxPooling2D(pool_size=(nb_pool, nb_pool))(out)
    out = Flatten()(out)
    out = Dense(32, bias=bias)(out)
    out = Activation(activation)(out)
    out = Dense(nb_classes)(out)
    out = Activation('softmax')(out)

    return Model(input=inp, output=out)


def load_mnist(img_cols, img_rows, nb_classes):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    tmp = []
    for x in X_train:
        tmp.append(imresize(x, (32, 32)))
    X_train = np.array(tmp)
    tmp = []
    for x in X_test:
        tmp.append(imresize(x, (32, 32)))
    X_test = np.array(tmp)
    print("shapes {} {}".format(X_train.shape, X_test.shape))

    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    return (X_train, Y_train), (X_test, Y_test)

def test_masci():
    bias = True
    batch_size = 100
    nb_epoch = 1
    nb_epoch_sup = 10
    nb_classes = 10
    img_rows, img_cols = 32, 32
    small_set_len = 1000 
    
    (X_train, Y_train), (X_test, Y_test) = load_mnist(img_rows, img_cols, nb_classes)

    print("X shape {}".format(X_train.shape))
    X_train_small = X_train[range(small_set_len)]
    Y_train_small = Y_train[range(small_set_len)]
    print("X shape {}".format(X_train.shape))

    #model = base(img_rows, img_cols, nb_classes, bias)
    model = masci(img_rows, img_cols, nb_classes, bias)
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    pretrain_layerwise(model, X_train, 3, batch_size, nb_epoch, nb_classes)

    print("Sup: layer weigths {}".format(model.layers[1].get_weights()[0][0]))
    model.fit(X_train_small, Y_train_small, batch_size=batch_size, nb_epoch=nb_epoch_sup, verbose=1, validation_data=(X_test, Y_test))
    score = model.evaluate(X_test, Y_test, verbose=0)

    print('Test cae score:', score[0])
    print('Test cae accuracy:', score[1])

def test_unsupervised_pretraining(mode='cae', unsup_epochs=1, sup_epochs=30, small_set_len=1000, fixed_conv=False, finetune=True):
    bias = True
    batch_size = 100
    nb_classes = 10
    img_rows, img_cols = 32, 32
 
    (X_train, Y_train), (X_test, Y_test) = load_mnist(img_rows, img_cols, nb_classes)

    print("X shape {}".format(X_train.shape))
    X_train_small = X_train[range(small_set_len)]
    Y_train_small = Y_train[range(small_set_len)]
    print("X shape {}".format(X_train.shape))

    #model = where_mnist_model(img_rows, img_cols, nb_classes, bias)
    #model = masci(img_rows, img_cols, nb_classes, bias)
    model = swwae_mnist(img_rows, img_cols, nb_classes, bias)
    optimizer = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    ae = None
    if mode == 'cae':
        ae = pretrain_layerwise(model, X_train, 2, batch_size, unsup_epochs, nb_classes, bias=bias)
    elif mode == 'swwae':
        print('fixed_conv {}'.format(fixed_conv))
        ae = pretrain_swwae(model, X_train, 2, batch_size, unsup_epochs, nb_classes, bias=bias, fixed_conv=fixed_conv)
    elif mode == 'ortho':
        print("No pretraining")
    else:
        raise Exception("Undefined mode: {}".format(mode))

    '''
    # Test what-where pooling-unpooling
    X_test_sample = X_test[np.random.uniform(0, len(X_test), size=(100)).astype(np.int)]
    X_rec = ae.predict(X_test_sample)
    
    for i in range(len(X_rec)):
        util.imwrite("{}_where_cae.jpg".format(i), X_test_sample[i][0])
        util.imwrite("{}_where_cae_rec.jpg".format(i), X_rec[i][0])
    '''

    print("Unsupervised weights")
    for layer in model.layers:
        if isinstance(layer, Convolution2D):
            print('Layer (trainable: {}) values {}'.format(layer.trainable, layer.get_weights()[0][0][0]))
            break

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.fit(X_train_small, Y_train_small, batch_size=batch_size, nb_epoch=sup_epochs, verbose=1, validation_data=(X_test, Y_test))
    score = model.evaluate(X_test, Y_test, verbose=0)
    print("Mode: {}".format(mode))
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    print("Supervised weights")
    for layer in model.layers:
        if isinstance(layer, Convolution2D):
            print('Layer values {}'.format(layer.get_weights()[0][0][0]))
            break

    if finetune:
        optimizer.lr.set_value(0.1)
        for layer in model.layers:
            if isinstance(layer, Convolution2D):
                print('Check trainable layers {}'.format(layer.trainable))
                layer.trainable = True

        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        model.fit(X_train_small, Y_train_small, batch_size=batch_size, nb_epoch=sup_epochs, verbose=1, validation_data=(X_test, Y_test))
        score = model.evaluate(X_test, Y_test, verbose=0)
        print("Finetune:")
        print('Test score:', score[0])
        print('Test accuracy:', score[1])

        print('\nFinetuned weights')
        for layer in model.layers:
            if isinstance(layer, Convolution2D):
                print('Layer values {}'.format(layer.get_weights()[0][0][0]))
                break

    return score[1]

def test_compare_unsups():
    modes = ['ortho', 'cae', 'swwae']
    unsup_epochs = [1, 2, 4, 8, 16] 
    sup_epochs = [2, 4, 8, 16, 32]
    sup_set_lens = [325, 750, 1500, 3000, 6000]
    print('experiment, pretraining mode, sup set len, unsup epochs, sup epochs, accuracy')
    for mode, sup_len, unsup, sup in product(modes, sup_set_lens, unsup_epochs, sup_epochs):
        np.random.seed(1337)
        acc = test_unsupervised_pretraining(mode=mode, unsup_epochs=unsup, sup_epochs=sup, small_set_len=sup_len)
        print('detailed comparison,{},{},{},{},{}'.format(mode, sup_len, unsup, sup, acc))

def test_augment_unsup(mode='layerwise', unsup_epochs=1, sup_epochs=10, small_set_len=1000, unsupervised_loss=l22_loss):
    bias = True
    batch_size = 100
    nb_classes = 10
    img_rows, img_cols = 32, 32
 
    (X_train, Y_train), (X_test, Y_test) = load_mnist(img_rows, img_cols, nb_classes)

    print("X shape {}".format(X_train.shape))
    X_train_small = X_train[range(small_set_len)]
    Y_train_small = Y_train[range(small_set_len)]
    print("X shape {}".format(X_train.shape))

    model = swwae_mnist(img_rows, img_cols, nb_classes, bias)

    # Supervised initialization
    print(">> Supervised initialization:")
    optimizer = SGD(lr=0.1)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.fit(X_train_small, Y_train_small, batch_size=batch_size, nb_epoch=sup_epochs, verbose=1, validation_data=(X_test, Y_test))
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('>> Test score:', score[0])
    print('>> Test accuracy:', score[1])
    print_trainable_state(model.layers)
    # SWWAE augment
    if mode in {'layerwise', 'all', 'first'}:
        print(">> SWWAE-augment")
        #swwae_augment(model, X_train, Y_train, mode=mode, nb_epoch=unsup_epochs, loss=loss)
        #score = model.evaluate(X_test, Y_test, verbose=0)
        swwae_augment(model, X_train_small, Y_train_small, mode=mode, nb_epoch=unsup_epochs, unsupervised_loss=unsupervised_loss)
        score = model.evaluate(X_test, Y_test, verbose=0)
        print('>> Test score:', score[0])
        print('>> Test accuracy:', score[1])
        print_trainable_state(model.layers)

    # Supervised finetuning
    print(">> Supervised finetuning:")
    optimizer = SGD(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.fit(X_train_small, Y_train_small, batch_size=batch_size, nb_epoch=sup_epochs, verbose=1, validation_data=(X_test, Y_test))
    print_trainable_state(model.layers)
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('>> Test score:', score[0])
    print('>> Test accuracy:', score[1])
    return score[1]

if __name__ == "__main__":
    #test_compare_unsups()
    unsup_epochs = [8] * 5
    sup_epochs=20

    accs = []
    for i in range(len(unsup_epochs)):
        #np.random.seed(1337)
        print(">> Unsup {} sup {}".format(unsup_epochs[i], sup_epochs))
        acc = test_augment_unsup(mode='all', unsup_epochs=unsup_epochs[i], sup_epochs=sup_epochs, small_set_len=60000, unsupervised_loss=l22_loss)
        accs.append(acc)
        #acc = test_augment_unsup(mode='none', unsup_epochs=2, sup_epochs=20, small_set_len=1000)
        #acc = test_unsupervised_pretraining(mode='swwae', unsup_epochs=2, sup_epochs=1, small_set_len=1000)
    accs = np.array(accs)
    print("accs mean {}, accs std {}".format(accs.mean(), accs.std()))
