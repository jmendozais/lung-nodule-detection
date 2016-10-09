from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.engine import Layer
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

class KMaxPooling2D(_Pooling2D):
    def __init__(self, pool_size=(2, 2), strides=None, border_mode='valid',
                 dim_ordering='th', **kwargs):
        super(KMaxPooling2D, self).__init__(pool_size, strides, border_mode,
                                           dim_ordering, **kwargs)

    def _pooling_function(self, inputs, pool_size, strides,
                          border_mode, dim_ordering):
        output = K.pool2d(inputs, pool_size, strides,
                          border_mode, dim_ordering, pool_mode='max')
        print("keras OUT TYPE", type(output))
        #print("keras OUT SHAPE", output.shape.eval())
        return output

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
        #y = K.resize_images(x, self.size[0], self.size[1], self.dim_ordering)
        y = T.nnet.abstract_conv.bilinear_upsampling(x, self.size[0])
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

def pretrain_swwae(network, X, nb_modules=-1, batch_size=32, nb_epoch=12, nb_classes=2, bias=True):
    nb_layers = len(network.layers)
    assert nb_layers > 2
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
            
            if isinstance(new_layer, MaxPooling2D):
                new_layer = WhatWhereMaxPooling2D(**modules[i][j].get_config())
                #new_layer = KMaxPooling2D(**modules[i][j].get_config())
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
        print("cnn layer {} trainable {}".format(i,layer.trainable))

    return model


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
def where_mnist_model(img_rows, img_cols, nb_classes, bias=True):
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

def test_unsupervised_pretraining(mode='cae', unsup_epochs=1, sup_epochs=30, small_set_len=1000):
    bias = True
    batch_size = 100
    nb_classes = 10
    img_rows, img_cols = 32, 32
    small_set_len = 1000 
 
    (X_train, Y_train), (X_test, Y_test) = load_mnist(img_rows, img_cols, nb_classes)

    print("X shape {}".format(X_train.shape))
    X_train_small = X_train[range(small_set_len)]
    Y_train_small = Y_train[range(small_set_len)]
    print("X shape {}".format(X_train.shape))

    #model = where_mnist_model(img_rows, img_cols, nb_classes, bias)
    model = masci(img_rows, img_cols, nb_classes, bias)
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    ae = None
    if mode == 'cae':
        ae = pretrain_layerwise(model, X_train, 2, batch_size, unsup_epochs, nb_classes)
    elif mode == 'swwae':
        ae = pretrain_swwae(model, X_train, 2, batch_size, unsup_epochs, nb_classes)
    elif mode == 'ortho':
        print("No pretraiing")
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

    model.fit(X_train_small, Y_train_small, batch_size=batch_size, nb_epoch=sup_epochs, verbose=1, validation_data=(X_test, Y_test))
    score = model.evaluate(X_test, Y_test, verbose=0)
    print("Mode: {}".format(mode))
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    return score[1]

if __name__ == "__main__":
    modes = ['ortho', 'cae', 'swwae']
    unsup_epochs = [1, 2, 4, 8] 
    sup_epochs = [2, 4, 8, 16, 32]
    for mode, unsup, sup in product(modes, unsup_epochs, sup_epochs):
        acc = test_unsupervised_pretraining(mode=mode, unsup_epochs=unsup, sup_epochs=sup)
        print('pretraining mode {}, unsup {}, sup {}, acc={}'.format(mode, unsup, sup, acc))
        
    '''
    CAE: 0.87, 0.8549, 0.945
    SWWAE: 0.86, 0.8649, 0.946
    '''
