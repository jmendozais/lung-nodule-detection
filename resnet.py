from __future__ import print_function
import numpy as np
import keras
from keras.layers import Input, merge
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers import BatchNormalization, Flatten, Dense, Dropout, Activation, MaxoutDense
from keras.models import Model, Sequential
from keras.models import model_from_json
from keras.layers.advanced_activations import LeakyReLU, PReLU

from keras.optimizers import SGD, Adadelta, Adagrad, Adam
from keras.utils import np_utils, generic_utils
from keras.regularizers import l1_l2
from six.moves import range


init = 'he_normal'
l1 = 0.
l2 = 0.0001
def bottleneck(input, nb_filters, stride, increase_dim=True):

    nb_bottleneck_filters = nb_filters / 4

    if not increase_dim:
        output = BatchNormalization()(input)
        output = Activation('relu')(output)
        output = Convolution2D(nb_bottleneck_filters, 1, 1, subsample=(stride, stride), border_mode='same', init=init, W_regularizer=l1_l2(l1, l2))(output)

        output = BatchNormalization()(output)
        output = Activation('relu')(output)
        output = Convolution2D(nb_bottleneck_filters, 3, 3, subsample=(1, 1), border_mode='same', init=init, W_regularizer=l1_l2(l1, l2))(output)

        output = BatchNormalization()(output)
        output = Activation('relu')(output)
        output = Convolution2D(nb_filters, 1, 1, subsample=(1, 1), border_mode='same', init=init, W_regularizer=l1_l2(l1, l2))(output)

        # Implicit identity mapping
        output = [input, output]
        output = merge(output, 'sum')
    else:
        block_input = BatchNormalization()(input)
        block_input = Activation('relu')(block_input)

        output = Convolution2D(nb_bottleneck_filters, 1, 1, subsample=(stride, stride), border_mode='same', init=init, W_regularizer=l1_l2(l1, l2))(block_input)

        output = BatchNormalization()(output)
        output = Activation('relu')(output)
        output = Convolution2D(nb_bottleneck_filters, 3, 3, subsample=(1, 1), border_mode='same', init=init, W_regularizer=l1_l2(l1, l2))(output)

        output = BatchNormalization()(output)
        output = Activation('relu')(output)
        output = Convolution2D(nb_filters, 1, 1, subsample=(1, 1), border_mode='same', init=init, W_regularizer=l1_l2(l1, l2))(output)

        # Conv 1x1 shortcut
        block_input = Convolution2D(nb_filters, 1, 1, subsample=(stride, stride), border_mode='same', init=init, W_regularizer=l1_l2(l1, l2))(block_input)

        output = [block_input, output]
        output = merge(output, 'sum')

    return output


def layer(block, input, nb_filters, count, stride):
    assert count > 0
    output = block(input, nb_filters, stride, increase_dim=True)
    for i in range(count-1):
        output = block(output, nb_filters, 1, increase_dim=False)
    return output


def resnet_cifar10(input_shape, nb_classes, depth):
    input = Input(shape=input_shape, dtype='float32', name='input_layer')
    repeats = (depth - 2) / 9
    nb_filters = [16, 64, 128, 256]
    output = Convolution2D(nb_filters[0], 3, 3, subsample=(1, 1), border_mode='same', W_regularizer=l1_l2(l1, l2))(input)
    output = layer(bottleneck, output, nb_filters[1], repeats, 1)
    output = layer(bottleneck, output, nb_filters[2], repeats, 2)
    output = layer(bottleneck, output, nb_filters[3], repeats, 2)
    output = BatchNormalization()(output)
    output = Activation('relu')(output)
    output = AveragePooling2D((8, 8))(output)
    output = Flatten()(output)
    output = Dense(nb_classes, init=init, W_regularizer=l1_l2(l1, l2))(output)

    return Model(input=input, output=output)


if __name__ == '__main__':
    from keras.datasets import cifar10
    from keras.preprocessing.image import ImageDataGenerator
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation, Flatten
    from keras.layers import Convolution2D, MaxPooling2D
    from keras.optimizers import SGD
    from keras.utils import np_utils

    batch_size = 128
    nb_classes = 10
    nb_epoch = 200
    data_augmentation = True

    # input image dimensions
    img_rows, img_cols = 32, 32
    # the CIFAR10 images are RGB
    img_channels = 3

    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    model = resnet_cifar10((3, 32, 32), 10, 11)

    from keras.utils.visualize_util import plot
    plot(model, to_file='resnet.png', show_shapes=True)
    # let's train the model using SGD + momentum (how original).
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(X_train, Y_train,
                  batch_size=batch_size,
                  nb_epoch=nb_epoch,
                  validation_data=(X_test, Y_test),
                  shuffle=True)
    else:
        print('Using real-time data augmentation.')

        # this will do preprocessing and realtime data augmentation
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=True,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images

        # compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied)
        datagen.fit(X_train)

        # fit the model on the batches generated by datagen.flow()
        model.fit_generator(datagen.flow(X_train, Y_train,
                            batch_size=batch_size),
                            samples_per_epoch=X_train.shape[0],
                            nb_epoch=nb_epoch,
                            validation_data=(X_test, Y_test))    
