import numpy as np
import neural 
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD

def transfer_with_imagenet(network_name):
    network = neural.create_network(network_name, (1, 1))

    # train
    train_datagen = ImageDataGenerator(
        rescale=1./255, 
        horizontal_flip=True)

    train_generator = train_datagen.flow_from_directory(
        '/home/juliomb/dbs/ILSVRC2012_img_train',
        target_size=(32, 32), 
        color_mode='grayscale',
        class_mode='categorical',
        batch_size=32,
        shuffle=True)


    optimizer = SGD(lr=0.01)
    network.network.compile(loss='categorical_crossentropy', optimizer=optimizer)

    network.network.fit_generator(
        train_generator,
        samples_per_epoch=3200000,
        nb_epoch = 2) 

    network.save(network_name)

if __name__ == '__main__':
    transfer_with_imagenet('3P-imagenet')

