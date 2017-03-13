
from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import numpy as np
np.random.seed(1000003) 
from keras.datasets import mnist
from keras.utils import np_utils, layer_utils
from scipy.misc import imresize

class SemiSupervisedData:
    def __init__(self, nb_labeled, img_rows=28, img_cols=28):
        nb_classes = 10
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        tmp = []
        for x in X_train:
            tmp.append(imresize(x, (img_rows, img_cols)))
        X_train = np.array(tmp)
        tmp = []
        for x in X_test:
            tmp.append(imresize(x, (img_rows, img_cols)))
        X_test = np.array(tmp)

        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train /= 255
        X_test /= 255

        # convert class vectors to binary class matrices
        Y_train = np_utils.to_categorical(y_train, nb_classes)
        Y_test = np_utils.to_categorical(y_test, nb_classes)
        
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.X_train_labeled = self.X_train[:nb_labeled]
        self.Y_train_labeled = self.Y_train[:nb_labeled]
        self.X_train_unlabeled = self.X_train[nb_labeled:]
        self.Y_train_unlabeled = self.Y_train[nb_labeled:]
        self.labeled_index = 0
        self.unlabeled_index = 0
        
    def next_batch(self, batch_size):
        start = self.labeled_index
        self.labeled_index += batch_size
        end = self.labeled_index
        
        if self.labeled_index > len(self.X_train_labeled):
            end = self.labeled_index = batch_size
            start = 0

        X_labeled_batch = self.X_train_labeled[start:end] 
        Y_labeled_batch = self.Y_train_labeled[start:end]
        
        start = self.unlabeled_index
        self.unlabeled_index += batch_size
        end = self.unlabeled_index
        
        if self.unlabeled_index > len(self.X_train_unlabeled):
            end = self.unlabeled_index = batch_size
            start = 0
            
        X_unlabeled_batch = self.X_train_unlabeled[start:end]
        #return X_labeled_batch, Y_labeled_batch
        return np.vstack((X_labeled_batch, X_unlabeled_batch)), Y_labeled_batch
        
print("Loaded MNIST")
import tensorflow as tf

# Utils 

class AttributeDict(dict):
    __getattr__ = dict.__getitem__

    def __setattr__(self, a, b):
        self.__setitem__(a, b)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def resize_images(X, height_factor, width_factor):
    shape = X.get_shape()
    original_shape = tuple([i.__int__() for i in shape])
    
    new_shape = tf.shape(X)[1:3]
    new_shape *= tf.constant(np.array([height_factor, width_factor]).astype('int32'))
    X = tf.image.resize_nearest_neighbor(X, new_shape)
    X.set_shape((None, original_shape[1] * height_factor if original_shape[1] is not None else None,
                original_shape[2] * width_factor if original_shape[2] is not None else None, None))
    return X

def up_sample_2x2(X):
    return resize_images(X, 2, 2)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def create_network(input_shape, filters, filters_size, dense, dropout):
    d = AttributeDict()
    
    d.x = tf.placeholder(tf.float32, shape=[None] + list(input_shape))
    d.y = tf.placeholder(tf.float32, shape=[None, dense[-1]])
    
    d.input_shape = input_shape
    d.do_tr = dropout
    d.filters = filters
    d.filters_size = filters_size
    d.dense = dense

    d.W = []
    d.b = []
    d.do = []
    
    d.h_prepool = [None]
    d.h = [d.x]
    d.mp_h = []
    d.wheres = []
    
    L_conv = len(filters)
    filters = [input_shape[-1]] + filters
    
    for i in range(L_conv):
        # Convolution
        wi = weight_variable((filters_size[i], filters_size[i], filters[i], filters[i+1]))
        bi = bias_variable((filters[i+1],))
        hi = tf.nn.relu(conv2d(d['h'][-1], wi) + bi)
        
       
        hi_prepool = hi
        hi = max_pool_2x2(hi)

        d.wheres.append(tf.stop_gradient(tf.gradients(tf.reduce_sum(hi), hi_prepool, colocate_gradients_with_ops=True)[0]))

        if dropout[i] != .0:
            doi = tf.placeholder(tf.float32)
            hi = tf.nn.dropout(hi, keep_prob=1.0 - doi)
            d.do.append(doi)
        else:
            d.do.append(None)

        d.W.append(wi)
        d.b.append(bi)
        d.h.append(hi)
    
    L_dense = len(dense)
    conv_output_shape = input_shape[0] / (2**L_conv)
    dense = [filters[-1] * (conv_output_shape ** 2),] + dense
    
    for i in range(L_dense):
        if i == 0:
            hi = tf.reshape(d.h[-1], shape=(-1, dense[0]))
        wi = weight_variable((dense[i], dense[i+1]))
        bi = bias_variable((dense[i+1],))
        hi = tf.nn.relu(tf.matmul(hi, wi) + bi)
        
        if dropout[L_conv + i] != .0:
            doi = tf.placeholder(tf.float32)
            hi = tf.nn.dropout(hi, keep_prob=1.0 - doi)
            d.do.append(doi)
        else:
            d.do.append(None)
        
        d.W.append(wi)
        d.b.append(bi)
        d.h.append(hi)
    return d

def fit(sess, net, loss, data, lr, metrics={}, batch_size=50, iterations=500, var_list=None):
    prev_vars = set(tf.global_variables())
    train_step = tf.train.AdamOptimizer(lr).minimize(loss, var_list=var_list)
    cur_vars = set(tf.global_variables())
    sess.run(tf.variables_initializer(cur_vars - prev_vars))

    to_fetch = [loss]
    to_fetch_labels = ['loss']
    for k in metrics:
        if isinstance(metrics[k], list):
            for i in range(len(metrics[k])):
                to_fetch_labels.append('{}_{}'.format(k, i))
                to_fetch.append(metrics[k][i])
        else:
            to_fetch_labels.append(k)
            to_fetch.append(metrics[k])
    
    print "Training stage ..."
    for i in range(iterations):
        batch = data.next_batch(batch_size)  
        feed_dict = {} 
        for j in range(len(net.do_tr)):
            if net.do[j] != None:
                feed_dict[net.do[j]] = net.do_tr[j]
        feed_dict[net.x] = batch[0]
        feed_dict[net.y] = batch[1]

        if i%100 == 0:
            fetched = sess.run(to_fetch, feed_dict=feed_dict)
            print 'Iteration {}:'.format(i),
            for j in range(len(fetched)):
                print "train {} {} ".format(to_fetch_labels[j], fetched[j]),
            print ''

        train_step.run(feed_dict=feed_dict)
            
        if i%1000 == 0 and i > 0:
            n = len(data.X_test)
            bs = 1000
            for j in range(len(net.do)):
                if net.do[j] != None:
                    feed_dict[net.do[j]] = .0

            fetched_v = []
            for j in range(n/bs):
                start = j*bs
                end = (j+1)*bs
                feed_dict[net.x] = data.X_test[start:end]
                feed_dict[net.y] = data.Y_test[start:end]
                fetched = sess.run(to_fetch, feed_dict=feed_dict)
                fetched_v.append(fetched)

            fetched = np.mean(np.array(fetched_v), axis=0)
            print 'Iteration {}:'.format(i),
            for j in range(len(fetched)):
                print "test {} {} ".format(to_fetch_labels[j], fetched[j]),
            print ''

        
    print "Testing stage ..."
    feed_dict = {net.x: data.X_test, net.y: data.Y_test}
    for i in range(len(net.do)):
        if net.do[i] != None:
            feed_dict[net.do[i]] = .0

    n = len(data.X_test)
    bs = 1000
    fetched_v = []
    for j in range(n/bs):
        start = j*bs
        end = (j+1)*bs
        feed_dict[net.x] = data.X_test[start:end]
        feed_dict[net.y] = data.Y_test[start:end]
        fetched = sess.run(to_fetch, feed_dict=feed_dict)
        fetched_v.append(fetched)

    fetched = np.mean(np.array(fetched_v), axis=0)
    for j in range(len(fetched)):
        print "test {} {} ".format(to_fetch_labels[j], fetched[j]),
    print ''

    '''
    feed_dict[net.x] = data.X_test
    feed_dict[net.y] = data.Y_test
    fetched = sess.run(to_fetch, feed_dict=feed_dict)
    print "test {} {} ".format(to_fetch_labels[j], fetched[j]),
    '''
    return fetched

def augment_supervised_with_semisupervised(sess, net, loss, data, metrics={}, batch_size=50, multipliers=None, layerwise_iters=1000, dec_iters=1000, all_iters=1000, mode='first'):
    L_conv = len(net.filters)
    layerwise_lr = 3e-4
    decoder_lr = 3e-4
    all_lr = 3e-5

    net.d = AttributeDict()
    net.d.multipliers = [tf.constant(multipliers[i]) for i in range(len(multipliers))]
    #sess.run(tf.variables_initializer(net.d.multipliers))
    net.d.W = [None for i in range(L_conv)]
    net.d.b = [None for i in range(L_conv)]
    net.d.h = [None for i in range(L_conv+1)]
    
    # Layerwise initialization
    filters = [net.input_shape[-1]] + net.filters

    for i in range(L_conv):
        print i
        net.d.h[i+1] = net.h[i+1]
        print net.d.h[i+1]

        hi = up_sample_2x2(net.d.h[i+1])
        hi = tf.multiply(net.wheres[i], hi)
        
        wi = weight_variable((net.filters_size[i], net.filters_size[i], filters[i+1], filters[i]))
        bi = bias_variable((filters[i],))
        if i > 0:
            hi = tf.nn.relu(conv2d(hi, wi) + bi)
        else:
            hi = conv2d(hi, wi) + bi
            
        net.d.W[i] = wi
        net.d.b[i] = bi
        net.d.h[i] = hi
        
        for j in reversed(range(i)):
            net.d.h[j] = up_sample_2x2(net.d.h[j+1])
            net.d.h[j] = tf.multiply(net.wheres[j], net.d.h[j])
            if j > 0:
                net.d.h[j] = tf.nn.relu(conv2d(net.d.h[j], net.d.W[j]) + net.d.b[j])
            else:
                net.d.h[j] = conv2d(net.d.h[j], net.d.W[j]) + net.d.b[j]
        
        # fit
        layerwise_loss = tf.reduce_mean(tf.reduce_sum(tf.square(net.d.h[0] - net.x), axis=(1, 2)))
        sess.run(tf.variables_initializer([wi, bi]))
        fit(sess, net, layerwise_loss, data, batch_size=batch_size, lr=layerwise_lr, iterations=layerwise_iters, var_list=[wi, bi])

    # Decoding pathway training
    u_loss = .0
    if mode == 'first':
        u_loss = net.d.multipliers[0] * tf.reduce_mean(tf.reduce_sum(tf.square(net.d.h[0] - net.h[0]), axis=(1, 2)))
    elif mode == 'all':
        for i in range(L_conv):
            u_loss += net.d.multipliers[i] * tf.reduce_mean(tf.reduce_sum(tf.square(net.d.h[i] - net.h[i]), axis=(1, 2)))
    fit(sess, net, u_loss, data, batch_size=batch_size, lr=decoder_lr, iterations=dec_iters, var_list=net.d.W + net.d.b)
    
    y_pred = tf.slice(net.h[-1], [0, 0], [tf.shape(net.y)[0], -1])
    s_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_pred, net.y))
    correct_prediction = tf.equal(tf.argmax(y_pred,1), tf.argmax(net.y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Enconding/decoding finetuning

    loss = u_loss + s_loss
    fetched = fit(sess, net, loss, data, batch_size=batch_size, lr=all_lr, iterations=all_iters, metrics={'accuracy':accuracy, 'u_loss':u_loss, 's_loss':s_loss})
    
    return fetched



cnn_1 = {'input_shape':(28, 28, 1), 'filters':[32, 64], 'filters_size':[5, 5], 'dense':[1024, 10], 'dropout':[.0, .0, .5, .0]}
cnn_2 = {'input_shape':(32, 32, 1), 'filters':[64, 64, 64], 'filters_size':[5, 3, 3], 'dense':[10], 'dropout':[.0, .0, .5, .0]}

def augment_test(data):
    architecture = cnn_1 
    #uloss_multipliers = [[1e-5, 1e-5], [3e-4, 3e-4], [1e-4, 1e-4], [3e-3, 3e-3], [1e-3, 1e-3]]
    #uloss_multipliers = [[3e-8, 3e-8], [1e-7, 1e-7], [3e-7, 3e-7], [1e-6, 1e-6], [3e-6, 3e-6]]
    uloss_multipliers = list(reversed([[3e-6, 3e-6], [1e-5, 1e-5], [3e-5, 3e-5], [1e-4, 1e-4], [3e-4, 3e-4]]))
    sess = tf.InteractiveSession() 
    res = []
    for k in range(len(uloss_multipliers)):
        acc = []
        acc_ft = []
        for i in range(5):
            net = create_network(**architecture)
            y_pred = tf.slice(net.h[-1], [0, 0], [tf.shape(net.y)[0], -1])
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_pred, net.y))
            correct_prediction = tf.equal(tf.argmax(y_pred,1), tf.argmax(net.y,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            sess.run(tf.global_variables_initializer())
            print data.Y_train_labeled
            fetched = fit(sess, net, loss, data, lr=3e-5, metrics={'accuracy':accuracy}, iterations=20000)
            acc.append(fetched[-1])
            fetched = augment_supervised_with_semisupervised(sess, net, loss, data, batch_size=50, multipliers=uloss_multipliers[k], layerwise_iters=5000, dec_iters=12000, all_iters=10000)
            #fetched = fit(sess, net, loss, data, lr=1e-5, metrics={'accuracy':accuracy}, iterations=12000)
            acc_ft.append(fetched[-1])

        acc = np.array(acc)
        acc_ft = np.array(acc_ft)
        print("Acc {} +- {}".format(acc.mean(), acc.std()))
        print("Acc ft {} +- {}".format(acc_ft.mean(), acc_ft.std()))
        res.append([acc.mean(), acc.std(), acc_ft.mean(), acc_ft.std()])
    sess.close()

    for i in range(len(res)):
        print "Multiplier {} Acc {} +- {}, Acc ft {} +- {}".format(uloss_multipliers[i], res[i][0], res[i][1], res[i][2], res[i][3])

def fit_semisupervised(sess, net, data, multipliers, batch_size=50, lr=1e-5, iters=1000):
    L_conv = len(net.filters)

    net.d = AttributeDict()
    net.d.multipliers = [tf.constant(multipliers[i]) for i in range(len(multipliers))]
    net.d.W = [None for i in range(L_conv)]
    net.d.b = [None for i in range(L_conv)]
    net.d.h = [None for i in range(L_conv+1)]
    
    filters = [net.input_shape[-1]] + net.filters
    net.d.h[L_conv] = net.h[L_conv]
    u_loss = .0
    layerwise_losses = []
    for i in reversed(range(L_conv)):
        hi = up_sample_2x2(net.d.h[i+1])
        hi = tf.multiply(net.wheres[i], hi)
        wi = weight_variable((net.filters_size[i], net.filters_size[i], filters[i+1], filters[i]))
        bi = bias_variable((filters[i],))
        sess.run(tf.variables_initializer([wi, bi]))
        if i > 0:
            hi = tf.nn.relu(conv2d(hi, wi) + bi)
        else:
            hi = conv2d(hi, wi) + bi
            
        net.d.W[i] = wi
        net.d.b[i] = bi
        net.d.h[i] = hi
        print(net.d.multipliers[i])        
        #layerwise_losses.append(net.d.multipliers[i] * tf.reduce_mean(tf.reduce_sum(tf.square(net.d.h[i] - net.h[i]), axis=(1, 2))))
        layerwise_losses.insert(0, net.d.multipliers[i] * tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(net.d.h[i] - net.h[i]), axis=(1, 2)))))
        u_loss += layerwise_losses[0]

    y_pred = tf.slice(net.h[-1], [0, 0], [tf.shape(net.y)[0], -1])
    s_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_pred, net.y))
    correct_prediction = tf.equal(tf.argmax(y_pred,1), tf.argmax(net.y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    loss = u_loss + s_loss

    fetched = fit(sess, net, loss, data, batch_size=batch_size, lr=lr, iterations=iters, metrics={'accuracy':accuracy, 'u_loss':u_loss, 's_loss':s_loss, 'l_loss':layerwise_losses})
    
    return fetched

def test_semisupervised(sess, data, iters=20000):
    architecture = cnn_2
    net = create_network(**architecture)
    sess.run(tf.global_variables_initializer())
    multipliers =  [3e-6, 3e-6, 3e-6]
    res = fit_semisupervised(sess, net, data, multipliers, lr=1e-4, iters=iters, batch_size=50)
    print('Acc {}'.format(res[-1]))

def test_supervised(sess, data, iters=40000, batch_size=50, lr=1e-4):
    architecture = cnn_2
    acc = []
    for i in range(5):
        net = create_network(**architecture)
        sess.run(tf.global_variables_initializer())
        y_pred = tf.slice(net.h[-1], [0, 0], [tf.shape(net.y)[0], -1])
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_pred, net.y))
        correct_prediction = tf.equal(tf.argmax(y_pred,1), tf.argmax(net.y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        result = fit(sess, net, loss, data, batch_size=batch_size, lr=lr, iterations=iters, metrics={'accuracy':accuracy, 'loss':loss})
        acc.append(result[-1])
        print('Acc {}'.format(result[-1]))

    acc = np.array(acc)
    print "Acc {} +- {}".format(acc.mean(), acc.std())

if __name__ == '__main__':
    data = SemiSupervisedData(100, 32, 32)
    sess = tf.InteractiveSession()     
    #test_supervised(sess, data, 20000)
    test_semisupervised(sess, data, 40000)
    sess.close()

''' 
How to implement s_loss + u_loss?

Papers suchs as SWWAE[1] mentions that u and s objetives should be calculated in the same phase. I assumed
that the labeled data used on a SGD iteration should be used for class/rec phase, and unlabeled data should 
used just for rec phase. However, ladder networks implementation[2] shows that labeled could be used for
classification only and unlabeled stream could include labeled and unlabeled data, computing only the 
reconstruction loss. So we have two options:

1) s_loss(labeled) + u_loss(unlabeled + labeled)
2) s_loss(labeled) + u_lsss(unlabeled*) # unlabeled* iterated over the whole dataset (labeled+unlabeled)

# The diference between both is that in 2) u_loss will not necessarily consider the same labeled images used
to compute s_loss

[1] https://arxiv.org/pdf/1506.02351v8.pdf
[2] https://github.com/rinuboney/ladder/blob/master/input_data.py
'''
'''
for i in range(L):
    diff_i = augmented_network['h'][i] - augmented_network['ih'][i]
    loss_i = tf.reduce_mean(tf.reduce_sum(tf.square(diff_i), axis=(1, 2)))
    fit(sess, data, loss_i)
'''
'''
s_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
u_loss = tf.reduce_mean(tf.reduce_sum(tf.square(x - x_reconstructed), axis=(1, 2)))
loss = s_loss + u_loss
'''

