import math
from random import *
import numpy as np
import numpy.linalg as la
from skimage import draw
import cv2
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize

font = {'family' : 'normal',
#        'weight' : 'bold',
        'size'   : 15}

matplotlib.rc('font', **font)

#import seaborn as sns

plt.switch_backend('agg')
plt.ioff()
import time
import csv

# Data utils
EPS = 1e-9
def load_list(path, blob_type='rad'):
    detect_f = open(path, 'r+') 
    paths = []
    blobs = []  

    for line in detect_f:
        toks = line.split(' ')  
        path = toks[0]
        blob_dim = 3
        _blobs = []
        argc = int(toks[1])
        if blob_dim == 'rect':
            blob_dim = 4
        for i in range(argc):
            blob = []
            for j in range(blob_dim):
                blob.append(int(toks[2 + blob_dim*i + j]))
            _blobs.append(blob)
        paths.append(path)
        blobs.append(_blobs)
    return paths, blobs

def save_list(paths, blobs, path):
    size = len(paths)
    blob_dim = len(blobs[0][0])
    for i in range(size):
        print paths[i],
        print len(blobs),
        for j in range(len(blobs[i])):
            for k in range(blob_dim):
                print path,
        print ''

def scale_list(path, factor):
    paths, tls, sizes = load_list(path)
    for i in range(len(tls)):
        tls[i][0] = math.round(tls[i][0] * 1.0 / factor)
        tls[i][1] = math.round(tls[i][1] * 1.0 / factor)
        for j in range(len(sizes[0])):
            sizes[i][j] = math.round(sizes[i][0] * 1.0 / factor)
            sizes[i][j] = math.round(sizes[i][1] * 1.0 / factor)

    save_list(paths, tls, sizes)

# Display utils 

def imshow(windowName,  _img, wait=True):
    img = np.array(_img).astype(np.float64)
    a = np.min(img)
    b = np.max(img)
    img = (img - a) / (b - a + EPS);
    print np.min(img), np.max(img)
    
    cv2.imshow(windowName, img)
    if wait:
        cv2.waitKey()

def imwrite(fname, _img):
    img = np.array(_img).astype(np.float64)
    a = np.min(img)
    b = np.max(img)
    img = (img - a) / (b - a + EPS);

    cv2.imwrite(fname, 255 * img)

def label_blob(img, blob, color=(255, 0, 0), margin=0):
    if len(img.shape) == 2:
        img = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)

    ex, ey = draw.circle_perimeter(blob[0], blob[1], blob[2] + margin)
    if np.max(ex) + 3 + margin >= img.shape[0] or np.max(ey) + 3 + margin >= img.shape[1]:
        return img

    img[ex, ey] = color 
    ex, ey = draw.circle_perimeter(blob[0], blob[1], blob[2]+1+margin)
    img[ex, ey] = color 
    '''
    ex, ey = draw.circle_perimeter(blob[0], blob[1], blob[2]+2+margin)
    img[ex, ey] = color 
    ex, ey = draw.circle_perimeter(blob[0], blob[1], blob[2]+3+margin)
    img[ex, ey] = color 
    '''
    return img

def show_blobs(windowName, img, blobs):
  labeled = np.array(img).astype(np.float32)
  maxima = np.max(labeled)
  for blob in blobs:
    labeled = label_blob(labeled, blob, color=(maxima, 0, 0), margin=-5)

  imshow(windowName, labeled)

def imwrite_with_blobs(fname, img, blobs):
    labeled = np.array(img).astype(np.float32)
    maxima = np.max(labeled)
    for blob in blobs:
        labeled = label_blob(labeled, blob, color=(maxima, 0, 0), margin=-5)

    imwrite(fname, labeled)

def show_blobs_real_predicted(path, res1, res2):
    img = np.load(path, 0)
    resized_img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)
    color_img = cv2.cvtColor(resized_img.copy(), cv2.COLOR_GRAY2BGR) 
    
    print "Real vs predicted .."
    print 'real',
    for res in res1:
        print res,
        if res[0] == -1:
            continue
        color_img = label_blob(color_img, res, (255, 0, 0))
    print ''
    print 'predicted',
    for res in res2:
        print res,
        if res[0] == -1:
            continue
        color_img = util.label_blob(color_img, res, (0, 255, 255))
    print ''
    util.imshow('real vs predicted', color_img)
    
def show_nodule(roi, mask, scale=4):
    dsize = (mask.shape[0] * scale, mask.shape[0] * scale)
    roi = cv2.resize(roi, dsize)
    mask = cv2.resize(mask, dsize)
    _max = np.max(roi)
    drawing = cv2.cvtColor(roi.copy().astype(np.float32), cv2.COLOR_GRAY2BGR)
    color = (uniform(0, _max), uniform(0, _max), uniform(0, _max))
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(drawing, contours, -1, color, 1)
    imshow("nodule", drawing)

def print_detection(path, blobs):
    print path,
    print len(blobs),
    if len(blobs) > 0:
        blob_dim = len(blobs[0])
        for i in range(len(blobs)):
            for j in range(blob_dim):
                print blobs[i][j],
    print ''

def print_list(paths, blobs):
    size = len(paths)
    blob_dim = len(blobs[0][0])
    for i in range(size):
        print_detection(paths[i], blobs[i])

def save_blob(path, img, blob):
    x, y, r = blob
    shift = 0
    side = 2 * shift + 2 * r + 1
    dsize = (128, 128)

    tl = (x - shift - r, y - shift - r)
    ntl = (max(0, tl[0]), max(0, tl[1]))
    br = (x + shift + r + 1, y + shift + r + 1)
    nbr = (min(img.shape[0], br[0]), min(img.shape[1], br[1]))

    img_roi = img[ntl[0]:nbr[0], ntl[1]:nbr[1]]
    img_roi = cv2.resize(img_roi, dsize, interpolation=cv2.INTER_CUBIC)

    imwrite(path, img_roi)

def show_blob(path, img, blob):
    x, y, r = blob
    shift = 0
    side = 2 * shift + 2 * r + 1
    dsize = (128, 128)

    tl = (x - shift - r, y - shift - r)
    ntl = (max(0, tl[0]), max(0, tl[1]))
    br = (x + shift + r + 1, y + shift + r + 1)
    nbr = (min(img.shape[0], br[0]), min(img.shape[1], br[1]))

    img_roi = img[ntl[0]:nbr[0], ntl[1]:nbr[1]]
    img_roi = cv2.resize(img_roi, dsize, interpolation=cv2.INTER_CUBIC)

    imshow(path, img_roi)

def save_froc_mixed(froc_ops, froc_legend, scatter_ops, scatter_legend, name, unique=True, with_std=False):
    ax = plt.gca()
    ax.grid(True)

    line_format = ['b.-', 'g.-', 'r.-', 'c.-', 'm.-', 'y.-', 'k.-', 
                                 'b.--', 'g.--', 'r.--', 'c.--', 'm.--', 'y.--', 'k.--',
                                 'b.-.', 'g.-.', 'r.-.', 'c.-.', 'm.-.', 'y.-.', 'k.-.',
                                 'b.:', 'g.:', 'r.:', 'c.:', 'm.:', 'y.:', 'k.:']

    idx = 0
    legend = []

    markers = ['o', 'v', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', '^', '<', '>']
    for i in range(len(scatter_ops)):
        ops = scatter_ops[i].T
        print ops[0], ops[1]
        plt.plot(ops[0], ops[1], '{}{}'.format(line_format[idx%28][0], markers[idx%13]), markersize=5, markeredgewidth=1)
        idx += 1
        legend.append(scatter_legend[i])

    for i in range(len(froc_ops)):
        ops = np.array(froc_ops[i]).T
        plt.plot(ops[0], ops[1] * 0.9091, line_format[i%28], marker='x', markersize=3)
        idx += 1
        legend.append(froc_legend[i])

    x_ticks = np.linspace(0.0, 10.0, 11)
    y_ticks = np.linspace(0.0, 1.0, 11)
    plt.xticks(x_ticks, x_ticks)
    plt.yticks(y_ticks, y_ticks)
    plt.xlim([0, 10.0])
    plt.ylim([0, 1.00])
    plt.ylabel('Sensitivity')
    plt.xlabel('Average FPs per Image')
    plt.legend(legend, loc=4, fontsize='small', numpoints=1)

    plt.savefig('{}_cmp.eps'.format(name))
    plt.clf()

def save_froc(op_set, name, legend=None, unique=True, with_std=False):
    ax = plt.gca()
    ax.grid(True)

    line_format = ['b.-', 'g.-', 'r.-', 'c.-', 'm.-', 'y.-', 'k.-', 
                                 'b.--', 'g.--', 'r.--', 'c.--', 'm.--', 'y.--', 'k.--',
                                 'b.-.', 'g.-.', 'r.-.', 'c.-.', 'm.-.', 'y.-.', 'k.-.',
                                 'b.:', 'g.:', 'r.:', 'c.:', 'm.:', 'y.:', 'k.:']

    for i in range(len(op_set)):
        ops = np.array(op_set[i]).T
        if with_std and i > 0:
            y_lower = ops[1] - ops[2]
            plt.errorbar(ops[0], ops[1], fmt=line_format[i%28], yerr=ops[2], marker='x', markersize=3)
        else:
            plt.plot(ops[0], ops[1], line_format[i%28], marker='x', markersize=3)

    '''
    import baseline
    for fp in baseline.interesting_fps:
        print 'fp {}: {}'.format(fp, ops[1][int(fp * 10)])
    '''

    x_ticks = np.linspace(0.0, 10.0, 11)
    y_ticks = np.linspace(0.0, 1.0, 11)
    plt.xticks(x_ticks, x_ticks)
    plt.yticks(y_ticks, y_ticks)
    plt.xlim([0, 10.0])
    plt.ylim([0, 1.00])
    plt.xlabel('Average FPs per Image')
    plt.ylabel('Sensitivity')

    if legend != None:
        assert len(legend) == len(op_set)
        plt.legend(legend, loc=4, fontsize='small', numpoints=1)

    if not unique:
        name='{}_{}'.format(name, time.clock())

    plt.savefig('{}_cnn_clf.eps'.format(name))
    plt.clf()

class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)
    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

def save_grid(scores, name, labels, ranges, title):
    min_score = np.min(scores.flatten())
    max_score = np.max(scores.flatten())
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
               norm=MidpointNormalize(vmin=0.2 * max_score, midpoint=0.92 * max_score, vmax=max_score))
    plt.xlabel(labels[1])
    plt.ylabel(labels[0])
    plt.colorbar()
    plt.xticks(np.arange(len(ranges[1])), np.round(ranges[1], 5), rotation=45)
    plt.yticks(np.arange(len(ranges[0])), np.round(ranges[0], 5))
    plt.title(title)
    name='{}_{}'.format(name, time.clock())
    plt.savefig('{}_grid.jpg'.format(name))
    plt.clf()

def save_weights(weights, name):
    weights = np.swapaxes(weights, 0, 1)
    mean1 = np.mean(weights[0], axis=0)
    std1 = np.std(weights[0], axis=0)
    mean2 = np.mean(weights[1], axis=0)
    std2 = np.std(weights[1], axis=0)

    idx = np.arange(len(mean1))
    plt.figure(figsize=(16, 12))
    plt.bar(idx + 0.3, mean1, edgecolor='none', width=0.3, yerr=std1, label='coef_', color='b', error_kw = {'elinewidth':0.3})
    plt.bar(idx + 0.6, mean2, edgecolor='none', width=0.3, yerr=std2, label='anova', color='g', error_kw = {'elinewidth':0.3})
    plt.title("Feature weights")
    plt.xlabel('Feature number')
    plt.xticks(np.arange(0, len(mean1), 10))
    plt.yticks(())
    plt.axis('tight')
    plt.legend(loc='upper right')
    plt.savefig('data/{}.jpg'.format(name), dpi=1000)
    plt.clf()
    plt.figure(figsize=None)

def save_loss(history, name):
    train_loss = history['loss']    
    train_loss_detail = history['loss_detail']
    test_loss = history['val_loss'] 

    print "loss shapes"
    print len(train_loss_detail), len(train_loss), len(test_loss)

    trx_init = len(train_loss) * 1.0 / len(train_loss_detail)
    trx = np.linspace(trx_init, len(train_loss), len(train_loss_detail));
    tex = np.linspace(1, len(test_loss), len(test_loss))

    plt.plot(trx, train_loss_detail, alpha=0.5, label = 'train loss by batch')
    plt.plot(tex, train_loss, label = 'train loss')
    plt.plot(tex, test_loss, label = 'test loss')
    plt.legend(loc = 2)

    plt.ylabel('Loss')
    plt.xlabel('Epochs')

    #plt.twinx()
    plt.grid()
    plt.ylim([0, 0.8])
    plt.legend(loc = 1)
    plt.savefig('{}.jpg'.format(name))
    plt.clf()

def save_acc(history, name):
    train_acc = history['acc']    
    test_acc = history['val_acc'] 

    tex = np.linspace(1, len(test_acc), len(test_acc))
    plt.plot(tex, train_acc, label = 'train acc')
    plt.plot(tex, test_acc, label = 'test acc')
    plt.legend(loc = 2)

    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')

    #plt.twinx()
    plt.grid()
    plt.ylim([0., 1.])
    plt.legend(loc = 4)
    plt.savefig('{}.jpg'.format(name))
    plt.clf()

def save_loss_acc(history, name):
    save_loss(history, name + '_loss')
    save_acc(history, name + '_acc')

if __name__ == '__main__':
    import jsrt
    from sklearn.cross_validation import StratifiedKFold
    paths, locs, rads, subs = jsrt.jsrt(set='jsrt140')
    folds = StratifiedKFold(subs, n_folds=10, shuffle=True, random_state=113)

    for tr, te in folds:
        print 'tr'
        print subs[tr]
        print 'te'
        print subs[te]
