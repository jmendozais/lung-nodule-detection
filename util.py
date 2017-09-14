import math
from random import *
import numpy as np
import numpy.linalg as la
from skimage import draw
import cv2
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from skimage.draw import circle_perimeter_aa
from skimage.draw import line
from sklearn import metrics
from sklearn.cross_validation import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.cross_validation import StratifiedShuffleSplit
from skimage.segmentation import find_boundaries
from scipy.interpolate import interp1d

import argparse
import os

from preprocess import *
font = {
#       'family' : 'normal',
#       'weight' : 'bold',
        'size'   : 15}

matplotlib.rc('font', **font)
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True

plt.switch_backend('agg')
plt.ioff()

import time
import csv

'''
Data utils
'''
EPS = 1e-9
FOLDS_SEED = 113
NUM_VAL_FOLDS = 5
DETECTOR_FPPI_RANGE = np.linspace(0.0, 100.0, 101)

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

def split_data_pos_neg(X, Y):
    is_pos = Y.T[1]
    end_pos = 0
    for i in range(len(is_pos)):
        if is_pos[i] < 1:
            end_pos = i
            break

    return (X[:end_pos], X[end_pos:]), (Y[:end_pos], Y[end_pos:])

def save_dataset(V_tr, pred_blobs_tr, blobs_tr, V_te, pred_blobs_te, blobs_te, name):
    np.save('{}-vtr.npy'.format(name), V_tr)
    np.save('{}-pbtr.npy'.format(name), pred_blobs_tr)
    np.save('{}-btr.npy'.format(name), blobs_tr)
    np.save('{}-vte.npy'.format(name), V_te)
    np.save('{}-pbte.npy'.format(name), pred_blobs_te)
    np.save('{}-bte.npy'.format(name), blobs_te)

def load_dataset(name):
    V_tr = np.load('{}-vtr.npy'.format(name))
    pred_blobs_tr = np.load('{}-pbtr.npy'.format(name))
    blobs_tr = np.load('{}-btr.npy'.format(name))
    V_te = np.load('{}-vte.npy'.format(name))
    pred_blobs_te = np.load('{}-pbte.npy'.format(name))
    blobs_te = np.load('{}-bte.npy'.format(name))
    return V_tr, pred_blobs_tr, blobs_tr, V_te, pred_blobs_te, blobs_te

'''
Display utils
'''

def imshow(windowName,  _img, wait=True, display_shape=(1024, 1024)):
    img = np.array(_img).astype(np.float64)
    a = np.min(img)
    b = np.max(img)
    img = (img - a) / (b - a + EPS);
    print np.min(img), np.max(img)
    img = cv2.resize(img, display_shape, interpolation=cv2.INTER_CUBIC)
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    print 'shape {}'.format(img.shape)
    cv2.imshow(windowName, img)
    if wait:
        cv2.waitKey()

def imwrite(fname, _img):
    img = np.array(_img).astype(np.float64)
    a = np.min(img)
    b = np.max(img)
    img = (img - a) / (b - a + EPS);

    cv2.imwrite(fname, 255 * img)

from skimage import io
def imwrite_as_pdf(name, _img):
    img = np.array(_img).astype(np.float64)
    a = np.min(img)
    b = np.max(img)
    img = (img - a) / (b - a + EPS);

    if len(img.shape) == 3:
        tmp = img[:,:,0].copy()
        img[:,:,0] = img[:,:,2]
        img[:,:,2] = tmp
    
    print img.shape, np.min(img), np.max(img)
    io.imsave(name + '.pdf', img)

'''
blob: tuple or list with (x, y, radius) values of the blob
'''
def label_blob(img, blob, border='square', proba=-1, color=(255, 0, 0), margin=0, width=1):
    if len(img.shape) == 2:
        img = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
    if border == 'circle':
        ex, ey = draw.circle_perimeter(blob[0], blob[1], blob[2] + margin)
        img[ex, ey] = color 
    elif border == 'square':
        for i in range(width):
            coord_x = np.array([blob[0]-blob[2]-margin+i, blob[0]-blob[2]-margin+i, blob[0]+blob[2]+margin+i, blob[0]+blob[2]+margin+i])
            coord_y = np.array([blob[1]-blob[2]-margin+i, blob[1]+blob[2]+margin+i, blob[1]+blob[2]+margin+i, blob[1]-blob[2]-margin+i])
            ex, ey = draw.polygon_perimeter(coord_x, coord_y, img.shape)
            img[ex, ey] = color 
    if proba != -1:
        cv2.putText(image,str(proba), (blob[0] + blob[2], blob[1] - blob[2]), cv2.FONT_HERSHEY_SIMPLEX, 2, color)
    return img

def show_blobs(windowName, img, blobs):
    labeled = np.array(img).astype(np.float32)
    maxima = np.max(labeled)
    for blob in blobs:
        labeled = label_blob(labeled, blob, color=(maxima, 0, 0), margin=-5)
    imshow(windowName, labeled)


def show_blobs_with_proba(windowName, img, blobs, probs):
    labeled = np.array(img).astype(np.float32)
    maxima = np.max(labeled)
    for i in range(len(blobs)):
        labeled = label_blob(labeled, blobs[i], color=(maxima, 0, 0), margin=-5, border='square', proba=probs[i])
    imshow(windowName, labeled)

def imwrite_with_blobs(fname, img, blobs):
    labeled = np.array(img).astype(np.float32)
    maxima = np.max(labeled)
    print 'imwrite input shape {}'.format(img.shape)
    for blob in blobs:
        labeled = label_blob(labeled, blob, color=(maxima, 0, 0), margin=5)
    imwrite_as_pdf(fname, labeled)

def show_blobs_real_predicted(img, idx, res1, res2):
    resized_img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)
    color_img = cv2.cvtColor(resized_img.copy().astype(np.float32), cv2.COLOR_GRAY2BGR) 
    max_value = np.max(img)
    
    print "Real vs predicted .."
    print 'real',
    for res in res1:
        print res,
        if res[0] == -1:
            continue
        color_img = label_blob(color_img, res, (max_value, 0, 0))
    print ''
    print 'predicted',
    for res in res2:
        print res,
        if res[0] == -1:
            continue
        color_img = label_blob(color_img, res, (0, max_value, max_value))
    print ''
    imwrite('jsrt-{}.jpg'.format(idx), color_img)
    
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

def imshow_with_mask(image, mask):
    boundaries = find_boundaries(mask)
    max_value = np.max(image)
    image[boundaries] = max_value
    imshow('Image with mask', np.array(image))

def show_landmarks(image, landmarks):
    print "show landmarks"
    max_value = np.max(image)
    print image.shape, np.array(landmarks.shape)
    for i in range(len(landmarks)):
        for j in range(len(landmarks[i])):
            #print landmark
            a, b = int(landmarks[i][j][0]), int(landmarks[i][j][1])
            c, d = int(landmarks[i][(j+1)%len(landmarks[i])][0]), int(landmarks[i][(j+1)%len(landmarks[i])][1])
            print a, b, c, d
            rr, cc, val = circle_perimeter_aa(a, b, 3)
            image[rr, cc] = (1-val) * max_value
            rr, cc = line(a, b, c, d)
            image[rr, cc] = 0
            
    imshow('Image with mask', np.array(image))
    
def save_image_with_landmarks(path, image, landmarks):
    print "show landmarks"
    max_value = np.max(image)
    print image.shape, np.array(landmarks.shape)
    for i in range(len(landmarks)):
        for j in range(len(landmarks[i])):
            #print landmark
            a, b = int(landmarks[i][j][0]), int(landmarks[i][j][1])
            c, d = int(landmarks[i][(j+1)%len(landmarks[i])][0]), int(landmarks[i][(j+1)%len(landmarks[i])][1])
            print a, b, c, d
            rr, cc, val = circle_perimeter_aa(a, b, 3)
            image[rr, cc] = (1-val) * max_value
            rr, cc = line(a, b, c, d)
            image[rr, cc] = 0
            
    imwrite_as_pdf(path, np.array(image))
        
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

def extract_roi(img, blob, dsize=(32, 32)):
    x, y, r = blob
    shift = 0
    side = 2 * shift + 2 * r + 1

    tl = (x - shift - r, y - shift - r)
    ntl = (max(0, tl[0]), max(0, tl[1]))
    br = (x + shift + r + 1, y + shift + r + 1)
    nbr = (min(img.shape[0], br[0]), min(img.shape[1], br[1]))

    img_roi = img[ntl[0]:nbr[0], ntl[1]:nbr[1]]
    img_roi = cv2.resize(img_roi, dsize, interpolation=cv2.INTER_CUBIC)

    return img_roi

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

def save_froc_mixed(froc_ops, froc_legend, scatter_ops, scatter_legend, name, unique=True, with_std=False, fppi_max=10.0):
    ax = plt.gca()
    ax.grid(True)

    line_format = ['b.-', 'g.-', 'r.-', 'c.-', 'm.-', 'y.-', 'k.-', 
                                 'b.--', 'g.--', 'r.--', 'c.--', 'm.--', 'y.--', 'k.--',
                                 'b.-.', 'g.-.', 'r.-.', 'c.-.', 'm.-.', 'y.-.', 'k.-.',
                                 'b.:', 'g.:', 'r.:', 'c.:', 'm.:', 'y.:', 'k.:']

    idx = 0
    legend = []

    markers = ['o', 's', '8', 'v', 'p', '*', 'h', 'H', 'D', 'd', '^', '<', '>']

    for i in range(len(scatter_ops)):
        ops = scatter_ops[i].T
        plt.plot(ops[0], ops[1], '{}{}'.format(line_format[idx%28][0], markers[idx%13]), markersize=5, markeredgewidth=1)
        idx += 1
        legend.append(scatter_legend[i])

    for i in range(len(froc_ops)):
        ops = np.array(froc_ops[i]).T
        plt.plot(ops[0], ops[1] * 0.9091, line_format[i%28], marker='x')
        idx += 1
        legend.append(froc_legend[i])

    x_ticks = np.linspace(0.0, fppi_max, 11)
    y_ticks = np.linspace(0.0, 1.0, 11)
    plt.xticks(x_ticks, x_ticks)
    plt.yticks(y_ticks, y_ticks)
    plt.xlim([0, fppi_max])
    plt.ylim([0, 1.00])
    plt.ylabel('Sensitivity')
    plt.xlabel('Average FPs per Image')
    plt.legend(legend, loc=4, numpoints=1, prop={'size':4})

    print("Saving at {}_sota.pdf".format(name))
    plt.savefig('{}_sota.pdf'.format(name))
    plt.clf()


def subsample_operating_points(ops, num_fppi):
    print np.min(ops[0]), np.max(ops[0]), num_fppi
    fppi_range = np.linspace(np.min(ops[0]), np.max(ops[0]), num_fppi)
    f1 = interp1d(ops[0], ops[1], kind='cubic', fill_value=0.0, bounds_error=False)
    sen = f1(fppi_range)

    if len(ops) == 2:
        return (fppi_range, sen)
    elif len(ops) == 3:
        f2 = interp1d(ops[0], ops[2], kind='cubic', fill_value=0.0, bounds_error=False)
        err = f2(fppi_range)
        return (fppi_range, sen, err)
    

def save_froc(op_set, name, legend=None, unique=True, with_std=False, use_markers=True, fppi_max=10.0, with_auc=True, size='normal'):
    """ Save the plot of the FROC curves in a pdf file and the interpolated operating points (n, 3, 101) for the 'n' FROC curves in a numpy array file.

    Parameters
    ----------
    op_set : a list of arrays of shape (3, 101) containing the operating points of each the FROC curves

    name : a string with the name pdf file

    legend : a list of strings with the lengend names for the FROC curves
    """

    if legend != None:
        assert len(legend) == len(op_set)
        
    legend = list(legend)

    # Size params
    legend_size = 12
    marker_size = 6
    line_width = 2
    num_fppi = 25

    print "save froc params {} {} {}, {}, {} {}".format(len(op_set), op_set[0].shape, op_set[0].dtype, name, len(legend), type(legend))

    ax = plt.gca()
    ax.grid(True)

    line_format = [
        'k.-', 'm.-', 'b.-', 'c.-', 'g.-', 'y.-', 'r.-', 
        'k.--', 'm.--', 'b.--', 'c.--', 'g.--', 'y.--', 'r.--', 
        'k.-.', 'm.-.', 'b.-.', 'c.-.', 'g.-.', 'y.-.', 'r.-.', 
        'k.:', 'm.:', 'b.:', 'c.:', 'g.:', 'y.:', 'r.:']

    if use_markers == True:
        markers = ['o', 's', '8', 'v', 'p', '*', 'h', 'H', 'D', 'd', '^', '<', '>']
    else:
        markers = '.............'
        
    for i in range(len(op_set)):
        ops = np.array(op_set[i]).T
        sops = subsample_operating_points(ops, num_fppi)
        if with_std and len(sops) > 2:
            plt.plot(sops[0], sops[1], line_format[i%28], marker=markers[i%13], markersize=marker_size, fillstyle='none', linewidth=line_width, mew=line_width)
            plt.fill_between(sops[0], sops[1] - sops[2], sops[1] + sops[2], facecolor=line_format[i%13][0], alpha=0.3)  
        else:
            plt.plot(sops[0], sops[1], line_format[i%28], marker=markers[i%13], markersize=marker_size, fillstyle='none', linewidth=line_width, mew=line_width)

        if legend != None and with_auc:
            auc_ = auc(np.array(op_set[i]), range=(0.0, fppi_max))
            legend[i] = r'{}, AUC = {:.2f}'.format(legend[i], auc_)

    x_ticks = np.linspace(0.0, fppi_max, 11)
    y_ticks = np.linspace(0.0, 1.0, 11)
    plt.xticks(x_ticks, x_ticks)
    plt.yticks(y_ticks, y_ticks)
    plt.xlim([0, fppi_max])
    plt.ylim([0, 1.00])
    plt.xlabel('Average FPs per Image')
    plt.ylabel('Sensitivity')

    if legend != None:
        plt.legend(legend, loc=4, prop={'size':legend_size}, numpoints=1)

    if not unique:
        name='{}-{}'.format(name, time.clock())

    np.save(name + '.npy', ops)
    print 'Saving to {}.pdf'.format(name)
    try: 
        print 
        plt.savefig('{}.pdf'.format(name))
    except :
        print "Error saving froc image."

    plt.clf()

def auc(ops, range):
    ops = ops.T
    lower = ops[0].searchsorted(range[0])
    upper = ops[0].searchsorted(range[-1])
    return metrics.auc(ops[0][lower:upper], ops[1][lower:upper])

def save_auc(epochs, aucs, name):
    ax = plt.gca()
    ax.grid(True)

    plt.plot(epochs, aucs)
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.savefig('{}.pdf'.format(name))
    plt.clf()

def save_aucs(epochs, auc_history, name, legend):
    auc_history = np.array(auc_history).T
    ax = plt.gca()
    ax.grid(True)
    for i in range(len(auc_history)):
        plt.plot(epochs, auc_history[i], label=legend[i])

    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.savefig('{}.pdf'.format(name))
    np.savez('{}.npz'.format(name), np.array(epochs), np.array(auc_history), np.array(legend))
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

    print "Loss shapes {} {} {}".format(len(train_loss_detail), len(train_loss), len(test_loss))

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

def add_random_blobs(data, blobs, blobs_by_image=100, rng=np.random, pred_blobs=None, blob_rad=24):
    assert len(data) == len(blobs)
    augmented_blobs = []
    for i in range(len(data)):
        img, lung_mask = data.get(i)
        side = lung_mask.shape[0]
        #assert len(blobs[i]) == 1

        augmented_blobs.append([])
        #rx, ry, _ = blobs[i][0]
        rx, ry, _ = blobs[i]
        #TODO :assert blobs[i][2] == 25, "roi rad check failed {}".format(blobs[i])

        if pred_blobs != None:
            augmented_blobs[i] += list(pred_blobs[i])
        cnt = len(augmented_blobs[i])

        while cnt < blobs_by_image:
            rx = int(rng.uniform(0, side))
            ry = int(rng.uniform(0, side))
            if lung_mask[rx, ry] > 0:
                augmented_blobs[i].append([rx, ry, blob_rad])
                cnt += 1
        augmented_blobs[i] = np.array(augmented_blobs[i])

    return np.array(augmented_blobs)

def extract_random_rois(data, dsize, rois_by_image=1000, rng=np.random, flat=True):
    rois = []
    if data != None:
        for i in range(len(data)):
            img, lung_mask = data.get(i)
            sampled, lce, norm = preprocess(img, lung_mask)
            # Pick LCE images
            side = lce.shape[0]
            assert lung_mask.shape[0] == lce.shape[0]
            #rois = []
            cnt = 0
            while cnt < rois_by_image:
                rx = int(rng.uniform(0, side))
                ry = int(rng.uniform(0, side))
                if lung_mask[rx, ry] > 0:
                    '''
                    print "img shape {}".format(img.shape)
                    print "lce shape {}".format(lce.shape)
                    print "lung_mask shape {}".format(lce.shape)
                    print "lung_mask corner_value {} max_value {}".format(lung_mask[0][0], np.max(lung_mask))
                    print "point {} {}".format(rx, ry)
                    #print 'roi-{}-{}.jpg'.format(i, cnt)
                    #imwrite('roi-{}-{}.jpg'.format(i, cnt), util.extract_roi(lce, (rx, ry, 25), dsize))
                    '''
                    rois.append([util.extract_roi(lce, (rx, ry, 25), dsize)])
                    cnt += 1
            #roi_set.append(rois)
    return np.array(rois)

'''
Validation utils
'''

def bootstrap_sets(size, num_sets=1000):
    bootstrapped_scores = []
    rng = np.random.RandomState(100003)
    sets = []

    for i in range(num_sets):
        indices = rng.random_integers(0, size - 1, size)
        sets.append(indices)

    return sets

def ci(scores, confidence=0.95):
    assert confidence < 1.0 and confidence > 0

    delta = (1 - confidence)/2
    sorted_scores = np.array(scores)
    sorted_scores.sort()
    confidence_lower = sorted_scores[int(delta * len(sorted_scores))]
    confidence_upper = sorted_scores[int((1.0 - delta) * len(sorted_scores))]

    return confidence_lower, confidence_upper

def stratified_kfold_holdout(stratified_labels, n_folds, shuffle=True, random_state=FOLDS_SEED):
    split = StratifiedShuffleSplit(stratified_labels, 1, test_size=0.3, random_state=random_state)
    tr, te = list(split)[0]
    folds = StratifiedKFold(stratified_labels[tr], n_folds=n_folds, shuffle=shuffle, random_state=random_state)
    tr_val_folds = []
    for tr_tr, tr_val in folds:
        tr_val_folds.append((tr[tr_tr], tr[tr_val]))

    return tr_val_folds, tr, te

def model_selection_folds(data, n_folds=NUM_VAL_FOLDS):
    return list(KFold(n_splits=n_folds, shuffle=True, random_state=FOLDS_SEED).split(data))

def froc_folds(real_blobs, blobs, probs, folds):
    frocs = []
    for tr_idx, te_idx in folds:    
        froc = eval.froc(real_blobs[te_idx], blobs[te_idx], probs[te_idx])
        frocs.append(froc)

    av_froc = eval.average_froc(frocs)
    return av_froc

def join_frocs(listname, bpiname, outname, max_fppi):
    frocf = open(listname, 'r')
    froc_lines = [line for line in frocf]
    if bpiname != None:
        bpif = open(bpiname, 'r')
        bpi_lines = [line for line in bpif]

    frocs = []
    names = []

    for i in range(len(froc_lines)):
        toks = froc_lines[i].strip().split(';')
        print "loading {}".format(toks)
        frocs.append(np.load(toks[0]).T)
        names.append(toks[1])
        print frocs[-1][:10]
        print names[-1]
        if bpiname != None:
            toks = bpi_lines[i].strip().split(';')
            names[-1] += ', ABPI={:.2f}'.format(float(np.loadtxt(toks[0])))
        
    util.save_froc(np.array(frocs), outname, names, fppi_max=max_fppi)

def single_froc(modelname, legendname, max_fppi):
    valname = 'data/' + modelname + '-sbf-0.7-aam-val-froc'
    froc = np.load(valname + '.npy').T
    util.save_froc(np.array([froc]), valname, np.array([legendname]), fppi_max=max_fppi)

def plot_abpi(listname, outname, maxy):
    f = open(listname, 'r')
    abpi = []
    names = []
    for line in f:
        toks = line.strip().split(',')
        print np.loadtxt(toks[0])
        abpi.append(np.loadtxt(toks[0]))
        names.append(toks[1])

    index = np.arange(len(names))
    bar_width = 0.35
    opacity = 0.4
    rects1 = plt.bar(index, abpi, bar_width,
                 alpha=opacity,
                 color='b')

    plt.xlabel('Threshold')
    plt.ylabel('Average False Positive per Image')
    plt.xticks(index + bar_width / 2, names)
    plt.savefig('{}.pdf'.format(outname))
    plt.clf()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='util.py')
    parser.add_argument('file', nargs='?', default=os.getcwd())
    parser.add_argument('--list', default=None, type=str)
    parser.add_argument('--bpif', default=None, type=str)
    parser.add_argument('--out', default=None, type=str)
    parser.add_argument('--max', default=10.0, type=int)
    parser.add_argument('--froc', action='store_true')
    parser.add_argument('--abpi', action='store_true')
    parser.add_argument('--single-froc', action='store_true')

    parser.add_argument('--model', default=None, type=str)
    #@parser.add_argument('--legend', default=None, type=str)

    args = parser.parse_args()
    
    if args.froc == True:
        join_frocs(args.list, args.bpif, args.out, args.max)
    elif args.single_froc == True:
        single_froc(args.model, args.model, args.max)
    elif args.abpi == True:
        plot_abpi(args.list, args.out, args.max)
 
    '''
    import jsrt
    
    from sklearn.cross_validation import StratifiedKFold
    paths, locs, rads, subs, sizes, kinds = jsrt.jsrt(set='jsrt140')

    data = np.array(range(20))
    strat = np.zeros((20,)).astype(int)
    strat[10:] = 1

    folds = stratified_kfold_train_val_test(strat, n_folds=5, shuffle=True, random_state=113)

    for tr, val, te in folds:
        print tr, val, te
    '''
