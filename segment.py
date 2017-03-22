import numpy as np
from itertools import *
from math import *
import cv2
import util
import scr
import argparse
import os
import pickle
import jsrt

from skimage import draw
from skimage.segmentation import find_boundaries
from sklearn.cross_validation import StratifiedKFold

# Util functions 
SEGMENTATION_IMAGE_SHAPE = (512, 512)
def finite_derivatives(img):
    size = img.shape
    dx = img.copy()
    dy = img.copy()

    for i, j in product(range(1, size[0] - 1), range(1, size[1] - 1)):
        dy[i, j] = (img[i, j + 1] - img[i, j - 1]) / 2.0
        dx[i, j] = (img[i + 1, j] - img[i - 1, j]) / 2.0
    mag = (dx ** 2 + dy ** 2) ** 0.5

    return mag, dx, dy

def circunference(img, blobs):
    masks = []
    for blob in blobs:
        x, y, r = blob
        mask = np.zeros((2*r + 1, 2*r + 1), np.uint8)
        for i, j in product(range(2*r + 1), range(2*r + 1)):
            if r  ** 2 > (r - i) ** 2  + (r - j) ** 2:
                mask[i][j] = 1
        masks.append(mask)
        
    return blobs, np.array(masks)

def cos_angle(a, b):
    dot = 1.0 * (a[0] * b[0] + a[1] * b[1])
    len_a = sqrt(a[0] * a[0] + a[1] * a[1])
    len_b = sqrt(b[0] * b[0] + b[1] * b[1])

    if len_a * len_b == 0:
        return 0

    tmp = dot/(len_a * len_b)
    if tmp > 1:
        tmp = 1
    elif tmp < -1:
        tmp = -1
    
    # cos(acos(tmp)
    return tmp

''' Nodule segmentation using Adaptive Distance thresold '''

def distance_thold(img, point, grad, dx, dy, t0=0,):
    point = (int(point[0]), int(point[1]))
    size = img.shape
    rmax = 25
    diam = 2 * rmax + 1
    tdelta = 1.7
    mask = np.full((diam, diam), 0, dtype=np.uint8)

    rx = np.linspace(-1 * rmax, rmax, 2 * rmax + 1)
    ry = np.linspace(-1 * rmax, rmax, 2 * rmax + 1)
    ry, rx = np.meshgrid(rx, ry)
    dist = rx ** 2 + ry ** 2
    dist = dist.astype(np.float64)
    ndist = dist / (rmax ** 2)
    ratio = (1.0 - np.exp(-1 * ndist)) / ( 1.0 - exp(-1))
    T = t0 + tdelta * ratio

    mask = dist < rmax ** 2
    mask = mask.astype(np.uint8)
    T = T * mask
    T[T == 0] = 1e10

    roi = img[point[0] - rmax:point[0] + rmax + 1, point[1] - rmax:point[1] + rmax + 1]
    s = roi > T
    s = s.astype(np.uint8)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # TODO:  Hole filling via morphological op ( it can be done with findConturs retrccomps )
    s = cv2.morphologyEx(s, cv2.MORPH_OPEN, kernel)

    smask = np.full((s.shape[0] + 2, s.shape[1] + 2), dtype=np.uint8, fill_value=0)
    s[rmax, rmax] = 1
    cv2.floodFill(s, smask, (rmax, rmax), 255, loDiff=0, upDiff=0, flags=4|cv2.FLOODFILL_FIXED_RANGE)
    s = smask[1:s.shape[0] + 1, 1:s.shape[1] + 1]
    
    # Calculate the ARG of t0
    rx = -1 * np.linspace(-1 * rmax, rmax, 2 * rmax + 1)
    ry = -1 * np.linspace(-1 * rmax, rmax, 2 * rmax + 1)
    ry, rx = np.meshgrid(rx, ry)
    rmag = (rx ** 2 + ry ** 2) ** 0.5

    rdx = dx[point[0] - rmax:point[0] + rmax + 1, point[1] - rmax:point[1] + rmax + 1]
    rdy = dy[point[0] - rmax:point[0] + rmax + 1, point[1] - rmax:point[1] + rmax + 1]
    rgrad = grad[point[0] - rmax:point[0] + rmax + 1, point[1] - rmax:point[1] + rmax + 1]

    rphase = (rx * rdx + ry * rdy) / ( rmag * rgrad + util.EPS)
    rarg = rgrad * rphase

    rarg = rarg * s
    rgrad = rgrad * s
    a = np.sum(rarg)
    b = np.sum(s)

    if b == 0: 
        return -1e10, s
    else:
        return (a * 1.0 / (b + util.EPS)), s

def _adaptive_thold(img, point, grad, dx, dy):
    coorners = [[1e10, 1e10], [-1e10, -1e10]]
    best_arg = -1e10
    best_t = 0
    for t in np.arange(0, -4, -0.1):
        arg, _ = distance_thold(img, point, grad, dx, dy, t)
        #print("{} {}".format(t, arg))      
        #util.imshow('mask', _)

        if best_arg < arg:
            best_arg = arg
            best_t = t

    _, mask = distance_thold(img, point, grad, dx, dy, best_t)

    return np.array(mask)

def adaptive_distance_thold(img, blobs):
    masks = []
    mag, dx, dy = finite_derivatives(img)

    for blob in blobs:
        x, y, r = blob
        mask = _adaptive_thold(img, (x, y), mag, dx, dy)
        masks.append(mask)

    return blobs, np.array(masks)

''' 
Lung segmentation
'''
FOLDS_SEED = 113

def create_mask_from_landmarks(landmarks, mask_shape=(512, 512)):
    landmarks = (landmarks/2).astype(np.int)
    mask = np.full(shape=mask_shape, fill_value=False, dtype=np.bool)

    rr, cc = draw.polygon(landmarks.T[1], landmarks.T[0])
    mask[rr, cc] = True

    return mask

class MeanShape:
    def __init__(self):
        self.mean_landmarks = None 
        self.image_shape = None

    def fit(self, images, landmarks):
        num_samples = len(landmarks)
        num_landmarks = len(landmarks[0])

        self.mean_landmarks = np.zeros(shape=(num_landmarks, 2))
        self.image_shape = images[0].shape

        for i in range(num_samples):
            for j in range(num_landmarks):
                self.mean_landmarks[j][0] += landmarks[i][j][0]
                self.mean_landmarks[j][1] += landmarks[i][j][1]

        for j in range(num_landmarks):
            self.mean_landmarks[j][0] /= num_samples
            self.mean_landmarks[j][1] /= num_samples

    def transform(self, images):
        if len(images[0].shape) == 2:
            num_samples = len(images)
            masks = np.zeros(shape=(num_samples,) + self.image_shape)
            for i in range(num_samples):
                masks[i] = create_mask_from_landmarks(self.mean_landmarks)
            return masks
        elif len(images.shape) == 2:
            return create_mask_from_landmarks(self.mean_landmarks)


def get_shape_model(model_name):
    if model_name == 'mean-shape':
        return MeanShape()
    else:
        raise Exception("{} not implemented yet!".format(model_name))

def load_shape_models_by_fold(model_name, fold=0):
    raise Exception("{} not implemented yet!".format(model_name))

def load_masks_sets(model_name):
    masks_sets = []
    for i in range(len(masks_sets)): 
        masks = np.load('data/{}-fold-{}'.format(model_name, i))
        masks_sets.append(masks)

    return masks_sets 

def train_with_method(model_name):
    set_name = 'jsrt140'
    paths, locs, rads, subs, sizes, kinds = jsrt.jsrt(set=set_name)
    images = jsrt.images_from_paths(paths, dsize=(512, 512))
    landmarks = scr.load_data(set=set_name)

    folds = StratifiedKFold(subs, n_folds=10, shuffle=True, random_state=FOLDS_SEED)

    i = 1
    for tr_idx, te_idx in folds:    
        left_shape_model = get_shape_model(model_name)
        right_shape_model = get_shape_model(model_name)

        left_shape_model.fit(images[tr_idx], landmarks[0][tr_idx])
        left_pred_masks = left_shape_model.transform(landmarks[0][te_idx])

        right_shape_model.fit(images[tr_idx], landmarks[1][tr_idx])
        right_pred_masks = right_shape_model.transform(landmarks[1][te_idx])

        pred_masks = []
        for j in range(len(te_idx)):
            pred_masks.append(np.logical_or(left_pred_masks[j], right_pred_masks[j]))

        lm_file = open('data/{}-lmodel-f{}.pkl'.format(model_name, i), 'wb')
        pickle.dump(left_shape_model, lm_file)
        lm_file.close()
        rm_file = open('data/{}-rmodel-f{}.pkl'.format(model_name, i), 'wb')
        pickle.dump(right_shape_model, rm_file)
        rm_file.close()

        np.save('data/{}-pred-masks-f{}'.format(model_name, i), np.array(pred_masks))

        i += 1

def segment(image, model_name, display=True):
    lmodel = pickle.load(open('data/{}-lmodel-f1.pkl'.format(model_name), 'rb'))
    rmodel = pickle.load(open('data/{}-rmodel-f1.pkl'.format(model_name), 'rb'))
    image = cv2.resize(image, SEGMENTATION_IMAGE_SHAPE, interpolation=cv2.INTER_CUBIC)
    lmask = lmodel.transform(image)
    rmask = rmodel.transform(image)
    if display:
        lboundary = find_boundaries(lmask)
        rboundary = find_boundaries(rmask)
        max_value = np.max(image)
        image[lboundary] = max_value
        image[rboundary] = max_value
        util.imshow('Segment with model {}'.format(model_name), image)
    return np.logical_or(lmask, rmask)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='segment.py')
    parser.add_argument('file', nargs='?', default=os.getcwd())
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--method', default='mean-shape')
    args = parser.parse_args()
    
    if args.train:
        train_with_method(args.method)
    elif args.file:
        image = np.load(args.file).astype('float32')
        segment(image, args.method)
