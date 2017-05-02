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

from collections import OrderedDict
import menpo.io as mio
from menpo.image import Image
from menpo.landmark import *
from menpo.shape import *

from menpofit.aam import HolisticAAM, PatchAAM
from menpo.feature import *
from menpofit.aam import LucasKanadeAAMFitter, WibergInverseCompositional
from matplotlib import pyplot as plt

'''
Constants
'''
SEGMENTATION_IMAGE_SHAPE = (512, 512)

'''
Nodule segmentation utils
'''

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

''' 
Nodule segmentation using Adaptive Distance threshold 
'''

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
Lung segmentation utils
'''

def create_mask_from_landmarks(landmarks, mask_shape=(512, 512)):
    mask = np.full(shape=mask_shape, fill_value=False, dtype=np.bool)
    rr, cc = draw.polygon(landmarks.T[0], landmarks.T[1], shape=mask_shape)
    '''
    for it in rr:
        print it,
    print ''
    '''
    mask[rr, cc] = True

    return mask

''' 
Lung segmentation model base class 
'''

class SegmentationModel:

    def fit(self, images, landmarks):
        raise Exception('Fit method not implemented')

    def transform_one_image(self, image):
        raise Exception('Transform_one_image method not implemented')

    def transform(self, images):
        if len(images[0].shape) == 2:
            num_samples = len(images)
            masks = []
            for i in range(num_samples):
                masks.append(self.transform_one_image(images[i]))
            return np.array(masks)

        elif len(images.shape) == 2:
            return self.transform_one_image(images)
        
'''
Use mean shape as predicted lung shape
'''
class MeanShape(SegmentationModel):
    def __init__(self):
        self.mean_shapes = None 
        self.image_shape = None

    def fit(self, images, landmarks):
        num_shapes = len(landmarks)
        num_samples = len(landmarks[0])
        self.mean_shapes = []
        for i in range(num_shapes):
            landmarks[i] /= 2
            self.mean_shapes.append(np.zeros(shape=(len(landmarks[i][0]), 2)))
        self.mean_shapes = np.array(self.mean_shapes)

        self.image_shape = images[0].shape
        for s in range(num_shapes):
            for i in range(num_samples):
                for j in range(len(landmarks[s][i])):
                    self.mean_shapes[s][j][1] += landmarks[s][i][j][0]
                    self.mean_shapes[s][j][0] += landmarks[s][i][j][1]

        for s in range(num_shapes):
            for j in range(len(landmarks[s][0])):
                self.mean_shapes[s][j][1] /= num_samples
                self.mean_shapes[s][j][0] /= num_samples

    def transform_one_image(self, image):
        mask = create_mask_from_landmarks(self.mean_shapes[0])
        for i in range(1, len(self.mean_shapes)):
            mask = np.logical_or(mask, create_mask_from_landmarks(self.mean_shapes[i]))
        return mask

'''
Lung segmentation based on Active Appearance Models
'''

def normalize_for_amm(image):
    mean = np.mean(image)
    std = np.std(image)
    image = (image - mean)/std
    min_ = np.min(image)
    max_ = np.max(image)
    image = (image - min_)/(max_ - min_ + 1e-7)
    image = image.reshape((1,) + image.shape)
    return image

def prepare_data_for_aam(images, landmarks):
    images_aam = []
    for i in range(len(images)):
        images[i] = normalize_for_amm(images[i])
        images_aam.append(Image(images[i]))
        lmx = landmarks[0][i].T[0]
        lmy = landmarks[0][i].T[1]
        num_points = len(landmarks[0])
        rmx = landmarks[1][i].T[0]
        rmy = landmarks[1][i].T[1]
        pc = PointCloud(points=np.vstack((np.array([lmy, lmx]).T, np.array([rmy, rmx]).T))/2)
        lg = LandmarkGroup.init_from_indices_mapping(pc, 
            OrderedDict({'left':range(len(landmarks[0][i])), 'right':range(len(landmarks[0][i]), len(landmarks[0][i]) + len(landmarks[1][i]))}))
        lm = LandmarkManager()
        lm.setdefault('left', lg)
        images_aam[-1].landmarks = lm
    return images_aam

class ActiveAppearanceModel(SegmentationModel):
    def __init__(self):
        self.mean_shape = None 
        self.image_shape = None
        self.fitter = None
        self.num_landmarks_by_shape = None

    def fit(self, images, landmarks):
        num_samples = len(landmarks[0])

        self.num_landmarks_by_shape = []
        for i in range(len(landmarks)):
            self.num_landmarks_by_shape.append(len(landmarks[i][0]))
        
        aam_images = prepare_data_for_aam(images, landmarks)
        aam = PatchAAM(aam_images, group=None, patch_shape=[(15, 15), (23, 23)],
                         diagonal=150, scales=(0.5, 1.0), holistic_features=fast_dsift,
                         max_shape_components=20, max_appearance_components=150,
                         verbose=True)

        self.fitter = LucasKanadeAAMFitter(aam,
                                  lk_algorithm_cls=WibergInverseCompositional,
                                  n_shape=[5, 20], n_appearance=[30, 150])
        
        pc = []
        for img in aam_images:
            pc.append(img.landmarks[None].lms)
            
        self.mean_shape = mean_pointcloud(pc)
        self.image_shape = images[0].shape

        fitting_results = []
        for img in aam_images[:10]:
            fr = self.fitter.fit_from_shape(img, self.mean_shape, gt_shape=img.landmarks[None].lms) 
            fitting_results.append(fr)

    def transform_one_image(self, image):
        image = Image(normalize_for_amm(image))
        fr = self.fitter.fit_from_shape(image, self.mean_shape) 
        pred_landmarks = fr.final_shape.points

        begin = 0
        mask = create_mask_from_landmarks(pred_landmarks[begin:begin + self.num_landmarks_by_shape[0]])
        begin += self.num_landmarks_by_shape[0]

        for i in range(1, len(self.num_landmarks_by_shape)):
            mask = np.logical_or(mask, create_mask_from_landmarks(pred_landmarks[begin:begin + self.num_landmarks_by_shape[i]]))
            begin += self.num_landmarks_by_shape[i]
        return mask

''' 
segment.py protocol functions
'''

def get_shape_model(model_name):
    if model_name == 'mean-shape':
        return MeanShape()
    elif model_name.find('aam') != -1:
        return ActiveAppearanceModel()
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

def train_and_save(model, tr_images, tr_landmarks, te_images, model_name):
    print "Fit model ... tr {}, te {}".format(len(tr_images), len(te_images))
    model.fit(tr_images, tr_landmarks)

    print "\nRun model on test set ..."
    pred_masks = model.transform(te_images)

    print "Save model ...".format(len(te_images))
    model_file = open('data/{}.pkl'.format(model_name), 'wb')
    pickle.dump(model, model_file)
    model_file.close()

    print "Save masks ..."
    np.save('data/{}-pred-masks'.format(model_name), np.array(pred_masks))
 
def train_with_method(model_name):
    set_name = 'jsrt140'
    paths, locs, rads, subs, sizes, kinds = jsrt.jsrt(set=set_name)
    images = jsrt.images_from_paths(paths, dsize=(512, 512))
    landmarks = scr.load_data(set=set_name)

    tr_val_folds, tr_val, te = util.stratified_kfold_holdout(subs, n_folds=5)

    i = 1
    for tr, val in tr_val_folds:    
        print "Fold {}".format(i)
        landmarks_tr = [landmarks[0][tr], landmarks[1][tr]]
        model = get_shape_model(model_name)
        train_and_save(model, images[tr], landmarks_tr, images[val], '{}-f{}-train'.format(model_name, i))
        i += 1

    landmarks_tr_val = [landmarks[0][tr_val], landmarks[1][tr_val]]
    model = get_shape_model(model_name)
    train_and_save(model, images[tr_val], landmarks_tr_val, images[te], '{}-train-val'.format(model_name))

def segment(image, model_name, display=True):
    print("Loading model ...")
    model = pickle.load(open('data/{}-model-f1.pkl'.format(model_name), 'rb'))

    print("Segment input ...")
    image = cv2.resize(image, SEGMENTATION_IMAGE_SHAPE, interpolation=cv2.INTER_CUBIC)
    mask = model.transform(image)
    if display:
        boundaries = find_boundaries(mask)
        max_value = np.max(image)
        image[boundaries] = max_value
        util.imshow('Segment with model {}'.format(model_name), image)
    return mask

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='segment.py')
    parser.add_argument('file', nargs='?', default=os.getcwd())
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--method', default='aam')
    args = parser.parse_args()
    
    if args.train:
        train_with_method(args.method)
    elif args.file:
        image = np.load(args.file).astype('float32')
        segment(image, args.method)
