
from itertools import *
from math import *
import numpy as np
import cv2
import argparse
import os
import pickle
import jsrt
import lidc
import scr
import util

import pathos.multiprocessing as mp

from skimage import draw
from skimage.segmentation import find_boundaries
import sklearn.metrics as metrics

from sklearn.cross_validation import StratifiedKFold

from collections import OrderedDict
import menpo.io as mio
from menpo.image import Image
from menpo.landmark import *
from menpo.shape import *

from menpofit.aam import HolisticAAM, PatchAAM
from menpo.feature import *
from menpofit.aam import LucasKanadeAAMFitter, WibergInverseCompositional

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
Lung segmentation model base class 
'''

class SegmentationModel:

    def create_mask_from_landmarks(self, landmarks, mask_shape=(512, 512)):
        mask = np.full(shape=mask_shape, fill_value=False, dtype=np.bool)
        rr, cc = draw.polygon(landmarks.T[0], landmarks.T[1], shape=mask_shape)
        mask[rr, cc] = True
        return mask

    def fit(self, images, landmarks):
        raise Exception('Fit method not implemented')

    def transform_one_image(self, image):
        raise Exception('Transform_one_image method not implemented')

    def transform(self, images):
        if len(images[0].shape) == 3:
            num_samples = len(images)
            masks = []
            for i in range(num_samples):
                masks.append(self.transform_one_image(images[i]))
            return np.array(masks)
            '''
            pool = mp.Pool(5)     
            return pool.map(self.transform_one_image, images)
            '''

        elif len(images.shape) == 3:
            return self.transform_one_image(images)
        
'''
Use mean shape as predicted lung shape
'''
class MeanShape(SegmentationModel):
    def __init__(self, join_masks=True):
        self.mean_shapes = None 
        self.image_shape = None
        self.join_masks = join_masks

    def fit(self, images, landmarks):
        num_shapes = len(landmarks)
        num_samples = len(landmarks[0])
        self.mean_shapes = []
        for i in range(num_shapes):
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
        masks = []
        mask = np.zeros(shape=image.shape)
        for i in range(0, len(self.mean_shapes)):
            masks.append(self.create_mask_from_landmarks(self.mean_shapes[i]))
            mask = np.logical_or(mask, masks[-1])
        if self.join_masks:
            return mask
        else:
            return masks

'''
Lung segmentation based on Active Appearance Models
'''

class ActiveAppearanceModel(SegmentationModel):
    def __init__(self, join_masks=True, scales=[15, 23]):
        self.mean_shape = None 
        self.image_shape = None
        self.fitter = None
        self.num_landmarks_by_shape = None
        self.join_masks = join_masks
        self.scales = scales

    def normalize(self, image):
        mean = np.mean(image)
        std = np.std(image)
        image = (image - mean)/std
        # FIX:
        min_ = np.min(image)
        max_ = np.max(image)
        image = (image - min_)/(max_ - min_ + 1e-7)
        return image

    def prepare_data_for_aam(self, images, landmarks):
        images_aam = []
        for i in range(len(images)):
            images[i] = self.normalize(images[i])
            images_aam.append(Image(images[i]))
            lmx = landmarks[0][i].T[0]
            lmy = landmarks[0][i].T[1]
            num_points = len(landmarks[0])
            rmx = landmarks[1][i].T[0]
            rmy = landmarks[1][i].T[1]
            pc = PointCloud(points=np.vstack((np.array([lmy, lmx]).T, np.array([rmy, rmx]).T)))
            lg = LandmarkGroup.init_from_indices_mapping(pc, 
                OrderedDict({'left':range(len(landmarks[0][i])), 'right':range(len(landmarks[0][i]), len(landmarks[0][i]) + len(landmarks[1][i]))}))
            lm = LandmarkManager()
            lm.setdefault('left', lg)
            images_aam[-1].landmarks = lm
        return images_aam

    def fit(self, images, landmarks):
        num_samples = len(landmarks[0])

        self.num_landmarks_by_shape = []
        for i in range(len(landmarks)):
            self.num_landmarks_by_shape.append(len(landmarks[i][0]))
        
        aam_images = self.prepare_data_for_aam(images, landmarks)
        aam = PatchAAM(aam_images, group=None, patch_shape=[(self.scales[0], self.scales[0]), (self.scales[1], self.scales[1])],
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
        print 'init ...',
        mask = np.zeros(shape=image.shape)
        image = Image(self.normalize(image))
        fr = self.fitter.fit_from_shape(image, self.mean_shape) 
        pred_landmarks = fr.final_shape.points
        begin = 0
        masks = []
        for i in range(0, len(self.num_landmarks_by_shape)):
            masks.append(self.create_mask_from_landmarks(pred_landmarks[begin:begin + self.num_landmarks_by_shape[i]]))
            mask = np.logical_or(mask, masks[-1])
            begin += self.num_landmarks_by_shape[i]

        print 'done'
        if self.join_masks:
            return mask
        else:
            return masks

''' 
segment.py protocol functions
'''

def get_shape_model(model_name, **kwargs):
    if model_name == 'mean-shape':
        return MeanShape(**kwargs)
    elif model_name.find('aam') != -1:
        return ActiveAppearanceModel(**kwargs)
    else:
        raise Exception("{} not implemented yet!".format(model_name))

def train_and_save(model, tr_images, tr_landmarks, te_images, model_name):
    print "Fit model ... tr {}, te {}".format(len(tr_images), len(te_images))
    model.fit(tr_images, tr_landmarks)

    print "\nRun model on test set ..."
    pred_masks = model.transform(te_images)

    for i in range(len(te_images)):
        image = te_images[i]
        boundaries = find_boundaries(pred_masks[i])
        max_value = np.max(image)
        image[boundaries] = max_value
        util.imshow('Segment with model {}'.format(model_name), image)

    print "Save model ...".format(len(te_images))
    model_file = open('data/{}.pkl'.format(model_name), 'wb')
    pickle.dump(model, model_file, -1)
    model_file.close()

    print "Save masks ..."
    np.save('data/{}-pred-masks'.format(model_name), np.array(pred_masks))
 
def train_with_method(model_name):
    set_name = 'jsrt140'
    paths, locs, rads, subs, sizes, kinds = jsrt.jsrt(set=set_name)
    images = jsrt.images_from_paths(paths, dsize=(512, 512))
    landmarks = scr.load(set=set_name)

    tr_val_folds, tr_val, te = util.stratified_kfold_holdout(subs, n_folds=5)

    i = 1
    for tr, val in tr_val_folds:    
        print "Fold {}".format(i)
        landmarks_tr = [landmarks[0][tr], landmarks[1][tr]]
        model = get_shape_model(model_name, join_masks=False)
        train_and_save(model, images[tr], landmarks_tr, images[val], '{}-f{}-train'.format(model_name, i))
        i += 1

    landmarks_tr_val = [landmarks[0][tr_val], landmarks[1][tr_val]]
    model = get_shape_model(model_name)
    train_and_save(model, images[tr_val], landmarks_tr_val, images[te], '{}-train-val'.format(model_name))

''' 
Functions for JSRT-LIDC
'''

def train(model_name):
    images, _ = jsrt.load(set_name='jsrt140n')
    landmarks = scr.load(set_name='jsrt140n')
    # TODO: check the masks returned by join masks
    model = get_shape_model(model_name)

    print 'Training model ...' 
    model.fit(images, landmarks)

    print 'Saving model ...'
    model_file = open('data/{}-{}-model.pkl'.format(model_name, 'jsrt140n'), 'wb')
    pickle.dump(model, model_file, -1)
    model_file.close()

def segment_datasets(model_name):
    model = pickle.load(open('data/{}-{}-model.pkl'.format(model_name, 'jsrt140n'), 'rb'))

    print('Segment lidc')
    lidc_images, _ = lidc.load()
    pred_masks = model.transform(lidc_images)
    np.save('data/{}-{}-pred-masks'.format(model_name, 'lidc'), np.array(pred_masks))

    print('Segment jsrt positives')
    jsrt_images, _ = jsrt.load(set_name='jsrt140p')
    pred_masks = model.transform(jsrt_images)
    np.save('data/{}-{}-pred-masks'.format(model_name, 'jsrt140p'), np.array(pred_masks))

    '''
    print('Segment jsrt')
    jsrt_images, _ = jsrt.load(set_name='jsrt140')
    pred_masks = model.transform(jsrt_images)
    np.save('data/{}-{}-pred-masks'.format(model_name, 'jsrt140'), np.array(pred_masks))
    '''

def segment_func(image, model_name, display=True):
    print("Loading model ...")
    model = pickle.load(open('data/{}-jsrt140n-model.pkl'.format(model_name), 'rb'))
    model.join_masks = True

    print("Segment input ...")
    image = cv2.resize(image, SEGMENTATION_IMAGE_SHAPE, interpolation=cv2.INTER_CUBIC)
    mask = model.transform(np.array([image]))
    if display:
        boundaries = find_boundaries(mask)[0]
        tmp = np.full(boundaries.shape, dtype=np.bool, fill_value=False)
        b1 = np.array([boundaries, tmp, tmp])
        b2 = np.array([boundaries, boundaries, boundaries])
        b1 = np.swapaxes(b1, 0, 2)
        b1 = np.swapaxes(b1, 0, 1)
        b2 = np.swapaxes(b2, 0, 2)
        b2 = np.swapaxes(b2, 0, 1)

        util.imwrite_as_pdf('data/original', image)

        image = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)
        max_value = np.max(image)
        image[b2] = 0.0
        image[b1] = max_value

        util.imwrite_as_pdf('data/segmented', image)
        print 'mask shape', mask.shape, mask.dtype
        util.imwrite_as_pdf('data/mask', mask[0])
        util.imshow('Segment with model {}'.format(model_name), image)
    return mask

def eval_by_missed_nodules():
    images, blobs = lidc.load()
    masks = np.load('data/aam-lidc-pred-masks.npy')
    p = 0
    tp = 0
    for i in range(len(images)):
        assert masks[i].shape[:2] == images[i][0].shape[:2]
        p += len(blobs[i])
        for j in range(len(blobs[i])): 
            if masks[i][blobs[i][j][0], blobs[i][j][1]] > 0:
                tp += 1
    print 'Total nodules {}, missed nodules {}'.format(p, p - tp)

def iou(mask1, mask2):
    mask2 = mask2.astype(np.uint64)
    intersection = mask1 * mask2
    intersection = np.sum(intersection)
    union = mask1 + mask2
    union[union > 0] = 1
    union = np.sum(union)
    return intersection*1.0/union
                
def eval_by_IOU(model_name, set_name, images_tr, landmarks_tr, images_te, masks_te, scales=[10, 15]):
    model = get_shape_model(model_name, join_masks=False, scales=scales)

    print 'Training model ...' 
    model.fit(images_tr, landmarks_tr)

    print 'Saving model ...'
    model_file = open('data/{}-{}-model.pkl'.format(model_name, set_name), 'wb')
    pickle.dump(model, model_file, -1)
    model_file.close()

    pred_masks = model.transform(images_te)
    np.save('data/{}-{}-pred-masks'.format(model_name, set_name + '_inv'), np.array(pred_masks))
    
    ious = []
    for i in range(len(masks_te)):
        iou_l = iou(masks_te[i][0], pred_masks[i][0])
        iou_r = iou(masks_te[i][1], pred_masks[i][1])
        ious.append((iou_l + iou_r)/2.0)
    return ious

def eval(model_name, args):
    images_tr, _ = jsrt.load(set_name='jsrt_od')
    landmarks_tr = scr.load(set_name='jsrt_od')
    masks_tr = jsrt.masks(set_name='jsrt_od', join_masks=False)
    images_te, _ = jsrt.load(set_name='jsrt_ev')
    landmarks_te = scr.load(set_name='jsrt_ev')
    masks_te = jsrt.masks(set_name='jsrt_ev', join_masks=False)
    
    scales = [args.scale1, args.scale2]
    ious_ev = eval_by_IOU(model_name, 'jsrt_od', images_tr, landmarks_tr, images_te, masks_te, scales=scales)
    ious_od = eval_by_IOU(model_name, 'jsrt_ev', images_te, landmarks_te, images_tr, masks_tr, scales=scales)

    ious = ious_ev + ious_od
    ious.sort()
    ious = np.array(ious)
    
    q1 = len(ious)/4
    med = len(ious)/2
    q3 = med + q1
    results = np.array([ious.mean(), ious.std(), ious.min(), ious[q1], ious[med], ious[q3], ious.max()])
    results = np.round(results, decimals=3)
    np.savetxt('{}_sc_{}_{}_eval.txt'.format(model_name, args.scale1, args.scale2), results)
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='segment.py')
    parser.add_argument('file', nargs='?', default=os.getcwd())
    parser.add_argument('--train-jsrt', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--segment', action='store_true')
    parser.add_argument('--method', default='aam')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--scale1', default=10, type=int)
    parser.add_argument('--scale2', default=15, type=int)

    args = parser.parse_args()
    
    if args.train:
        train(args.method)
    if args.segment:
        segment_datasets(args.method)
    if args.eval:
        eval(args.method, args)
    if args.train_jsrt:
        train_with_model(args.method)
    elif args.file != os.getcwd():
        image = np.load(args.file).astype('float32')
        segment_func(image, args.method)
