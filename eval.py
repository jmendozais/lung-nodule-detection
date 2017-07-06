import os

import os.path as path
import numpy as np
import cv2
import sys
import scipy.stats as stats
from scipy.interpolate import interp1d
from operator import itemgetter


'''
from shapely.geometry import Polygon
from shapely.geometry import Point
'''

import util
from preprocess import *
from segment import adaptive_distance_thold as segment

def _poly(res):
    tl = (res[0], res[1])
    size = (res[2], res[3])

    pts = [tl, (tl[0] + size[0], tl[1]), (tl[0] + size[0], tl[1] + size[1]), (tl[0], tl[1] + size[1])]

    return Polygon(pts)

def _iou(res1, res2):
    roi1 = _poly(res1)
    roi2 = _poly(res2) 

    inter = roi1.intersection(roi2)
    overlap = inter.area / (roi1.area + roi2.area - inter.area)

    return overlap

def _iou_circle(res1, res2):
    r1 = max(res1[2], res1[3])/2
    r2 = max(res2[2], res2[3])/2
    p1 = Point(res1[0] + r1, res1[1] + r1).buffer(r1)
    p2 = Point(res2[0] + r2, res2[1] + r2).buffer(r2)

    return p1.intersection(p2).area / p1.union(p2).area

def _iou_circle(res1, res2):
    r1 = max(res1[2], res1[3])/2
    r2 = max(res2[2], res2[3])/2
    p1 = Point(res1[0] + r1, res1[1] + r1).buffer(r1)
    p2 = Point(res2[0] + r2, res2[1] + r2).buffer(r2)

    return p1.intersection(p2).area / p1.union(p2).area

def _dist(blob1, blob2):
    return (blob1[0] - blob2[0]) ** 2 + (blob1[1] - blob2[1]) ** 2

def _load_results(results_path, factor=1.0):
    fin = open(results_path)
    results = []
    for line in fin:
        toks = line.split(' ')
        name = toks[0]

        if len(toks) == 1:
            num_rois = 0
        else:
            num_rois = int(toks[1])

        results.append([])

        for i in range(num_rois):
            results[-1].append([])
            for j in range(4):
                results[-1][-1].append(int(factor * int(toks[2 + i * 4 + j])))
        print num_rois, len(results[-1])

    return results

def _get_paths(results_path):
    fin = open(results_path)
    paths = []

    for line in fin:
        toks = line.split(' ')
        paths.append(toks[0])

    return paths

def evaluate(real, predicted, data=None, sample=False):
    assert len(real) == len(predicted)
    num_imgs = len(real)
    sensitivity = 0
    fppi = []
    iou = []
    iou_pos = []
    tp = 0
    p = 0
    MAX_DIST = 35.7142 # 25 mm

    for i in range(num_imgs):
        if sample: 
            img, mask = data.get(i)
            sampled, lce, norm = preprocess(img, mask)
            img = lce

        found = False
        found_blob_idx = -1
        overlap = -1e10
        for j in range(len(real[i])):
            if real[i][j][0] == -1:
                continue

            p += 1
            for k in range(len(predicted[i])):
                #overlap = _iou_circle(real[i][j], predicted[i][k])
                dist = _dist(real[i][j], predicted[i][k])
                iou.append(dist)
                if dist < MAX_DIST * MAX_DIST:
                    iou_pos.append(overlap) 
                    found = True
                    found_blob_idx = k
                    break
            if found:
                break


        #print "real blob {}".format(real[i][0])
        # Assuming that we just have one true object per image at most
        if found:
            fppi.append(len(predicted[i]) - 1)
            tp += 1
            if sample:
                for k in range(len(predicted[i])):
                    if k != found_blob_idx:
                        util.save_blob('data/fp/{}_{}_{}.jpg'.format(i, k, data.img_paths[i].split('/')[-1]), img, predicted[i][k])
                        _, masks = segment(img, [predicted[i][k]])
                        rmask = cv2.resize(masks[0], (128, 128), interpolation=cv2.INTER_CUBIC)
                        util.imwrite('data/fp/{}_{}_{}_mask.jpg'.format(i, k, data.img_paths[i].split('/')[-1]), rmask) 
                    else:
                        print 'real, predicted ->', real[i][0], predicted[i][k]
                        util.save_blob('data/tp/real_{}_{}.jpg'.format(i, data.img_paths[i].split('/')[-1]), img, real[i][0])
                        util.save_blob('data/tp/{}_{}_{}.jpg'.format(i, k, data.img_paths[i].split('/')[-1]), img, predicted[i][k])
        else:
            fppi.append(len(predicted[i]))
            if sample:
                if real[i][0][0] != -1:     
                    util.save_blob('data/fn/{}_{}.jpg'.format(i, data.img_paths[i].split('/')[-1]), img, real[i][0])
                for k in range(len(predicted[i])):
                    util.save_blob('data/fp/{}_{}_{}.jpg'.format(i, k, data.img_paths[i].split('/')[-1]), img, predicted[i][k])
                    _, masks = segment(img, [predicted[i][k]])
                    rmask = cv2.resize(masks[0], (128, 128), interpolation=cv2.INTER_CUBIC)
                    util.imwrite('data/fp/{}_{}_{}_mask.jpg'.format(i, k, data.img_paths[i].split('/')[-1]), rmask) 
        
        #print "found {}, overlap {}".format(found, overlap)
        
        #if paths != None:
        #   util.show_blobs_real_predicted(paths[i], [real[i]], predicted[i])

    fppi = np.array(fppi)
    iou = np.array(iou)
    iou_pos = np.array(iou_pos)

    sensitivity = tp * 1.0 / p
    #return sensitivity, np.mean(fppi), np.std(fppi), np.mean(iou), np.std(iou), np.mean(iou_pos), np.std(iou_pos)
    return sensitivity, np.mean(fppi), np.std(fppi)

# TODO: Adapt the code to calculate the exact vertical average FROC curve, instead of mix tav and vav

def extract_random_rois(data, dsize, rois_by_image=1000, rng=np.random):
    roi_set = []
    if data != None:
        for i in range(len(data)):
            img, lung_mask = data.get(i)
            sampled, lce, norm = preprocess(img, lung_mask)

            # Pick LCE images
            side = lce.shape[0]
            assert lung_mask.shape[0] == lce.shape[0]
            rois = []
            cnt = 0
            while cnt < rois_by_image:
                rx = rng.uniform(0, side)
                ry = rng.uniform(0, side)
                if lung_mask[rx, ry] > 0:
                    rois.append(util.extract_roi((rx, ry), img))
                    cnt += 1
            roi_set.append(rois)
    
    return np.array(roi_set)

''' 
ISSUE: We cant compare a multi-object FROC curve and a single-object one straighforwardly
'''

def froc(real, pred, probs, rois=None, data=None, jsrt_idx=None, save_rois=False, verbose=True, distance=31.43):
    img_set = []
    if data != None:
        for i in range(len(data)):
            img, lung_mask = data.get(i)
            sampled, lce, norm = preprocess(img, lung_mask)
            img_set.append(np.array([lce]))
    img_set = np.array(img_set)

    n = len(real)       
    p = 0

    entries = []
    if jsrt_idx != None: 
        entries = np.full((0,6), fill_value=0, dtype=float)
    else: 
        entries = np.full((0,5), fill_value=0, dtype=float)
    
    pred_blob_idx = 0
    
    num_imgs_with_nods = len(real)
    blob_rads = []
    for i in range(n):
        # Remove
        if data != None:
            util.show_blobs_real_predicted(img_set[i][0], jsrt_idx[i], real[i], pred[i])

        if len(real[i]) == 0:
            num_imgs_with_nods -= 1

        for j in range(len(real[i])):
            if len(pred[i]) > 0:
                dist = np.linalg.norm(pred[i][:,:2] - real[i][j][:2], axis=1)
            else:
                dist = np.array([])

            entry = []
            entry.append(probs[i])
            entry.append(dist)
            entry.append(np.full((probs[i].shape), fill_value=p, dtype=np.float))
            entry.append(np.full((probs[i].shape), fill_value=i, dtype=np.float))
            entry.append(np.arange(pred_blob_idx, pred_blob_idx + len(pred[i]), dtype=np.float))
            if jsrt_idx != None:
                entry.append(np.full((probs[i].shape), fill_value=jsrt_idx[i], dtype=np.float))

            entries = np.append(entries, np.array(entry).T, axis=0)
            p += 1
            blob_rads.append(real[i][j][2]/2)

        pred_blob_idx += len(pred[i])
    
    def compare_entries(a, b):
        if a[0] == b[0]:
            if a[1] == b[1]:
                return 0
            elif a[1] < b[1]:
                return -1
            else:
                return 1
        elif a[0] < b[0]:
            return 1
        else:
            return -1

    entries = sorted(entries, cmp=compare_entries)
    entries = np.array(entries)

    tp = 0.0
    fppi = np.full((n,), fill_value=0, dtype=np.float)
    seen_blob = np.full((p,), fill_value=False, dtype=np.bool)
    seen_pred_blob = np.full((pred_blob_idx,), fill_value=False, dtype=np.bool)

    froc = []
    f_prev = -1.0
    
    for i in range(entries.shape[0]):
        blob_idx = int(entries[i][2])
        img_idx = int(entries[i][3])
        pred_blob_idx = int(entries[i][4])

        if entries[i][0] != f_prev:
            froc.append([np.sum(fppi)/num_imgs_with_nods, tp / p])
            f_prev = entries[i][0]

        if not seen_pred_blob[pred_blob_idx]:
            seen_pred_blob[pred_blob_idx] = True

            threshold = distance
            if distance == 'rad':
                threshold = blob_rads[blob_idx]

            if entries[i][1] < threshold and not seen_blob[blob_idx]:
                seen_blob[blob_idx] = True
                tp += 1
            else:
                fppi[img_idx] += 1

    froc.append([np.sum(fppi)/num_imgs_with_nods, tp / p])
    froc.append([1000.0, tp / p])
    targets = [2., 4., 10.]
    ops = []
    for i in range(len(targets)):
        best_op = froc[0]
        for op in froc:
            if abs(op[0] - targets[i] + 0.5) <= 0.5:
                best_op = op
        ops.append(best_op)
    ops.append(froc[-1])

    if verbose:
        print "fppi operating point: {}".format(ops)
        print "mean fppi {}, tp {}, p {}".format(np.sum(fppi)/num_imgs_with_nods, tp, p)

    return np.array(froc)   

def froc_stratified(real, pred, probs, kind, num_frocs):
    n = len(real)       
    p = 0
    entries = np.full((0,3), fill_value=0, dtype=float)
    
    for i in range(n):
        if real[i][0][0] != -1:
            p += 1
        dist = np.linalg.norm(pred[i][:,:2] - real[i][0][:2], axis=1)
        entry = []
        entry.append(probs[i])
        entry.append(dist)
        entry.append(np.full((probs[i].shape), fill_value=p, dtype=np.float))
        entry.append(np.full((probs[i].shape), fill_value=i, dtype=np.float))
        entries = np.append(entries, np.array(entry).T, axis=0)
    
    entries = sorted(entries, key=itemgetter(0))
    entries.reverse()
    entries = np.array(entries)
    fppi = np.full((n,), fill_value=0, dtype=np.float)
    found = np.full((n,), fill_value=False, dtype=bool)

    froc = []
    tp = []
    p = []
    for i in xrange(num_frocs):
        froc.append([])
        tp.append(0.0)
        p.append(0.0)

    for val in kind:
        p[val] += 1

    print 'num frocs {}'.format(num_frocs)
    print 'values {}'.format(kind)
    for val in range(num_frocs):
        froc[val].append([0.0, 0.0])
        if p[val] == 0.0:
            print 'kind {}: p {}'.format(val, p[val])
            p[val] = util.EPS

    f_prev = -1.0
    for i in range(entries.shape[0]):
        blob_id = entries[i][2]
        img_id = entries[i][3]
        if entries[i][0] != f_prev:
            froc[kind[img_id]].append([np.mean(fppi), tp[kind[img_id]] / p[kind[img_id]]])
            f_prev = entries[i][0]
        if not found[blob_id] and entries[i][1] < 31.43:
            found[blob_id] = True
            tp[kind[img_id]] += 1
        else:
            fppi[img_id] += 1

    for i in xrange(num_frocs):
        froc[i].append([np.mean(fppi), tp[i] / p[i]])
        froc[i] = np.array(froc[i])
    return np.array(froc)   

def fppi_sensitivity(real, pred, data=None):
        n = len(real)       
        p = 0
        entries = np.full((0,3), fill_value=0, dtype=float)
        
        for i in range(n):
            if real[i][0][0] != -1:
                p += 1

            dist = np.linalg.norm(pred[i][:,:2] - real[i][0][:2], axis=1)
            
            entry = []
            entry.append(dist)
            entry.append(dist)
            entry.append(np.full((dist.shape), fill_value=i, dtype=np.float))

            entries = np.append(entries, np.array(entry).T, axis=0)
        
        entries = sorted(entries, key=itemgetter(0))
        entries.reverse()
        entries = np.array(entries)
    
        tp = 0.0
        fppi = np.full((n,), fill_value=0, dtype=np.float)
        found = np.full((n,), fill_value=False, dtype=bool)
        froc = []
        f_prev = -1.0
        for i in range(entries.shape[0]):
            idx = entries[i][2]
            if entries[i][0] != f_prev:
                froc.append([np.mean(fppi), tp / p])
                f_prev = entries[i][0]

            if not found[idx] and entries[i][1] < 31.43:
                found[idx] = True
                tp += 1
            else:
                fppi[idx] += 1

        froc.append([np.mean(fppi), tp / p])
        return froc[-1]
    
DEFAULT_FPPI_RANGE = np.linspace(0.0, 10.0, 101)
def average_froc(frocs, fppi_range=DEFAULT_FPPI_RANGE):
    av_sen = []

    print 'whole {}'.format(len(frocs))
    for i in range(len(frocs)):
        x = frocs[i].T
        print 'av froc {} -> {}'.format(i, x.shape)
        f = interp1d(x[0], x[1], kind='linear', fill_value=0.0, bounds_error=False)
        sen = f(fppi_range)
        print 'fppi range {}, sen {}'.format(fppi_range, sen.shape)
        av_sen.append(sen)

    av_sen = np.array(av_sen)
    av_froc = [fppi_range, np.mean(av_sen, axis=0), np.std(av_sen, axis=0)]
    av_froc = np.array(av_froc).T

    return av_froc

def average_froc_with_ci(frocs, fppi_range, confidence=0.95):
    av_sen = []

    for i in range(len(frocs)):
        x = frocs[i].T
        f = interp1d(x[0], x[1], kind='linear', fill_value=0.0, bounds_error=False)
        sen = f(fppi_range)
        av_sen.append(sen)

    av_sen = np.array(av_sen)
    assert len(fppi_range) == len(av_sen[i]) 

    aucs = []
    for i in range(len(av_sen)):
        froc = np.array([fppi_range, av_sen[i]]).T
        aucs.append(util.auc(froc, fppi_range))
    
    lo, up = util.ci(aucs, confidence)

    av_froc = [fppi_range, np.mean(av_sen, axis=0), np.std(av_sen, axis=0)]
    av_froc = np.array(av_froc).T

    return av_froc, lo, up

def froc_on_folds(real_blobs, blobs, probs, folds):
    frocs = []
    for tr_idx, te_idx in folds:    
        froc = eval.froc(real_blobs[te_idx], blobs[te_idx], probs[te_idx])
        frocs.append(froc)

    av_froc = eval.average_froc(frocs, fppi_range)
    return av_froc

def sublety_stratified_kfold(y, sublety):
    nidx = np.arange(len(y))[y == 0]
    pidx = np.arange(len(y))[y == 1]
    pfolds = StratifiedKFold(sublety[pidx], n_folds=10, shuffle=True, random_state=113)
    nfolds = KFold(y, n_folds=10, shuffle=True, random_state=113)
    pfolds = [x for x in pfolds]
    nfolds = [x for x in nfolds]
    folds = []

    for i in range(len(pfolds)):
        tr = np.concatenate(pfolds[i][0], nfolds[i][0])
        te = np.concatenate(pfolds[i][1], nfolds[i][1])
        folds.append((tr, te))

    return folds
    
if __name__ == "__main__":
    real_path = sys.argv[1]
    predicted_path = sys.argv[2]

    paths, real = util.load_list(real_path)
    _, predicted = util.load_list(predicted_path)

    mea = evaluate(real, predicted, paths)
    print "Sensitivity {:.2f}, \nfppi mean {:.2f}, fppi std {:.2f} \n".format(mea[0], mea[1], mea[2])
    #print "Sensitivity {:.2f}, \nfppi mean {:.2f}, fppi std {:.2f} \niou mean {:.2f} iou std {:.2f}, iou+ mean {:.2f}, iou+ std {:.2f}".format(mea[0], mea[1], mea[2], mea[3], mea[4], mea[5], mea[6])
