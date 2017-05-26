import argparse
import numpy as np
import cv2
import gc

from skimage.exposure import equalize_hist
from skimage.restoration import denoise_nl_means
from sklearn.cross_validation import KFold
#import matplotlib.pyplot as plt

import preprocess
import detect
import neural
import eval
import util

import lidc
import jsrt

def preprocess_rois(rois, methods):
    assert methods != 'none'
    result = []

    for j in range(len(rois)): 
        result.append([])
        for k in range(len(rois[j])):
            if 'norm' in methods:
                result[-1].append(preprocess.normalize(rois[j][k], np.ones(shape=rois[j][k].shape)))
            if 'norm3' in methods:
                norm = preprocess.normalize(rois[j][k], np.ones(shape=rois[j][k].shape))
                result[-1].append(norm)
                result[-1].append(norm)
                result[-1].append(norm)
            if 'heq' in methods:
                result[-1].append(equalize_hist(rois[j][k]))
            if 'nlm' in methods:
                result[-1].append(denoise_nl_means(rois[j][k]))
        result[-1] = np.array(result[-1])

    return np.array(result)

def create_rois(imgs, masks, blob_set, args, save=False):
    pad_size = 1.15
    dsize = (int(args.roi_size * pad_size), int(args.roi_size * pad_size))

    print 'Preprocess images: # images {}'.format(len(imgs))
    num_blobs = 0

    stats = np.zeros(shape=(len(imgs), 4))
    for i in range(len(imgs)):
        num_blobs += len(blob_set[i])
        if i % 10 == 0:
            print 'Preprocess image {}.'.format(i)

        imgs[i] = preprocess.normalize(imgs[i], masks[i])
        stats[i] = (np.min(imgs[i]), np.max(imgs[i]), np.mean(imgs[i]), np.std(imgs[i]))

        '''
        util.imshow_with_mask(imgs[i][0], masks[i])
        plt.hist(np.array(imgs[i]).ravel(), 256, range=(-10,+10)); 
        plt.ylim(0, 12000)
        plt.show()
        '''

    print 'stats avg', np.mean(stats, axis=0)
    roi_set = []

    print 'Extract rois'
    for i in range(len(imgs)):
        img = imgs[i]
        if i % 10 == 0:
            print 'extract rois {}'.format(i)

        rois = []
        for j in range(len(blob_set[i])):
            x, y, _ = blob_set[i][j]
            r = args.blob_rad
            r = int(r * pad_size)

            shift = 0 
            tl = (x - shift - r, y - shift - r)
            ntl = (max(0, tl[0]), max(0, tl[1]))
            br = (x + shift + r + 1, y + shift + r + 1)
            nbr = (min(img.shape[1], br[0]), min(img.shape[2], br[1]))

            roi = []
            for k in range(img.shape[0]):
                tmp = img[k][ntl[0]:nbr[0], ntl[1]:nbr[1]]
                tmp = cv2.resize(tmp, dsize, interpolation=cv2.INTER_CUBIC)
                roi.append(tmp)
                
            rois.append(roi)
        rois = np.array(rois)

        if save:
            rois = args.file.create_dataset('tmp_{}'.format(i), data=rois)

        roi_set.append(rois)
        gc.collect()

    if args.preprocess_roi != 'none':
        print 'Preprocess rois: {}'.format(args.preprocess_roi)
        # stats
        tmp = np.full((4, len(roi_set)), fill_value=0, dtype=np.float)
        for i in range(len(roi_set)):
            if i % 10 == 0: 
                print 'preprocess rois {} '.format(i)
            if save:
                roi_set[i] = args.file.create_dataset('rois_{}'.format(i), data=preprocess_rois(roi_set[i], args.preprocess_roi))
                args.file.create_dataset('pred_blobs_{}'.format(i), data=blob_set[i])
            else:
                roi_set[i] = preprocess_rois(roi_set[i], args.preprocess_roi)

            tmp[0][i] = np.min(roi_set[i]) if len(roi_set[i]) > 0 else 0
            tmp[1][i] = np.max(roi_set[i]) if len(roi_set[i]) > 0 else 0
            tmp[2][i] = np.mean(roi_set[i])if len(roi_set[i]) > 0 else 0
            tmp[3][i] = np.std(roi_set[i]) if len(roi_set[i]) > 0 else 0

        print("norm rois: min {}, max {}, mean {}, std {}".format(np.mean(tmp[0]), np.mean(tmp[1]), np.mean(tmp[2]), np.mean(tmp[3])))
    return np.array(roi_set)

def evaluate_model(model, real_blobs_tr, pred_blobs_tr, rois_tr, real_blobs_te, pred_blobs_te, rois_te):
    X_tr, Y_tr, X_te, Y_te= neural.create_train_test_sets(real_blobs_tr, pred_blobs_tr, rois_tr, real_blobs_te, pred_blobs_te, rois_te)
    history = model.fit(X_tr, Y_tr, X_te, Y_te)
    pred_blobs_te, probs_te = neural.predict_proba(model, pred_blobs_te, rois_te)
    return eval.froc(real_blobs_te, pred_blobs_te, probs_te)

def model_selection(model_name, args):
    '''
    imgs, blobs = lidc.load()
    pred_blobs = detect.read_blobs('data/wmci-aam-lidc-pred-blobs.pkl')
    masks = np.load('data/aam-lidc-pred-masks.npy')
    '''
    imgs, blobs = jsrt.load(set_name='jsrt140')
    pred_blobs = detect.read_blobs('data/wmci-aam-jsrt140-pred-blobs.pkl')
    masks = np.load('data/aam-jsrt140-pred-masks.npy')
    folds = util.model_selection_folds(imgs)

    #gt_rois = create_rois(imgs, masks, blobs, args)
    rois = create_rois(imgs, masks, pred_blobs, args)
    '''
    for i in range(len(blobs)):
        for j in range(len(blobs[i])):
            print blobs[i][j]
            print rois[i][j].shape
            print rois[i][j].shape
            util.imshow("img", imgs[i][0], display_shape=(400, 400))
            util.imshow("gt ROI", gt_rois[i][j][0], display_shape=(200, 200))
            for k in range(len(pred_blobs[i])):
                if ((blobs[i][j][0] - pred_blobs[i][k][0])**2 + (blobs[i][j][1] - pred_blobs[i][k][1])**2)**.5 < 31.43:
                    util.imshow("candidate ROI", rois[i][k][0], display_shape=(200, 200))
    '''

    frocs = []
    legends = ['Fold {}'.format(i + 1) for i in range(util.NUM_VAL_FOLDS)] 

    fold_idx = 0
    for tr, te in folds:
        model = neural.create_network(model_name, (1, args.roi_size, args.roi_size)) 
        model.name = model.name + '.fold-{}'.format(fold_idx + 1)
        froc = evaluate_model(model, blobs[tr], pred_blobs[tr], rois[tr], blobs[te], pred_blobs[te], rois[te])
        frocs.append(froc)

        current_frocs = [eval.average_froc([froc_i]) for froc_i in frocs]
        util.save_froc(current_frocs, 'data/{}-folds-froc'.format(model_name), legends[:len(frocs)], with_std=False)
        model.save('data/' + model.name)
        fold_idx += 1

    legends = ['Val FROC (LIDC-IDRI)']
    average_froc = eval.average_froc(frocs, np.linspace(0.0, 10.0, 101))
    util.save_froc([average_froc], 'data/{}-val-froc'.format(model_name), legends, with_std=True)

def model_evaluation(model_name, args):
    imgs_tr, blobs_tr = lidc.load()
    pred_blobs_tr = detect.read_blobs('data/wmci-aam-lidc-pred-blobs.pkl')
    masks_tr = np.load('data/aam-lidc-pred-masks.npy')
    imgs_te, blobs_te = jsrt.load(set_name='jsrt140p')
    pred_blobs_te = detect.read_blobs('data/wmci-aam-jsrt140p-pred-blobs.pkl')
    masks_te = np.load('data/aam-jsrt140p-pred-masks.npy')

    rois_tr = create_rois(imgs_tr, masks_tr, pred_blobs_tr, args)
    rois_te = create_rois(imgs_te, masks_te, pred_blobs_te, args)

    model = neural.create_network(model_name, (1, args.roi_size, args.roi_size)) 
    froc = evaluate_model(model, blobs_tr, pred_blobs_tr, rois_tr, blobs_te, pred_blobs_te, rois_te)
    froc = eval.average_froc([froc])

    legends = ['Test FROC (JSRT positives)']
    util.save_froc([froc], 'data/{}-jsrt140p-froc'.format(model_name), legends, with_std=False)

def eval_trained_model(model_name, args):
    imgs, blobs = lidc.load()
    pred_blobs = detect.read_blobs('data/wmci-aam-lidc-pred-blobs.pkl')
    masks = np.load('data/aam-lidc-pred-masks.npy')

    folds = util.model_selection_folds(imgs)
    rois = create_rois(imgs, masks, pred_blobs, args)

    frocs = []
    legends = []

    fold_idx = 0
    
    model = neural.create_network(model_name, (1, args.roi_size, args.roi_size)) 
    model_name = model.name

    epochs = model.training_params['nb_epoch']
    
    frocs = []
    for tr, te in folds:
        model.load('data/' + model_name + '.fold-{}'.format(fold_idx + 1))
        frocs.append([])
        for epoch in range(1, epochs + 1, 2):
            weights_file_name = 'data/{}.weights.{:02d}.hdf5'.format(model.name, epoch)
            model.network.load_weights(weights_file_name)
            pred_blobs_te, probs_te = neural.predict_proba(model, pred_blobs[te], rois[te])
            frocs[fold_idx].append(eval.froc(blobs[te], pred_blobs_te, probs_te))
        fold_idx += 1

    frocs = np.array(frocs)
    froc_history = []
    legends = []

    i = 0
    for epoch in range(1, epochs + 1, 2):
        frocs_by_epoch = frocs[:,i]
        froc_history.append(eval.average_froc(np.array(frocs_by_epoch), np.linspace(0.0, 10.0, 101)))
        legends.append('Val FROC (LIDC-IDRI), epoch {}'.format(epoch))
        i += 1

    util.save_froc(froc_history, 'data/{}-val-froc-by-epoch'.format(model_name), legends, with_std=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='lnd.py')

    parser.add_argument('--model', help='Evaluate convnet.', default='none')
    parser.add_argument('--model-selection', help='Perform model selection protocol', action='store_true') 
    parser.add_argument('--model-selection-detailed', help='Perform model selection protocol', action='store_true') 
    parser.add_argument('--model-evaluation', help='Perform model evaluation protocol', action='store_true') 
    
    parser.add_argument('--roi-size', help='Size of ROIs after scaling', default=32, type=int)
    parser.add_argument('--blob-rad', help='Radius used to extract blobs', default=32, type=int)

    parser.add_argument('--preprocess-roi', help='Preproc ROIs with a given method', default='norm')
    parser.add_argument('-b', '--blob-detector', help='Options: wmci-mean-shape, wmci-aam.', default='wmci-aam')

    args = parser.parse_args() 

    if args.model_selection_detailed:
        eval_trained_model(args.model, args)
    if args.model_selection:
        model_selection(args.model, args)
    if args.model_evaluation:
        model_evaluation(args.model, args)
