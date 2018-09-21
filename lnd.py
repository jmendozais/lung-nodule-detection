import argparse
import numpy as np
import cv2
import gc

from skimage.exposure import equalize_hist
from skimage.restoration import denoise_nl_means
from skimage import draw
from sklearn.model_selection import KFold

import preprocess
import detect
import neural
import eval
import util

import lidc
import jsrt

from operator import itemgetter
from menpo.image import Image

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

def create_rois(imgs, masks, blob_set, args, save=False, real_blobs=None, paths=None):
    pad_size = 1.5
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
            x, y, r = blob_set[i][j]
            x = int(x)
            y = int(y)
            if args.blob_rad > 0:
                r = args.blob_rad
            else:
                r *= 1.5
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
        for i in range(len(roi_set)):
            if i % 10 == 0: 
                print 'preprocess rois {} '.format(i)
            if save:
                roi_set[i] = args.file.create_dataset('rois_{}'.format(i), data=preprocess_rois(roi_set[i], args.preprocess_roi))
                args.file.create_dataset('pred_blobs_{}'.format(i), data=blob_set[i])
            else:
                roi_set[i] = preprocess_rois(roi_set[i], args.preprocess_roi)

    return np.array(roi_set)

def save_rois(args):
    imgs_tr, blobs_tr = lidc.load(pts=False)
    pred_blobs_tr = detect.read_blobs('data/sbf-aam-lidc-pred-blobs.pkl')
    masks_tr = np.load('data/aam-lidc-pred-masks.npy')

    imgs_te, blobs_te = jsrt.load(set_name='jsrt140p')
    pred_blobs_te = detect.read_blobs('data/sbf-aam-jsrt140p-pred-blobs.pkl')
    masks_te = np.load('data/aam-jsrt140p-pred-masks.npy')

    rois_tr = create_rois(imgs_tr, masks_tr, pred_blobs_tr, args, real_blobs=blobs_tr)
    rois_te = create_rois(imgs_te, masks_te, pred_blobs_te, args, real_blobs=blobs_te)
    X_tr, Y_tr, X_te, Y_te = neural.create_train_test_sets(blobs_tr, pred_blobs_tr, rois_tr, blobs_te, pred_blobs_te, rois_te)
    X_tr, Y_tr = util.split_data_pos_neg(X_tr, Y_tr)
    X_te, Y_te = util.split_data_pos_neg(X_te, Y_te)

    X_pos = X_tr[0]
    idx = np.random.randint(0, len(X_tr[1]), len(X_pos))
    X_neg = X_tr[1][idx]

    print len(X_pos), len(X_neg)
    for i in  range(len(X_pos)):
        util.imwrite('data/lidc/roi{}p.jpg'.format(i), X_pos[i][0])
        np.save('data/lidc/roi{}p.npy'.format(i), X_pos[i])
        util.imwrite('data/lidc/roi{}n.jpg'.format(i), X_neg[i][0])
        np.save('data/lidc/roi{}n.npy'.format(i), X_neg[i])
    
    X_pos = X_te[0]
    idx = np.random.randint(0, len(X_te[1]), len(X_pos))
    X_neg = X_te[1][idx]

    print len(X_pos), len(X_neg)
    for i in  range(len(X_pos)):
        util.imwrite('data/jsrt140/roi{}p.jpg'.format(i), X_pos[i][0])
        np.save('data/jsrt140/roi{}p.npy'.format(i), X_pos[i])
        util.imwrite('data/jsrt140/roi{}n.jpg'.format(i), X_neg[i][0])
        np.save('data/jsrt140/roi{}n.npy'.format(i), X_neg[i])

def evaluate_model(model, real_blobs_tr, pred_blobs_tr, rois_tr, real_blobs_te, pred_blobs_te, rois_te, load_model=False):
    X_tr, Y_tr, X_te, Y_te = neural.create_train_test_sets(real_blobs_tr, pred_blobs_tr, rois_tr, real_blobs_te, pred_blobs_te, rois_te)

    if load_model == True:
        print 'load weights {}'.format(model.name)
        model.network.load_weights('data/{}_weights.h5'.format(model.name))
        # FIX: remove and add zmuv mean and zmuv std no Preprocessor augment.py
        if not hasattr(model.preprocessor, 'zmuv_mean'):
            model.preprocessor.fit(X_tr, Y_tr)
    else:
        _ = model.fit(X_tr, Y_tr, X_te, Y_te)

    model.save('data/' + model.name)

    pred_blobs_te, probs_te, _ = neural.predict_proba(model, pred_blobs_te, rois_te)
    return eval.froc(real_blobs_te, pred_blobs_te, probs_te)

def model_selection(model_name, args):
    # Load img, blobs and masks
    imgs, blobs, paths = lidc.load(pts=True, set_name=args.ds_tr)
    if args.ds_tr != args.ds_val:
        _, blobs_val,_  = lidc.load(pts=True, set_name=args.ds_val)
    else:
        blobs_val = blobs

    pred_blobs = detect.read_blobs('data/{}-lidc-pred-blobs.pkl'.format(args.detector))
    masks = np.load('data/aam-lidc-pred-masks.npy')
    assert len(imgs) == len(masks) and len(pred_blobs) == len(masks)
    
    # Load folds
    folds = util.model_selection_folds(imgs)

    # Create rois
    rois = create_rois(imgs, masks, pred_blobs, args, real_blobs=blobs)
    rois_val = create_rois(imgs, masks, pred_blobs, args, real_blobs=blobs_val)

    #  Set up CV
    frocs = []
    legends = ['Fold {}'.format(i + 1) for i in range(util.NUM_VAL_FOLDS)] 
    fold_idx = 0

    for tr, te in folds:
        # Load and setup model
        model = neural.create_network(model_name, args, (1, args.roi_size, args.roi_size)) 
        model.network.summary()
        model.name = model.name + '.fold-{}'.format(fold_idx + 1)
        if args.load_model:
            print "Loading model: data/{}".format(model.name)
            model.load('data/' + model.name)

        # Train/test model
        froc = evaluate_model(model, blobs[tr], pred_blobs[tr], rois[tr], blobs_val[te], pred_blobs[te], rois_val[te], args.load_model)
        frocs.append(froc)

        # Record model results
        current_frocs = [eval.average_froc([froc_i]) for froc_i in frocs]
        util.save_froc(current_frocs, 'data/{}-{}-folds-froc'.format(model.name[:-7], args.detector), legends[:len(frocs)], with_std=False)
        model.save('data/' + model.name)
        fold_idx += 1

    legends = ['Val FROC (LIDC-IDRI)']
    average_froc = eval.average_froc(frocs, np.linspace(0.0, 10.0, 101))
    util.save_froc([average_froc], 'data/{}-{}-val-froc'.format(model.name[:-7], args.detector), legends, with_std=True)

    #save_performance_history(model_name, args, rois, blobs, pred_blobs, folds)

def model_selection_unsup(model_name, args):
    imgs, blobs, paths = lidc.load(pts=True)
    pred_blobs = detect.read_blobs('data/{}-lidc-pred-blobs.pkl'.format(args.detector))
    masks = np.load('data/aam-lidc-pred-masks.npy')

    assert len(imgs) == len(masks) and len(pred_blobs) == len(masks)
    
    folds = util.model_selection_folds(imgs)
    rois = create_rois(imgs, masks, pred_blobs, args, real_blobs=blobs)

    frocs = []
    legends = ['Fold {}'.format(i + 1) for i in range(util.NUM_VAL_FOLDS)] 

    fold_idx = 0
    for tr, te in folds:
        model = neural.create_network(model_name, args, (1, args.roi_size, args.roi_size)) 
        model.name = model.name + '.fold-{}'.format(fold_idx + 1)
        froc = evaluate_model(model, blobs[tr], pred_blobs[tr], rois[tr], blobs[te], pred_blobs[te], rois[te])
        frocs.append(froc)

        current_frocs = [eval.average_froc([froc_i]) for froc_i in frocs]
        util.save_froc(current_frocs, 'data/{}-{}-folds-froc'.format(model_name, args.detector), legends[:len(frocs)], with_std=False)
        model.save('data/' + model.name)
        fold_idx += 1

    legends = ['Val FROC (LIDC-IDRI)']
    average_froc = eval.average_froc(frocs, np.linspace(0.0, 10.0, 101))
    util.save_froc([average_froc], 'data/{}-{}-val-froc'.format(model_name, args.detector), legends, with_std=True)

def model_evaluation_tr_lidc_te_jsrt(model_name, args):
    imgs_tr, blobs_tr = lidc.load()
    pred_blobs_tr = detect.read_blobs('data/{}-lidc-pred-blobs.pkl'.format(args.detector))
    masks_tr = np.load('data/aam-lidc-pred-masks.npy')
    imgs_te, blobs_te = jsrt.load(set_name='jsrt140p')
    pred_blobs_te = detect.read_blobs('data/{}-jsrt140p-pred-blobs.pkl'.format(args.detector))
    masks_te = np.load('data/aam-jsrt140p-pred-masks.npy')

    rois_tr = create_rois(imgs_tr, masks_tr, pred_blobs_tr, args)
    rois_te = create_rois(imgs_te, masks_te, pred_blobs_te, args)

    model = neural.create_network(model_name, args, (1, args.roi_size, args.roi_size)) 
    model.name += '-{}-lidc'.format(args.detector)
    froc = evaluate_model(model, blobs_tr, pred_blobs_tr, rois_tr, blobs_te, pred_blobs_te, rois_te)
    froc = eval.average_froc([froc])

    legends = ['Test FROC (JSRT positives)']
    util.save_froc([froc], 'data/{}-{}-lidc-jsrt-froc'.format(model.name, args.detector), legends, with_std=False)

def model_evaluation_jsrt_only(model_name, args):
    print "Model Evaluation Protocol 2"
    imgs, blobs = jsrt.load(set_name='jsrt140p')
    pred_blobs = detect.read_blobs('data/{}-jsrt140p-pred-blobs.pkl'.format(args.detector))
    masks = np.load('data/aam-jsrt140p-pred-masks.npy')
    rois = create_rois(imgs, masks, pred_blobs, args)
    folds = KFold(n_splits=5, shuffle=True, random_state=util.FOLDS_SEED).split(imgs)

    fold_idx = 0
    frocs = []
    legends = ['Fold {}'.format(i + 1) for i in range(5)] 
    for tr, te in folds:
        model = neural.create_network(model_name, args, (1, args.roi_size, args.roi_size)) 
        model.name = model.name + '-{}-lidc.fold-{}'.format(args.detector, fold_idx + 1)
        froc = evaluate_model(model, blobs[tr], pred_blobs[tr], rois[tr], blobs[te], pred_blobs[te], rois[te])
        frocs.append(froc)

        current_frocs = [eval.average_froc([froc_i]) for froc_i in frocs]
        util.save_froc(current_frocs, 'data/{}-{}-only-jsrt-folds'.format(model.name[:-7], args.detector), legends[:len(frocs)], with_std=False)
        model.save('data/' + model.name)
        fold_idx += 1

    froc = eval.average_froc(frocs)
    legends = ['Test FROC (JSRT positives)']
    util.save_froc([froc], 'data/{}-{}-only-jsrt'.format(model.name[:-7], args.detector), legends, with_std=True)

def model_output(model_name, args):
    print "Model Outputs"
    imgs, blobs = jsrt.load(set_name='jsrt140p')
    pred_blobs = detect.read_blobs('data/{}-jsrt140p-pred-blobs.pkl'.format(args.detector))
    masks = np.load('data/aam-jsrt140p-pred-masks.npy')
    rois = create_rois(imgs, masks, pred_blobs, args)
    folds = KFold(n_splits=5, shuffle=True, random_state=util.FOLDS_SEED).split(imgs)

    fold_idx = 0
    frocs = []
    legends = ['Fold {}'.format(i + 1) for i in range(5)] 

    index = np.array(range(len(imgs)))
    for tr, te in folds:
        X_tr, Y_tr, _, _ = neural.create_train_test_sets(blobs[tr], pred_blobs[tr], rois[tr], blobs[te], pred_blobs[te], rois[te])
        model = neural.create_network(model_name, args, (1, args.roi_size, args.roi_size)) 
        model.name = model.name + '-{}-lidc.fold-{}'.format(args.detector, fold_idx + 1)
        model.network.load_weights('data/{}_weights.h5'.format(model.name))
        if not hasattr(model.preprocessor, 'zmuv_mean'):
            model.preprocessor.fit(X_tr, Y_tr)

        print "Predict ..." 
        pred_blobs_te, probs_te, rois_te = neural.predict_proba(model, pred_blobs[te], rois[te])

        print "Save ..." 
        eval.save_outputs(imgs[te], blobs[te], pred_blobs_te, probs_te, rois_te, index[te])

def model_evaluation(model_name, args):
    model_evaluation_tr_lidc_te_jsrt(model_name, args)
    model_evaluation_jsrt_only(model_name, args)

def visual_results_jsrt_only(model_name, args):
    print "Visual results for model {} JSRT only".format(model_name)
    imgs, blobs = jsrt.load(set_name='jsrt140p')
    pred_blobs = detect.read_blobs('data/{}-jsrt140p-pred-blobs.pkl'.format(args.detector))
    masks = np.load('data/aam-jsrt140p-pred-masks.npy')
    rois = create_rois(imgs, masks, pred_blobs, args)
    folds = KFold(n_splits=5, shuffle=True, random_state=util.FOLDS_SEED).split(imgs)
    fold_idx = 0
    for tr, te in folds:
        model.load('data/' + model.name + '.fold-{}'.format(fold_idx + 1))
        model = neural.create_network(model_name, args, (1, args.roi_size, args.roi_size)) 
        X_tr, Y_tr, X_te, Y_te = neural.create_train_test_sets(real_blobs_tr, pred_blobs_tr, rois_tr, real_blobs_te, pred_blobs_te, rois_te)

        print 'load weights {}'.format(model.name)
        model.network.load_weights('data/{}_weights.h5'.format(model.name))
        # FIX: remove and add zmuv mean and zmuv std no Preprocessor augment.py
        if not hasattr(model.preprocessor, 'zmuv_mean'):
            model.preprocessor.fit(X_tr, Y_tr)

        model.save('data/' + model.name)
        pred_blobs_te, probs_te = neural.predict_proba(model, pred_blobs_te, rois_te)
        util.save_rois_with_probs(rois_te, probs_te)
        fold_idx += 1

def save_performance_history(model_name, args, rois, blobs, pred_blobs, folds):
    model = neural.create_network(model_name, args, (1, args.roi_size, args.roi_size)) 
    model_name = model.name
    epochs = model.training_params['nb_epoch']
    frocs = []
    legends = []

    fold_idx = 0
    for tr, te in folds:
        model.load('data/' + model_name + '.fold-{}'.format(fold_idx + 1))
        frocs.append([])
        epochs_set = list(range(1, epochs + 1, 2))

        for epoch in epochs_set:
            weights_file_name = 'data/{}.weights.{:02d}.hdf5'.format(model.name, epoch)
            model.network.load_weights(weights_file_name)
            pred_blobs_te, probs_te = neural.predict_proba(model, pred_blobs[te], rois[te])
            frocs[fold_idx].append(eval.froc(blobs[te], pred_blobs_te, probs_te))
        fold_idx += 1

    frocs = np.array(frocs)
    froc_history = []
    aucs_history = []
    legends = []

    i = 0
    print "check -> frocs.shape {}".format(frocs.shape)
    for epoch in range(1, epochs + 1, 2):
        frocs_by_epoch = frocs[:,i]
        froc_history.append(eval.average_froc(np.array(frocs_by_epoch), np.linspace(0.0, 10.0, 101)))
        aucs_history.append([])
        aucs_history[-1].append(util.auc(froc_history[-1], np.linspace(0.2, 4.0, 101))**2)
        aucs_history[-1].append(util.auc(froc_history[-1], np.linspace(0.0, 5.0, 101))**2)
        aucs_history[-1].append(util.auc(froc_history[-1], np.linspace(0.0, 10.0, 101))**2)
        legends.append('Val FROC (LIDC-IDRI), epoch {}'.format(epoch))
        i += 1

    util.save_froc(froc_history, 'data/{}-val-froc-by-epoch'.format(model_name), legends, with_std=False)
    util.save_aucs(list(range(1, epochs + 1, 2)), aucs_history, 'data/{}-val-aucs'.format(model_name), ['AUC between 2-4', 'AUC between 0-5', 'AUC between 0-10'])

# Deprecated
def eval_trained_model(model_name, args):
    imgs, blobs = lidc.load()
    pred_blobs = detect.read_blobs('data/{}-lidc-pred-blobs.pkl'.format(args.detector))
    masks = np.load('data/aam-lidc-pred-masks.npy')

    folds = util.model_selection_folds(imgs)
    rois = create_rois(imgs, masks, pred_blobs, args)
    save_performance_history(model_name, args, rois, blobs, pred_blobs, folds)
    
# TODO: name of model to load
def classify(image, args):
    image = preprocess.antialiasing_dowsample(image, downsample=True)
    image = np.array([image])
    blobs, probs, mask = detect.detect_func(image[0], 'sbf', 'aam', 0.5) 
    rois = create_rois([image], mask, [blobs], args)
    model = neural.create_network(args.model, args, (1, args.roi_size, args.roi_size)) 
    model.name += '-lidc'
    model.load('data/' + model.name)
    blobs, probs = neural.predict_proba(model, [blobs], rois)
    blobs, probs = blobs[0], probs[0]
    entries = [[blobs[i], probs[i]] for i in range(len(blobs))]
    entries = list(reversed(sorted(entries, key=itemgetter(1))))
    top_blobs = []
    top_probs = []
    for i in range(args.fppi):
        top_blobs.append(entries[i][0])
        top_probs.append([entries[i][1]])

    util.imwrite_with_blobs('data/classified', image[0], top_blobs)
    return blobs, probs, mask

def add_feed_forward_convnet_args(parser):
    parser.add_argument('file', nargs='?', default=None, type=str)
    parser.add_argument('--save-blobs', help='Use the detector and segmentator to generate blobs', action='store_true')
    parser.add_argument('--model', help='Evaluate convnet.', default='none')
    parser.add_argument('--load-model', action='store_true')

    # Protocols
    parser.add_argument('--model-selection', help='Perform model selection protocol', action='store_true') 
    parser.add_argument('--model-selection-detailed', help='Perform model selection protocol', action='store_true') 
    parser.add_argument('--model-evaluation', help='Perform model evaluation protocol', action='store_true') 
    parser.add_argument('--model-evaluation2', help='Perform model evaluation protocol', action='store_true') 
    parser.add_argument('--model-output', help='Perform model evaluation protocol', action='store_true') 
    parser.add_argument('--model-eval-jsrt', help='Perform model evaluation protocol', action='store_true') 

    # Model params
    parser.add_argument('--roi-size', help='Size of ROIs after scaling', default=64, type=int)
    parser.add_argument('--blob-rad', help='Radius used to extract blobs', default=32, type=int)
    parser.add_argument('--preprocess-roi', help='Preproc ROIs with a given method', default='norm')
    parser.add_argument('--save-rois', help='Perform model evaluation protocol', action='store_true') 
    parser.add_argument('--detector', help='Detector', default='sbf-0.7-aam')
    parser.add_argument('--ds-tr', help='Detector', default='lidc-idri-npy-r1-r2')
    parser.add_argument('--ds-val', help='Detector', default='lidc-idri-npy-r1-r2')
    parser.add_argument('--fppi', help='False positives per image', default=4)
    
    # Network params: Default Convnet(6, 32, 1)
    parser.add_argument('--lr', help='Learning rate', default=0.001, type=float)
    parser.add_argument('--epochs', help='Number of epochs', default=70, type=int)
    parser.add_argument('--da-rot', help='Rotation range in data augmentation', default=18, type=int)
    parser.add_argument('--da-tr', help='Translation range in data augmentation', default=0.12, type=float)
    parser.add_argument('--da-zoom', help='Zoom upper bound data augmentation', default=1.25, type=float)
    parser.add_argument('--da-is', help='Intesity shift data augmentation', default=0.2, type=float)
    parser.add_argument('--da-flip', help='Flip data augmentation', default=1, type=int)

    parser.add_argument('--dropout', help='Fixed dp for conv layers or slope on variable dp', default=0.05, type=float)
    parser.add_argument('--dp-intercept', help='Incercept on variable dp', default=-0.1, type=float)
    parser.add_argument('--lidp', help='Linear increasing dropout', action='store_true')

    parser.add_argument('--conv', help='Number of conv layers', default=6, type=int)
    parser.add_argument('--filters', help='Number of filters on conv layers', default=32, type=int)
    parser.add_argument('--fc', help='Number of fully connected layers', default=1, type=int)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='lnd.py')
    add_feed_forward_convnet_args(parser)
    args = parser.parse_args() 

    if args.file:
        image = np.load(args.file).astype('float32')
        classify(image, args)
    elif args.model_selection_detailed:
        eval_trained_model(args.model, args)
    elif args.model_selection:
        model_selection(args.model, args)
    elif args.model_output:
        model_output(args.model, args)
    elif args.model_evaluation:
        model_evaluation(args.model, args)
    elif args.model_evaluation2:
        model_evaluation2(args.model, args)
    elif args.save_rois:
        save_rois(args)
