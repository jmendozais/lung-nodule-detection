import argparse
import numpy as np
import cv2
import gc

from skimage.exposure import equalize_hist
from skimage.restoration import denoise_nl_means
from skimage import draw
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
#import matplotlib.pyplot as plt

import preprocess
import detect
import neural
import eval
import util
import augment

import lidc
import jsrt
import lnd

from operator import itemgetter
from menpo.image import Image

from keras.applications import vgg16

def VGG16(mode='ots-feat', filename=None):
    if mode == 'ots-feat':
        network = vgg16.VGG16(include_top=False)
    elif mode == 'ft-fc':
        network = vgg16.VGG16(classes=2)
        for i in range(len(network.layers-6)):
            network.layers[i] = False
    elif mode == 'ft-all':
        network = vgg16.VGG16(classes=2)
    else:
        raise Exception("Undefined model")
    return network

# Off the shelf features
def save_features(filename, feats_tr, y_tr):
    np.save('{}.xtr'.format(filename), feats_tr)
    np.save('{}.ytr'.format(filename), y_tr)

def load_features(filename):
    feats_tr = np.load('{}.xtr.npy'.format(filename))
    y_tr = np.load('{}.ytr.npy'.format(filename))
    return feats_tr, y_tr

def evaluate_classifier(clf, feats_tr, y_tr, real_blobs_te, pred_blobs_te, feats_te):
    y_tr = y_tr.T[1]
    print 'feats {}, {}'.format(feats_tr.shape, y_tr.shape)
    print "raw values ", np.mean(feats_tr, axis=0), np.std(feats_tr, axis=0), np.min(feats_tr), np.max(feats_tr)

    '''
    preproc = augment.Preprocessor()
    feats_tr = preproc.fit_transform(feats_tr, y_tr)
    print "pre values ", np.mean(feats_tr, axis=0), np.std(feats_tr, axis=0), np.min(feats_tr), np.max(feats_tr)
    '''

    #scaler = StandardScaler()
    scaler = MinMaxScaler(feature_range=(-1, 1))
    feats_tr = scaler.fit_transform(feats_tr)
    print "z-scaled ", np.mean(feats_tr, axis=0), np.std(feats_tr, axis=0), np.min(feats_tr), np.max(feats_tr)

    print "Train SVM"
    #clf.fit(feats_tr, y_tr)
    #clf_prob = CalibratedClassifierCV(base_estimator=clf, cv='prefit')
    clf_prob = CalibratedClassifierCV(base_estimator=clf)
    clf_prob.fit(feats_tr, y_tr)

    minv, maxv = 1e10, -1e10
    probs_te = []
    for feats in feats_te:
        print "feats te i {}".format(feats.shape)
        #feats = preproc.transform(feats)
        feats = scaler.transform(feats)
        minv = min(minv, feats.min())
        maxv = max(maxv, feats.max())
        probs_te.append(clf_prob.predict_proba(feats).T[1])
        print 'probs ->', probs_te[-1]

    probs_te = np.array(probs_te)
    print "min, max feats fed on svm {}, {}".format(minv, maxv)
    return eval.froc(real_blobs_te, pred_blobs_te, probs_te)

def scale_vgg(image): 
    roi = cv2.resize(image[0], (224, 224), interpolation=cv2.INTER_CUBIC)
    roi = np.array([roi, roi, roi])
    return roi

def extract_convfeats(network, X, intensity_range):
    # Scale image
    X_scaled = []
    for i in range(len(X)):
        x = scale_vgg(X[i])
        X_scaled.append(x)
    X_scaled = np.array(X_scaled)
    
    # Imagenet preproc 
    X_scaled -= intensity_range[0]
    gc.collect()
    X_scaled /= (intensity_range[1] - intensity_range[0] + 1e-6)
    gc.collect()
    X_scaled *= 255 

    X_scaled[:, 0, :, :] -= 103.939
    X_scaled[:, 1, :, :] -= 116.779
    X_scaled[:, 2, :, :] -= 123.68
    gc.collect()

    # Extract feats
    feats = np.array([])
    if len(X_scaled) > 0:
        feats = network.predict(X_scaled)
        feats = feats.reshape((feats.shape[0], feats.shape[1] * feats.shape[2] * feats.shape[3]))
    gc.collect()
    return feats
 
def extract_convfeats_from_rois(network, rois, intensity_range):
    feats = []
    for i in range(len(rois)): 
        feats.append(extract_convfeats(network, rois[i], intensity_range))
    return np.array(feats)

def extract_features_from_convnet(args): # Load img, blobs and masks
    imgs, blobs, paths = lidc.load(pts=True, set_name=args.ds_tr)
    pred_blobs = detect.read_blobs('data/{}-lidc-pred-blobs.pkl'.format(args.detector))
    masks = np.load('data/aam-lidc-pred-masks.npy')

    assert len(imgs) == len(masks) and len(pred_blobs) == len(masks)
    
    # Load folds
    folds = util.model_selection_folds(imgs)

    # Create rois
    rois = lnd.create_rois(imgs, masks, pred_blobs, args, real_blobs=blobs)
    
    # Load model
    network = VGG16(mode='ots-feat')
    network.summary()

    #  Set up CV
    frocs = []
    legends = ['Fold {}'.format(i + 1) for i in range(util.NUM_VAL_FOLDS)] 
    fold_idx = 0

    for tr, te in folds:
        # TODO: apply extract convfeats funcs for tr and te sets
        print "Fold {}".format(fold_idx + 1)
        X_tr, Y_tr, _, _ = neural.create_train_test_sets(blobs[tr], pred_blobs[tr], rois[tr], blobs[te], pred_blobs[te], rois[te])
        gc.collect()

        generator = augment.get_default_generator((args.roi_size, args.roi_size))
        X_tr, Y_tr = augment.balance_and_perturb(X_tr, Y_tr, generator)
        gc.collect()
        
        ''''
        counta = 0
        countb = 0
        count = 0

        while counta < 10 and countb < 10: 
            if Y_tr[count][1] > 0 and counta < 10:
                util.imshow("positives", X_tr[count][0], display_shape=(256, 256))
                counta += 1
            elif Y_tr[count][1] == 0.0 and countb < 10:
                util.imshow("negatives", X_tr[count][0], display_shape=(256, 256))
                countb += 1
            count += 1
        '''

        range_tr = (X_tr.min(), X_tr.max())
        print "Range {}".format(range_tr)
        print "Extract feats on balanced tr set"
        feats_tr = extract_convfeats(network, X_tr, range_tr)
        save_features("data/{}-f{}-lidc-feats".format(args.detector, fold_idx), feats_tr, Y_tr) 
        gc.collect()

        print "Extract feats on te set"
        feats_te = extract_convfeats_from_rois(network, rois[te], range_tr)
        print "Test feats to save shape {}".format(feats_te.shape)
        np.save("data/{}-f{}-te-lidc-feats.npy".format(args.detector, fold_idx), feats_te)
        gc.collect()

        fold_idx += 1

def model_selection_with_convfeats(args):
    # Load img, blobs and masks
    imgs, blobs, paths = lidc.load(pts=True, set_name=args.ds_tr)
    pred_blobs = detect.read_blobs('data/{}-lidc-pred-blobs.pkl'.format(args.detector))
    masks = np.load('data/aam-lidc-pred-masks.npy')
    assert len(imgs) == len(masks) and len(pred_blobs) == len(masks)
    
    # Load folds
    folds = util.model_selection_folds(imgs)

    # Create rois
    #rois = create_rois(imgs, masks, pred_blobs, args, real_blobs=blobs)

    #  Set up CV
    frocs = []
    legends = ['Fold {}'.format(i + 1) for i in range(util.NUM_VAL_FOLDS)] 
    fold_idx = 0

    import time
    from sklearn.neighbors import KNeighborsClassifier
    for tr, te in folds:
        print "Load features fold {}".format(fold_idx)

        start = time.time()
        feats_tr, Y_tr = load_features('data/{}-f{}-lidc-feats'.format(args.detector, fold_idx))
        print 'tr time {}'.format(time.time() - start)

        start = time.time()
        feats_te = np.load('data/{}-f{}-te-lidc-feats.npy'.format(args.detector, fold_idx))
        print 'te time {}'.format(time.time() - start)

        print "-> tr {}, {}, te {}".format(feats_tr.shape, Y_tr.shape, feats_te.shape)

        # Train/test model
        print "Evaluate clf"
        #clf = KNeighborsClassifier(n_neighbors=3)
        clf = LinearSVC(C=args.svm_C)
        froc = evaluate_classifier(clf, feats_tr, Y_tr, blobs[te], pred_blobs[te], feats_te)
        frocs.append(froc)

        # Record model results
        current_frocs = [eval.average_froc([froc_i]) for froc_i in frocs]
        util.save_froc(current_frocs, 'data/lsvm-C{}-{}-folds-froc'.format(args.svm_C, args.detector), legends[:len(frocs)], with_std=False)
        fold_idx += 1

    legends = ['Val FROC (LIDC-IDRI)']
    average_froc = eval.average_froc(frocs, np.linspace(0.0, 10.0, 101))
    util.save_froc([average_froc], 'data/lsvm-C{}-{}-val-froc'.format(args.svm_C, args.detector), legends, with_std=True)

def model_evaluation_with_convfeats(): 
    raise NotImplemented()

def model_evaluation2_with_convfeats(): 
    raise NotImplemented()

def exp_convfeats(args):
    C_set = np.logspace(-3, 4, 8)
    for i in range(len(C_set)):
        args.svm_C = C_set[i]
        model_selection_with_convfeats(args)

def exp_eval_ots_lidc_jsrt(args):
    # load LIDC & JSRT-positives data
    imgs_tr, blobs_tr = lidc.load()
    pred_blobs_tr = detect.read_blobs('data/{}-lidc-pred-blobs.pkl'.format(args.detector))
    masks_tr = np.load('data/aam-lidc-pred-masks.npy')
    imgs_te, blobs_te = jsrt.load(set_name='jsrt140p')
    pred_blobs_te = detect.read_blobs('data/{}-jsrt140p-pred-blobs.pkl'.format(args.detector))
    masks_te = np.load('data/aam-jsrt140p-pred-masks.npy')
    rois_tr = create_rois(imgs_tr, masks_tr, pred_blobs_tr, args)
    rois_te = create_rois(imgs_te, masks_te, pred_blobs_te, args)

    # Extract features
    network = VGG16(mode='ots-feat')
    feats_tr = extract_convfeats_from_rois(network, rois_tr)
    feats_te = extract_convfeats_from_rois(network, rois_te)
    np.save('data/{}-lidc-feats.npy'.format(args.detector, fold_idx), feats_tr)
    np.save('data/{}-jsrt140p-feats.npy'.format(args.detector, fold_idx), feats_te)

    # Eval classifier
    feats_tr, Y_tr, _, _ = neural.create_train_test_sets(blobs_tr, pred_blobs_tr, rois_tr, None, None, None)
    feats_tr, Y_tr = augment.balance_and_perturb(X_tr, Y_tr)
    clf = LinearSVC(C=args.svm_C)
    froc = evaluate_classifier(clf, feats_tr, Y_tr, blobs_te, pred_blobs_te, feats_te)

def exp_eval_ots_jsrt_only(args):
    # load LIDC & JSRT-positives data
    network = VGG16(mode='ots-feat')
    print "Model Evaluation Protocol 2"
    imgs, blobs= jsrt.load(set_name='jsrt140p')
    pred_blobs = detect.read_blobs('data/{}-jsrt140p-pred-blobs.pkl'.format(args.detector))
    masks = np.load('data/aam-jsrt140p-pred-masks.npy')
    rois = create_rois(imgs, masks, pred_blobs, args)
    folds = KFold(n_splits=5, shuffle=True, random_state=util.FOLDS_SEED).split(imgs)
    #feats = extract_convfeats_from_rois(network, rois)
    feats = extract_convfeats_from_rois(network, rois)

    fold_idx = 0
    frocs = []
    legends = ['Fold {}'.format(i + 1) for i in range(5)] 
    for tr, te in folds:
        # Eval classifier
        feats_tr, Y_tr, _, _ = neural.create_train_test_sets(blobs_tr, pred_blobs_tr, feats[tr], 
            None, None, None)

        feats_tr, Y_tr = augment.balance_and_perturb(feats_tr, Y_tr)

        clf = LinearSVC(C=1.0)
        froc = evaluate_classifier(clf, feats_tr, Y_tr, blobs[te], pred_blobs[te], feats[te])
        frocs.append(froc)
        current_frocs = [eval.average_froc([froc_i]) for froc_i in frocs]
        util.save_froc(current_frocs, 'data/lsvm-{}-jsrtonly-folds'.format(args.detector), legends[:len(frocs)], with_std=False)
        fold_idx += 1

    froc = eval.average_froc(frocs)
    legends = ['Test FROC (JSRT positives)']
    util.save_froc([froc], 'data/lsvm-{}-jsrtonly'.format(args.detector), legends, with_std=True)

# Fine-tuning convnet

def model_selection_fclayers(model):
    for tr, te in folds:
        # Fix conv layers
        # Train classification layers 
        print ''
    # Perform epoch analysis
    raise NotImplemented()

def model_selection_finetuning(model):
    for tr, te in folds:
        # Fix conv layers
        # Train all layers 
        print '' 
    raise NotImplemented()
    
def model_evaluation_finetuning():
    for tr, te in folds:
        # Fix conv layers
        # Train all layers 
        print ''
    raise NotImplemented()

def model_evaluation2_finetuning():
    for tr, te in folds:
        # Fix conv layers
        # Train all layers 
        print '' 
    raise NotImplemented()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='lnd-tl.py')
    lnd.add_feed_forward_convnet_args(parser)
    parser.add_argument('--mode', help='ots-feat, ots-clf, ft-fc, ft-all, ots, ft', default='ots-feat')
    args = parser.parse_args() 

    print args

    if args.file:
        image = np.load(args.file).astype('float32')
        classify(image, args)
    elif args.model_selection:
        if args.mode == 'ots-feat':
            extract_features_from_convnet(args)
        elif args.mode == 'ots-clf':
            args.svm_C = 1.0
            model_selection_with_convfeats(args)
        elif args.mode == 'ft-fc':
            model_selection_ft_fc(args)
        elif args.mode == 'ft-all':
            model_selection_ft_all(args)
        elif args.mode == 'exp-convfeats':
            exp_convfeats(args)
    elif args.model_evaluation:
        if args.mode == 'ots':
            model_evaluation_with_convfeats(args)
        elif args.mode =='ft':
            model_evaluation_finetuning(args)
    elif args.model_evaluation2:
        if args.mode == 'ots':
            model_evaluation2_with_convfeats(args)
        elif args.mode =='ft':
            model_evaluation2_finetuning(args)
    elif args.save_rois:
        save_rois(args)
    else:
        print "Invalid command", args.model_selection, args.mode
