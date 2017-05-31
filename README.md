# LND: Lung Nodule Detection

LND is a basic framework to evaluate lung segmentation, candidate detection and false positive reduction (nodule classification) on x-ray images using JSRT and LIDC-IDRI datasets.

## Cite
```
@Inbook{Bobadilla2017,
author="Bobadilla, Julio Cesar Mendoza
and Pedrini, Helio",
title="Lung Nodule Classification Based on Deep Convolutional Neural Networks",
bookTitle="Progress in Pattern Recognition, Image Analysis, Computer Vision, and Applications: 21st Iberoamerican Congress, CIARP 2016, Lima, Peru, November 8--11, 2016, Proceedings",
year="2017",
publisher="Springer International Publishing",
url="http://dx.doi.org/10.1007/978-3-319-52277-7_15"
}
```


## Usage

```bash
usage: lnd.py [-h] [--preprocess-lung PREPROCESS_LUNG]
              [--preprocess-roi PREPROCESS_ROI] [-b BLOB_DETECTOR]
              [--eval-wmci] [-d DESCRIPTOR] [-c CLASSIFIER] [-r REDUCTOR]
              [--fts] [--clf] [--hyp] [-t TARGET] [--cmp CMP] [--fw]
              [--label LABEL] [--bovw BOVW] [--clf-foldwise CLF_FOLDWISE]
              [--cnn CNN] [-l LAYER] [--hybrid] [--sota] [--trf-channels]
              [--streams STREAMS] [--early EARLY] [--late LATE]
              [--cmp-cnn CMP_CNN] [--pre-tr] [--uns-aug] [--rpi RPI]
              [--init INIT] [--pos-neg-ratio POS_NEG_RATIO]
              [--init-transfer INIT_TRANSFER] [--opt OPT] [-a AUGMENT]
              [--roi-size ROI_SIZE] [--blob-rad BLOB_RAD] [--epochs EPOCHS]
              [--frocs-by-epoch]

optional arguments:
  -h, --help            show this help message and exit
  --preprocess-lung PREPROCESS_LUNG
                        Generates a bunch preprocessed images for the input.
                        Exemple --preprocessor norm,lce,ci (generates 3
                        images)
  --preprocess-roi PREPROCESS_ROI
                        Generates a bunch preprocessed images for the input.
                        Exemple --preprocessor norm,lce,ci (generates 3
                        images)
  -b BLOB_DETECTOR, --blob_detector BLOB_DETECTOR
                        Options: wmci(default), TODO hog, log.
  --eval-wmci           Measure sensitivity and fppi without classification
  -d DESCRIPTOR, --descriptor DESCRIPTOR
                        Options: baseline.hardie(default), hog, hogio, lbpio,
                        zernike, shape, all, set1, overf, overfin.
  -c CLASSIFIER, --classifier CLASSIFIER
                        Options: lda(default), svm.
  -r REDUCTOR, --reductor REDUCTOR
                        Feature reductor or selector. Options: none(default),
                        pca, lda, rfe, rlr.
  --fts                 Performs feature extraction.
  --clf                 Performs classification.
  --hyp                 Performs hyperparameter search. The target method to
                        evaluate should be specified using -t.
  -t TARGET, --target TARGET
                        Method to be optimized. Options wmci, pca, lda, rlr,
                        rfe, svm,
  --cmp CMP             Compares results of different models via froc.
                        Options: hog, hog-impls, lbp, clf.
  --fw                  Plot the importance of individual features ( selected
                        clf coef vs anova )
  --label LABEL         Options: nodule, sublety.
  --bovw BOVW           Options: check available configs on bovw.py.
  --clf-foldwise CLF_FOLDWISE
                        Performs classification loading features foldwise.
  --cnn CNN             Evaluate convnet. Options: shallow_1, shallow_2.
  -l LAYER, --layer LAYER
                        Layer index used to extract feature from cnn model.
  --hybrid              Evaluate hybrid approach: convnet + descriptor.
  --sota                Performs classification.
  --trf-channels
  --streams STREAMS     Options: trf (transformations), seg (segmentation),
                        fovea (center-scaling), none (default)
  --early EARLY         Options: trf, seg, fovea, none(lce, default)
  --late LATE           Options: trf, seg, fovea, none(lce, default)
  --cmp-cnn CMP_CNN     Compare models (mod), preprocessing (pre),
                        regularization (reg), optimization (opt), max-pooling
                        stages (mp), number of feature maps (nfm), dropout
                        (dp), mlp width (clf-width, deprecated), common
                        classifiers (skl), hybrid cnn + features (hyb), hybrid
                        model evaluated with linear SVM grid search on C (hyp-
                        hyb)
  --pre-tr              Enable pretraining
  --uns-aug             Enable pretraining
  --rpi RPI             Number of random regions of interest per image you
                        want to augment
  --init INIT           Enable initialization from a existing network
  --pos-neg-ratio POS_NEG_RATIO
                        Set the positive/negative ratio for unsupervised aug
                        experiments
  --init-transfer INIT_TRANSFER
                        Enable initialization from a existing network
  --opt OPT             Select an optimization algorithm: sgd-nesterov,
                        adagrad, adadelta, adam.
  -a AUGMENT, --augment AUGMENT
                        Augmentation configurations: bt, zcabt, xbt.
  --roi-size ROI_SIZE   Size of ROIs after scaling
  --blob-rad BLOB_RAD   Radius used to extract blobs
  --epochs EPOCHS       Number of epochs for pretraining.
  --frocs-by-epoch      Generate a figure with froc curves every 5 epochs

```
