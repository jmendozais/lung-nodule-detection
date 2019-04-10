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

## How to run the project

1. Download the datasets
- JSRT: http://db.jsrt.or.jp/eng.php
- LIDC-IDRI: https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI (Images and annotations)
- SCR: https://www.isi.uu.nl/Research/Databases/SCR/

2. Prepare LIDC-IDRI dataset with the script lidc.py

3. Train and evaluate the segmentation model with the script segment.py

4. Get nodule candidates for evaluation with the script detect.py

5. Train and evaluate the convnet to classify candidates with lnd.py
