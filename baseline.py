#!/usr/bin/env python
import sys
import time
from itertools import product

import numpy as np
from scipy.interpolate import interp1d
from sklearn.cross_validation import StratifiedKFold
from sklearn import svm
from sklearn import lda
from sklearn import decomposition
from sklearn import feature_selection as selection
from sklearn import linear_model
from sklearn import ensemble
from sklearn.metrics import auc
import matplotlib.pyplot as plt

import sys
import argparse

step = 10
fppi_range = np.linspace(0.0, 10.0, 101)

''' false positive values for comparison '''

interesting_fps = [2, 4, 4.3, 5, 5.4]

''' reported operating points '''
''' Using the whole JSRT database or using 0.9... * whole database. '''

sota_authors = [
    'Chen and Suzuki.',
    'Hardie et al.',
    'Schilham et al.',
    'Shiraishi et al.',
    'Coppini et al.',
    'Wei et al.',
    'Wang et al.'
]

scale_factor = 0.9091
#chen = np.array([[5, 0.85 * scale_factor ], [2, 0.779 * scale_factor]])
chen = np.array([[5, 0.85 * scale_factor ]])
hardie = np.array([[2, 0.571], [4, 0.71], [4.3, 0.714], [5, 0.728], [5.4, 0.756]])
shiraishi = np.array([[4.2, 0.704 * scale_factor], [5, 0.701 * scale_factor]])
schilham = np.array([[2, 0.51], [4, 0.67], [5.4, 0.73]])
coppini = np.array([[4.3, 0.60], [7.6, 0.70], [10.2, 0.75]])
wei = np.array([[5.4, 0.80]])
wang = np.array([[1.0, 0.692 * scale_factor]])

sota_ops = [
    chen,
    hardie,
    shiraishi,
    schilham,
    coppini,
    wei,
    wang
]

hardie = np.array([
[0.0, 0.0], [0.1, 0.2], [0.2, 0.3],[0.3, 0.38], [0.4, 0.415], [0.5, 0.46], [0.6, 0.48], [0.7, 0.51], [0.9, 0.53], 
[1.0, 0.565], [1.18, 0.585], [1.25, 0.6], [1.3, 0.615], [1.5, 0.642], [1.57, 0.675], [1.70, 0.685],
[2.0, 0.71], [2.23, 0.712], [2.44, 0.74], [2.54, 0.75], [2.64, 0.75], [2.75, 0.755], [2.8,0.77],
[3.0, 0.77], [3.15, 0.775], [3.73, 0.775], [3.80, 0.783],
[4.0, 0.785], [4.12, 0.79], [4.7, 0.79], [4.8, 0.80], [4.9, 0.81],
[5.0, 0.81], [5.05, 0.82 ], [5.95, 0.83],
[6.0, 0.83], [6.1, 0.835 ], [6.45,0.835], [6.55, 0.84], [6.7, 0.85], 
[7.0, 0.85], [7.6, 0.85], [7.65, 0.855],
[8.0, 0.855], [8.15, 0.855], [8.23, 0.863], [8.36, 0.866],  [8.7, 0.89],
[9.0, 0.89], [9.2, 0.89], [9.3, 0.905], [9.4, 0.905], [9.6, 0.91], [9.7, 0.92], [9.9, 0.93],
[10.0, 0.93]
])

fun = interp1d(hardie.T[0], hardie.T[1], kind='linear', fill_value=0, bounds_error=False)
hardie = np.array([fppi_range, fppi_range.copy()])
hardie[1] = fun(hardie[0])
hardie = hardie.T

horvath = np.array([[0.0, 0.0], [0.5, 0.49], [1.0, 0.63], [1.5, 0.68], [2.0, 0.72], [2.5, 0.75], [3.0, 0.78], [3.5, 0.79], [4.0, 0.81], [10.0, 0.81]])
fun = interp1d(horvath.T[0], horvath.T[1], kind='linear', fill_value=0, bounds_error=False)
horvath = np.array([fppi_range, fppi_range.copy()])
horvath[1] = fun(horvath[0])
horvath = horvath.T

vde = np.array([[0.0, 0.0], [0.5, 0.56], [1.0, 0.66], [1.5, 0.68], [2.0, 0.78], [2.5, 0.79], [3.0, 0.81], [3.5, 0.84], [4.0, 0.85], [5.0, 0.86], [10.0, 0.86]])
fun = interp1d(vde.T[0], vde.T[1], kind='linear', fill_value=0, bounds_error=False)
vde = np.array([fppi_range, fppi_range.copy()])
vde[1] = fun(vde[0])
vde = vde.T

op_set = [hardie, horvath, vde]
legend = ['Hardie', 'Horvath', 'VDE']

#if __name__ == '__main__':
#    util.save_froc(op_set, 'baselines', legend)



