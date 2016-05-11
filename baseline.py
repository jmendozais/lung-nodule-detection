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

from data import DataProvider
import model
import eval
import util
import sys
import argparse

import jsrt

step = 10
fppi_range = np.linspace(0.0, 5.0, 101)

hardie = np.array([[0.0, 0.0], [0.1, 0.2], [0.2, 0.3],[0.3, 0.38], [0.4, 0.415], [0.5, 0.46], [0.6, 0.48], [0.7, 0.51], [0.9, 0.53], [1.0, 0.57], [1.5, 0.67], [2.0, 0.72], [2.5, 0.75],[3.0, 0.78], [4.0, 0.79], [5.0, 0.81], [6.0, 0.82], [7.0, 0.85], [8.0, 0.86], [9.0, 0.895], [10.0, 0.93]])
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
util.save_froc(op_set, 'baselines', legend)



