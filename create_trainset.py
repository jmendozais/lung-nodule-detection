import sys
from random import *
from math import *
from time import *

import numpy as np
import cv2

import jsrt
from util import *
from model import *
from classify import create_training_set

if __name__ == '__main__':
	out_file = sys.argv[1]
	X, Y = create_training_set()
	np.save(out_file, [X, Y])


