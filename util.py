import math

import numpy as np
import numpy.linalg as la
from skimage import draw
import cv2

# Data utils

def load_list(path, blob_type='rad'):
  detect_f = open(path, 'r+') 
  paths = []
  blobs = []  

  for line in detect_f:
    toks = line.split(' ')  
    path = toks[0]
    blob_dim = 3
    _blobs = []
    argc = int(toks[1])
    if blob_dim == 'rect':
      blob_dim = 4
    for i in range(argc):
      blob = []
      for j in range(blob_dim):
        blob.append(int(toks[2 + blob_dim*i + j]))
      _blobs.append(blob)
    paths.append(path)
    blobs.append(_blobs)
  return paths, blobs

def save_list(paths, blobs, path):
  size = len(paths)
  blob_dim = len(blobs[0][0])
  for i in range(size):
    print paths[i],
    print len(blobs),
    for j in range(len(blobs[i])):
      for k in range(blob_dim):
        print path,
    print ''

def scale_list(path, factor):
  paths, tls, sizes = load_list(path)
  for i in range(len(tls)):
    tls[i][0] = math.round(tls[i][0] * 1.0 / factor)
    tls[i][1] = math.round(tls[i][1] * 1.0 / factor)
    for j in range(len(sizes[0])):
      sizes[i][j] = math.round(sizes[i][0] * 1.0 / factor)
      sizes[i][j] = math.round(sizes[i][1] * 1.0 / factor)

  save_list(paths, tls, sizes)

# Display utils 

def imshow(windowName,  _img):
	img = np.array(_img).astype(np.float64)
	a = np.min(img)
	b = np.max(img)
	img = (img - a) / (b - a);
	print np.min(img), np.max(img)

	cv2.imshow(windowName, img)
	cv2.waitKey()

def label_blob(img, blob, color=(255, 0, 0), margin=0):
	if len(img.shape) == 2:
		img = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)

	ex, ey = draw.circle_perimeter(blob[0], blob[1], blob[2] + margin)

	if np.max(ex) + 3 + margin >= img.shape[0] or np.max(ey) + 3 + margin >= img.shape[1]:
		return img

	img[ex, ey] = color 
	ex, ey = draw.circle_perimeter(blob[0], blob[1], blob[2]+1+margin)
	img[ex, ey] = color 

	'''
	ex, ey = draw.circle_perimeter(blob[0], blob[1], blob[2]+2+margin)
	img[ex, ey] = color 
	ex, ey = draw.circle_perimeter(blob[0], blob[1], blob[2]+3+margin)
	img[ex, ey] = color 
	'''

	return img

def show_blobs(windowName, img, blobs):
	labeled = np.array(img).astype(np.float32)
	maxima = np.max(labeled)
	for blob in blobs:
		labeled = label_blob(labeled, blob, color=(maxima, 0, 0))

	imshow(windowName, labeled)

def print_detection(path, blobs):
  print path,
  print len(blobs),

  if len(blobs) > 0:
    blob_dim = len(blobs[0])
    for i in range(len(blobs)):
      for j in range(blob_dim):
        print blobs[i][j],
  print ''

def print_list(paths, blobs):
  size = len(paths)
  blob_dim = len(blobs[0][0])
  for i in range(size):
    print_detection(paths[i], blobs[i])
