import math
from random import *
import numpy as np
import numpy.linalg as la
from skimage import draw
import cv2
import matplotlib.pyplot as plt
import time
import csv

# Data utils
EPS = 1e-9
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
	img = (img - a) / (b - a + EPS);
	print np.min(img), np.max(img)
  
	cv2.imshow(windowName, img)
	cv2.waitKey()

def imwrite(fname, _img):
  img = np.array(_img).astype(np.float64)
  a = np.min(img)
  b = np.max(img)
  img = (img - a) / (b - a + EPS);

  cv2.imwrite(fname, 255 * img)

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
		labeled = label_blob(labeled, blob, color=(maxima, 0, 0), margin=-5)

	imshow(windowName, labeled)

def imwrite_with_blobs(fname, img, blobs):
  labeled = np.array(img).astype(np.float32)
  maxima = np.max(labeled)
  for blob in blobs:
    labeled = label_blob(labeled, blob, color=(maxima, 0, 0), margin=-5)

  imwrite(fname, labeled)

def show_blobs_real_predicted(path, res1, res2):
  img = np.load(path, 0)
  resized_img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)
  color_img = cv2.cvtColor(resized_img.copy(), cv2.COLOR_GRAY2BGR) 
  
  print "Real vs predicted .."
  print 'real',
  for res in res1:
    print res,
    if res[0] == -1:
      continue
    color_img = label_blob(color_img, res, (255, 0, 0))
  print ''
  print 'predicted',
  for res in res2:
    print res,
    if res[0] == -1:
      continue
    color_img = util.label_blob(color_img, res, (0, 255, 255))
  print ''
  util.imshow('real vs predicted', color_img)
  
def show_nodule(roi, mask, scale=4):
  dsize = (mask.shape[0] * scale, mask.shape[0] * scale)
  roi = cv2.resize(roi, dsize)
  mask = cv2.resize(mask, dsize)
  _max = np.max(roi)
  drawing = cv2.cvtColor(roi.copy().astype(np.float32), cv2.COLOR_GRAY2BGR)
  color = (uniform(0, _max), uniform(0, _max), uniform(0, _max))
  contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cv2.drawContours(drawing, contours, -1, color, 1)
  imshow("nodule", drawing)

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

def save_blob(path, img, blob):
  x, y, r = blob
  shift = 0
  side = 2 * shift + 2 * r + 1
  dsize = (128, 128)

  tl = (x - shift - r, y - shift - r)
  ntl = (max(0, tl[0]), max(0, tl[1]))
  br = (x + shift + r + 1, y + shift + r + 1)
  nbr = (min(img.shape[0], br[0]), min(img.shape[1], br[1]))

  img_roi = img[ntl[0]:nbr[0], ntl[1]:nbr[1]]
  img_roi = cv2.resize(img_roi, dsize, interpolation=cv2.INTER_CUBIC)

  imwrite(path, img_roi)

def show_blob(path, img, blob):
  x, y, r = blob
  shift = 0
  side = 2 * shift + 2 * r + 1
  dsize = (128, 128)

  tl = (x - shift - r, y - shift - r)
  ntl = (max(0, tl[0]), max(0, tl[1]))
  br = (x + shift + r + 1, y + shift + r + 1)
  nbr = (min(img.shape[0], br[0]), min(img.shape[1], br[1]))

  img_roi = img[ntl[0]:nbr[0], ntl[1]:nbr[1]]
  img_roi = cv2.resize(img_roi, dsize, interpolation=cv2.INTER_CUBIC)

  imshow(path, img_roi)

def save_froc(op_set, name, legend=None):
  ax = plt.gca()
  ax.grid(True)

  op_set = np.array(op_set)

  line_format = ['b.-', 'g.-', 'r.-', 'c.-', 'm.-', 'y.-', 'k.-', 
                 'b.--', 'g.--', 'r.--', 'c.--', 'm.--', 'y.--', 'k.--',
                 'b.-.', 'g.-.', 'r.-.', 'c.-.', 'm.-.', 'y.-.', 'k.-.']

  for i in range(len(op_set)):
    ops = np.array(op_set[i]).T
    plt.plot(ops[0], ops[1], line_format[i%14])

  plt.title(name)
  plt.ylabel('Sensitivity')
  plt.xlabel('Average FPPI')
  if legend != None:
    assert len(legend) == len(op_set)
    plt.legend(legend, loc=4, fontsize='small')

  name='{}_{}'.format(name, time.clock())
  plt.savefig('{}_froc.jpg'.format(name))

  file = open('{}_ops.txt'.format(name), "wb")
  writer = csv.writer(file, delimiter=",")
  writer.writerows(op_set)

