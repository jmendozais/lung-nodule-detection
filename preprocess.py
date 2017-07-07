import cv2
import numpy as np
import util
import time
from math import sqrt
from itertools import product
EPS = 1e-9

# check a better low-pass anti-aliasing filter, boxfilter is better
def antialiasing(img):
    ksize = (11, 11)
    sigma = 0.5
    smt = cv2.GaussianBlur(img, ksize, sigma)
    return smt

def _downsample(img):
    dsize = (512, 512)
    img = antialiasing(img)
    return cv2.resize(img, dsize, interpolation=cv2.INTER_CUBIC)

def antialiasing_dowsample(img, downsample=True):
    if downsample:
	    img = _downsample(img)
    else:
        img = antialiasing(img)
    return img

def lce(img, downsample=True):
    if downsample:
        hsize = (33, 33)
        hsigma = 16
    else:
        hsize = (32 * 4 + 1, 32 * 4 + 1)
        hsigma = 16 * 4

    img = img.astype(np.float64)
    mu = cv2.GaussianBlur(img, hsize, hsigma)
    ro2 = cv2.GaussianBlur(pow(img, 2), hsize, hsigma) - pow(mu, 2) + EPS

    assert np.min(ro2) >= 0

    res = (img - mu) / pow(ro2, 0.5)
    return res.astype(np.float32)

def normalize(img, lung_mask):
    count = np.count_nonzero(lung_mask)
    mean = np.sum(img * lung_mask) * 1.0 / count
    std = (np.sum(lung_mask * ((img - mean) ** 2)) * 1.0 / (count-1)) ** 0.5
    normalized = (img - mean) / std
    return normalized

def preprocess_hardie(img, lung_mask, downsample=True):
    img = antialiasing_dowsample(img, downsample)
    enhanced = lce(img, downsample)
    normalized = normalize(img, lung_mask)
    return img, enhanced, normalized

# Weighted Multi-Scale Convergence Index filter (Hardie et al.)
def finite_derivatives(img):
    size = img.shape

    dx = np.empty(img.shape, dtype=np.double)
    dx[0, :] = 0
    dx[-1, :] = 0
    dx[1:-1, :] = (img[2:, :] - img[:-2, :]) / 2.0

    dy = np.empty(img.shape, dtype=np.double)
    dy[:, 0] = 0
    dy[:, -1] = 0
    dy[:, 1:-1] = (img[:, 2:] - img[:, :-2]) / 2.0

    mag = (dx ** 2 + dy ** 2) ** 0.5 + 1e-9
    return mag, dx, dy

def hardie_filters():
    sizes = [7, 10, 13]
    energy = [1.0, 0.47, 0.41]
    k = sizes[2] * 2 + 1
    filters = []

    for idx in range(3):
        filter = np.empty((k, k), dtype=np.float64)
        for i in range(k):
            for j in range(k):
                if ((i - k/2) * (i - k/2) + (j - k/2) * (j - k/2) <= sizes[idx] * sizes[idx]):
                    filter[i, j] = 1
                else:
                    filter[i, j] = 0

        filters.append(filter); 
        _sum = np.sum(filter);
        filter /= _sum
        filter *= energy[idx];
    
    filters[1] += filters[0];
    filters[2] += filters[1];
    return filters

def grad_to_center_filter(size):
    assert size[0]%2 == 1
    fx = np.empty(size, dtype=np.float64)
    fy = np.empty(size, dtype=np.float64)
    for i, j in product(range(size[0]), range(size[0])):
        x = -1 * (i - size[0] / 2)
        y = -1 * (j - size[1] / 2)
        mu = sqrt(x * x + y * y) + 1e-9;    
        fx[i, j] = x * 1.0 / mu
        fy[i, j] = y * 1.0 / mu
    return fx, fy, np.ones(size, dtype=np.float64)

def wci(img, filter):
    size = filter.shape
    magnitude, dx, dy = finite_derivatives(img)
    
    fx = np.empty(size, dtype=np.float64)
    fy = np.empty(size, dtype=np.float64)
    ax = np.empty(size, dtype=np.float64)
    ay = np.empty(size, dtype=np.float64)

    for i in range(size[0]):
        for j in range(size[1]):
            x = -1 * (i - size[0] / 2)
            y = -1 * (j - size[1] / 2)
            mu = sqrt(x * x + y * y) + 1e-9;    
            fx[i, j] = filter[i, j] * x * 1.0 / mu
            fy[i, j] = filter[i, j] * y * 1.0 / mu

    nx = dx / magnitude
    ny = dy / magnitude

    ax = cv2.filter2D(nx, -1, fx)
    ay = cv2.filter2D(ny, -1, fy)
    return ax + ay

def wmci(img, mask, threshold=0.5):
    filters = hardie_filters()

    ans = wci(img, filters[0])
    for i in range(1, len(filters)):
        tmp = wci(img, filters[i])
        ans = np.maximum(tmp, ans)
    return ans

def lce_wmci(img, mask, threshold=0.5):
    _, lce, _ = preprocess.preprocess_hardie(img, lung_mask)
    wmci(lce, mask, threshold)

# Sliding Band Filter

def sliding_band_filter(img, num_rays = 256, rad_range = (2, 22), band=5):
    np.set_printoptions(suppress=True)

    theta = np.linspace(-np.pi, np.pi, num_rays, endpoint=False)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    rads = np.arange(rad_range[0], rad_range[1] + band)
    r_cos_t = cos_t[:,None] * rads[None,:]
    r_sin_t = sin_t[:,None] * rads[None,:]
  
    # TODO: remove astype for interp
    x_step = np.round(r_cos_t).astype(int).flatten()
    y_step = np.round(r_sin_t).astype(int).flatten()
    
    side = int(2*rad_range[1] + 2*band + 1)

    magnitude, dx, dy = finite_derivatives(img)
    cx, cy, cmagnitude = grad_to_center_filter((side, side))
    
    sbf = np.zeros(img.shape, dtype=np.double)
    rad = np.zeros(img.shape, dtype=np.double)
    rows, cols = img.shape

    bx = cx[x_step + side/2, y_step + side/2]
    by = cy[x_step + side/2, y_step + side/2]
    bmag = cmagnitude[x_step + side/2, y_step + side/2]
    
    '''
    simple 17 sec
    with if 15.5 sec
    with padding 14 sec
    with take 11.5 sec
    '''
    for i, j in product(range(rows), range(cols)):
        x_idx = x_step + i
        y_idx = y_step + j
        if i < side/2 or i >= rows - side/2:
            x_idx[x_idx<0] = 0
            x_idx[x_idx>=rows] = rows - 1
        if j < side/2 or j >= cols - side/2:
            y_idx[y_idx<0] = 0
            y_idx[y_idx>=cols] = cols - 1

        ax = dx.take(x_idx*cols + y_idx)
        ay = dy.take(x_idx*cols + y_idx)
        amag = magnitude.take(x_idx*cols + y_idx)

        contrib = (ax * bx + ay * by) / (amag * bmag)
        contrib = contrib.reshape(len(theta), len(rads)) 
        contrib_acc = np.add.accumulate(contrib, axis=1)
        acc = contrib_acc[:,band:] - contrib_acc[:,:-band] 

        argmax_band_by_angle = np.argmax(acc, axis=1)
        max_band_by_angle = np.max(acc, axis=1)
        sbf[i][j] = np.sum(max_band_by_angle)
        rad[i][j] = np.mean(argmax_band_by_angle) + band * 0.5

        '''
        if j % 8 == 0 and i % 8 == 0 and i > 0 and j > 0 and i  > 22 and j > 22 and i < rows - 22 and j < cols - 22:
            print sbf[i][j]
            print i, j
            roi = img[i-rad_range[1] - band:i + rad_range[1] + band, j-rad_range[1] - band: j + rad_range[1] + band]
            img2 = np.copy(img)
            
            x_idx2 = []
            y_idx2 = []
            x_step2 = x_step.reshape(len(theta), len(rads))
            y_step2 = y_step.reshape(len(theta), len(rads))
            for k in range(len(argmax_band_by_angle)):
                x_idx2.append(x_step2[k][argmax_band_by_angle[k]] + i)
                y_idx2.append(y_step2[k][argmax_band_by_angle[k]] + j)
                
            print 'contour'
            print x_idx2, y_idx2

            img2[x_idx2, y_idx2] = np.max(img2)
            roi2 = img2[i-rad_range[1] - band:i + rad_range[1] + band, j-rad_range[1] - band: j + rad_range[1] + band]
            
            print "contrib shape {}, roi shape {}".format(contrib.shape, roi.shape)
            util.imshow('roi', roi, display_shape=(256, 256))
            util.imshow('roi2', roi2, display_shape=(256, 256))
 
        '''

        '''
        if j % 64 == 0 and i % 64 == 0 and i > 0 and j > 0:
            print np.round(contrib, decimals=2)
            print np.round(contrib_acc, decimals=2)
            print contrib.dtype, contrib_acc.dtype
            print np.round(acc, decimals=2)
            print argmax_band_by_angle
            print np.round(max_band_by_angle, decimals=2)

            roi = img[i-rad_range[1] - band:i + rad_range[1] + band, j-rad_range[1] - band: j + rad_range[1] + band]
            img2 = np.copy(img)
            
            x_idx2 = []
            y_idx2 = []
            x_step2 = x_step.reshape(len(theta), len(rads))
            y_step2 = y_step.reshape(len(theta), len(rads))
            for k in range(len(argmax_band_by_angle)):
                x_idx2.append(x_step2[k][argmax_band_by_angle[k]] + i)
                y_idx2.append(y_step2[k][argmax_band_by_angle[k]] + j)
                
            print 'contour'
            print x_idx2, y_idx2

            img2[x_idx2, y_idx2] = np.max(img2)
            roi2 = img2[i-rad_range[1] - band:i + rad_range[1] + band, j-rad_range[1] - band: j + rad_range[1] + band]
            
            print "contrib shape {}, roi shape {}".format(contrib.shape, roi.shape)
            util.imshow('roi', roi, display_shape=(256, 256))
            util.imshow('roi2', roi2, display_shape=(256, 256))
        '''

    return sbf, rad

# Algorithms adapted from https://gist.github.com/shunsukeaihara/4603234 

def stretch_pre(nimg, min_value, max_value):
    scaled_img = (nimg - min_value)/(max_value - min_value)
    for i in range(len(scaled_img)):
        scaled_img[i] = np.maximum(scaled_img[i]-scaled_img[i].min(), 0.0)
    return scaled_img * (max_value - min_value) + min_value

def grey_world(nimg, min_value, max_value):
    scaled_img = (nimg - min_value)/(max_value - min_value)
    mu_0 = np.average(scaled_img[0])
    for i in range(1, len(scaled_img)):
        scaled_img[i] = np.minimum(scaled_img[i] * (mu_0 / np.average(scaled_img[i])), 0.0)
    return scaled_img * (max_value - min_value) + min_value

def max_white(nimg, min_value, max_value):
    scaled_img = (nimg - min_value)/(max_value - min_value)
    brightest = max_value
    for i in range(len(scaled_img)):
        scaled_img[i] = np.minimum(scaled_img[i] * (1.0 / scaled_img[i].max()), 1.0)
    return scaled_img * (max_value - min_value) + min_value

def stretch(nimg, min_value, max_value):
    return max_white(stretch_pre(nimg, min_value, max_value), min_value, max_value)

def retinex(nimg, min_value, max_value):
    scaled_img = (nimg - min_value)/(max_value - min_value)
    mu_0 = scaled_img[0].max()
    for i in range(1, len(scaled_img)):
        scaled_img[i] = np.minimum(scaled_img[i] * (mu_0 / scaled_img[i].max()), 1.0)
    return scaled_img * (max_value - min_value) + min_value

def retinex_adjust(nimg, min_value, max_value):
    """
    from 'Combining Gray World and Retinex Theory for Automatic White Balance in Digital Photography'
    """
    scaled_img = (nimg - min_value)/(max_value - min_value)

    sum_0 = np.sum(scaled_img[0])
    max_0 = scaled_img[0].max()
    
    for i in range(1, len(scaled_img)):
        sum_i = np.sum(scaled_img[i])
        sum2_i = np.sum(scaled_img[i] ** 2)
        max_i = scaled_img[i].max()
        max2_i = max_i ** 2
        coefficient = np.linalg.solve(np.array([[sum2_i,sum_i],[max2_i,max_i]]), np.array([sum_0,max_0]))
        scaled_img[i] = np.minimum((scaled_img[i]**2)*coefficient[0] + scaled_img[i]*coefficient[1],1.0)

    return scaled_img * (max_value - min_value) + min_value

def retinex_with_adjust(nimg, min_value, max_value):
    return retinex_adjust(retinex(nimg, min_value, max_value), min_value, max_value)

# TODO: Re-implement
def standard_deviation_weighted_grey_world(nimg,subwidth,subheight):
    """
    This function does not work correctly
    """
    nimg = nimg.astype(np.uint32)
    height, width,ch = nimg.shape
    strides = nimg.itemsize*np.array([width*subheight,subwidth,width,3,1])
    shape = (height/subheight, width/subwidth, subheight, subwidth,3)
    blocks = np.lib.stride_tricks.as_strided(nimg, shape=shape, strides=strides)
    y,x = blocks.shape[:2]
    std_r = np.zeros([y,x],dtype=np.float16)
    std_g = np.zeros([y,x],dtype=np.float16)
    std_b = np.zeros([y,x],dtype=np.float16)
    std_r_sum = 0.0
    std_g_sum = 0.0
    std_b_sum = 0.0
    for i in xrange(y):
        for j in xrange(x):
            subblock = blocks[i,j]
            subb = subblock.transpose(2, 0, 1)
            std_r[i,j]=np.std(subb[0])
            std_g[i,j]=np.std(subb[1])
            std_b[i,j]=np.std(subb[2])
            std_r_sum += std_r[i,j]
            std_g_sum += std_g[i,j]
            std_b_sum += std_b[i,j]
    sdwa_r = 0.0
    sdwa_g = 0.0
    sdwa_b = 0.0
    for i in xrange(y):
        for j in xrange(x):
            subblock = blocks[i,j]
            subb = subblock.transpose(2, 0, 1)
            mean_r=np.mean(subb[0])
            mean_g=np.mean(subb[1])
            mean_b=np.mean(subb[2])
            sdwa_r += (std_r[i,j]/std_r_sum)*mean_r
            sdwa_g += (std_g[i,j]/std_g_sum)*mean_g
            sdwa_b += (std_b[i,j]/std_b_sum)*mean_b
    sdwa_avg = (sdwa_r+sdwa_g+sdwa_b)/3
    gain_r = sdwa_avg/sdwa_r
    gain_g = sdwa_avg/sdwa_g
    gain_b = sdwa_avg/sdwa_b
    nimg = nimg.transpose(2, 0, 1)
    nimg[0] = np.minimum(nimg[0]*gain_r,255)
    nimg[1] = np.minimum(nimg[1]*gain_g,255)
    nimg[2] = np.minimum(nimg[2]*gain_b,255)
    return nimg.transpose(1, 2, 0).astype(np.uint8)

def preprocess_with_method(method, **kwargs):
    if method == 'lce_wmci':
        return lce_wmci(**kwargs)
    if method == 'wmci':
        return wmci(**kwargs)
    elif method == 'lce':
        return lce(**kwargs)

    elif method == 'norm':
        return normalize(**kwargs)
    elif method == 'norm3':
        norm = normalize(**kwargs)
        return [norm, norm, norm]
    elif method == 'heq':
        return equalize_hist(**kwargs)
    elif method == 'nlm':
        return denoise_nl_means(**kwargs)
    elif method == 'max_white':
        return max_white(**kwargs)
    elif method == 'stretch':
        return stretch(**kwargs)
    elif mehtod == 'grey_world':
        return grey_world(**kwargs)
    elif method == 'retinex':
        return retinex(**kwargs)
    elif method == 'retinex_adjust':
        return retinex_adjust(**kwargs)

PREPROCESS_METHODS_WITH_MIN_MAX = ['stretch', 'max_white', 'grey_world', 'retinex', 'retinex_adjust']

if __name__ == '__main__':
    image = np.load('../dbs/lidc-idri-npy/LIDC0014.npy')
    rois = np.load('../dbs/lidc-idri-npy/LIDC0014-rois.npy')
    util.imshow('roi', image, display_shape=(256, 256))
    sbf = sliding_band_filter(image, num_rays=256, rad_range=(2,21), band=3)
    util.imshow('SBF image', sbf, display_shape=(256, 256))
