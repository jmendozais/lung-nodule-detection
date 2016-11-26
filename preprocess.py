import cv2
import numpy as np
import util
import time
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

    mu = cv2.GaussianBlur(img, hsize, hsigma)
    ro2 = cv2.GaussianBlur(pow(img, 2), hsize, hsigma) - pow(mu, 2) + EPS
    assert np.min(ro2) >= 0
    res = (img - mu) / pow(ro2, 0.5)
    return res


def normalize(img, lung_mask):
    count = np.count_nonzero(lung_mask)
    mean = np.sum(img * lung_mask) * 1.0 / count
    std = (np.sum(lung_mask * ((img - mean) ** 2)) * 1.0 / (count-1)) ** 0.5
    normalized = (img - mean) / std
    return normalized


def preprocess(img, lung_mask, downsample=True):
    img = antialiasing_dowsample(img, downsample)
    enhanced = lce(img, downsample)
    normalized = normalize(img, lung_mask)
    return img, enhanced, normalized

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

'''
    # channels > 0
    stretch(from_pil(img))).show()
    max_white(from_pil(img))).show()

    # channels > 1
    grey_world(from_pil(img))).show()
    retinex(from_pil(img))).show()
    retinex_adjust(retinex(from_pil(img)))).show()
    standard_deviation_weighted_grey_world(from_pil(img),50,50)).show()
'''

