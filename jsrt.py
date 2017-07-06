from os import listdir
from os.path import isfile, join
import cv2
import sklearn.cross_validation as cross_val
import numpy as np

import preprocess

PATH = '../dbs/jsrt-npy'
NNINFO = '../dbs/jsrt-info/CNNDAT_EN.TXT'
LNINFO = '../dbs/jsrt-info/CLNDAT_EN.txt'
LMPATH = '../dbs/jsrt-masks/left_masks.txt'
RMPATH = '../dbs/jsrt-masks/right_masks.txt'
PIXEL_SPACING = 0.175

DATASET_LEN = 257
NUM_POSITIVES = 154

sublety_labels = ['False positive', 'Extremely subtle', 'Very subtle', 'Subtle', 'Relatively obvious', 'Obvious']
size_labels = [u'False positive', u'< 10 mm.', u'>= 10 mm. and < 20 mm.', u'>= 20mm.']
severity_labels = ['False positive', 'Benign', 'Malignant']

def get_paths(root = PATH, suffix = 'npy'):
        paths = []
        for doc in listdir(root):
                file_path = join(root, doc)
                if len(doc) > len(suffix):
                        name_len = len(file_path)
                        if isfile(file_path) and (file_path[name_len - len(suffix):name_len] == suffix):
                                paths.append(join(root, doc))
        paths = sorted(paths)
        return np.array(paths)

def get_metadata():
        lnfile = file(LNINFO, 'rb')
        nnfile = file(NNINFO, 'rb')
        size = []
        location = []
        sublety = []
        kind = []

        for line in lnfile:
                toks = line.split('\t')
                sublety.append(int(toks[1]))

                if toks[2] == '?' and toks[3] == '?':
                        toks[2] = 30
                        toks[3] = 30
                elif toks[2] == '?':
                        toks[2] = toks[3]
                elif toks[3] == '?':
                        toks[3] = toks[2]

                size.append([float(toks[2]) / PIXEL_SPACING, float(toks[3]) / PIXEL_SPACING])
                location.append([int(toks[5]), int(toks[6])])

                if toks[7] == 'benign':
                    kind.append(1)
                elif toks[7] == 'malignant':
                    kind.append(2)

        for line in nnfile:
                size.append([-1, -1])
                location.append([-1, -1])
                sublety.append(0)
                kind.append(0)

        lnfile.close()
        nnfile.close()
        return np.array(sublety), np.array(size), np.array(location), np.array(kind)

# Method to get lists
overlapped = ['LN060','LN065','LN105','LN108','LN112','LN113','LN115','LN126','LN130','LN133','LN136','LN149','LN151','LN152']

def set_index(paths, set_name=None):
    index = [] 
    for i in range(len(paths)):
        valid = True
        if set_name == 'jsrt140':
            for tok in overlapped:
                if paths[i].find(tok) != -1:
                    valid = False
                    break
        elif set_name == 'jsrt140p':
            for tok in overlapped:
                if paths[i].find(tok) != -1 or paths[i].find('NN') != -1:
                    valid = False
                    break
        elif set_name == 'jsrt140n':
            for tok in overlapped:
                if paths[i].find(tok) != -1 or paths[i].find('LN') != -1:
                    valid = False
                    break
        elif set_name == 'jsrt_od':
            if i%2 == 1:
                valid = False
        elif set_name == 'jsrt_ev':
            if i%2 == 0:
                valid = False

        if valid:
            index.append(i)
    return np.array(index)

def jsrt(set=None):
    paths = get_paths()
    sub, siz, loc, kind = get_metadata() 
    xfactor = 0.25
    yfactor = 0.25

    npaths = []
    nloc = []
    diam = []

    subs = []
    tholds = [14.29, 28.57]
    sizes = []
    kinds = []

    idx = set_index(paths, set)
    
    for i in idx:
        count = 1 if siz[i][0] != -1 else 0
        npaths.append(paths[i])
        if count > 0:
            nloc.append([int(round(loc[i][1] * xfactor)), int(round(loc[i][0] * yfactor))])
            diam.append(int(round(xfactor * siz[i][0])))
            subs.append(sub[i])
            kinds.append(kind[i])

            if siz[i][0] < tholds[0]:
                sizes.append(1)
            elif siz[i][0] < tholds[1]:
                sizes.append(2)
            else:
                sizes.append(3)

        else:
            nloc.append([-1, -1])
            diam.append(-1)
            subs.append(0)
            sizes.append(0)
            kinds.append(0)
    
    print len(npaths), len(diam)
    return np.array(npaths), np.array(nloc), np.array(diam), np.array(subs), np.array(sizes), np.array(kinds)

def left_lung(set=None):
    f = open(LMPATH)
    paths = []
    lines = np.array([line for line in f])
    idx = set_index(lines, set)
    return lines[idx]

def right_lung(set=None):
    f = open(RMPATH)
    paths = []
    lines = np.array([line for line in f])
    idx = set_index(lines, set)
    return lines[idx]

def images_from_paths(paths, dsize=(512, 512)): 
    imgs = []
    for i in range(len(paths)):
        img = np.load(paths[i]).astype(np.float)
        img = cv2.resize(img, dsize, interpolation=cv2.INTER_CUBIC)
        imgs.append(img)
    return np.array(imgs)

def load(set_name=None, dsize=(512, 512)):
    paths = get_paths()
    sub, siz, loc, kind = get_metadata() 
    factor = 0.25
    idx = set_index(paths, set_name)
    blobs = []
    imgs = []
    for i in idx:
        img = np.load(paths[i]).astype(np.float)
        img = preprocess.antialiasing_dowsample(img, downsample=True)
        imgs.append(img)
        if loc[i][0] != -1:
            blobs.append([[int(round(loc[i][1] * factor)), int(round(loc[i][0] * factor)), int(round(factor * siz[i][0]))]])
        else:
            blobs.append([])

    imgs = np.array(imgs)
    return imgs.reshape((imgs.shape[0],) + (1,) + imgs.shape[1:]), np.array(blobs)

def masks(set_name=None, dsize=(512, 512), join_masks=True):
    ll_paths = left_lung(set_name)
    lr_paths = right_lung(set_name)

    resp = []
    for i in range(len(ll_paths)):
        ll_mask = cv2.imread(ll_paths[i].rstrip())
        lr_mask = cv2.imread(lr_paths[i].rstrip())
        if join_masks:
            lung_mask = ll_mask + lr_mask
            lung_mask = cv2.resize(lung_mask, dsize, interpolation=cv2.INTER_CUBIC)
            lung_mask = cv2.cvtColor(lung_mask, cv2.COLOR_BGR2GRAY)
            lung_mask = (lung_mask > 0).astype(np.uint8)
            resp.append(lung_mask)
        else:
            ll_mask = cv2.resize(ll_mask, dsize, interpolation=cv2.INTER_CUBIC)
            ll_mask = cv2.cvtColor(ll_mask, cv2.COLOR_BGR2GRAY)
            ll_mask = (ll_mask > 0).astype(np.uint8)
            lr_mask = cv2.resize(lr_mask, dsize, interpolation=cv2.INTER_CUBIC)
            lr_mask = cv2.cvtColor(lr_mask, cv2.COLOR_BGR2GRAY)
            lr_mask = (lr_mask > 0).astype(np.uint8)
            resp.append([lr_mask, ll_mask])
    return resp

class DataProvider:
    def __init__(self, img_paths, lm_paths, rm_paths):
        self.img_paths = img_paths
        self.ll_paths = lm_paths
        self.lr_paths = rm_paths

    def __len__(self):
        return len(self.img_paths)

    def get(self, i, downsample=True, dsize=(512, 512)):
        img = np.load(self.img_paths[i]).astype(np.float)

        ll_mask = cv2.imread(self.ll_paths[i])
        lr_mask = cv2.imread(self.lr_paths[i])
        lung_mask = ll_mask + lr_mask

        if downsample:
            lung_mask = cv2.resize(lung_mask, dsize, interpolation=cv2.INTER_CUBIC)
        else:
            lung_mask = cv2.resize(lung_mask, img.shape, interpolation=cv2.INTER_CUBIC)

        lung_mask = cv2.cvtColor(lung_mask, cv2.COLOR_BGR2GRAY)
        lung_mask = (lung_mask > 0).astype(np.uint8)

        return img, lung_mask

import matplotlib.pyplot as plt

def subtlety_by_size():
    set_name = 'jsrt'
    paths, loc, diams, subs, sizes, kinds = jsrt(set=set_name)
    sizes = [0, 10, 15, 20, 25, 30, 60]
    num_itv = len(sizes) - 1
    num_subs = 5
    
    tab = np.zeros(shape=(6, 7))
    for i in range(len(paths)):
        if subs[i] == 0:
            continue

        diams[i] *= (PIXEL_SPACING * 4)
        col = 0
        for k in range(num_itv):
            if diams[i] > sizes[k] and diams[i] <= sizes[k + 1]:
                col = k
        tab[subs[i] - 1][col] += 1

    for i in range(num_subs):
        tab[i][num_itv] = np.sum(tab[i,:])

    for i in range(num_itv):
        tab[num_subs][i] = np.sum(tab[:,i])

    tab[num_subs][num_itv] = np.sum(tab[:num_subs,:num_itv])

def subtlety_by_size():
    set_name = 'jsrt'
    paths, loc, diams, subs, sizes, kinds = jsrt(set=set_name)
    sizes = [0, 10, 15, 20, 25, 30, 60]
    num_itv = len(sizes) - 1
    num_subs = 5
    
    tab = np.zeros(shape=(6, 7))
    for i in range(len(paths)):
        if subs[i] == 0:
            continue

        diams[i] *= (PIXEL_SPACING * 4)
        col = 0
        for k in range(num_itv):
            if diams[i] > sizes[k] and diams[i] <= sizes[k + 1]:
                col = k
        tab[subs[i] - 1][col] += 1

    for i in range(num_subs):
        tab[i][num_itv] = np.sum(tab[i,:])

    for i in range(num_itv):
        tab[num_subs][i] = np.sum(tab[:,i])

    tab[num_subs][num_itv] = np.sum(tab[:num_subs,:num_itv])

def save_samples_by_subt():
    import util

    set_name = 'jsrt'
    paths, loc, diams, subs, sizes, kinds = jsrt(set=set_name)
    idxs_by_sub = [[],[],[],[],[]]
    for i in range(len(paths)): 
        if subs[i] == 0:
            continue
        idxs_by_sub[subs[i] - 1].append(i)
    for i in range(5):
        print len(idxs_by_sub[i])
        idx = np.random.randint(0, len(idxs_by_sub[i]))
        idx = idxs_by_sub[i][idx]
        img = np.load(paths[idx])
        blob = (loc[idx][0], loc[idx][1], diams[idx])
        roi = util.extract_roi(img, blob)
        util.imwrite_as_pdf('data/sub_{}_{}'.format(i + 1, idx), roi)

def test_show_blobs():
    import util
    import detect
    import preprocess
    set_name = 'jsrt140'
    imgs, blobs = load(set_name=set_name)
    pred_blobs = detect.read_blobs('data/wmci-aam-jsrt140-blobs-gt.pkl')

    for i in range(len(imgs)):
        max_intensity = np.max(imgs[i])
        img = util.label_blob(imgs[i].astype(np.float32), list(blobs[i][0]), color=(max_intensity, 0, 0))
        plt.hist(np.array(img).ravel(), 256, range=(0,4096)); 
        plt.show()

if __name__ == '__main__':
    save_samples_by_subt()
