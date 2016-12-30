from os import listdir
from os.path import isfile, join
import cv2
import sklearn.cross_validation as cross_val
import numpy as np

PATH = '../dbs/jsrt-npy'
NNINFO = '../dbs/jsrt-info/CNNDAT_EN.TXT'
LNINFO = '../dbs/jsrt-info/CLNDAT_EN.txt'
LMPATH = '../dbs/scr/masks/left_masks.txt'
RMPATH = '../dbs/scr/masks/right_masks.txt'

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

                size.append([int(toks[2]), int(toks[3])])
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

def split_index(Y, train_percent, n_iter=1, seed=113): 
    labels = []
    for y in Y:
        if y[0] == -1:
            labels.append(0)
        else:
            labels.append(1)
    return list(cross_val.StratifiedShuffleSplit(labels, n_iter, train_size=train_percent, shuffle=True, random_state=seed))

def load_data():
    paths = get_paths(PATH)
    X = []
    Y = [None]*DATASET_LEN

    for i in range(len(paths)):
        path = paths[i]
        img = io.imread(path, as_grey=True)
        X.append(np.array([img]))

    _, _, location = get_metadata()

    for i in range(10):
        Y[i] = location[i]
        img = np.copy(X[i])
        gap = 20

        if location[i][0] == -1:
            continue

        '''
        x = np.array([location[i][0] - gap, location[i][0] - gap, location[i][0] + size[i][0] + gap, location[i][0] + size[i][0] + gap])
        y = np.array([location[i][1] - gap, location[i][1] + size[i][1] + gap, location[i][1] - gap, location[i][1] + size[i][1] + gap])
        '''

        # draw bounding circle
        ex, ey = draw.circle_perimeter(location[i][1], location[i][0], max(size[i])+gap)
        img[ex, ey] = 0 
        ex, ey = draw.circle_perimeter(location[i][1], location[i][0], max(size[i])+1+gap)
        img[ex, ey] = 0 
        ex, ey = draw.circle_perimeter(location[i][1], location[i][0], max(size[i])+2+gap)
        img[ex, ey] = 0 
        ex, ey = draw.circle_perimeter(location[i][1], location[i][0], max(size[i])+3+gap)
        img[ex, ey] = 0
        cv2.imshow(img, 0)

    return X, Y

def split_data(prop=0.7, seed=113):
    paths, X = raw_data()
    idx = [[]]*DATASET_LEN

    for i in range(len(X)):
        filename = paths[i].split('/')[-1]
        jsrt_idx = int(filename[2:5]) 
        if len(idx[jsrt_idx]) == 0:
            idx[jsrt_idx] = []
        idx[jsrt_idx].append(i)

    X = np.array(X)
    Y = np.zeros(len(X))
    idx = np.array(idx)
    idx1 = np.arange(0, NUM_POSITIVES, 1, dtype=np.int32)
    idx2 = np.arange(NUM_POSITIVES, DATASET_LEN, 1, dtype=np.int32)
    i1 = np.array([int(x) for x in np.concatenate(idx[idx1])])
    i2 = np.array([int(x) for x in np.concatenate(idx[idx2])])
    if len(i1) > 0:
        Y[i1] = 0
    if len(i2) > 0:
        Y[i2] = 1

    lim1 = int(NUM_POSITIVES * prop)
    lim2 = int(93*prop)

    tr_idx = np.concatenate((idx1[:lim1],idx2[:lim2]))
    te_idx = np.concatenate((idx1[lim1:],idx2[lim2:]))
    X_train = X[np.concatenate(idx[tr_idx]).astype(np.int32)]
    X_test = X[np.concatenate(idx[te_idx]).astype(np.int32)]
    Y_train = Y[np.concatenate(idx[tr_idx]).astype(np.int32)]
    Y_test = Y[np.concatenate(idx[te_idx]).astype(np.int32)]

    np.random.seed(seed)
    np.random.shuffle(X_train)
    np.random.seed(seed)
    np.random.shuffle(Y_train)

    return (X_train, np.array([Y_train]).T), (X_test, np.array([Y_test]).T)

# Method to get lists
overlapped = ['LN060','LN065','LN105','LN108','LN112','LN113','LN115','LN126','LN130','LN133','LN136','LN149','LN151','LN152']

def jsrt(set=None):
    paths = get_paths()
    sub, siz, loc, kind = get_metadata()
    xfactor = 0.25
    yfactor = 0.25

    npaths = []
    nloc = []
    rads = []

    subs = []
    tholds = [14.29, 28.57]
    sizes = []
    kinds = []
    
    for i in range(len(paths)):
        count = 1 if siz[i][0] != -1 else 0
        if set == 'jsrt140':
            valid = True
            for tok in overlapped:
                if paths[i].find(tok) != -1:
                    valid = False
                    break
            if not valid:
                continue
        npaths.append(paths[i])
        if count > 0:
            nloc.append([int(round(loc[i][1] * xfactor)), int(round(loc[i][0] * yfactor))])
            rads.append(int(round(xfactor * max(siz[i][1], siz[i][0]))))
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
            rads.append(-1)
            subs.append(0)
            sizes.append(0)
            kinds.append(0)
    
    return np.array(npaths), np.array(nloc), np.array(rads), np.array(subs), np.array(sizes), np.array(kinds)

def left_lung(set=None):
    f = open(LMPATH)
    paths = []
    for line in f:
        if set=='jsrt140':
            valid = True
            for tok in overlapped:
                if line.find(tok) != -1:
                    valid = False
                    break
            if not valid:
                continue
        paths.append(line.rstrip())
    return np.array(paths)

def right_lung(set=None):
    f = open(RMPATH)
    paths = []
    for line in f:
        if set=='jsrt140':
            valid = True
            for tok in overlapped:
                if line.find(tok) != -1:
                    valid = False
                    break
            if not valid:
                continue
        paths.append(line.rstrip())
    return np.array(paths)
