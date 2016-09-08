from __future__ import print_function
import numpy as np
np.random.seed(1337)
from itertools import product
from sklearn import cluster
from sklearn.externals import joblib
from keras.datasets import mnist
from sklearn.neighbors import KNeighborsClassifier
from scipy.misc import imresize
from keras.utils import np_utils
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold



EPS = 1e-9
class IMG: 
    def extract(self, img):
        return img.flatten()

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

    mag = (dx ** 2 + dy ** 2) ** 0.5
    return mag, dx, dy

class HOG:
    def __init__(self, orientations=9, cell=(8,8)):
        self.orientations = orientations
        self.cell = cell

    def extract(self, img, mask=None):
        if len(img.shape) == 3:
            img = img[0]
        if mask == None:
            mask = np.ones(shape=img.shape, dtype=img.dtype)

        mag, dx, dy = finite_derivatives(img)
        phase = np.arctan2(dy, dx)
        phase = phase.astype(np.float64)    
        #phase = np.abs(phase)

        size = img.shape
        size = (size[0] / self.cell[0], size[1] / self.cell[1])
        w = mask.astype(np.float64)
        w *= mag

        if np.sum(w) > EPS:
            w /= np.sum(w)

        ans = np.array([])
        for i, j in product(range(size[0]), range(size[1])):
            tl = (i * self.cell[0], j * self.cell[1])
            br = ((i + 1) * self.cell[0], (j + 1) * self.cell[1])
            roi = phase[tl[0]:br[0], tl[1]:br[1]]
            wroi = w[tl[0]:br[0], tl[1]:br[1]]
            hist, _ = np.histogram(roi, bins=self.orientations, range=(-np.pi, np.pi), weights=wroi, density=True)
            #hist /= (np.sum(hist) + util.EPS)
            if np.sum(wroi) < EPS:
                hist = np.zeros(hist.shape, dtype=hist.dtype)
            
            ans = np.hstack((ans, hist))
        ans /= (np.sum(ans) + EPS)
        return ans

class BOVW:
    def __init__(self, extractor, k=10, size=(8, 8), pad=(4, 4)):
        self.k = k
        self.pad = pad
        self.size = size

        self.extractor = extractor
        print("K ------> {}".format(self.k))
        self.clusterer = cluster.KMeans(self.k, max_iter=40, n_init=1)

    def load(self, name):
        self.k, self.pad, self.size = joblib.load('{}_pms.pkl'.format(name))
        self.extractor = joblib.load('{}_ext.pkl'.format(name))
        self.clusterer = joblib.load('{}_clu.pkl'.format(name))

    def save(self, name):
        joblib.load((self.k, self.pad, self.size), '{}_pms.pkl'.format(name))
        joblib.load(self.extractor, '{}_ext.pkl'.format(name))
        joblib.load(self.clusterer, '{}_clu.pkl'.format(name))
    
    def fit(self, X):
        assert len(X) > 0
        xr = np.linspace(0, X[0].shape[0] - self.size[0], self.pad[0])
        yr = np.linspace(0, X[0].shape[1] - self.size[1], self.pad[1])

        v_len = len(self.extractor.extract(X[0][0:self.size[0], 0:self.size[1]]))
        V = np.zeros(shape=(len(X) * len(xr) * len(yr), v_len), dtype='float32')
        it = 0
        for img in X:
            for i, j in product(xr, yr):
                V[it] = self.extractor.extract(img[i:i + self.size[0], j:j + self.size[1]]) 
                it += 1
        
        assert len(V) == it
        self.clusterer.fit(V) 
        
    def transform(self, X):
        assert len(X) > 0
        xr = np.linspace(0, X[0].shape[0] - self.size[0], self.pad[0])
        yr = np.linspace(0, X[0].shape[1] - self.size[1], self.pad[1])

        v_len = len(self.extractor.extract(X[0][0:self.size[0], 0:self.size[1]]))
        ft = np.zeros(shape=(len(xr) * len(yr), v_len), dtype='float32')
        V = np.zeros(shape=(len(X), self.k), dtype='float32')
        
        for k in range(len(X)):
            it = 0
            for i, j in product(xr, yr):
                ft[it] = self.extractor.extract(X[k][i:i + self.size[0], j:j + self.size[1]]) 
                it += 1
            assert len(ft) == it
            idx = self.clusterer.predict(ft)
            V[k], _ = np.histogram(idx, bins=self.k, range=(0, self.k))  
        
        return V

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
        
def load_mnist(img_cols, img_rows, nb_classes):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    tmp = []
    for x in X_train:
        tmp.append(imresize(x, (32, 32)))
    X_train = np.array(tmp)
    tmp = []
    for x in X_test:
        tmp.append(imresize(x, (32, 32)))
    X_test = np.array(tmp)
    print("shapes {} {}".format(X_train.shape, X_test.shape))

    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    #Y_train = np_utils.to_categorical(y_train, nb_classes)
    #Y_test = np_utils.to_categorical(y_test, nb_classes)
    return (X_train, y_train), (X_test, y_test)

def extract_IMG(X):
    assert len(X) != 0
    ext = IMG()
    v_len = len(ext.extract(X[0]))
    V = np.zeros(shape=(len(X), v_len), dtype="float32")
    for i in range(len(X)):
        V[i] = ext.extract(X[i])
    return V

def extract_HOG(X):
    assert len(X) != 0
    ext = HOG()
    v_len = len(ext.extract(X[0]))
    V = np.zeros(shape=(len(X), v_len), dtype="float32")
    for i in range(len(X)):
        V[i] = ext.extract(X[i])
    return V

if __name__ == '__main__':
    bias = True
    batch_size = 100
    nb_epoch = 1
    nb_classes = 10
    img_rows, img_cols = 32, 32
    
    (X_train, Y_train), (X_test, Y_test) = load_mnist(img_rows, img_cols, nb_classes)

    print("X shape {}".format(X_train.shape))
    X_train_small = X_train[range(10000)]
    Y_train_small = Y_train[range(10000)]
    print("X shape {}".format(X_train_small.shape))

    bow = BOVW(HOG(cell=(5,5)), k=320, size=(15, 15), pad=(1,1))
    print("BOVW fit transform ...")
    V_train = bow.fit_transform(X_train_small)
    print("BOVW transform ...")
    V_test = bow.transform(X_test)

    '''
    # 32x32 feature vector, 0.94
    V_train = extract_IMG(X_train_small)
    V_test = extract_IMG(X_test)
     '''
    '''
    #  feature vector, 0.94
    V_train = extract_HOG(X_train_small)
    V_test = extract_HOG(X_test)
    '''
    #clf = KNeighborsClassifier()
    '''
    clf = SVC(kernel='rbf')
    parameters = {'C':10. ** np.arange(-3,3), 'gamma':2. ** np.arange(-5, 1)}
    grid = GridSearchCV(clf, parameters, cv=StratifiedKFold(Y_train_small, 5), verbose=3, n_jobs=-1)
    grid.fit(V_train, Y_train_small)
    print("predicting")
    print("score: {}".format(grid.score(X_test, y_test)))
    print(grid.best_estimator_)
    '''
    for c in range(-3, 3):
        c = 10 ** c
        clf = SVC(kernel='rbf', C=c)
        
        print("C = {}, clf fit ...".format(c))
        clf.fit(V_train, Y_train_small)
        print("clf predict ...")
        Y_pred = clf.predict(V_test)

        print("Y test: {}".format(Y_test))
        print("Y pred: {}".format(Y_pred))
        acc = np.mean(Y_test == Y_pred)
        print("Accuracy: {}".format(acc))
  

   #'''
