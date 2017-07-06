import numpy as np
from os import listdir
from os.path import isfile, join
from sklearn.manifold import TSNE
from skimage.feature import hog
from sklearn.decomposition import LatentDirichletAllocation as LDA

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


LIDC_ROIS_PATH = '/home/juliomb/lung-nodule-detection/data/lidc'
JSRT_ROIS_PATH = '/home/juliomb/lung-nodule-detection/data/jsrt140'

def get_paths(root, suffix = '.npy'):
    paths = []
    for doc in listdir(root):
        file_path = join(root, doc)
        if len(doc) > len(suffix):
            name_len = len(file_path)
            if isfile(file_path) and (file_path[name_len - len(suffix):name_len] == suffix):
                #paths.append(join(root, doc[:len(doc)-9] + '.npy'))
                paths.append(file_path)

    paths = sorted(paths)
    return np.array(paths)

def load_dataset(root):
    paths = get_paths(root)
    X_pos = []
    X_neg = []
    for i in range(len(paths)):
        if paths[i][len(paths[i]) - 5] == 'p':
            X_pos.append(np.load(paths[i]))
        else:
            X_neg.append(np.load(paths[i]))

    print len(X_pos), len(X_neg)
    assert len(X_pos) == len(X_neg)
    return np.array(X_pos), np.array(X_neg)
            
def project_datasets():
    X_pos, X_neg = load_dataset(JSRT_ROIS_PATH) 
    print X_pos.shape, X_neg.shape
    len_pos = len(X_pos)
    X = np.concatenate([X_pos, X_neg], 0)

    X_hog = []
    for i in range(len(X)):
        pad = int(X[i][0].shape[0]/6)
        roi = X[i][0][pad:-pad, pad:-pad]
        #plt.imshow(roi, cmap='gray')
        #plt.show()
        fd, hog_image = hog(roi, orientations=16, pixels_per_cell=(12, 12),
                        cells_per_block=(1, 1), visualise=True)
        X_hog.append(fd)
    #X = X.reshape((X.shape[0], 36 * 36))

    X = np.array(X_hog)

    lda = LDA(n_topics=10)
    X = lda.fit_transform(X)

    print X.shape

    model = TSNE(n_components=3, random_state=0, init='random')
    np.set_printoptions(suppress=True)
    X_pr = model.fit_transform(X) 

    X_pr_pos = X_pr[:len_pos]
    X_pr_neg = X_pr[len_pos:]

    fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    ax = fig.add_subplot(111)
    #ax.scatter(X_pr_pos[:,0], X_pr_pos[:,1], X_pr_pos[:,2], c='r', label='positives')
    #ax.scatter(X_pr_neg[:,0], X_pr_neg[:,1], X_pr_neg[:,2], c='b', label='negatives')
    ax.scatter(X_pr_pos[:,0], X_pr_pos[:,1], c='r', label='positives')
    ax.scatter(X_pr_neg[:,0], X_pr_neg[:,1], c='b', label='negatives')
    ax.legend()
    ax.grid()
    plt.show()

if __name__ == '__main__':
    project_datasets()
    
