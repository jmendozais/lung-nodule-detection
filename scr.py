''' 
Methods to access to the SCR (Segmentation in Chest Radiographs) database.
'''

SCR_LANDMARKS_DIR = '../dbs/scr/landmarks'
SCR_MASKS_DIR = '../dbs/scr/masks'

LMPATH = '../dbs/jsrt-masks/left_masks.txt'
RMPATH = '../dbs/jsrt-masks/right_masks.txt'

overlapped = ['LN060','LN065','LN105','LN108','LN112','LN113','LN115','LN126','LN130','LN133','LN136','LN149','LN151','LN152']

#DATASET_LABELS = ['right lung', 'left lung', 'heart', 'right clavicle', 'left clavicle']
DATASET_LABELS = ['right lung', 'left lung']

from os import listdir
from os.path import isfile, join
import re
import numpy as np

def get_paths(root=SCR_LANDMARKS_DIR, suffix='pfs'):
    paths = []
    for file_ in listdir(root):
        if len(file_) > len(suffix):
            file_ = join(root, file_)
            cur_suffix = file_[len(file_) - len(suffix):]
            if isfile(file_) and cur_suffix == 'pfs':
                paths.append(file_)
    return paths

def parse_point(point_string):
    coords = point_string.split(',')
    return [float(coords[0]), float(coords[1])]

def read_dataset(dataset_string):
    begin_idx = dataset_string.find('[')
    end_idx = dataset_string.find(']')
    label = dataset_string[begin_idx:end_idx]
    label = label.split('=')[-1]

    begin_idx = dataset_string.find('{')
    end_idx = dataset_string.rfind('}')
    dataset_string = dataset_string[begin_idx:end_idx]
    entries = dataset_string.split('},  {')
    begin_idx = entries[0].find('{')
    end_idx = entries[0].rfind('}')
    entries[0] = entries[0][begin_idx+1:end_idx]

    points = [] 
    for i in range(len(entries)):
        points.append(parse_point(entries[i]))

    return label, np.array(points)

def read_pfs(path):
    pfs_file = file(path, 'rb') 

    content_pfs = ''
    lines = pfs_file.read().splitlines()
    for line in lines:
        content_pfs += line

    begin_idx = content_pfs.find('{')
    end_idx = content_pfs.rfind('}')
    content_pfs = content_pfs[begin_idx+1:end_idx]

    datasets_pfs = re.split('},{|}{', content_pfs)

    datasets = dict()

    for dataset_pfs in datasets_pfs:
        label, points = read_dataset(dataset_pfs)
        datasets[label] = points

    return datasets
    
def load_data(set='jsrt140'):
    paths = get_paths(root=SCR_LANDMARKS_DIR, suffix='pfs')
    scr_landmarks = [[] for i in range(len(DATASET_LABELS))]
    for path in paths:
        point_sets = read_pfs(path)
        for i in range(len(DATASET_LABELS)):
            scr_landmarks[i].append(point_sets[DATASET_LABELS[i]])
    scr_landmarks[0] = np.array(scr_landmarks[0])
    scr_landmarks[1] = np.array(scr_landmarks[1])

    '''
    TODO: implement mask reading
    paths = get_paths(root=SCR_MASKS_DIR, suffix='bmp')
    scr_masks = []
    for path in paths:
        print path
        point_sets = read_pfs(path)
        print "keys {}".format(point_sets.keys())
        entry = []
        for label in DATASET_LABELS:
            entry.append(point_sets[label])
            print "label {} len {}".format(label, len(point_sets[label]))
        scr_masks.append(entry)
    '''

    print ("landmarks 1-d {}, 2-n-d {}".format(len(scr_landmarks), scr_landmarks[0].shape))
    return scr_landmarks
