''' 
Methods to access to the SCR (Segmentation in Chest Radiographs) database.
'''

SCR_PATH = '/home/juliomb/dbs/scr/landmarks'

#DATASET_LABELS = ['right lung', 'left lung', 'heart', 'right clavicle', 'left clavicle']
DATASET_LABELS = ['right lung', 'left lung']

from os import listdir
from os.path import isfile, join
import re

def get_paths(scr_path=SCR_PATH, suffix='pfs'):
    paths = []
    for pfs_file in listdir(scr_path):
        if len(pfs_file) > len(suffix):
            pfs_file = join(scr_path, pfs_file)
            cur_suffix = pfs_file[len(pfs_file) - len(suffix):]
            if isfile(pfs_file) and cur_suffix == 'pfs':
                paths.append(pfs_file)

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

    return label, points

def read_pfs(path):
    pfs_file = file(path, 'rb') 

    content_pfs = ''
    lines = pfs_file.read().splitlines()
    for line in lines:
        content_pfs += line

    #print "pre content pfs {}".format(content_pfs)
    begin_idx = content_pfs.find('{')
    end_idx = content_pfs.rfind('}')
    content_pfs = content_pfs[begin_idx+1:end_idx]

    #print "content pfs {}".format(content_pfs)
    datasets_pfs = re.split('},{|}{', content_pfs)

    #print("datasets num {}".format(len(datasets_pfs)))
    datasets = dict()

    for dataset_pfs in datasets_pfs:
        label, points = read_dataset(dataset_pfs)
        datasets[label] = points

    return datasets
    
def load_data():
    paths = get_paths()
    scr_dataset = []
    for path in paths:
        print path
        point_sets = read_pfs(path)
        print "keys {}".format(point_sets.keys())
        entry = []
        for label in DATASET_LABELS:
            entry.append(point_sets[label])
            print "label {} len {}".format(label, len(point_sets[label]))
        scr_dataset.append(entry)

    return scr_dataset
