import xml.etree.ElementTree as ET
from os import listdir
from os.path import isfile, isdir, join
import numpy as np

import dicom
from dicom.tag import Tag

import cv2
from skimage import draw
import csv

import matplotlib
from matplotlib import pyplot as plt

import util
import preprocess

LIDC_XML_PATH='../dbs/lidc-idri/LIDC-XML-only/tcia-lidc-xml'
LIDC_PATH = '../dbs/lidc-idri/DOI'
LIDC_NPY_PATH = '../dbs/lidc-idri-npy-r1-r2'
LIDC_NOD_SIZES = '../dbs/lidc-idri/nodule-sizes.csv'
LIDC_SMALL_NON_NODULES = '../dbs/lidc-idri/small-or-non-nodules.csv'
LIDC_IMAGE_SIZE = 400 # 400 milimeters
LIDC_EXCLUDED_CASES = ['LIDC-IDRI-0115', 'LIDC-IDRI-0103', 'LIDC-IDRI-0036', 'LIDC-IDRI-0034']

#EXCLUDE_NODS= ['0008-3', '0013-0', '0013-9', '0020-IL057_132682', '0080-IL057_203646', '0137-126768', '0144-Nodule 002', '0162-Nodule 005', '0162-Nodule 001', '0170-Nodule 002', '0236-168238', '0270-172423', '0060-Nodule 001']

#WEAK_NODS = ['0286-Nodule 003', '0017-IL057_130635', '0078-16332', '0083-4', '0099-4', '0117-3179', '0119-124241', '0132-18375', '0144-Nodule 003', '0145-IL057_167661', '0147-2983', '0148-126799', '0150-13017', '0155-IL057_167405', '0184-163136', '0272-172543', '0275-168130', '0286-Nodule 003']

#INCLUDE_NODS = ['0006-Nodule 005', '0017-IL057_130635', '0027-IL057_130662',  '0117-3179', '0184-163136', '0145-IL057_167661', '0155-IL057_167405', '0247-172283', '0275-168130', '0286-Nodule 003']
#INCLUDE_NODS = ['0017-IL057_130635', '0027-IL057_130662',  '0117-3179', '0184-163136', '0145-IL057_167661', '0155-IL057_167405', '0247-172283', '0275-168130', '0286-Nodule 003']


def size_by_case_and_noid_map():
    size_by_case_noid = dict()
    with open(LIDC_NOD_SIZES, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        header = reader.next()
        for row in reader:
            assert len(row) > 9
            for j in range(9, len(row)):
                if row[j]  != '':
                    size_by_case_noid[row[0] + "-" + row[j]] = row[4]
    return size_by_case_noid

SIZE_BY_NOD = size_by_case_and_noid_map()

def lidc_nod_id(case_id, roi_id):
    roi_id = roi_id.replace('.', '_')
    idx = case_id.find('LIDC-IDRI-')
    if idx != -1:
        case_id = case_id[idx + 10: idx + 14]
    roi_id = case_id + '-' + roi_id
    if not roi_id in SIZE_BY_NOD.keys():
        idx = roi_id.rfind('_')
        if idx != -1:
            roi_id = case_id + '-' + roi_id[idx+1:]

    if not roi_id in SIZE_BY_NOD.keys():
        print '\n\n Key not found in nodule-size csv file: {}!\n\n'.format(roi_id)

    return roi_id

def small_or_non_nod_cases():
    cases = None
    with open(LIDC_SMALL_NON_NODULES, 'rb') as csvfile:
        cases = [case.strip() for case in csvfile]
    return cases

'''
Functions for all data
'''

def is_cxr(path):
    tree = ET.parse(path)
    root = tree.getroot()
    for child in root:
        if child.tag == '{http://www.nih.gov/idri}ResponseHeader':
            return True
    return False

def read_rois(path):
    tree = ET.parse(path)
    root = tree.getroot()
    rois = dict()

    for child in root:
        if child.tag == '{http://www.nih.gov/idri}CXRreadingSession':
            read = child.find('{http://www.nih.gov/idri}unblindedRead')
            if read != None:
                nodule_id = read.find('{http://www.nih.gov/idri}noduleID').text
                point = read.find('{http://www.nih.gov/idri}roi').find('{http://www.nih.gov/idri}edgeMap') 
                x = int(point.find('{http://www.nih.gov/idri}xCoord').text)
                y = int(point.find('{http://www.nih.gov/idri}yCoord').text)
                if not nodule_id in rois.keys():
                    rois[nodule_id] = []
                rois[nodule_id].append([y, x])

    result = []
    ids = []
    num_reads = []
    for k in rois.keys():
        result.append(np.mean(rois[k], axis=0))
        ids.append(k)
        num_reads.append(len(rois[k]))
    return result, ids, num_reads

def is_valid_dcm(dcm):
    orientation = None
    if [0x0020, 0x0020] in dcm:
        orientation = dcm[0x0020, 0x0020]
    if [0x0018, 0x5101] in dcm and (orientation == None or orientation.value == ''):
        orientation = dcm[0x0018, 0x5101]
    if [0x0008, 0x103e] in dcm and (orientation == None or orientation.value == ''):
        orientation = dcm[0x0008, 0x103e]
        
    if orientation == None or (isinstance(orientation.value, list) and orientation[0] == 'P'):
        return False
    else:
        return True

def medoid(pts):
    min_dist = 1e9
    best_pt = None

    for i in range(len(pts)):
        dist = 0
        for j in range(len(pts)):
            dist += (pts[i][0] - pts[j][0]) ** 2 + (pts[i][1] - pts[j][1]) ** 2

        if dist < min_dist:
            min_dist = dist
            best_pt = pts[i]

    return best_pt

def in_nod_set(path, nod_id, nod_set):
    lst = [] 

    nod_from_case = dict()
    for entry in nod_set:
        case = entry.split('-')[0]
        nod = entry.split('-')[1]
        if case not in nod_from_case.keys():
            nod_from_case[case] = set()
        nod_from_case[case].add(nod)
    
    idx = path.find('LIDC-IDRI-')
    include = False
    if idx != -1:
        case_id = path[idx+10:idx+14]
        nod_id = fix_key(case_id, nod_id)[5:]
        if case in nod_from_case.keys():
            if nod_id in nod_from_case[case]:
                include = True

    return include

def rois_by_dcm_map(path, min_agreement=3):
    tree = ET.parse(path)
    root = tree.getroot()
    pts_by_roi = dict()
    rois_by_dcm = dict()
    conf_by_roi = dict()
    subt_by_roi = dict()

    print '\n ######'
    for child in root:
        if child.tag == '{http://www.nih.gov/idri}CXRreadingSession':
            for read in child:
                if read.tag == '{http://www.nih.gov/idri}unblindedRead':
                    nodule_id = read.find('{http://www.nih.gov/idri}noduleID').text
                    chars = read.find('{http://www.nih.gov/idri}characteristics')
                    confidence = int(chars.find('{http://www.nih.gov/idri}confidence').text)

                    subtlety = chars.find('{http://www.nih.gov/idri}subtlety')
                    subtlety = -1 if subtlety == None else int(subtlety.text)

                    obscuration = chars.find('{http://www.nih.gov/idri}obscuration')
                    obscuration = -1 if obscuration == None else int(obscuration.text)

                    reason = chars.find('{http://www.nih.gov/idri}reason')
                    reason = -1 if reason == None else int(reason.text)

                    #print 'nodi {}, conf {}, subt {}, obsc {}, reas {}'.format(nodule_id, confidence, subtlety, obscuration, reason),
                    ''' 
                    if confidence < 2:
                        continue
                    '''

                    roi = read.find('{http://www.nih.gov/idri}roi')
                    point = roi.find('{http://www.nih.gov/idri}edgeMap') 
                    x = int(point.find('{http://www.nih.gov/idri}xCoord').text)
                    y = int(point.find('{http://www.nih.gov/idri}yCoord').text) 
                    if nodule_id not in pts_by_roi.keys():
                        pts_by_roi[nodule_id] = [[y, x]]
                        conf_by_roi[nodule_id] = [confidence]
                    else:
                        pts_by_roi[nodule_id].append([y, x])
                        conf_by_roi[nodule_id].append(confidence)

                    if nodule_id not in subt_by_roi.keys() and subtlety != -1:
                        subt_by_roi[nodule_id] = [subtlety]
                    elif nodule_id in subt_by_roi.keys() and subtlety != -1:
                        subt_by_roi[nodule_id].append(subtlety)

                    iid = roi.find('{http://www.nih.gov/idri}imageSOP_UID')
                    if iid.text not in rois_by_dcm.keys():
                        rois_by_dcm[iid.text] = [nodule_id]
                    else:
                        rois_by_dcm[iid.text].append(nodule_id)

    #print 'rois by dcm', rois_by_dcm
    #print 'pts by roi', pts_by_roi
    #print conf_by_roi
    #print subt_by_roi
    dcm_ids = list(rois_by_dcm.keys())

    min_cf = 1e10 
    max_cf = -min_cf

    if len(dcm_ids) > 0:
        for i in range(len(dcm_ids)):
            pos = []
            num_reads = []
            roi_ids = set(rois_by_dcm[dcm_ids[i]])

            #print roi_ids
            valid_roi_ids = []
            for k in roi_ids:
                conf_sum = np.sum(conf_by_roi[k])
                subt_mean = 1e10
                if k in subt_by_roi.keys():
                    subt_mean = np.mean(subt_by_roi[k])

                min_cf = min(min_cf, conf_sum)
                max_cf = max(max_cf, conf_sum)

                # Include rois with conf sum >= 7
                if conf_sum < 8:
                    continue 

                # Exclude rois with subtlety mean < 2
                print subt_mean
                if subt_mean < 2:
                    continue

                if len(pts_by_roi[k]) >= min_agreement:
                    if len(conf_by_roi[k]) < 4:
                        print 'HERE {}, {}'.format(k, conf_by_roi[k])

                    '''
                    print min_cf, max_cf
                    print k, conf_by_roi[k]
                    '''
                    valid_roi_ids.append(k)
                    pos.append(medoid(pts_by_roi[k]))

            rois_by_dcm[dcm_ids[i]] = (valid_roi_ids, pos)

    '''
    print 'cf: ', min_cf, max_cf
    print 'valid by dcm:', rois_by_dcm
    '''
    return rois_by_dcm

'''
Return the paths, roi ids, and roi position to the PA x-rais with largest number of valid rois
'''

def paths_and_rois(root = LIDC_PATH, min_agreement=3):
    data = []
    paths = [] 
    paths_xml = []
    cxr_paths = []

    patients_with_cxr = 0 
    patient_paths = []
    for patient_dir in listdir(root):
        patient_path = join(root, patient_dir)
        if isdir(patient_path):
            patient_paths.append(patient_path)
    
    patient_paths.sort()
    for i in range(len(patient_paths)):
        studies = []
        study_idx = -1
        series_idx = -1
        
        if i > 300:
            break
        for study_dir in listdir(patient_paths[i]):
            study_path = join(patient_paths[i], study_dir)
            if isdir(study_path):
                series = []
                for series_dir in listdir(study_path):
                    series_path = join(study_path, series_dir)
                    if isdir(series_path):
                        dcm_files = []
                        xml_files = []
                        for filename in listdir(series_path):
                            if len(filename) > 3:
                                file_path = join(series_path, filename)
                                if isfile(file_path) and filename[len(filename) - 3:] == 'xml':
                                    xml_files.append(file_path)
                                    if study_idx == -1 and is_cxr(file_path):
                                        cxr_paths.append(file_path)
                                        study_idx = len(studies)
                                        series_idx = len(series)

                                if isfile(file_path) and filename[len(filename) - 3:] == 'dcm':
                                    dcm_files.append(file_path)
                        series.append([dcm_files, xml_files])
                studies.append(series)
        data.append(studies)

        # Get best PA x-ray with rois
        if study_idx != -1:
            xml_files = studies[study_idx][series_idx][1]
            dcm_files = studies[study_idx][series_idx][0]

            if len(xml_files) == 1 and is_cxr(xml_files[0]):
                assert len(dcm_files) > 0
                rois_by_dcm = rois_by_dcm_map(xml_files[0], min_agreement=min_agreement)
                max_valid_nods = 0
                dcm_file_id = -1

                valid_dcms = []

                for i in range(len(dcm_files)):
                    dcm = dicom.read_file(dcm_files[i])
                    if is_valid_dcm(dcm):
                        valid_dcms.append((dcm, dcm_files[i]))

                if len(valid_dcms) > 0:
                    best_dcm_file = valid_dcms[0][1]
                    best_coords = []
                    best_roi_ids = []
                    max_rois = -1
                    for i in range(len(valid_dcms)):
                        dcm, dcm_file = valid_dcms[i]
                        if dcm.SOPInstanceUID in rois_by_dcm.keys():
                            roi_ids, coords = rois_by_dcm[dcm.SOPInstanceUID]
                            if len(coords) > max_rois:
                                max_rois = len(coords)
                                best_dcm_file = dcm_file
                                best_coords = coords
                                best_roi_ids = roi_ids
                            
                    #print 'append', best_dcm_file, best_roi_ids, best_coords
                    paths.append((best_dcm_file, best_roi_ids, best_coords))

    return paths

def generate_npy_dataset():
    paths_and_rois_ = paths_and_rois()
    no_nodules = 0
    single_nodule = 0
    multiple_nodules = 0
    roi_counter = 0

    mins = []
    maxs = []
    size_map = size_by_case_and_noid_map()
    small_or_nn = small_or_non_nod_cases()
    lidc_rois = []
    for i in range(len(paths_and_rois_)):
        dcm_path, roi_ids, roi_coords = paths_and_rois_[i]

        print '>', type(roi_ids), type(roi_coords), '<'

        case_id_pos = dcm_path.find('LIDC-IDRI-')
        case_id = dcm_path[case_id_pos:case_id_pos + 14]

        print '-> ', case_id
        if case_id in LIDC_EXCLUDED_CASES:
            continue

        img = dicom.read_file(dcm_path)
        img = img.pixel_array
        img = img.astype('float32')

        pixel_spacing = LIDC_IMAGE_SIZE * 1.0 / np.mean(img.shape)

        #rois, roi_ids, num_reads = read_rois(xml_path)
        if case_id in small_or_nn:
            roi_ids = np.array([])
            roi_coords = np.array([])

        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)

        rois = []
        for j in range(len(roi_ids)):
            key = lidc_nod_id(case_id, roi_ids[j])
            print 'Nod ID: {}'.format(key)
            size = 15
            if key in size_map.keys():
                size = int(float(size_map[key]) * 1.0 /pixel_spacing)
            rois.append(np.append(roi_coords[j], size))

        rois = np.array(rois)

        '''
        tmp = img
        for roi in rois:
            max_intensity = np.max(img)
            real_roi = img[roi[0]-roi[2]:roi[0]+roi[2], roi[1]-roi[2]:roi[1] + roi[2]]
            #util.imshow("gt ROI", real_roi, display_shape=(200, 200))
            img = util.label_blob(img, list(roi), color=(max_intensity, 0, 0))

        if len(rois) > 0:
            util.imshow('lidc', img, display_shape=(1000, 1000))
        img = tmp
        '''

        if len(rois) == 0:
            no_nodules += 1
        elif len(rois) == 1:
            single_nodule += 1
        else:
            multiple_nodules += 1

        # crop images to be square
        print 'img shape {}'.format(img.shape)
        rows, cols = img.shape 
        offset = abs(rows-cols)/2
        if rows > cols:
            img = img[offset:offset + cols,:]
            for j in range(len(rois)):
                rois[j][0] -= offset
        else:
            img = img[:,offset:offset + rows]
            for j in range(len(rois)):
                rois[j][1] -= offset

        factor = 2048.0/img.shape[0]

        # resize and preprocess imgs
        img = cv2.resize(img, (2048, 2048), interpolation=cv2.INTER_CUBIC)
        img = preprocess.antialiasing_dowsample(img)

        '''
        min_value = np.min(img)
        max_value = np.max(img)
        img = (img - min_value)/(max_value - min_value) * 4096
        '''

        mins.append(np.min(img))
        maxs.append(np.max(img))
        print 'min {}, max {}'.format(np.min(img), np.max(img))
        
        # resize rois
        resized_rois = []
        for roi in rois:
            resized_rois.append([int(roi[0]*factor*0.25), int(roi[1]*factor*0.25), int(roi[2]*factor*0.25)])
        roi_counter += len(rois)

        #plt.hist(np.array(img).ravel(), 256, range=(0,4096)); 
        #plt.show()

        #np.save('{}/{}.npy'.format(LIDC_NPY_PATH, case_id), img)
        #np.save('{}/{}-rois.npy'.format(LIDC_NPY_PATH, case_id), resized_rois)
         
        lidc_rois.append(resized_rois)
        print resized_rois

    sizes = []
    for rpi in lidc_rois:
        for roi in rpi:
            sizes.append(roi[2])

    sizes.sort()
    print 'average rad {}, median rad {}'.format(np.mean(sizes), sizes[int(len(sizes)/2)])
    plt.hist(np.array(sizes).ravel(), 60, range=(0, 60)); 
    plt.show()

    print 'av min, av max, min, max {} {} {} {}'.format(np.mean(mins), np.mean(maxs), np.min(mins), np.max(maxs))
    print 'multi {}, single {}, no nods {}, total rois {}'.format(multiple_nodules, single_nodule, no_nodules, roi_counter)

def get_paths(root = LIDC_NPY_PATH, suffix = '-rois.npy'):
    print 'root {}'.format(root)
    paths = []
    for doc in listdir(root):
        file_path = join(root, doc)
        if len(doc) > len(suffix):
            name_len = len(file_path)
            if isfile(file_path) and (file_path[name_len - len(suffix):name_len] == suffix):
                paths.append(join(root, doc[:len(doc)-9] + '.npy'))

    paths = sorted(paths)
    return np.array(paths)

def load(pts=False, set_name='lidc-idri-npy-r1-r2'):
    idx = LIDC_NPY_PATH.find('lidc-idri-npy')
    path = LIDC_NPY_PATH[:idx] + set_name
    paths = get_paths(root=path)
    images = []
    blobs = []

    for i in range(len(paths)):
        blobs_path = paths[i][:len(paths[i])-4] + '-rois.npy'
        images.append(np.load(paths[i]))
        blobs.append(np.load(blobs_path))

    images = np.array(images)
    if pts:
        return images.reshape((images.shape[0],) + (1,) + images.shape[1:]), np.array(blobs), paths
    else:
        return images.reshape((images.shape[0],) + (1,) + images.shape[1:]), np.array(blobs)

if __name__ == '__main__':
    #plt.switch_backend('Qt4Agg')
    #plt.ion()
    generate_npy_dataset()
    #res = size_by_case_and_noid_map()
    #load()
