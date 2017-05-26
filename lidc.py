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

LIDC_XML_PATH='/home/juliomb/dbs/lidc-idri/LIDC-XML-only/tcia-lidc-xml'
LIDC_PATH = '/home/juliomb/dbs/lidc-idri/DOI'
LIDC_NPY_PATH = '/home/juliomb/dbs/lidc-idri-npy'
LIDC_NOD_SIZES = '/home/juliomb/dbs/lidc-idri/nodule-sizes.csv'
LIDC_IMAGE_SIZE = 400 # 400 milimeters
LIDC_EXCLUDED_CASES = ['0115', '0103', '0036', '0034']

'''
Missed nodules

/home/juliomb/dbs/lidc-idri/DOI/LIDC-IDRI-0098/1.3.6.1.4.1.14519.5.2.1.6279.6001.843174581147844155586402949269/1.3.6.1.4.1.14519.5.2.1.6279.6001.850400065999974190169503174211/264.xml

'''

'''
Functions for only XML data
'''

def idx_excluding_hard_cases():
    cases = [4, 12, 35, 36, 40, 54, 65, 73, 80, 87, 90, 99, 107, 108, 114, 125, 130, 139, 141, 142, 152, 177, 178, 182, 201, 211, 212, 213, 217, 225, 235, 239, 251, 256, 263]

    idx = []
    for i in range(265):
        if (i + 1) not in cases:
            idx.append(i) 
    return np.array(idx)

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
    
def get_xml_paths(root = LIDC_XML_PATH, suffix = 'xml'):
    paths = []
    for directory in listdir(root):
        dir_path = join(root, directory)
        if isdir(dir_path):
            for xml_file in listdir(dir_path):
                if len(xml_file) > len(suffix):
                    xml_path = join(dir_path, xml_file)
                    name_len = len(xml_file)
                    if isfile(xml_path) and xml_file[(name_len - len(suffix)):name_len] == suffix:
                        paths.append(xml_path)
    return np.array(paths)

def filter_xml_by_cxr(paths):
    ct = 0;
    xr = 0;
    ct_series = set()
    xr_series = set()
    cxr_xmls = []
    for path in paths:
        tree = ET.parse(path)
        root = tree.getroot()
        for child in root:
            if child.tag == '{http://www.nih.gov/idri}ResponseHeader':
                series = child.find('{http://www.nih.gov/idri}CTSeriesInstanceUid')
                xr_series.add(series.text)
                xr += 1
                cxr_xmls.append(path)
            if child.tag == '{http://www.nih.gov}ResponseHeader':
                ct += 1
                series = child.find('{http://www.nih.gov}SeriesInstanceUid')
                ct_series.add(series.text)

    print("xr {}, # xr series {}, ct {}, # ct series {}".format(xr, len(xr_series), ct, len(ct_series)))
    return cxr_xmls

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
    print path
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
    for k in rois.keys():
        result.append(np.mean(rois[k], axis=0))
        ids.append(k)
    return result, ids

def imgs_referenced_on_xml_rois(path):
    print "xml rois {}".format(path)
    tree = ET.parse(path)
    root = tree.getroot()
    rois = dict()
    iid_set = set()
    for child in root:
        if child.tag == '{http://www.nih.gov/idri}CXRreadingSession':
            read = child.find('{http://www.nih.gov/idri}unblindedRead')
            if read != None:
                nodule_id = read.find('{http://www.nih.gov/idri}noduleID')
                roi = read.find('{http://www.nih.gov/idri}roi')

                # Filter by sublety

                iid = roi.find('{http://www.nih.gov/idri}imageSOP_UID')
                iid_set.add(iid.text)
    if len(iid_set) > 0:
        return iid_set.pop()
    else: 
        return -1

def paths_first_cxr_per_patient(root = LIDC_PATH):
    data = []
    paths = []
    paths_xml = []
    cxr_paths = []

    tmp = 0
    patients_with_cxr = 0

    for patient_dir in listdir(root):
        patient_path = join(root, patient_dir)
        if isdir(patient_path):
            studies = []

            study_idx = -1
            series_idx = -1
            
            for study_dir in listdir(patient_path):
                study_path = join(patient_path, study_dir)
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

            # First CXR per patient
            if study_idx != -1:
                xml_files = studies[study_idx][series_idx][1]
                dcm_files = studies[study_idx][series_idx][0]
                #assert len(xml_files) < 2 and dcm_files > 0, 'Assertion failed # xml {}, # dcm {} {}'.format(len(xml_files), len(dcm_files), dcm_files[0])

                if len(xml_files) == 1 and is_cxr(xml_files[0]):
                    assert len(dcm_files) > 0
                    img_id = imgs_referenced_on_xml_rois(xml_files[0])
                    for dcm_file in dcm_files:
                        dcm = dicom.read_file(dcm_file)
                        if dcm.SOPInstanceUID == img_id or img_id == -1:
                            print '\n{} {} {}'.format(tmp, dcm.SOPInstanceUID, xml_files[0])
                            orientation = None
                            if [0x0020, 0x0020] in dcm:
                                print "a. {}".format(dcm[0x0020, 0x0020])
                                orientation = dcm[0x0020, 0x0020]
                            if [0x0018, 0x5101] in dcm and (orientation == None or orientation.value == ''):
                                print "b. {}".format(dcm[0x0018, 0x5101])
                                orientation = dcm[0x0018, 0x5101]
                            if [0x0008, 0x103e] in dcm and (orientation == None or orientation.value == ''):
                                print "c. {}".format(dcm[0x0008, 0x103e])
                                orientation = dcm[0x0008, 0x103e]
                                
                            if orientation != None:
                                print 'orientation <{}>'.format(orientation.value)
                            if orientation == None or (isinstance(orientation.value, list) and orientation[0] == 'P'):
                                print "\nLATERAL\n"
                            else:
                                paths.append([dcm_file, xml_files[0]])
                                tmp += 1
                                break

    return np.array(paths)

def generate_npy_dataset():
    paths = paths_first_cxr_per_patient()
    no_nodules = 0
    single_nodule = 0
    multiple_nodules = 0
    roi_counter = 0

    mins = []
    maxs = []
    for i in range(len(paths)):
        dcm_path, xml_path = paths[i]
        case_id_pos = dcm_path.find('LIDC-IDRI-') + 10
        case_id = dcm_path[case_id_pos:case_id_pos + 4]
        if case_id in LIDC_EXCLUDED_CASES:
            continue

        img = dicom.read_file(dcm_path)
        img = img.pixel_array
        img = img.astype('float32')

        pixel_spacing = LIDC_IMAGE_SIZE * 1.0 / np.mean(img.shape)

        rois, rois_id = read_rois(xml_path)
        size_map = size_by_case_and_noid_map()

        if len(rois) == 0:
            no_nodules += 1
        elif len(rois) == 1:
            single_nodule += 1
        else:
            multiple_nodules += 1

        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)

        img2 = img.copy()
        for j in range(len(rois)):
            rois_id[j] = rois_id[j].replace('.', '_')
            key = case_id + '-' + rois_id[j]
            if not key in size_map.keys():
                idx = rois_id[j].rfind('_')
                if idx == -1:
                    raise Exception('Cant find an alternative key name for {}'.format(rois_id[j]))
                key = case_id + '-' + rois_id[j][idx+1:]

            size = 10
            if not key in size_map:
                print '\n\nCheck this {}!\n\n'.format(key)
            else:
                size = int(float(size_map[key]) * 1.0 /pixel_spacing)

            rois[j] = np.append(rois[j], size)

        # crop images to be square
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

        plt.hist(np.array(img).ravel(), 256, range=(0,4096)); 
        plt.show()

        np.save('{}/LIDC{}.npy'.format(LIDC_NPY_PATH, case_id), img)
        np.save('{}/LIDC{}-rois.npy'.format(LIDC_NPY_PATH, case_id), resized_rois)
         
        '''
        max_intensity = np.max(img)
        for roi in resized_rois:
            roi[2] = 32
            img = util.label_blob(img, list(roi), color=(max_intensity, 0, 0))
        lidc_rois.append(resized_rois)
        util.imshow('lidc', img)
        '''
        
    print 'av min, av max, min, max {} {} {} {}'.format(np.mean(mins), np.mean(maxs), np.min(mins), np.max(maxs))

    '''
    sizes = []
    for rpi in lidc_rois:
        for roi in rpi:
            sizes.append(roi[2])

    plt.hist(np.array(sizes).ravel(), 48, range=(0, 48)); 
    plt.show()
    '''

    print "multi {}, single {}, no nods {}, total rois {}".format(multiple_nodules, single_nodule, no_nodules, roi_counter)

def get_paths(root = LIDC_NPY_PATH, suffix = '-rois.npy'):
    paths = []
    for doc in listdir(root):
        file_path = join(root, doc)
        if len(doc) > len(suffix):
            name_len = len(file_path)
            if isfile(file_path) and (file_path[name_len - len(suffix):name_len] == suffix):
                paths.append(join(root, doc[:len(doc)-9] + '.npy'))

    paths = sorted(paths)
    return np.array(paths)

def load():
    paths = get_paths()
    images = []
    blobs = []

    for i in range(len(paths)):
        blobs_path = paths[i][:len(paths[i])-4] + '-rois.npy'
        images.append(np.load(paths[i]))
        blobs.append(np.load(blobs_path))

    images = np.array(images)
    return images.reshape((images.shape[0],) + (1,) + images.shape[1:]), np.array(blobs)

if __name__ == '__main__':
    #plt.switch_backend('Qt4Agg')
    #plt.ion()
    generate_npy_dataset()
    #res = size_by_case_and_noid_map()
    #load()

