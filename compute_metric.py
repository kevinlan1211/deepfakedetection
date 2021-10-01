import numpy as np
import os
import json
import re
import cv2 as cv
import dlib
from imutils import face_utils
from numba import jit
from numba import cuda
import sklearn
from sklearn.ensemble import RandomForestClassifier
import tqdm

metadatas = {}
img_paths = []

fast = cv.FastFeatureDetector_create()
brief = cv.xfeatures2d.BriefDescriptorExtractor_create()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')\

unzero = np.vectorize(lambda x: x if x > 0 else 1)

def detect_face(img):
    gray = None
    if len(img.shape) == 3:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    else:
        gray = img
    faces = detector(gray, 1)
    return faces, gray

@jit
def rect_contains(rect, point):
    return rect[0] < point[0] < rect[0] + rect[2] and rect[1] < point[1] < rect[1] + rect[3]

@jit
def add_to_row(metric, row, vector):
    metric[row, :] += vector

@jit
def create_metric(size):
    return np.zeros((8, size))

@jit
def take_avg(matrix, column):
    column = unzero(column)
    matrix /= column

def get_label(filepath):
    numbers = re.findall('[0-9]+', filepath)
    number = int(''.join(numbers)[0:2])
    key = filepath.split("\\")[3][:-4] + '.mp4'
    return 0 if metadatas[number][key]['label'] == 'REAL' else 1

for dirname, _, filenames in os.walk('archive'):
    for filename in filenames:
        if "metadata" in filename:
            numbers = re.findall('[0-9]+', filename)
            number = int(''.join(numbers))
            os.path.join(dirname, filename)
            with open(os.path.join(dirname, filename)) as f:
                metadatas[number] = json.load(f)

        else:
            img_paths.append(os.path.join(dirname, filename))

labels = list(map(get_label, img_paths))

def create_data(indices, avg=False, extra_column=False, rows=range(7)):
    data = []
    for i in tqdm.tqdm(indices):
        ip = img_paths[i]

        img = cv.imread(ip, 0)
        fp = fast.detect(img, None)
        fp, des = brief.compute(img, fp)
        descriptor_size = brief.descriptorSize()
        metric = create_metric(descriptor_size)
        counts_column = np.zeros((8, 1))
        faces, gray = detect_face(img)
        if len(faces) == 0:
            '''for j, p in enumerate(fp):
                des_vector = des[j, :]
                metric += des_vector
                counts_column += [1]
            if avg:
                take_avg(metric, counts_column)
            data.append(metric.flatten())'''
            continue
        shape = predictor(gray, faces[0])
        shape = face_utils.shape_to_np(shape)

        for l, (name, (j, k)) in enumerate(face_utils.FACIAL_LANDMARKS_IDXS.items()):
            if name == 'jaw':
                break
            b_rect = cv.boundingRect(np.array([shape[j:k]]))
            whole_face_rect = face_utils.rect_to_bb(faces[0])
            for j, p in enumerate(fp):
                if rect_contains(whole_face_rect, p.pt):
                    des_vector = des[j, :]
                    add_to_row(metric, 7, des_vector)
                    add_to_row(counts_column, 7, [1])
                    w = b_rect[2]
                    h = b_rect[3]
                    if rect_contains((b_rect[0] - w/10, b_rect[1] - h/10, 1.1 * w, 1.1 * h), p.pt):
                        add_to_row(metric, l, des_vector)
                        add_to_row(counts_column, l, [1])

        if avg:
            take_avg(metric, counts_column)
        if extra_column:
            metric = np.concatenate((metric, counts_column), axis=1)
        data.append(np.take(metric, rows, 0).flatten())
    return data







