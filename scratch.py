import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import json
import re
import cv2 as cv
import dlib
import imutils
from imutils import face_utils
import random

def rect_contains(rect, point):
    return rect[0] < point[0] < rect[0] + rect[2] and rect[1] < point[1] < rect[1] + rect[3]

metadatas = {}
img_paths = []
fast = cv.FastFeatureDetector_create()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

for dirname, _, filenames in os.walk('archive'):
    for filename in filenames:
        if "metadata" in filename:
            numbers = re.findall('[0-9]+', filename)
            number = int(''.join(numbers))
            with open(os.path.join(dirname, filename)) as f:
                metadatas[number] = json.load(f)
        else:
            img_paths.append(os.path.join(dirname, filename))

img = cv.imread(img_paths[0], 0)
fp = fast.detect(img, None)
img2 = cv.drawKeypoints(img, fp, None, color=(255, 0, 0))
cv.imwrite('fast_fake.png', img2)
print(len(fp))

real_path = ''
for path in img_paths:
    if 'xugmhbetrw' in path:
        real_path = path

img_real = cv.imread(real_path)
fp_real = fast.detect(img_real, None)
img2_real = cv.drawKeypoints(img_real, fp_real, None, color=(255, 0, 0))
cv.imwrite('fast_real.png', img2_real)
print(len(fp_real))

img3 = cv.imread(img_paths[0])
gray = cv.cvtColor(img3, cv.COLOR_BGR2GRAY)

faces = detector(gray, 1)

for result in faces:
    (x, y, w, h) = face_utils.rect_to_bb(result)
    cv.rectangle(img3, (x, y), (x + w, y + h), (0, 0, 255), 2)

shape = predictor(gray, faces[0])
print(faces[0])
shape = face_utils.shape_to_np(shape)

for (name, (j, k)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
    if name == 'jaw':
        break
    rgb = [255, 0, 0]
    random.shuffle(rgb)
    color = tuple(rgb)
    b_rect = cv.boundingRect(np.array([shape[j:k]]))
    (x, y, w, h) = b_rect
    cv.rectangle(img3, (x, y), (x + w, y + h), color, 2)
    for p in fp:
        if rect_contains(b_rect, p.pt):
            cv.circle(img3, tuple(map(lambda x: int(x), p.pt)), radius=0, color=color, thickness=-1)

cv.imwrite('facial_detection.png', img3)

print(face_utils.FACIAL_LANDMARKS_IDXS.items())