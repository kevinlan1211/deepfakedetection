import numpy as np
from compute_metric import create_data, labels, detector, detect_face, img_paths
import sklearn
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import random
import cv2 as cv
import json
import dlib
import os
import tqdm
import math
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import StackingClassifier

random.seed('1')

#remove images with undetectable faces
print("removing images with undetectable faces")
if not os.path.exists('pruned.json'):
    kept_image_indices = [i for i, ip in tqdm.tqdm(enumerate(img_paths)) if len(detect_face(cv.imread(ip, 0))[0]) != 0]
    with open('pruned.json', 'x') as outfile:
        json.dump(kept_image_indices, outfile)

kept_set = None
with open('pruned.json') as infile:
    kept_set = set(json.load(infile))

#balance dataset

real_indices = [index for index, label in enumerate(labels) if label == 0 and index in kept_set]
print(len(real_indices))
fake_indices = [index for index, label in enumerate(labels) if label == 1 and index in kept_set]
print(len(fake_indices))

print("balancing dataset")
fake_sample = random.sample(fake_indices, len(real_indices))

balanced_data = real_indices + fake_sample

random.shuffle(balanced_data)

print("pruned dataset size", len(balanced_data))

all_data = balanced_data

#all_data = list(kept_set)

random.shuffle(all_data)

def create_model(num_data, avg=False, custom=False, rows=range(7)):
    print("creating training data")
    data = all_data[0:num_data]
    X = create_data(data, avg=avg, extra_column=custom, rows=rows)
    print("dataset of our metric created")

    print("standardizing dataset")
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    print("creating labels")
    print("finished creating labels")

    y = [labels[i] for i in data]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1
    )

    depths = [15]
    y_preds = []
    y_train_preds = []
    clf = None
    print("fitting classifiers")
    for d in depths:
        clf_rf = RandomForestClassifier(n_estimators=500, max_depth=d, class_weight='balanced')
        clf_lr = LogisticRegression(max_iter=1000)
        clf_svm = svm.SVC()
        clf_nb = GaussianNB()
        clfs = [('rf', clf_rf), ('lr', clf_lr), ('svm', clf_svm), ('nb', clf_nb)]
        clfs = [clfs[0], clfs[2]]
        clf = StackingClassifier(
            estimators=clfs, final_estimator=LogisticRegression())

        clf = clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_train_pred = clf.predict(X_train)

        y_preds.append(y_pred)
        y_train_preds.append(y_train_pred)

    if not avg:
        joblib.dump(clf, 'FFR_FD_no_ave_model_' + str(num_data) + '.pkl')
        print('exported model file as ', 'FFR_FD_no_ave_model_' + str(num_data) + '.pkl')
    elif not custom:
        joblib.dump(clf, 'FFR_FD_ave_model_' + str(num_data) + '.pkl')
        print('exported model file as ', 'FFR_FD_ave_model_' + str(num_data) + '.pkl')
    else:
        joblib.dump(clf, 'custom_model_' + str(num_data) + '.pkl')
        print('exported model file as ', 'custom_model_' + str(num_data) + '.pkl')

    return y_preds, y_train_preds, X_train, X_test, y_train, y_test, depths


num_data = 10000
avg = True
custom = True
rows = [0, 4, 6, 7]

def train():
    y_preds, y_train_preds, X_train, X_test, y_train, y_test, depths = create_model(num_data, avg=avg, custom=custom, rows=rows)

    for i in range(len(depths)):
        print(depths[i], f"Test set accuracy is {accuracy_score(y_preds[i], y_test) * 100:.2f} %")

        print(depths[i], f"Train set accuracy is {accuracy_score(y_train_preds[i], y_train) * 100:.2f} %")

def test():
    X_test = create_data(all_data[num_data:num_data + 100], avg=avg, extra_column=custom, rows=rows)
    scaler = StandardScaler()
    X_test = scaler.fit_transform(X_test)
    y_test = [labels[i] for i in all_data[num_data:num_data + 100]]
    print("loading model from file")
    clf = joblib.load('custom_model_' + str(num_data) + '.pkl')

    y_pred = clf.predict(X_test)

    print(list(y_pred))
    print(y_test)
    print(clf.final_estimator_.coef_)
    #print(np.sum(np.vectorize(abs)(clf.coef_.reshape(8, 33)), axis=1))
    #print(np.sum(np.vectorize(math.log)(clf.feature_importances_.reshape((8, 33))), axis=0))

    #print(np.sum(np.vectorize(math.log)(clf.feature_importances_.reshape((8, 33))), axis=1))

    print(f"Accuracy is {accuracy_score(y_pred, y_test)*100:.2f} %")

#train()
test()
