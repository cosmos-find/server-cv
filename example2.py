from sklearn. neighbors import KNeighborsClassifier
from sklearn import datasets
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d

def modelready():

    array = []
    array2 = []

    for i in range(30):
        im = cv2.imread(f'trash_{i}.jpg')
        src = im[0:250,:].copy()
        re_im = cv2.resize(src, dsize=(100, 100), interpolation=cv2.INTER_AREA)
        pix = np.array(re_im)
        array.append(pix)


    for j in range(30):
        im = cv2.imread(f'yellow_{j}.jpg')
        src = im[0:250,:].copy()
        re_im = cv2.resize(src, dsize=(100, 100), interpolation=cv2.INTER_AREA)
        pix = np.array(re_im)
        array.append(pix)

    array = np.array(array)
    array = array.reshape(array.shape[0],100*100*3)

    for i in range(30):
        im = Image.open(f'trash_{i}.jpg')
        pix = 0
        array2.append(pix)

    for j in range(30):
        im = Image.open(f'yellow_{j}.jpg')
        pix = 1
        array2.append(pix)

    X = array
    y = array2

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=1, stratify = y)
    
    y_train = np.array(y_train)

    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)

    pca = PCA(n_components=3)
    X_train_pca = pca.fit_transform(X_train_std)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train_pca, y_train)
    
    return knn

def newdata(frame):
    
    src = frame[0:250,:].copy()
    re_im = cv2.resize(src, dsize=(100, 100), interpolation=cv2.INTER_AREA)
    pix = np.array(re_im)

    sc = StandardScaler()
    sc.fit(pix)
    proto_std = sc.transform(pix)
    pca = PCA(n_components=3)
    proto = pca.fit_transform(proto_std)

    return proto


def predict(proto, knn):
    y_predict = knn.predict(proto)
    return y_predict


