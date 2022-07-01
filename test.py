import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC

label = ['Fake','Real']

X = []
Y = []

for i in range(len(label)):
    for root, dirs, directory in os.walk('dataset/'+label[i]):
        for j in range(len(directory)):
            img = cv2.imread('dataset/'+label[i]+"/"+directory[j])
            img = cv2.resize(img,(128,128))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pixel_vals = img.reshape((-1,3))
            pixel_vals = np.float32(pixel_vals)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85) 
            retval, labels, centers = cv2.kmeans(pixel_vals, 6, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS) 
            centers = np.uint8(centers) 
            segmented_data = centers[labels.flatten()]
            X.append(segmented_data.ravel())
            Y.append(i)
            print('dataset/'+label[i]+"/"+directory[j]+" "+str(X[j].shape))
                    
np.save("model/features.txt",X)
np.save("model/labels.txt",Y)

X = np.load("model/features.txt.npy")
Y = np.load("model/labels.txt.npy")
print(Y)



indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]
'''
first = X.shape[0]
second = X.shape[1] * X.shape[2]
X.resize((first,second))
'''
print(X.shape)
print(Y.shape)

#scaler = MinMaxScaler()
#X = scaler.fit_transform(X)