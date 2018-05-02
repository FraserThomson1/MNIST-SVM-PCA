# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import time
ids = []
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train_image = train.iloc[:,1:]
train_label = train.iloc[:,0]

train_image = train_image.values / 255.0
test_image = test.values / 255.0

train_label = train_label.values

X_train, X_val, y_train, y_val = train_test_split(train_image,train_label, test_size = 0.2,random_state = 0)

def n_component_analysis(n, X_train, y_train, X_val, y_val):
    start = time.time()
    pca = PCA(n_components=n)
    print("PCA begin with n_components: {}".format(n));
    pca.fit(X_train)
    
    X_train_pca = pca.transform(X_train)
    X_val_pca = pca.transform(X_val)

    print('SVC begin')
    clf1 = svm.SVC()
    clf1.fit(X_train_pca, y_train)
    #accuracy
    accuracy = clf1.score(X_val_pca, y_val)
    end = time.time()
    print("accuracy: {}, time elaps:{}".format(accuracy, int(end-start)))
    return accuracy

n_s = np.linspace(0.70, 0.85, num=15)
accuracy = []
for n in n_s:
    tmp = n_component_analysis(n, X_train, y_train, X_val, y_val)
    accuracy.append(tmp)


pca = PCA(n_components=0.75)#PCA type of dimensionality reduction, plots high dimension data
pca.fit(train_image)
X_train_pca = pca.transform(train_image)
X_test_pca = pca.transform(test_image)
clf1 = svm.SVC()
clf1.fit(X_train_pca, train_label)
predictions = clf1.predict(X_test_pca)
for i in range(1,28001):
    ids.append(i)

my_submission = pd.DataFrame({'ImageId': ids, 'Label': predictions})
my_submission.to_csv('submission.csv', index=False)