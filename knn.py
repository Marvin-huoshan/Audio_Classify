import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,\
    classification_report, confusion_matrix
import pylab as pl
from built_matrix import *
import os
from sklearn.decomposition import PCA



X, y = bulit_matrix()
lf = LabelEncoder().fit(y)
y = lf.transform(y)
print(len(y))

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)


print("Shape of the reduced data:", X_pca.shape)

X_train, X_test, y_train, y_test=train_test_split(X_pca, y, test_size=0.2, random_state=5)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


clf = KNeighborsClassifier(n_neighbors=15)
clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)


print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
clf.score(X_test, y_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


error = []
# Calculating error for K values between 1 and 40
for i in range(1, 41):
    knn = KNeighborsClassifier(n_neighbors=i)
    _ = knn.fit(X_train, y_train)
    y_pred_i = knn.predict(X_test)
    error.append(np.mean(y_pred_i != y_test))
print(error)
plt.figure(figsize=(12, 6))
plt.plot(range(1, 41), error, color='red',
    linestyle='dashed', marker='o',
    markerfacecolor='blue', markersize=10)
plt.title('Error Rate vs K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.figure()

# decision boundary
X=np.array(X_test)
y=np.array(y_pred)
x_min, x_max = X[:,0].min() - 0.1, X[:,0].max() + 0.1
y_min, y_max = X[:,1].min() - 0.1, X[:,1].max() + 0.1
h = 0.02 # step size in the mesh
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
Z = Z.astype(float)

pl.set_cmap(pl.cm.Paired)
pl.pcolormesh(xx, yy, Z)


for i in range(len(X)):
    if y[i] == 0:
        _ = pl.scatter(X[i,0], X[i,1], c='red', marker='o')
    elif y[i] == 1:
        _ = pl.scatter(X[i,0], X[i,1], c='blue', marker='x')
    elif y[i] == 2:
        _ = pl.scatter(X[i,0], X[i,1], c='green', marker='+')
    else:
        _ = pl.scatter(X[i,0], X[i,1], c='magenta', marker='v')

pl.xlabel('Petal length')
pl.ylabel('Petal width')
pl.xlim(xx.min(), xx.max())
pl.ylim(yy.min(), yy.max())
pl.show()