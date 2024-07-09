"""
4) Demonstrate the working of SVM classifier for a suitable dataset
"""

from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.svm import SVC

iris = load_iris()
X, y = iris.data[:, :2], iris.target

svm = SVC(kernel='linear', gamma=0.5, C=1.0).fit(X, y)

DecisionBoundaryDisplay.from_estimator(svm, X, cmap=plt.cm.Spectral, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.show()
