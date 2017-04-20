from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()

X = iris.data
y= iris.target

knn = KNeighborsClassifier(n_neighbors = 5)

knn.fit(X,y)

print knn.predict([[3,5,4,2],[5,4,3,2]])