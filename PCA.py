import numpy as np

arr = np.array([[1, 4 ,400], \
                [3, 5 ,500], \
                [3, 2 ,450], \
                [3, 1 ,400], \
                [1, 3 ,425]])

cov = np.dot(arr.T,arr)/5
##PCA
#Number_of_dimensions = 20
## calculate covariance matrix of centered matrix
CovMatrice = np.cov(arr.T)
## eigendecomposition of covariance matrix
#values, vectors = np.linalg.eig(CovMatrice)
## highest eigenvalue vectors
#new_order = (-values).argsort()[:Number_of_dimensions]
#new_vectors = vectors[new_order]
#cleanData = np.dot(cleanData,new_vectors.T)

values, vectors = np.linalg.eig(CovMatrice)

arr2 = np.array([[2,1], \
                [-2,-1], \
                [-2,4]])
cov2 = np.dot(arr2,arr2.T)/2
values, vectors = np.linalg.eig(cov2)