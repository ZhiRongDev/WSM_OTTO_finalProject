import numpy as np
from sklearn.decomposition import PCA

X = np.array([[-1, -1, 4], [-2, 3, -1]])

pca = PCA(n_components=2)


pca.fit(X)
newX = pca.fit_transform(X)

print(X)
print(newX)
