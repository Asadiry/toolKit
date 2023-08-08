import numpy as np
from sklearn.decomposition import PCA

class DR_PCA:
    def __init__(self):
        self.pca = None

    def fit(self, data, n_components):
        self.pca = PCA(n_components=n_components)
        dr_data = self.pca.fit_transform(data)
        return dr_data

    def trans(self, data):
        return self.pca.transform(data)

if __name__ == "__main__":
    pca = DR_PCA()
    x_train = np.random.random((3000, 404))
    dr_data = pca.fit(x_train, n_components=0.95)
    print(pca.trans(x_train).shape)
    print(dr_data.shape)