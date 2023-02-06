from warnings import simplefilter  # import warnings filter

from sklearn.decomposition import NMF

# ignore all future warnings
simplefilter(action="ignore", category=FutureWarning)


def get_NMF(X, n_components=2, max_iter=1000):
    """Gets DF and returns itself and its NMF matrix"""
    model = NMF(n_components=n_components, max_iter=max_iter)
    model.fit(X)
    nmf_features = model.transform(X)
    # print(X.shape, nmf_features.shape)
    return X, nmf_features
