import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score,
)


def get_cluster_indices(cluster_labels):
    """Returns dict with distinct cluster values as keys and values as lists of its indexes in original list"""
    indices = {}
    for value in np.unique(cluster_labels):
        indices[value] = np.where(cluster_labels == value)[
            0
        ].tolist()  # np.where returns tuple with ndarray, convert to list
    return indices


def get_norm_silhouette(indices, sil_score, sil_samples):
    """computes normalized version of SA"""
    ratios = []
    for cls in indices:
        inds = indices[cls]
        t_inds = []
        for ti in inds:
            if sil_samples[ti] - sil_score >= 1e-10:
                t_inds.append(ti)
        ratio = (1.0 * len(t_inds) / len(inds)) * (1.0 * len(inds) / len(sil_samples))
        ratios.append(ratio)
    return sum(ratios) * sil_score


def get_sa_normal_score(X, cluster_labels):  # ex. evaluate_clustering
    sil_score = silhouette_score(X, cluster_labels)
    sil_samples = silhouette_samples(X, cluster_labels)
    indices = get_cluster_indices(cluster_labels)
    norm_sil_score = get_norm_silhouette(indices, sil_score, sil_samples)
    return norm_sil_score


def compute_scores(X, cluster_labels):
    """computes SA, SA normal, CH and DB index scores"""
    sa = silhouette_score(X, cluster_labels)
    sa_normal = get_sa_normal_score(X, cluster_labels)
    ch = calinski_harabasz_score(X, cluster_labels)
    db = davies_bouldin_score(X, cluster_labels)
    return [round(sa, 4), round(sa_normal, 4), round(ch, 4), round(db, 4)]


# for each index get the k with the best score, return the most common k
def get_optimal_k(scores, low):
    """Returns most commonly recommended number for clusters

    Args:
        scores (list): 2d list of different scores
        low (int): minimal number of clusters

    Returns:
        int: optimal k
    """
    optimal_k = np.argmax(scores, axis=0)
    optimal_k_min = np.argmin(scores, axis=0)
    score_names = ["SA", "SA normal", "CH", "DB"]

    cluster_no_arr = []
    for i, score_name in enumerate(score_names):
        if score_name == "DB":
            cluster_no_arr.append(optimal_k_min[i])
            print(
                f"k = {optimal_k_min[i] + low}; {score_name}: {scores[optimal_k_min[i]][i]:.3f}"
            )
        else:
            cluster_no_arr.append(optimal_k[i] + low)
            print(
                f"k = {optimal_k[i] + low}; {score_name}: {scores[optimal_k[i]][i]:.3f}"
            )

    return max(cluster_no_arr, key=cluster_no_arr.count)


def get_optimal_cluster_no(df, clusters_min, clusters_max):
    """Returns optimal cluster number and df with all cluster predictions

    Args:
        df (pandas dataframe): normalized patient instances
        clusters_min (int, optional): minimal number of clusters.
        clusters_max (int, optional): maximal number of clusters.

    Returns:
        tuple: optimal k and nmf df with cluster predictions
    """

    # K means clustering and saving scores
    kmeans_scores = []  # each row: "SA", "SA normal", "CH", "DB"
    X_data = pd.DataFrame(
        df.copy()
    )  # convert to pd df just in case we have 2d NMF array
    X_data_no_cluster_labels = df.copy()
    for n_clusters in range(clusters_min, clusters_max):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X_data_no_cluster_labels)
        clustering_results_name = f"no_cluster_{n_clusters}"
        X_data[clustering_results_name] = cluster_labels
        kmeans_scores.append(compute_scores(X_data_no_cluster_labels, cluster_labels))

    k = get_optimal_k(kmeans_scores, clusters_min)
    return k, X_data
