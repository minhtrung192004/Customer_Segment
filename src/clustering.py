import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score
)


def scale_data(df):
    """Chuẩn hóa dữ liệu"""
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(df)
    return pd.DataFrame(scaled_array, columns=df.columns)


def apply_pca(data, n_components=4):
    """Giảm chiều dữ liệu với PCA"""
    pca = PCA(n_components=n_components)
    data_pca = pca.fit_transform(data)
    return data_pca, pca


def evaluate_clustering(X, clustering_algo):
    """Tính các chỉ số đánh giá mô hình phân cụm"""
    clustering_algo.fit(X)
    labels = clustering_algo.labels_
    if len(set(labels)) == 1:
        return None, None, None
    return (
        silhouette_score(X, labels),
        davies_bouldin_score(X, labels),
        calinski_harabasz_score(X, labels)
    )


def clustering_summary(data_pca):
    """Chạy các mô hình phân cụm và trả về bảng kết quả"""
    kmeans = KMeans(n_clusters=4, random_state=123)
    kmeans_metrics = evaluate_clustering(data_pca, kmeans)

    hierarchical = AgglomerativeClustering(n_clusters=4)
    hierarchical_metrics = evaluate_clustering(data_pca, hierarchical)

    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan.fit(data_pca)
    labels_dbscan = dbscan.labels_

    if len(set(labels_dbscan)) > 1:
        dbscan_metrics = (
            silhouette_score(data_pca, labels_dbscan),
            davies_bouldin_score(data_pca, labels_dbscan),
            calinski_harabasz_score(data_pca, labels_dbscan)
        )
    else:
        dbscan_metrics = ('N/A', 'N/A', 'N/A')

    results = {
        'Clustering Method': ['KMeans', 'Hierarchical', 'DBSCAN'],
        'Silhouette Score': [kmeans_metrics[0], hierarchical_metrics[0], dbscan_metrics[0]],
        'Davies-Bouldin Index': [kmeans_metrics[1], hierarchical_metrics[1], dbscan_metrics[1]],
        'Calinski-Harabasz Index': [kmeans_metrics[2], hierarchical_metrics[2], dbscan_metrics[2]],
    }
    return pd.DataFrame(results)


def calculate_inertia(data, max_clusters=20):
    """Tính inertia để vẽ Elbow Curve"""
    inertia = []
    for n in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=n, random_state=123)
        kmeans.fit(data)
        inertia.append(kmeans.inertia_)
    return inertia