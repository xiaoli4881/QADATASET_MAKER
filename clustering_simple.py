import numpy as np
from sklearn.cluster import KMeans

def kmeans_clustering_simple(data, n_clusters=2, random_state=42):
    """
    使用K-means算法对一维数据进行聚类，返回两个聚类列表
    
    参数:
    data: list - 输入的一维数据列表
    n_clusters: int - 聚类数量，默认为2
    random_state: int - 随机种子，默认为42
    
    返回:
    tuple: 包含两个列表的元组 (cluster_1, cluster_2)
    """

    X = np.array(data).reshape(-1, 1)
    
    print(X,"kmeans_clustering"*10)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(X)

    cluster_1 = [data[i] for i in range(len(data)) if labels[i] == 0]
    cluster_2 = [data[i] for i in range(len(data)) if labels[i] == 1]
    
    return cluster_1,cluster_2
    
