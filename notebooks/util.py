import numpy as np
import open3d as o3d


def cluster_pcd(pcd):
    cluster = pcd.cluster_dbscan(1, 2)
    point_num = len(cluster)
    clusters = np.asarray(cluster)
    cluster_indices = set(list(clusters))
    filtered_pcds = [[] for _ in cluster_indices]
    for k in cluster_indices:
        filtered_indices = filter(lambda x: clusters[x] == k, range(point_num))
        filtered_pcds[k] = pcd.select_down_sample(list(filtered_indices))
    return filtered_pcds
