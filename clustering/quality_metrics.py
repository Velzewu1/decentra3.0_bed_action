from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import numpy as np

def evaluate_clustering_quality(ml_features_scaled, labels, chosen_algorithm):
    """
    Evaluate clustering quality using multiple metrics
    """
    
    # Remove noise points for evaluation (-1 labels)
    non_noise_mask = labels != -1
    
    if np.sum(non_noise_mask) == 0:
        print("❌ No valid clusters found (all points are noise)")
        return {
            'silhouette_score': -1.0,
            'calinski_harabasz': 0.0,
            'davies_bouldin': float('inf'),
            'n_clusters': 0,
            'noise_ratio': 1.0,
            'largest_cluster_ratio': 0.0
        }
    
    X_filtered = ml_features_scaled[non_noise_mask]
    labels_filtered = labels[non_noise_mask]
    
    n_clusters = len(np.unique(labels_filtered))
    
    if n_clusters < 2:
        print(f"❌ Insufficient clusters ({n_clusters}) for quality evaluation")
        return {
            'silhouette_score': -1.0,
            'calinski_harabasz': 0.0,
            'davies_bouldin': float('inf'),
            'n_clusters': n_clusters,
            'noise_ratio': np.sum(labels == -1) / len(labels),
            'largest_cluster_ratio': 0.0
        }
    
    # Calculate metrics
    silhouette = silhouette_score(X_filtered, labels_filtered)
    calinski = calinski_harabasz_score(X_filtered, labels_filtered)
    davies_bouldin = davies_bouldin_score(X_filtered, labels_filtered)
    
    # Additional metrics
    noise_ratio = np.sum(labels == -1) / len(labels)
    cluster_sizes = np.bincount(labels_filtered)
    largest_cluster_ratio = np.max(cluster_sizes) / len(labels_filtered)
    
    metrics = {
        'silhouette_score': silhouette,
        'calinski_harabasz': calinski,
        'davies_bouldin': davies_bouldin,
        'n_clusters': n_clusters,
        'noise_ratio': noise_ratio,
        'largest_cluster_ratio': largest_cluster_ratio
    }
    
    print(f"\n📈 CLUSTERING QUALITY METRICS for {chosen_algorithm}:")
    print(f"   • silhouette_score: {silhouette:.3f}")
    print(f"   • calinski_harabasz: {calinski:.3f}")
    print(f"   • davies_bouldin: {davies_bouldin:.3f}")
    print(f"   • n_clusters: {n_clusters}")
    print(f"   • noise_ratio: {noise_ratio:.3f}")
    print(f"   • largest_cluster_ratio: {largest_cluster_ratio:.3f}")
    
    return metrics
