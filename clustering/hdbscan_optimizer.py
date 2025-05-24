import numpy as np
import hdbscan
from sklearn.decomposition import PCA
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import silhouette_score

def optimize_hdbscan_clustering(ml_features_scaled, features_df):
    """
    ULTRA-OPTIMIZED HDBSCAN clustering with advanced techniques:
    1. Feature selection and dimensionality reduction
    2. Multi-stage parameter optimization
    3. Ensemble clustering approach
    4. Advanced noise reduction
    5. Maximum cluster discovery
    """
    
    print("\n🚀 ULTRA-OPTIMIZED HDBSCAN CLUSTERING...")
    print("   🧬 Advanced preprocessing and optimization techniques...")
    
    # =====================================================================================
    # STAGE 1: ADVANCED DATA PREPROCESSING
    # =====================================================================================
    
    print("\n🔬 STAGE 1: ADVANCED DATA PREPROCESSING")
    
    # Feature selection - remove highly correlated features
    correlation_matrix = np.corrcoef(ml_features_scaled.T)
    correlation_threshold = 0.95
    to_remove = set()
    
    for i in range(len(correlation_matrix)):
        for j in range(i+1, len(correlation_matrix)):
            if abs(correlation_matrix[i][j]) > correlation_threshold:
                to_remove.add(j)
    
    if to_remove:
        keep_indices = [i for i in range(ml_features_scaled.shape[1]) if i not in to_remove]
        ml_features_optimized = ml_features_scaled[:, keep_indices]
        print(f"   🎯 Removed {len(to_remove)} highly correlated features")
    else:
        ml_features_optimized = ml_features_scaled
        print("   ✅ No highly correlated features found")
    
    # PCA for noise reduction while preserving variance
    pca = PCA(n_components=0.95, random_state=42)  # Keep 95% of variance
    ml_features_pca = pca.fit_transform(ml_features_optimized)
    explained_variance = np.sum(pca.explained_variance_ratio_)
    
    print(f"   📊 PCA: {ml_features_pca.shape[1]} components, {explained_variance:.1%} variance retained")
    
    # Additional scaling for better cluster separation
    power_transformer = PowerTransformer(method='yeo-johnson', standardize=True)
    try:
        ml_features_transformed = power_transformer.fit_transform(ml_features_pca)
        print("   ⚡ Applied Yeo-Johnson power transformation")
    except:
        ml_features_transformed = ml_features_pca
        print("   ⚠️ Power transformation failed, using PCA features")
    
    # =====================================================================================
    # STAGE 2: ULTRA-EXTENSIVE PARAMETER SEARCH
    # =====================================================================================
    
    print("\n🎯 STAGE 2: ULTRA-EXTENSIVE PARAMETER OPTIMIZATION")
    
    best_score = -1
    best_clusterer = None
    best_labels = None
    best_params = None
    best_metrics = None
    best_data = None
    
    # ULTRA-AGGRESSIVE parameter ranges for maximum clusters and minimum noise
    min_cluster_sizes = [3, 5, 7, 10, 12, 15, 18, 20, 25, 30]  # Very small clusters
    min_samples_range = [1, 2, 3, 4, 5, 7, 10]  # Very low min_samples
    cluster_selection_epsilons = [0.0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.15]  # Fine-grained epsilon
    cluster_selection_methods = ['eom', 'leaf']
    metrics = ['euclidean', 'manhattan', 'cosine']  # Different distance metrics
    
    # Test both original and transformed features
    feature_sets = [
        ('Original', ml_features_scaled),
        ('Optimized', ml_features_optimized),
        ('PCA', ml_features_pca),
        ('Transformed', ml_features_transformed)
    ]
    
    total_combinations = (len(min_cluster_sizes) * len(min_samples_range) * 
                         len(cluster_selection_epsilons) * len(cluster_selection_methods) * 
                         len(metrics) * len(feature_sets))
    
    print(f"   🔥 Testing {total_combinations} ultra-optimized combinations...")
    
    tested_combinations = 0
    valid_combinations = 0
    
    for feature_name, feature_data in feature_sets:
        print(f"   📊 Testing feature set: {feature_name}")
        
        for metric in metrics:
            for min_cluster_size in min_cluster_sizes:
                for min_samples in min_samples_range:
                    for epsilon in cluster_selection_epsilons:
                        for method in cluster_selection_methods:
                            tested_combinations += 1
                            
                            # Skip invalid combinations
                            if min_samples > min_cluster_size:
                                continue
                            
                            # Skip cosine metric for some configurations (can be unstable)
                            if metric == 'cosine' and min_cluster_size < 10:
                                continue
                                
                            try:
                                clusterer = hdbscan.HDBSCAN(
                                    min_cluster_size=min_cluster_size,
                                    min_samples=min_samples,
                                    cluster_selection_epsilon=epsilon,
                                    cluster_selection_method=method,
                                    metric=metric,
                                    algorithm='best',
                                    leaf_size=30,
                                    core_dist_n_jobs=-1  # Use all CPU cores
                                )
                                
                                labels = clusterer.fit_predict(feature_data)
                                
                                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                                noise_ratio = np.sum(labels == -1) / len(labels)
                                
                                # AGGRESSIVE criteria for maximum clusters and minimum noise
                                if n_clusters >= 3 and noise_ratio < 0.4:  # At least 3 clusters, max 40% noise
                                    non_noise = labels != -1
                                    non_noise_count = np.sum(non_noise)
                                    
                                    if non_noise_count > len(labels) * 0.6:  # At least 60% non-noise
                                        unique_clusters = len(set(labels[non_noise]))
                                        
                                        if unique_clusters >= 3:
                                            # Calculate silhouette score
                                            silhouette = silhouette_score(feature_data[non_noise], labels[non_noise])
                                            
                                            # ULTRA-OPTIMIZED scoring function
                                            # Heavily reward more clusters and penalize noise
                                            cluster_bonus = min(n_clusters / 5.0, 0.8)  # Up to 0.8 bonus for clusters
                                            noise_penalty = noise_ratio * 0.6  # Heavy noise penalty
                                            
                                            # Bonus for balanced cluster sizes
                                            cluster_sizes = np.bincount(labels[non_noise])
                                            cluster_balance = 1.0 - (np.std(cluster_sizes) / np.mean(cluster_sizes))
                                            balance_bonus = cluster_balance * 0.2
                                            
                                            # Bonus for higher membership probabilities
                                            try:
                                                avg_prob = np.mean(clusterer.probabilities_[clusterer.probabilities_ > 0])
                                                prob_bonus = avg_prob * 0.1
                                            except:
                                                prob_bonus = 0
                                            
                                            # Combined score
                                            custom_score = (silhouette + cluster_bonus + balance_bonus + 
                                                          prob_bonus - noise_penalty)
                                            
                                            if custom_score > best_score:
                                                best_score = custom_score
                                                best_clusterer = clusterer
                                                best_labels = labels
                                                best_data = feature_data
                                                best_params = {
                                                    'feature_set': feature_name,
                                                    'min_cluster_size': min_cluster_size,
                                                    'min_samples': min_samples,
                                                    'cluster_selection_epsilon': epsilon,
                                                    'cluster_selection_method': method,
                                                    'metric': metric
                                                }
                                                best_metrics = {
                                                    'n_clusters': n_clusters,
                                                    'noise_ratio': noise_ratio,
                                                    'silhouette': silhouette,
                                                    'custom_score': custom_score,
                                                    'cluster_balance': cluster_balance,
                                                    'avg_probability': avg_prob if 'avg_prob' in locals() else 0
                                                }
                                                
                                                print(f"   🏆 NEW BEST: {n_clusters} clusters, {noise_ratio:.1%} noise, score: {custom_score:.3f}")
                                                
                                            valid_combinations += 1
                                            
                            except Exception as e:
                                continue
    
    print(f"\n   ✅ Tested {tested_combinations} combinations, {valid_combinations} valid results")
    
    # =====================================================================================
    # STAGE 3: ENSEMBLE REFINEMENT (if we have good results)
    # =====================================================================================
    
    if best_clusterer is not None and best_metrics['n_clusters'] >= 3:
        print(f"\n🔧 STAGE 3: ENSEMBLE REFINEMENT")
        
        # Try ensemble approach with multiple similar configurations
        ensemble_results = []
        base_params = best_params.copy()
        
        # Test variations around the best configuration
        for size_var in [-2, -1, 0, 1, 2]:
            for sample_var in [-1, 0, 1]:
                try:
                    new_min_cluster = max(3, base_params['min_cluster_size'] + size_var)
                    new_min_samples = max(1, base_params['min_samples'] + sample_var)
                    
                    if new_min_samples <= new_min_cluster:
                        ensemble_clusterer = hdbscan.HDBSCAN(
                            min_cluster_size=new_min_cluster,
                            min_samples=new_min_samples,
                            cluster_selection_epsilon=base_params['cluster_selection_epsilon'],
                            cluster_selection_method=base_params['cluster_selection_method'],
                            metric=base_params['metric'],
                            core_dist_n_jobs=-1
                        )
                        
                        ensemble_labels = ensemble_clusterer.fit_predict(best_data)
                        ensemble_n_clusters = len(set(ensemble_labels)) - (1 if -1 in ensemble_labels else 0)
                        ensemble_noise = np.sum(ensemble_labels == -1) / len(ensemble_labels)
                        
                        if ensemble_n_clusters > best_metrics['n_clusters'] and ensemble_noise < best_metrics['noise_ratio']:
                            print(f"   🚀 Ensemble improvement: {ensemble_n_clusters} clusters, {ensemble_noise:.1%} noise")
                            best_clusterer = ensemble_clusterer
                            best_labels = ensemble_labels
                            best_metrics['n_clusters'] = ensemble_n_clusters
                            best_metrics['noise_ratio'] = ensemble_noise
                            
                except Exception as e:
                    continue
    
    # =====================================================================================
    # RESULTS
    # =====================================================================================
    
    if best_clusterer is not None:
        print(f"\n🏆 ULTRA-OPTIMIZED HDBSCAN RESULTS:")
        print(f"   🎯 Clusters: {best_metrics['n_clusters']}")
        print(f"   🔇 Noise Ratio: {best_metrics['noise_ratio']:.1%}")
        print(f"   📈 Silhouette Score: {best_metrics['silhouette']:.3f}")
        print(f"   🚀 Ultra Score: {best_metrics['custom_score']:.3f}")
        print(f"   ⚖️ Cluster Balance: {best_metrics.get('cluster_balance', 0):.3f}")
        print(f"   🎲 Avg Probability: {best_metrics.get('avg_probability', 0):.3f}")
        print(f"   🔧 Best Config: {best_params}")
        
        # Additional HDBSCAN-specific metrics
        try:
            cluster_persistence = best_clusterer.cluster_persistence_
            probabilities = best_clusterer.probabilities_
            outlier_scores = best_clusterer.outlier_scores_
            
            print(f"   📊 Stable Clusters: {len(cluster_persistence)}")
            print(f"   🎯 High Confidence Points: {np.sum(probabilities > 0.8)}")
            print(f"   📉 Outlier Score Range: [{np.min(outlier_scores):.3f}, {np.max(outlier_scores):.3f}]")
            
        except Exception as e:
            print(f"   ⚠️ Could not compute additional metrics: {e}")
            cluster_persistence = []
            probabilities = np.array([])
            outlier_scores = np.array([])
        
        # Map labels back to original feature space for consistency
        results = {
            'HDBSCAN_UltraOptimized': {
                'labels': best_labels,
                'clusterer': best_clusterer,
                'n_clusters': best_metrics['n_clusters'],
                'silhouette': best_metrics['silhouette'],
                'noise_ratio': best_metrics['noise_ratio'],
                'custom_score': best_metrics['custom_score'],
                'cluster_persistence': cluster_persistence,
                'best_params': best_params,
                'probabilities': probabilities,
                'outlier_scores': outlier_scores,
                'feature_set_used': best_params.get('feature_set', 'Unknown'),
                'pros': [
                    'Ultra-optimized for maximum clusters',
                    'Minimized noise ratio',
                    'Advanced preprocessing',
                    'Ensemble refinement',
                    'Multi-metric optimization'
                ],
                'cons': [
                    'Computationally intensive',
                    'Complex parameter tuning'
                ]
            }
        }
        
        return results, 'HDBSCAN_UltraOptimized', best_clusterer
    
    else:
        print("❌ Ultra-optimization failed to find valid configurations")
        print("🔄 Falling back to aggressive configuration...")
        
        # Last resort - very aggressive settings
        try:
            aggressive_clusterer = hdbscan.HDBSCAN(
                min_cluster_size=5,
                min_samples=1,
                cluster_selection_method='leaf',
                metric='euclidean'
            )
            
            aggressive_labels = aggressive_clusterer.fit_predict(ml_features_transformed)
            n_clusters = len(set(aggressive_labels)) - (1 if -1 in aggressive_labels else 0)
            noise_ratio = np.sum(aggressive_labels == -1) / len(aggressive_labels)
            
            print(f"   🔥 Aggressive Results: {n_clusters} clusters, {noise_ratio:.1%} noise")
            
            results = {
                'HDBSCAN_Aggressive': {
                    'labels': aggressive_labels,
                    'clusterer': aggressive_clusterer,
                    'n_clusters': n_clusters,
                    'silhouette': 0.0,
                    'noise_ratio': noise_ratio,
                    'best_params': {
                        'min_cluster_size': 5,
                        'min_samples': 1,
                        'cluster_selection_method': 'leaf'
                    },
                    'cluster_persistence': [],
                    'probabilities': np.array([]),
                    'outlier_scores': np.array([]),
                    'pros': ['Aggressive clustering'],
                    'cons': ['May be overfitted']
                }
            }
            
            return results, 'HDBSCAN_Aggressive', aggressive_clusterer
            
        except Exception as e:
            print(f"❌ Even aggressive configuration failed: {e}")
            raise Exception("All optimization attempts failed")
