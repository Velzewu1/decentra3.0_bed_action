#!/usr/bin/env python3
"""
Тестирование разных алгоритмов кластеризации для сбалансированных результатов
"""

import pandas as pd
import numpy as np
import hdbscan
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

# Загружаем данные
from core.data_processing import process_transaction_data
from utils.helpers import load_transaction_data, setup_logging

setup_logging('INFO')
print("🎯 ТЕСТИРОВАНИЕ АЛГОРИТМОВ ДЛЯ СБАЛАНСИРОВАННЫХ КЛАСТЕРОВ\n")

# Загружаем и обрабатываем данные
df = load_transaction_data('transactions.csv')
features_df, ml_features_processed, _ = process_transaction_data(df)

print(f"📊 Данные: {ml_features_processed.shape}")

def analyze_clustering_result(labels, algorithm_name):
    """Анализ результатов кластеризации"""
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    noise_ratio = np.sum(labels == -1) / len(labels) if -1 in labels else 0
    
    # Размеры кластеров
    cluster_sizes = []
    for label in unique_labels:
        if label != -1:
            size = np.sum(labels == label)
            cluster_sizes.append(size)
    
    if len(cluster_sizes) == 0:
        return None
        
    max_size = max(cluster_sizes)
    min_size = min(cluster_sizes)
    balance_ratio = min_size / max_size
    largest_pct = max_size / len(labels) * 100
    
    # Silhouette score
    silhouette = None
    if n_clusters > 1:
        non_noise = labels != -1
        if np.sum(non_noise) > 1:
            try:
                silhouette = silhouette_score(ml_features_processed[non_noise], labels[non_noise])
            except:
                pass
    
    return {
        'algorithm': algorithm_name,
        'n_clusters': n_clusters,
        'noise_ratio': noise_ratio,
        'cluster_sizes': sorted(cluster_sizes, reverse=True),
        'balance_ratio': balance_ratio,
        'largest_pct': largest_pct,
        'silhouette': silhouette
    }

results = []

print("🧪 ТЕСТИРОВАНИЕ РАЗНЫХ АЛГОРИТМОВ:")
print("="*60)

# 1. K-Means с разным количеством кластеров
print("\n📊 K-MEANS:")
for n_clusters in [3, 4, 5, 6, 7, 8]:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(ml_features_processed)
    
    result = analyze_clustering_result(labels, f'K-Means (k={n_clusters})')
    if result:
        results.append(result)
        print(f"  k={n_clusters}: {result['n_clusters']} кластеров, баланс={result['balance_ratio']:.3f}, "
              f"макс={result['largest_pct']:.1f}%, silhouette={result['silhouette']:.3f}")

# 2. Gaussian Mixture Model
print("\n🔮 GAUSSIAN MIXTURE:")
for n_components in [3, 4, 5, 6, 7, 8]:
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    labels = gmm.fit_predict(ml_features_processed)
    
    result = analyze_clustering_result(labels, f'GMM (n={n_components})')
    if result:
        results.append(result)
        print(f"  n={n_components}: {result['n_clusters']} кластеров, баланс={result['balance_ratio']:.3f}, "
              f"макс={result['largest_pct']:.1f}%, silhouette={result['silhouette']:.3f}")

# 3. Агломеративная кластеризация
print("\n🌳 AGGLOMERATIVE CLUSTERING:")
for n_clusters in [3, 4, 5, 6, 7, 8]:
    agg = AgglomerativeClustering(n_clusters=n_clusters)
    labels = agg.fit_predict(ml_features_processed)
    
    result = analyze_clustering_result(labels, f'Agglomerative (n={n_clusters})')
    if result:
        results.append(result)
        print(f"  n={n_clusters}: {result['n_clusters']} кластеров, баланс={result['balance_ratio']:.3f}, "
              f"макс={result['largest_pct']:.1f}%, silhouette={result['silhouette']:.3f}")

# 4. HDBSCAN с маленькими кластерами
print("\n🔬 HDBSCAN (маленькие кластеры):")
for min_cluster_size in [20, 30, 40, 50, 60]:
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=5
    )
    labels = clusterer.fit_predict(ml_features_processed)
    
    result = analyze_clustering_result(labels, f'HDBSCAN (min_size={min_cluster_size})')
    if result:
        results.append(result)
        print(f"  min_size={min_cluster_size}: {result['n_clusters']} кластеров, "
              f"баланс={result['balance_ratio']:.3f}, макс={result['largest_pct']:.1f}%, "
              f"шум={result['noise_ratio']:.1%}, silhouette={result['silhouette']:.3f}")

# Анализ лучших результатов
print(f"\n🏆 ЛУЧШИЕ РЕЗУЛЬТАТЫ ПО БАЛАНСУ:")
print("="*60)

# Фильтруем и сортируем результаты
good_results = [r for r in results if r['balance_ratio'] > 0.2 and r['largest_pct'] < 50]
good_results.sort(key=lambda x: (x['balance_ratio'], x['silhouette'] or 0), reverse=True)

for i, result in enumerate(good_results[:10], 1):
    print(f"\n{i}. {result['algorithm']}")
    print(f"   Кластеры: {result['n_clusters']}")
    print(f"   Размеры: {result['cluster_sizes']}")
    print(f"   Баланс (мин/макс): {result['balance_ratio']:.3f}")
    print(f"   Самый большой: {result['largest_pct']:.1f}%")
    print(f"   Silhouette: {result['silhouette']:.3f}" if result['silhouette'] else "   Silhouette: N/A")
    if result['noise_ratio'] > 0:
        print(f"   Шум: {result['noise_ratio']:.1%}")

print(f"\n📈 ПО SILHOUETTE SCORE:")
print("="*40)
silhouette_sorted = [r for r in results if r['silhouette'] is not None]
silhouette_sorted.sort(key=lambda x: x['silhouette'], reverse=True)

for i, result in enumerate(silhouette_sorted[:5], 1):
    print(f"{i}. {result['algorithm']}: {result['silhouette']:.3f} "
          f"(баланс={result['balance_ratio']:.3f}, макс={result['largest_pct']:.1f}%)") 