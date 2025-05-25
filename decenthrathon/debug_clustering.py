#!/usr/bin/env python3
"""
Debug HDBSCAN Clustering
Simple test to see what's happening
"""

import pandas as pd
import numpy as np
import hdbscan
from sklearn.metrics import silhouette_score

# Загружаем данные
from core.data_processing import process_transaction_data
from utils.helpers import load_transaction_data, setup_logging

setup_logging('INFO')
print("🔧 Loading data...")
df = load_transaction_data('transactions.csv')

print("🔧 Processing features...")
features_df, ml_features_processed, processing_info = process_transaction_data(df)

print(f"📊 Data shape: {ml_features_processed.shape}")

# Тестируем простой HDBSCAN
print("🔧 Testing simple HDBSCAN...")

for min_cluster_size in [5, 10, 20, 50]:
    for min_samples in [3, 5]:
        print(f"\n🎯 Testing: min_cluster_size={min_cluster_size}, min_samples={min_samples}")
        
        try:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples
            )
            
            labels = clusterer.fit_predict(ml_features_processed)
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            noise_ratio = np.sum(labels == -1) / len(labels)
            non_noise = labels != -1
            
            print(f"   Clusters: {n_clusters}")
            print(f"   Noise ratio: {noise_ratio:.1%}")
            print(f"   Non-noise points: {np.sum(non_noise)}")
            
            if n_clusters > 1 and np.sum(non_noise) > 1:
                try:
                    silhouette = silhouette_score(ml_features_processed[non_noise], labels[non_noise])
                    print(f"   Silhouette: {silhouette:.3f}")
                    print("   ✅ SUCCESS!")
                except Exception as e:
                    print(f"   ❌ Silhouette failed: {e}")
            else:
                print("   ❌ Not enough clusters/points for silhouette")
                
        except Exception as e:
            print(f"   ❌ Clustering failed: {e}")

print("\n🏆 Debug complete!") 