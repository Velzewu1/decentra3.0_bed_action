#!/usr/bin/env python3
"""
Анализ проблемы несбалансированности кластеров
"""

import pandas as pd
import numpy as np
import hdbscan
from sklearn.metrics import silhouette_score

# Загружаем данные
from core.data_processing import process_transaction_data
from utils.helpers import load_transaction_data, setup_logging

setup_logging('INFO')
print("🔍 АНАЛИЗ ПРОБЛЕМЫ НЕСБАЛАНСИРОВАННОСТИ КЛАСТЕРОВ\n")

# Загружаем и обрабатываем данные
df = load_transaction_data('transactions.csv')
features_df, ml_features_processed, _ = process_transaction_data(df)

print(f"📊 Данные: {ml_features_processed.shape}")

# Тестируем разные параметры для более сбалансированных кластеров
print("\n🎯 ТЕСТИРОВАНИЕ ПАРАМЕТРОВ ДЛЯ СБАЛАНСИРОВАННОСТИ:")
print("="*70)

balance_results = []

# Более агрессивные параметры для разделения большого кластера
test_params = [
    # Маленькие кластеры, строгие условия
    {'min_cluster_size': 50, 'min_samples': 10, 'cluster_selection_epsilon': 0.0},
    {'min_cluster_size': 100, 'min_samples': 20, 'cluster_selection_epsilon': 0.0},
    {'min_cluster_size': 150, 'min_samples': 30, 'cluster_selection_epsilon': 0.0},
    
    # Средние кластеры
    {'min_cluster_size': 200, 'min_samples': 50, 'cluster_selection_epsilon': 0.0},
    {'min_cluster_size': 300, 'min_samples': 75, 'cluster_selection_epsilon': 0.0},
    
    # С epsilon для более детального разделения
    {'min_cluster_size': 50, 'min_samples': 10, 'cluster_selection_epsilon': 0.1},
    {'min_cluster_size': 100, 'min_samples': 20, 'cluster_selection_epsilon': 0.2},
    {'min_cluster_size': 150, 'min_samples': 30, 'cluster_selection_epsilon': 0.3},
    
    # Leaf clustering для максимального разделения
    {'min_cluster_size': 100, 'min_samples': 20, 'cluster_selection_epsilon': 0.0, 'cluster_selection_method': 'leaf'},
    {'min_cluster_size': 200, 'min_samples': 40, 'cluster_selection_epsilon': 0.0, 'cluster_selection_method': 'leaf'},
]

for i, params in enumerate(test_params, 1):
    print(f"\n🔬 Тест {i}: {params}")
    
    try:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=params['min_cluster_size'],
            min_samples=params['min_samples'],
            cluster_selection_epsilon=params.get('cluster_selection_epsilon', 0.0),
            cluster_selection_method=params.get('cluster_selection_method', 'eom')
        )
        
        labels = clusterer.fit_predict(ml_features_processed)
        
        # Анализ результатов
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        noise_ratio = np.sum(labels == -1) / len(labels)
        
        # Распределение размеров кластеров
        cluster_sizes = []
        for label in unique_labels:
            if label != -1:
                size = np.sum(labels == label)
                cluster_sizes.append(size)
        
        if len(cluster_sizes) > 0:
            max_cluster_size = max(cluster_sizes)
            min_cluster_size = min(cluster_sizes)
            balance_ratio = min_cluster_size / max_cluster_size if max_cluster_size > 0 else 0
            largest_cluster_pct = max_cluster_size / len(labels) * 100
            
            print(f"   Кластеры: {n_clusters}")
            print(f"   Шум: {noise_ratio:.1%}")
            print(f"   Размеры кластеров: {sorted(cluster_sizes, reverse=True)}")
            print(f"   Баланс (мин/макс): {balance_ratio:.3f}")
            print(f"   Самый большой кластер: {largest_cluster_pct:.1f}%")
            
            # Silhouette только если достаточно кластеров
            silhouette = None
            if n_clusters > 1:
                non_noise = labels != -1
                if np.sum(non_noise) > n_clusters:
                    try:
                        silhouette = silhouette_score(ml_features_processed[non_noise], labels[non_noise])
                        print(f"   Silhouette: {silhouette:.3f}")
                    except:
                        print(f"   Silhouette: ошибка расчета")
            
            balance_results.append({
                'params': params,
                'n_clusters': n_clusters,
                'noise_ratio': noise_ratio,
                'balance_ratio': balance_ratio,
                'largest_cluster_pct': largest_cluster_pct,
                'silhouette': silhouette,
                'cluster_sizes': cluster_sizes
            })
            
            # Отмечаем хорошие результаты
            if largest_cluster_pct < 60 and n_clusters >= 3 and balance_ratio > 0.1:
                print(f"   ✅ ХОРОШИЙ БАЛАНС!")
        else:
            print(f"   ❌ Нет валидных кластеров")
            
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")

# Находим лучшие результаты по балансу
print(f"\n🏆 ЛУЧШИЕ РЕЗУЛЬТАТЫ ПО СБАЛАНСИРОВАННОСТИ:")
print("="*70)

good_results = [r for r in balance_results if r['largest_cluster_pct'] < 60 and r['n_clusters'] >= 3]
good_results.sort(key=lambda x: (x['balance_ratio'], -x['largest_cluster_pct']), reverse=True)

for i, result in enumerate(good_results[:5], 1):
    print(f"\n{i}. Кластеры: {result['n_clusters']}, Баланс: {result['balance_ratio']:.3f}")
    print(f"   Самый большой: {result['largest_cluster_pct']:.1f}%")
    print(f"   Размеры: {result['cluster_sizes']}")
    print(f"   Silhouette: {result['silhouette']:.3f}" if result['silhouette'] else "   Silhouette: N/A")
    print(f"   Параметры: {result['params']}")

print(f"\n💡 РЕКОМЕНДАЦИИ:")
print("="*50)
print("1. Использовать min_cluster_size 100-200 для более сбалансированных групп")
print("2. Увеличить min_samples для строгого разделения")
print("3. Экспериментировать с cluster_selection_epsilon")
print("4. Рассмотреть cluster_selection_method='leaf'")
print("5. Возможно, данные имеют естественную иерархическую структуру") 