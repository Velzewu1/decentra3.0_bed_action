#!/usr/bin/env python3
"""
Модуль кластеризации с поддержкой сбалансированных алгоритмов
"""

import numpy as np
import pandas as pd
import hdbscan
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import logging
from typing import Dict, Tuple, Any
import time

logger = logging.getLogger(__name__)

def evaluate_clustering_balance(labels: np.ndarray) -> Dict[str, float]:
    """Оценка сбалансированности кластеров"""
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    noise_ratio = np.sum(labels == -1) / len(labels) if -1 in labels else 0.0
    
    # Размеры кластеров (исключая шум)
    cluster_sizes = []
    for label in unique_labels:
        if label != -1:
            size = np.sum(labels == label)
            cluster_sizes.append(size)
    
    if len(cluster_sizes) == 0:
        return {
            'n_clusters': 0,
            'noise_ratio': noise_ratio,
            'balance_ratio': 0.0,
            'largest_cluster_pct': 0.0,
            'silhouette': None
        }
    
    max_size = max(cluster_sizes)
    min_size = min(cluster_sizes)
    balance_ratio = min_size / max_size if max_size > 0 else 0
    largest_cluster_pct = max_size / len(labels) * 100
    
    return {
        'n_clusters': n_clusters,
        'noise_ratio': noise_ratio,
        'balance_ratio': balance_ratio,
        'largest_cluster_pct': largest_cluster_pct,
        'cluster_sizes': sorted(cluster_sizes, reverse=True)
    }

def calculate_silhouette_safe(X: np.ndarray, labels: np.ndarray) -> float:
    """Безопасный расчет silhouette score"""
    try:
        # Убираем шум для расчета
        non_noise = labels != -1
        if np.sum(non_noise) < 2:
            return None
            
        unique_labels = np.unique(labels[non_noise])
        if len(unique_labels) < 2:
            return None
            
        return silhouette_score(X[non_noise], labels[non_noise])
    except Exception as e:
        logger.warning(f"Ошибка расчета silhouette score: {e}")
        return None

def run_gmm_clustering(X: np.ndarray, params: Dict) -> Dict[str, Any]:
    """Gaussian Mixture Model кластеризация"""
    logger.info("🔮 Запуск GMM кластеризации...")
    
    best_result = None
    best_score = -1
    results = []
    
    # Тестируем разные количества компонент
    for n_components in params.get('gmm_n_components_range', [3, 4, 5]):
        for covariance_type in params.get('gmm_covariance_type', ['full']):
            try:
                start_time = time.time()
                
                gmm = GaussianMixture(
                    n_components=n_components,
                    covariance_type=covariance_type,
                    init_params=params.get('gmm_init_params', 'kmeans'),
                    max_iter=params.get('gmm_max_iter', 100),
                    tol=params.get('gmm_tol', 1e-3),
                    random_state=params.get('gmm_random_state', 42)
                )
                
                labels = gmm.fit_predict(X)
                fit_time = time.time() - start_time
                
                # Оценка качества
                balance_metrics = evaluate_clustering_balance(labels)
                silhouette = calculate_silhouette_safe(X, labels)
                
                # Комбинированный score: баланс (60%) + silhouette (40%)
                if silhouette is not None:
                    combined_score = balance_metrics['balance_ratio'] * 0.6 + silhouette * 0.4
                else:
                    combined_score = balance_metrics['balance_ratio'] * 0.6
                
                result = {
                    'labels': labels,
                    'clusterer': gmm,
                    'params': {
                        'algorithm': 'gmm',
                        'n_components': n_components,
                        'covariance_type': covariance_type
                    },
                    'metrics': {
                        **balance_metrics,
                        'silhouette': silhouette,
                        'combined_score': combined_score,
                        'fit_time': fit_time
                    }
                }
                
                results.append(result)
                
                silhouette_str = f"{silhouette:.3f}" if silhouette is not None else "N/A"
                logger.info(f"  n_components={n_components}, cov={covariance_type}: "
                          f"{balance_metrics['n_clusters']} кластеров, "
                          f"баланс={balance_metrics['balance_ratio']:.3f}, "
                          f"макс={balance_metrics['largest_cluster_pct']:.1f}%, "
                          f"silhouette={silhouette_str}")
                
                # Обновляем лучший результат
                if combined_score > best_score:
                    best_score = combined_score
                    best_result = result
                    
            except Exception as e:
                logger.error(f"Ошибка GMM (n={n_components}, cov={covariance_type}): {e}")
    
    if best_result is None:
        raise ValueError("Не удалось выполнить GMM кластеризацию")
    
    logger.info(f"✅ Лучший GMM результат: "
              f"{best_result['metrics']['n_clusters']} кластеров, "
              f"баланс={best_result['metrics']['balance_ratio']:.3f}, "
              f"score={best_score:.3f}")
    
    return {'gmm_best': best_result, 'gmm_all_results': results}

def run_kmeans_clustering(X: np.ndarray, params: Dict) -> Dict[str, Any]:
    """K-Means кластеризация"""
    logger.info("📊 Запуск K-Means кластеризации...")
    
    best_result = None
    best_score = -1
    results = []
    
    # Тестируем разные количества кластеров
    for n_clusters in params.get('kmeans_n_clusters_range', [3, 4, 5, 6, 7]):
        try:
            start_time = time.time()
            
            kmeans = KMeans(
                n_clusters=n_clusters,
                init=params.get('kmeans_init', 'k-means++'),
                n_init=params.get('kmeans_n_init', 10),
                max_iter=params.get('kmeans_max_iter', 300),
                random_state=params.get('kmeans_random_state', 42)
            )
            
            labels = kmeans.fit_predict(X)
            fit_time = time.time() - start_time
            
            # Оценка качества
            balance_metrics = evaluate_clustering_balance(labels)
            silhouette = calculate_silhouette_safe(X, labels)
            
            # Комбинированный score
            if silhouette is not None:
                combined_score = balance_metrics['balance_ratio'] * 0.6 + silhouette * 0.4
            else:
                combined_score = balance_metrics['balance_ratio'] * 0.6
            
            result = {
                'labels': labels,
                'clusterer': kmeans,
                'params': {
                    'algorithm': 'kmeans',
                    'n_clusters': n_clusters
                },
                'metrics': {
                    **balance_metrics,
                    'silhouette': silhouette,
                    'combined_score': combined_score,
                    'fit_time': fit_time
                }
            }
            
            results.append(result)
            
            silhouette_str = f"{silhouette:.3f}" if silhouette is not None else "N/A"
            logger.info(f"  k={n_clusters}: "
                      f"{balance_metrics['n_clusters']} кластеров, "
                      f"баланс={balance_metrics['balance_ratio']:.3f}, "
                      f"макс={balance_metrics['largest_cluster_pct']:.1f}%, "
                      f"silhouette={silhouette_str}")
            
            # Обновляем лучший результат
            if combined_score > best_score:
                best_score = combined_score
                best_result = result
                
        except Exception as e:
            logger.error(f"Ошибка K-Means (k={n_clusters}): {e}")
    
    if best_result is None:
        raise ValueError("Не удалось выполнить K-Means кластеризацию")
    
    logger.info(f"✅ Лучший K-Means результат: "
              f"{best_result['metrics']['n_clusters']} кластеров, "
              f"баланс={best_result['metrics']['balance_ratio']:.3f}")
    
    return {'kmeans_best': best_result, 'kmeans_all_results': results}

def run_hdbscan_clustering(X: np.ndarray, params: Dict) -> Dict[str, Any]:
    """HDBSCAN кластеризация (оригинальная)"""
    logger.info("🔬 Запуск HDBSCAN кластеризации...")
    
    best_result = None
    best_score = -1
    results = []
    
    # Берем только первые значения для ускорения
    min_cluster_sizes = params.get('min_cluster_size_range', [5, 10])[:2]
    min_samples_list = params.get('min_samples_range', [3, 5])[:2]
    
    for min_cluster_size in min_cluster_sizes:
        for min_samples in min_samples_list:
            try:
                start_time = time.time()
                
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    cluster_selection_epsilon=0.0,
                    cluster_selection_method='eom',
                    metric='euclidean',
                    n_jobs=params.get('n_jobs', -1),
                    core_dist_n_jobs=params.get('core_dist_n_jobs', -1)
                )
                
                labels = clusterer.fit_predict(X)
                fit_time = time.time() - start_time
                
                # Оценка качества
                balance_metrics = evaluate_clustering_balance(labels)
                silhouette = calculate_silhouette_safe(X, labels)
                
                # Для HDBSCAN приоритет silhouette
                if silhouette is not None:
                    combined_score = silhouette * 0.7 + balance_metrics['balance_ratio'] * 0.3
                else:
                    combined_score = balance_metrics['balance_ratio'] * 0.3
                
                result = {
                    'labels': labels,
                    'clusterer': clusterer,
                    'params': {
                        'algorithm': 'hdbscan',
                        'min_cluster_size': min_cluster_size,
                        'min_samples': min_samples
                    },
                    'metrics': {
                        **balance_metrics,
                        'silhouette': silhouette,
                        'combined_score': combined_score,
                        'fit_time': fit_time
                    }
                }
                
                results.append(result)
                
                silhouette_str = f"{silhouette:.3f}" if silhouette is not None else "N/A"
                logger.info(f"  min_size={min_cluster_size}, min_samples={min_samples}: "
                          f"{balance_metrics['n_clusters']} кластеров, "
                          f"шум={balance_metrics['noise_ratio']:.1%}, "
                          f"silhouette={silhouette_str}")
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_result = result
                    
            except Exception as e:
                logger.error(f"Ошибка HDBSCAN: {e}")
    
    if best_result is None:
        raise ValueError("Не удалось выполнить HDBSCAN кластеризацию")
    
    return {'hdbscan_best': best_result, 'hdbscan_all_results': results}

def perform_clustering(X: np.ndarray, params: Dict) -> Tuple[np.ndarray, Any, Dict]:
    """
    Основная функция кластеризации с поддержкой разных алгоритмов
    
    Returns:
        Tuple[labels, clusterer, results_dict]
    """
    logger.info("🎯 Запуск сбалансированной кластеризации...")
    
    algorithm = params.get('algorithm', 'gmm').lower()
    
    if algorithm == 'gmm':
        results = run_gmm_clustering(X, params)
        best_result = results['gmm_best']
    elif algorithm == 'kmeans':
        results = run_kmeans_clustering(X, params)
        best_result = results['kmeans_best']
    elif algorithm == 'hdbscan':
        results = run_hdbscan_clustering(X, params)
        best_result = results['hdbscan_best']
    else:
        # Тестируем все алгоритмы и выбираем лучший
        logger.info("🧪 Тестирование всех алгоритмов...")
        
        all_results = {}
        best_overall = None
        best_overall_score = -1
        
        # GMM
        try:
            gmm_results = run_gmm_clustering(X, params)
            all_results.update(gmm_results)
            if gmm_results['gmm_best']['metrics']['combined_score'] > best_overall_score:
                best_overall_score = gmm_results['gmm_best']['metrics']['combined_score']
                best_overall = gmm_results['gmm_best']
        except Exception as e:
            logger.error(f"GMM failed: {e}")
        
        # K-Means
        try:
            kmeans_results = run_kmeans_clustering(X, params)
            all_results.update(kmeans_results)
            if kmeans_results['kmeans_best']['metrics']['combined_score'] > best_overall_score:
                best_overall_score = kmeans_results['kmeans_best']['metrics']['combined_score']
                best_overall = kmeans_results['kmeans_best']
        except Exception as e:
            logger.error(f"K-Means failed: {e}")
        
        # HDBSCAN
        try:
            hdbscan_results = run_hdbscan_clustering(X, params)
            all_results.update(hdbscan_results)
            if hdbscan_results['hdbscan_best']['metrics']['combined_score'] > best_overall_score:
                best_overall_score = hdbscan_results['hdbscan_best']['metrics']['combined_score']
                best_overall = hdbscan_results['hdbscan_best']
        except Exception as e:
            logger.error(f"HDBSCAN failed: {e}")
        
        if best_overall is None:
            raise ValueError("Все алгоритмы кластеризации завершились с ошибкой")
        
        results = all_results
        best_result = best_overall
    
    # Финальная информация
    metrics = best_result['metrics']
    silhouette_str = f"{metrics['silhouette']:.3f}" if metrics['silhouette'] is not None else "N/A"
    logger.info(f"🏆 ФИНАЛЬНЫЙ РЕЗУЛЬТАТ:")
    logger.info(f"   Алгоритм: {best_result['params']['algorithm'].upper()}")
    logger.info(f"   Кластеры: {metrics['n_clusters']}")
    logger.info(f"   Размеры: {metrics.get('cluster_sizes', 'N/A')}")
    logger.info(f"   Баланс: {metrics['balance_ratio']:.3f}")
    logger.info(f"   Самый большой: {metrics['largest_cluster_pct']:.1f}%")
    logger.info(f"   Silhouette: {silhouette_str}")
    logger.info(f"   Время: {metrics['fit_time']:.1f}с")
    
    return best_result['labels'], best_result['clusterer'], results 