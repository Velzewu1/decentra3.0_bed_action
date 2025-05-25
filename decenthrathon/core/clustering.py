#!/usr/bin/env python3
"""
Clustering Module
Реализация различных алгоритмов кластеризации с автоматическим выбором
"""

import numpy as np
import pandas as pd
import time
import logging
from typing import Dict, Any, Tuple, List, Optional
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

logger = logging.getLogger(__name__)

# Подавляем warnings от sklearn
warnings.filterwarnings('ignore', category=UserWarning)

def calculate_silhouette_safe(X: np.ndarray, labels: np.ndarray) -> Optional[float]:
    """Безопасный расчет silhouette score"""
    try:
        unique_labels = np.unique(labels)
        # Исключаем noise метки (-1) и проверяем достаточное количество кластеров
        valid_labels = unique_labels[unique_labels != -1]
        
        if len(valid_labels) < 2:
            return None
        
        # Проверяем, что каждый кластер содержит более 1 точки
        for label in valid_labels:
            if np.sum(labels == label) < 2:
                return None
        
        # Фильтруем данные без noise точек
        mask = labels != -1
        if np.sum(mask) < 2:
            return None
            
        return silhouette_score(X[mask], labels[mask])
    except:
        return None

def evaluate_clustering_balance(labels: np.ndarray) -> Dict[str, Any]:
    """Оценка сбалансированности кластеризации"""
    unique_labels = np.unique(labels)
    valid_labels = unique_labels[unique_labels != -1]  # Исключаем noise (-1)
    
    n_clusters = len(valid_labels)
    cluster_sizes = []
    
    for label in valid_labels:
        size = np.sum(labels == label)
        cluster_sizes.append(size)
    
    total_points = len(labels)
    noise_points = np.sum(labels == -1)
    valid_points = total_points - noise_points
    
    if n_clusters == 0 or valid_points == 0:
        return {
            'n_clusters': 0,
            'cluster_sizes': [],
            'balance_ratio': 0.0,
            'largest_cluster_pct': 100.0,
            'noise_ratio': noise_points / total_points
        }
    
    # Размеры кластеров в процентах от валидных точек
    cluster_percentages = [(size / valid_points) * 100 for size in cluster_sizes]
    largest_cluster_pct = max(cluster_percentages)
    
    # Balance ratio = 1 - normalized Gini coefficient
    # Чем ближе к 1, тем более сбалансированы кластеры
    sorted_sizes = sorted(cluster_sizes)
    cumsum = np.cumsum(sorted_sizes)
    n = len(cluster_sizes)
    gini = (2 * np.sum((np.arange(1, n + 1) * sorted_sizes)) / (n * sum(sorted_sizes))) - (n + 1) / n
    balance_ratio = 1 - gini
    
    return {
        'n_clusters': n_clusters,
        'cluster_sizes': cluster_sizes,
        'balance_ratio': balance_ratio,
        'largest_cluster_pct': largest_cluster_pct,
        'noise_ratio': noise_points / total_points
    }

def find_optimal_n_components_auto(X: np.ndarray, params: Dict) -> int:
    """
    Автоматическое определение оптимального количества компонентов для GMM
    через BIC/AIC критерии
    """
    logger.info("Ищем оптимальное количество компонентов через BIC/AIC")
    
    min_components = params.get('gmm_auto_min_components', 2)
    max_components = params.get('gmm_auto_max_components', 8)
    
    results = []
    
    for n_components in range(min_components, max_components + 1):
        for covariance_type in ['full', 'tied', 'diag', 'spherical']:
            try:
                gmm = GaussianMixture(
                    n_components=n_components,
                    covariance_type=covariance_type,
                    random_state=params.get('gmm_random_state', 42),
                    max_iter=params.get('gmm_max_iter', 100)
                )
                
                gmm.fit(X)
                labels = gmm.predict(X)
                
                bic = gmm.bic(X)
                aic = gmm.aic(X)
                
                # Оценка баланса
                balance_metrics = evaluate_clustering_balance(labels)
                
                # Silhouette
                silhouette = calculate_silhouette_safe(X, labels)
                
                results.append({
                    'n_components': n_components,
                    'covariance_type': covariance_type,
                    'bic': bic,
                    'aic': aic,
                    'balance_metrics': balance_metrics,
                    'silhouette': silhouette
                })
                
                silhouette_str = f"{silhouette:.3f}" if silhouette is not None else "N/A"
                logger.info(f"     n={n_components}, cov={covariance_type}: BIC={bic:.1f}, AIC={aic:.1f}, "
                          f"баланс={balance_metrics['balance_ratio']:.3f}, silhouette={silhouette_str}")
                
            except Exception as e:
                logger.error(f"   Ошибка для n_components={n_components}, cov={covariance_type}: {e}")
    
    if not results:
        logger.warning("Не удалось протестировать ни одной конфигурации, используем значение по умолчанию")
        return 3
    
    # Выбираем критерий
    criterion = params.get('gmm_auto_criterion', 'bic').lower()
    
    if criterion == 'bic':
        # Меньший BIC = лучше
        best_result = min(results, key=lambda x: x['bic'])
        logger.info(f"BIC выбрал: {best_result['n_components']} компонентов (BIC={best_result['bic']:.1f})")
    elif criterion == 'aic':
        # Меньший AIC = лучше  
        best_result = min(results, key=lambda x: x['aic'])
        logger.info(f"AIC выбрал: {best_result['n_components']} компонентов (AIC={best_result['aic']:.1f})")
    else:  # combined
        # Комбинированный критерий: BIC (50%) + balance (30%) + silhouette (20%)
        for result in results:
            # Нормализуем BIC (меньше = лучше, поэтому берем обратное)
            bic_score = 1.0 / (1.0 + result['bic'] / 1000)  # Нормализация
            balance_score = result['balance_metrics']['balance_ratio']
            silhouette_score = result['silhouette'] if result['silhouette'] is not None else 0.0
            
            combined_score = 0.5 * bic_score + 0.3 * balance_score + 0.2 * silhouette_score
            result['combined_score'] = combined_score
        
        best_result = max(results, key=lambda x: x['combined_score'])
        logger.info(f"Комбинированный критерий выбрал: {best_result['n_components']} компонентов "
                   f"(score={best_result['combined_score']:.3f})")
    
    return best_result['n_components']

def run_gmm_clustering_auto(X: np.ndarray, params: Dict) -> Dict[str, Any]:
    """
    GMM кластеризация с автоматическим определением количества компонентов
    """
    logger.info("Запуск GMM кластеризации с автоопределением компонентов")
    
    # Автоматически определяем оптимальное количество компонентов
    optimal_n_components = find_optimal_n_components_auto(X, params)
    
    logger.info(f"Оптимальное количество компонентов: {optimal_n_components}")
    
    # Теперь тестируем разные типы ковариации для найденного количества компонентов
    best_result = None
    best_score = -1
    results = []
    
    for covariance_type in params.get('gmm_covariance_type', ['full', 'tied', 'diag', 'spherical']):
        try:
            start_time = time.time()
            
            gmm = GaussianMixture(
                n_components=optimal_n_components,
                covariance_type=covariance_type,
                init_params=params.get('gmm_init_params', 'kmeans'),
                max_iter=params.get('gmm_max_iter', 100),
                tol=params.get('gmm_tol', 1e-3),
                random_state=params.get('gmm_random_state', 42)
            )
            
            labels = gmm.fit_predict(X)
            fit_time = time.time() - start_time
            
            # Дополнительные метрики
            bic = gmm.bic(X)
            aic = gmm.aic(X)
            
            # Оценка качества
            balance_metrics = evaluate_clustering_balance(labels)
            silhouette = calculate_silhouette_safe(X, labels)
            
            # Комбинированный score для выбора лучшего
            if silhouette is not None:
                combined_score = balance_metrics['balance_ratio'] * 0.6 + silhouette * 0.4
            else:
                combined_score = balance_metrics['balance_ratio'] * 0.6
            
            result = {
                'labels': labels,
                'clusterer': gmm,
                'params': {
                    'algorithm': 'gmm_auto',
                    'n_components': optimal_n_components,
                    'covariance_type': covariance_type
                },
                'metrics': {
                    **balance_metrics,
                    'silhouette': silhouette,
                    'combined_score': combined_score,
                    'fit_time': fit_time,
                    'bic': bic,
                    'aic': aic,
                    'optimal_n_components': optimal_n_components
                }
            }
            
            results.append(result)
            
            silhouette_str = f"{silhouette:.3f}" if silhouette is not None else "N/A"
            logger.info(f"  n_components={optimal_n_components}, cov={covariance_type}: "
                      f"{balance_metrics['n_clusters']} кластеров, "
                      f"баланс={balance_metrics['balance_ratio']:.3f}, "
                      f"макс={balance_metrics['largest_cluster_pct']:.1f}%, "
                      f"silhouette={silhouette_str}, BIC={bic:.1f}")
            
            # Обновляем лучший результат
            if combined_score > best_score:
                best_score = combined_score
                best_result = result
                
        except Exception as e:
            logger.error(f"Ошибка GMM (n={optimal_n_components}, cov={covariance_type}): {e}")
    
    if best_result is None:
        raise ValueError("Не удалось выполнить GMM кластеризацию")
    
    logger.info(f"Лучший автоматический GMM результат: "
              f"{best_result['metrics']['n_clusters']} кластеров, "
              f"баланс={best_result['metrics']['balance_ratio']:.3f}, "
              f"BIC={best_result['metrics']['bic']:.1f}, "
              f"score={best_score:.3f}")
    
    return {'gmm_auto_best': best_result, 'gmm_auto_all_results': results}

def run_gmm_clustering(X: np.ndarray, params: Dict) -> Dict[str, Any]:
    """Gaussian Mixture Model кластеризация"""
    logger.info("Запуск GMM кластеризации")
    
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
    
    logger.info(f"Лучший GMM результат: "
              f"{best_result['metrics']['n_clusters']} кластеров, "
              f"баланс={best_result['metrics']['balance_ratio']:.3f}, "
              f"score={best_score:.3f}")
    
    return {'gmm_best': best_result, 'gmm_all_results': results}

def run_kmeans_clustering(X: np.ndarray, params: Dict) -> Dict[str, Any]:
    """K-Means кластеризация"""
    logger.info("Запуск K-Means кластеризации")
    
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
    
    logger.info(f"Лучший K-Means результат: "
              f"{best_result['metrics']['n_clusters']} кластеров, "
              f"баланс={best_result['metrics']['balance_ratio']:.3f}")
    
    return {'kmeans_best': best_result, 'kmeans_all_results': results}

def run_hdbscan_clustering(X: np.ndarray, params: Dict) -> Dict[str, Any]:
    """HDBSCAN кластеризация (оригинальная)"""
    if not HDBSCAN_AVAILABLE:
        raise ImportError("HDBSCAN not available. Install with: pip install hdbscan")
    
    logger.info("Запуск HDBSCAN кластеризации")
    
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
    logger.info("Запуск кластеризации")
    
    algorithm = params.get('algorithm', 'gmm').lower()
    
    if algorithm == 'gmm':
        results = run_gmm_clustering(X, params)
        best_result = results['gmm_best']
    elif algorithm == 'gmm_auto':
        results = run_gmm_clustering_auto(X, params)
        best_result = results['gmm_auto_best']
    elif algorithm == 'kmeans':
        results = run_kmeans_clustering(X, params)
        best_result = results['kmeans_best']
    elif algorithm == 'hdbscan':
        results = run_hdbscan_clustering(X, params)
        best_result = results['hdbscan_best']
    else:
        # Тестируем все алгоритмы и выбираем лучший
        logger.info("Тестирование всех алгоритмов")
        
        all_results = {}
        best_overall = None
        best_overall_score = -1
        
        # GMM Auto (предпочтительный)
        try:
            gmm_auto_results = run_gmm_clustering_auto(X, params)
            all_results.update(gmm_auto_results)
            if gmm_auto_results['gmm_auto_best']['metrics']['combined_score'] > best_overall_score:
                best_overall_score = gmm_auto_results['gmm_auto_best']['metrics']['combined_score']
                best_overall = gmm_auto_results['gmm_auto_best']
        except Exception as e:
            logger.error(f"GMM Auto failed: {e}")
        
        # GMM (fallback)
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
        if HDBSCAN_AVAILABLE:
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
    logger.info(f"ФИНАЛЬНЫЙ РЕЗУЛЬТАТ:")
    logger.info(f"   Алгоритм: {best_result['params']['algorithm'].upper()}")
    logger.info(f"   Кластеры: {metrics['n_clusters']}")
    logger.info(f"   Размеры: {metrics.get('cluster_sizes', 'N/A')}")
    logger.info(f"   Баланс: {metrics['balance_ratio']:.3f}")
    logger.info(f"   Самый большой: {metrics['largest_cluster_pct']:.1f}%")
    logger.info(f"   Silhouette: {silhouette_str}")
    if 'bic' in metrics:
        logger.info(f"   BIC: {metrics['bic']:.1f}")
    if 'optimal_n_components' in metrics:
        logger.info(f"   Автоматически выбрано компонентов: {metrics['optimal_n_components']}")
    logger.info(f"   Время: {metrics['fit_time']:.1f}с")
    
    return best_result['labels'], best_result['clusterer'], results 