#!/usr/bin/env python3
"""
Тестирование воспроизводимости результатов кластеризации
"""

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from utils.helpers import set_random_state
from config import get_config
import time

def test_reproducibility_with_seed(seed=42, n_runs=3):
    """
    Тестируем воспроизводимость с фиксированным seed
    
    Args:
        seed: Random seed для тестирования
        n_runs: Количество запусков для проверки
    """
    print(f"ТЕСТИРОВАНИЕ С SEED={seed}")
    print("="*60)
    
    # Создаем тестовые данные
    np.random.seed(42)  # Фиксированный seed для генерации тестовых данных
    X_test = np.random.randn(1000, 10)
    
    results = []
    
    for run in range(n_runs):
        print(f"\nЗапуск {run + 1}/{n_runs}")
        
        # Устанавливаем seed перед каждым запуском
        set_random_state(seed)
        
        # Получаем конфигурацию
        config = get_config()
        config.update_random_state(seed)
        
        # PCA
        pca = PCA(n_components=5, random_state=config.RANDOM_STATE)
        X_pca = pca.fit_transform(X_test)
        
        # GMM кластеризация
        gmm = GaussianMixture(
            n_components=3,
            covariance_type='full',
            random_state=config.CLUSTERING_PARAMS['gmm_random_state']
        )
        labels = gmm.fit_predict(X_pca)
        
        # Сохраняем результаты
        result = {
            'run': run + 1,
            'pca_explained_variance': pca.explained_variance_ratio_.sum(),
            'first_10_labels': labels[:10].tolist(),
            'gmm_bic': gmm.bic(X_pca),
            'gmm_aic': gmm.aic(X_pca),
            'unique_labels': np.unique(labels).tolist(),
            'label_counts': np.bincount(labels).tolist()
        }
        
        results.append(result)
        
        print(f"   PCA variance: {result['pca_explained_variance']:.6f}")
        print(f"   Первые 10 меток: {result['first_10_labels']}")
        print(f"   GMM BIC: {result['gmm_bic']:.3f}")
    
    # Проверяем идентичность результатов
    print(f"\nПРОВЕРКА РЕЗУЛЬТАТОВ:")
    print("-" * 40)
    
    first_result = results[0]
    all_identical = True
    
    for i, result in enumerate(results[1:], 2):
        identical = True
        
        # Проверяем PCA
        if abs(result['pca_explained_variance'] - first_result['pca_explained_variance']) > 1e-10:
            print(f"PCA variance отличается в запуске {i}")
            identical = False
        
        # Проверяем первые 10 меток
        if result['first_10_labels'] != first_result['first_10_labels']:
            print(f"Метки отличаются в запуске {i}")
            identical = False
        
        # Проверяем BIC
        if abs(result['gmm_bic'] - first_result['gmm_bic']) > 1e-10:
            print(f"GMM BIC отличается в запуске {i}")
            identical = False
        
        # Проверяем количество меток
        if result['label_counts'] != first_result['label_counts']:
            print(f"Количество меток отличается в запуске {i}")
            identical = False
        
        if identical:
            print(f"Запуск {i} идентичен запуску 1")
        else:
            all_identical = False
    
    if all_identical:
        print(f"\nВОСПРОИЗВОДИМОСТЬ ОК!")
        print(f"   Все {n_runs} запусков дали одинаковые результаты с seed={seed}")
    else:
        print(f"\nЕСТЬ ПРОБЛЕМЫ!")
        print("   Некоторые запуски дали разные результаты")
    
    return all_identical, results

def test_different_seeds():
    """Тестируем с разными seed значениями"""
    print(f"\nТЕСТИРОВАНИЕ РАЗНЫХ SEED")
    print("="*60)
    
    seeds = [42, 123, 999]
    seed_results = {}
    
    for seed in seeds:
        print(f"\nТестируем seed={seed}")
        reproducible, results = test_reproducibility_with_seed(seed, n_runs=2)
        seed_results[seed] = {
            'reproducible': reproducible,
            'first_result': results[0]
        }
    
    # Проверяем, что разные seeds дают разные результаты
    print(f"\nПРОВЕРКА РАЗЛИЧИЙ:")
    print("-" * 40)
    
    seed_42_labels = seed_results[42]['first_result']['first_10_labels']
    
    for seed in seeds[1:]:
        other_labels = seed_results[seed]['first_result']['first_10_labels']
        if seed_42_labels != other_labels:
            print(f"Seed {seed} дает другие результаты чем seed=42")
        else:
            print(f"Seed {seed} дает такие же результаты как seed=42")
    
    return seed_results

if __name__ == "__main__":
    print("ТЕСТИРОВАНИЕ ВОСПРОИЗВОДИМОСТИ")
    print("="*60)
    
    start_time = time.time()
    
    # Тест 1: Воспроизводимость с фиксированным seed
    reproducible_42, _ = test_reproducibility_with_seed(42, n_runs=3)
    
    # Тест 2: Разные seed дают разные результаты
    seed_results = test_different_seeds()
    
    # Итоговый отчет
    total_time = time.time() - start_time
    
    print(f"\nИТОГ:")
    print("="*60)
    print(f"Время: {total_time:.2f} секунд")
    print(f"Воспроизводимость с seed=42: {'ОК' if reproducible_42 else 'ОШИБКА'}")
    
    all_seeds_reproducible = all(result['reproducible'] for result in seed_results.values())
    print(f"Воспроизводимость всех seeds: {'ОК' if all_seeds_reproducible else 'ОШИБКА'}")
    
    if reproducible_42 and all_seeds_reproducible:
        print(f"\nВСЕ ТЕСТЫ ПРОЙДЕНЫ!")
        print("   Система воспроизводимости работает правильно")
    else:
        print(f"\nНУЖНО РАЗБИРАТЬСЯ!")
        print("   Требуется дополнительная настройка") 