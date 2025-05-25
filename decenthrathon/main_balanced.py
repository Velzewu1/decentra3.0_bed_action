#!/usr/bin/env python3
"""
Главный файл для запуска улучшенного пайплайна кластеризации
"""

import time
import os
from core.data_processing import process_transaction_data
from core.clustering import perform_clustering
from utils.helpers import load_transaction_data, setup_logging, save_dataframe
from reporting.reports import generate_comprehensive_report

def main():
    """Основная функция пайплайна"""
    print("🚀 REFACTORED CUSTOMER SEGMENTATION PIPELINE")
    print("="*60)
    
    # Настройка логгирования
    logger = setup_logging('INFO')
    logger.info("🎯 Starting balanced clustering pipeline...")
    
    # Начинаем измерение времени
    start_time = time.time()
    
    # 1. Загрузка данных
    print("\n📂 ЭТАП 1: Загрузка данных")
    print("-" * 40)
    df = load_transaction_data('transactions.csv')
    print(f"✅ Загружено транзакций: {len(df):,}")
    
    # 2. Обработка данных и создание признаков
    print("\n🔧 ЭТАП 2: Обработка данных")
    print("-" * 40)
    features_df, ml_features_processed, preprocessing_pipeline = process_transaction_data(df)
    print(f"✅ Создано признаков: {ml_features_processed.shape[1]}")
    print(f"✅ Клиентов для кластеризации: {len(features_df):,}")
    
    # 3. Кластеризация с улучшенными алгоритмами
    print("\n🎯 ЭТАП 3: Сбалансированная кластеризация")
    print("-" * 40)
    
    from config import get_config
    config = get_config()
    
    labels, clusterer, clustering_results = perform_clustering(
        ml_features_processed, 
        config.CLUSTERING_PARAMS
    )
    
    # Добавляем метки кластеров к признакам
    features_df['segment'] = labels
    
    # 4. Генерация отчетов
    print("\n📊 ЭТАП 4: Генерация отчетов")
    print("-" * 40)
    
    # Формируем results в нужном формате для отчетов
    algorithm_name = clustering_results.get('gmm_best', clustering_results.get('kmeans_best', clustering_results.get('hdbscan_best', {})))
    algorithm_key = f"{algorithm_name.get('params', {}).get('algorithm', 'unknown')}_balanced"
    
    formatted_results = {
        algorithm_key: {
            'labels': labels,
            'clusterer': clusterer,
            'params': algorithm_name.get('params', {}),
            'metrics': algorithm_name.get('metrics', {}),
            'algorithm_name': f"Balanced {algorithm_name.get('params', {}).get('algorithm', 'Unknown').upper()}"
        }
    }
    
    final_report = generate_comprehensive_report(
        features_df, labels, ml_features_processed, formatted_results
    )
    
    # 5. Сохранение результатов
    print("\n💾 ЭТАП 5: Сохранение результатов")
    print("-" * 40)
    
    # Сохраняем основные файлы
    output_files = [
        'customer_segments_refactored.csv',
        'customer_segments_refactored.parquet'
    ]
    
    for filename in output_files:
        save_dataframe(features_df, filename)
        print(f"✅ Сохранено: {filename}")
    
    # Показываем время выполнения
    total_time = time.time() - start_time
    print(f"\n⏱️  ОБЩЕЕ ВРЕМЯ ВЫПОЛНЕНИЯ: {total_time:.1f} секунд")
    
    # Финальная сводка
    print(f"\n🎉 ПАЙПЛАЙН ЗАВЕРШЕН УСПЕШНО!")
    print(f"📊 Кластеров создано: {len(set(labels))}")
    print(f"👥 Клиентов сегментировано: {len(features_df):,}")
    print(f"📁 Файлы сохранены в: {os.getcwd()}")
    
    # Показываем сбалансированность
    import numpy as np
    unique_labels = np.unique(labels)
    cluster_sizes = [np.sum(labels == label) for label in unique_labels]
    max_size = max(cluster_sizes)
    min_size = min(cluster_sizes)
    balance_ratio = min_size / max_size
    largest_pct = max_size / len(labels) * 100
    
    print(f"\n📈 КАЧЕСТВО СЕГМЕНТАЦИИ:")
    print(f"   Размеры кластеров: {sorted(cluster_sizes, reverse=True)}")
    print(f"   Баланс (мин/макс): {balance_ratio:.3f}")
    print(f"   Самый большой кластер: {largest_pct:.1f}%")
    
    if largest_pct < 50:
        print(f"   ✅ Отличная сбалансированность!")
    elif largest_pct < 70:
        print(f"   ⚠️  Приемлемая сбалансированность")
    else:
        print(f"   ❌ Требует улучшения сбалансированности")

if __name__ == "__main__":
    main() 