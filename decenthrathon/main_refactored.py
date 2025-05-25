#!/usr/bin/env python3
"""
Main Entry Point
CLI Interface + Orchestration
"""

import argparse
import sys
import time
import logging
import os
from pathlib import Path
import numpy as np

# Local imports
from config import get_config
from utils.helpers import (
    setup_logging, load_transaction_data, validate_dataframe_schema, 
    validate_features_dataframe, clean_features_dataframe, 
    set_random_state, print_dataframe_info, save_dataframe
)
from core.data_processing import process_transaction_data
from core.clustering import perform_clustering
from reporting.reports import generate_comprehensive_report

def setup_cli_parser() -> argparse.ArgumentParser:
    """Настройка CLI интерфейса"""
    parser = argparse.ArgumentParser(
        description="🏆 Ultra-Optimized Customer Segmentation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --data transactions.csv                    # Basic run
  %(prog)s --data transactions.csv --random-state 123 # Fixed seed  
  %(prog)s --data transactions.csv --log-level DEBUG  # Detailed logs
  %(prog)s --data transactions.csv --no-gpt4          # Skip GPT-4 insights
        """
    )
    
    # Data arguments
    parser.add_argument(
        '--data', 
        type=str, 
        default='transactions.csv',
        help='Path to transactions CSV file (default: transactions.csv)'
    )
    
    # Reproducibility
    parser.add_argument(
        '--random-state', 
        type=int, 
        default=42,
        help='Random state for reproducibility (default: 42)'
    )
    
    # Logging
    parser.add_argument(
        '--log-level', 
        type=str, 
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--log-file', 
        type=str,
        help='Log file path (optional)'
    )
    
    # Features
    parser.add_argument(
        '--no-gpt4', 
        action='store_true',
        help='Skip GPT-4 insights generation'
    )
    
    parser.add_argument(
        '--validate-only', 
        action='store_true',
        help='Only validate data and exit'
    )
    
    return parser

def validate_input_args(args) -> bool:
    """Валидация входных аргументов"""
    logger = logging.getLogger(__name__)
    
    # Проверяем существование файла данных
    if not Path(args.data).exists():
        logger.error(f"❌ Data file not found: {args.data}")
        return False
    
    # Проверяем random state
    if args.random_state < 0:
        logger.error(f"❌ Random state must be non-negative: {args.random_state}")
        return False
    
    logger.info("✅ Input arguments validated")
    return True

def print_pipeline_header():
    """Красивый заголовок pipeline"""
    print("=" * 80)
    print("🏆 ULTRA-OPTIMIZED CUSTOMER SEGMENTATION PIPELINE")
    print("=" * 80)
    print("📊 Advanced HDBSCAN with Business Intelligence")
    print("🔧 Feature Engineering + Preprocessing + Ultra-Optimization")
    print("🧠 GPT-4 Insights + Comprehensive Reporting")
    print("=" * 80)

def print_pipeline_summary(report: dict, execution_time: float):
    """Итоговая сводка pipeline"""
    logger = logging.getLogger(__name__)
    
    print("\n" + "=" * 80)
    print("🏆 PIPELINE EXECUTION SUMMARY")
    print("=" * 80)
    
    # Основные результаты
    metrics = report['quality_metrics']
    logger.info(f"🎯 Algorithm: {report['algorithm']}")
    logger.info(f"🎯 Clusters Found: {metrics['n_clusters']}")
    logger.info(f"🔇 Noise Ratio: {metrics['noise_ratio']:.1%}")
    logger.info(f"📈 Silhouette Score: {metrics.get('silhouette_score', 'N/A')}")
    logger.info(f"📊 Davies-Bouldin: {metrics.get('davies_bouldin', 'N/A')}")
    logger.info(f"⚡ Calinski-Harabasz: {metrics.get('calinski_harabasz', 'N/A')}")
    
    # Бизнес инсайты
    insights = report['insights']
    logger.info(f"🧠 Business Segments: {len(insights)}")
    
    # Файлы
    if report['export_success']:
        logger.info("📁 Generated Files:")
        for name, path in report['files_created'].items():
            logger.info(f"   📋 {path}")
    
    # Время выполнения
    logger.info(f"⏱️ Total Execution Time: {execution_time:.1f} seconds")
    
    print("=" * 80)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)

def main():
    """Главная функция pipeline"""
    start_time = time.time()
    
    # Парсинг аргументов
    parser = setup_cli_parser()
    args = parser.parse_args()
    
    # Настройка логирования
    setup_logging(level=args.log_level, log_file=args.log_file)
    logger = logging.getLogger(__name__)
    
    # Заголовок
    print_pipeline_header()
    
    try:
        # Валидация аргументов
        if not validate_input_args(args):
            sys.exit(1)
        
        # Фиксируем random state
        set_random_state(args.random_state)
        
        # Загружаем конфигурацию
        config = get_config()
        config.print_config_summary()
        
        # ========================================================================
        # STAGE 1: DATA LOADING & VALIDATION
        # ========================================================================
        logger.info("🚀 STAGE 1: DATA LOADING & VALIDATION")
        
        # Загружаем данные
        df = load_transaction_data(args.data)
        print_dataframe_info(df, "Transaction Data")
        
        # Валидируем схему
        expected_columns = config.FEATURE_CONFIG['expected_columns']
        if not validate_dataframe_schema(df, expected_columns):
            logger.error("❌ Data validation failed")
            sys.exit(1)
        
        # Если только валидация - выходим
        if args.validate_only:
            logger.info("✅ Data validation completed successfully")
            sys.exit(0)
        
        # ========================================================================
        # STAGE 2: FEATURE ENGINEERING & PREPROCESSING
        # ========================================================================
        logger.info("🚀 STAGE 2: FEATURE ENGINEERING & PREPROCESSING")
        
        # Обрабатываем транзакционные данные
        features_df, ml_features_processed, preprocessing_pipeline = process_transaction_data(df)
        
        # Валидируем и очищаем фичи
        if not validate_features_dataframe(features_df):
            logger.error("❌ Features validation failed")
            sys.exit(1)
        
        features_df = clean_features_dataframe(features_df)
        print_dataframe_info(features_df, "Customer Features")
        
        # ========================================================================
        # STAGE 3: ULTRA-OPTIMIZED CLUSTERING
        # ========================================================================
        logger.info("🚀 STAGE 3: ULTRA-OPTIMIZED CLUSTERING")
        
        # Выполняем кластеризацию
        labels, clusterer, clustering_results = perform_clustering(
            ml_features_processed, 
            config.CLUSTERING_PARAMS
        )
        
        # Добавляем метки кластеров к признакам
        features_df['segment'] = labels
        
        # ========================================================================
        # STAGE 4: COMPREHENSIVE REPORTING
        # ========================================================================
        logger.info("🚀 STAGE 4: COMPREHENSIVE REPORTING")
        
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
        
        # ========================================================================
        # STAGE 5: PIPELINE SUMMARY
        # ========================================================================
        execution_time = time.time() - start_time
        print_pipeline_summary(final_report, execution_time)
        
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
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("\n⚠️ Pipeline interrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        logger.debug("Exception details:", exc_info=True)
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 