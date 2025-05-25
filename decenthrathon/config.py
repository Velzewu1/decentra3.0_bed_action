#!/usr/bin/env python3
"""
Configuration Module
Централизованная конфигурация для всего проекта
"""

import os
import warnings
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')

class Config:
    """Централизованная конфигурация проекта"""
    
    def __init__(self):
        self.load_environment()
        self.setup_paths()
        self.setup_clustering_params()
        self.setup_feature_params()
        
    def load_environment(self):
        """Загружаем переменные окружения"""
        try:
            from dotenv import load_dotenv
            load_dotenv(dotenv_path='../.env')
            print("Environment variables загружены из .env файла")
        except ImportError:
            print("python-dotenv не установлен, используем системные переменные")
        except Exception as e:
            print(f"Не удалось загрузить .env файл: {e}")
            
        # OpenAI Configuration
        self.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'your-api-key-here')
        if self.OPENAI_API_KEY == 'your-api-key-here':
            print("OpenAI API key не найден в переменных окружения")
            print("Добавьте ваш API key в .env файл: OPENAI_API_KEY=your-actual-key")
        else:
            print("OpenAI API key загружен успешно")
    
    def setup_paths(self):
        """Настройка путей к файлам"""
        self.DATA_FILE = "transactions.csv"
        self.OUTPUT_DIR = Path("output")
        self.OUTPUT_DIR.mkdir(exist_ok=True)
        
        # Output files
        self.SEGMENTS_FILE = "customer_segments.parquet"
        self.SEGMENTS_CSV = "customer_segments.csv"
        self.SUMMARY_FILE = "segment_summary.parquet"
        self.RESULTS_JSON = "hackathon_segmentation_results.json"
        self.INSIGHTS_JSON = "business_insights.json"
        
    def setup_clustering_params(self):
        """Параметры для кластеризации"""
        # Global random state for reproducibility
        self.RANDOM_STATE = 42
        
        # Clustering Configuration
        self.CLUSTERING_PARAMS = {
            # Выбор алгоритма: 'hdbscan', 'gmm', 'gmm_auto', 'kmeans'
            'algorithm': 'gmm_auto',  # Автоматическое определение количества компонентов через BIC/AIC
            
            # Global random state
            'random_state': self.RANDOM_STATE,
            
            # HDBSCAN параметры (оставляем для совместимости)
            'min_cluster_size_range': [5, 10, 15, 20, 30, 50],
            'min_samples_range': [3, 5, 10, 15],
            'cluster_selection_epsilon_range': [0.0, 0.1, 0.2],
            'cluster_selection_method': ['eom', 'leaf'],
            'metric': ['euclidean', 'manhattan'],
            'n_jobs': -1,
            'core_dist_n_jobs': -1,
            # Примечание: HDBSCAN в некоторых версиях не поддерживает random_state
            
            # GMM параметры (ручной режим)
            'gmm_n_components_range': [3, 4, 5, 6, 7],
            'gmm_covariance_type': ['full', 'tied', 'diag', 'spherical'],
            'gmm_init_params': 'kmeans',
            'gmm_max_iter': 100,
            'gmm_tol': 1e-3,
            'gmm_random_state': self.RANDOM_STATE,
            
            # GMM Auto параметры (автоматическое определение)
            'gmm_auto_min_components': 2,  # Минимальное количество компонентов для тестирования
            'gmm_auto_max_components': 8,  # Максимальное количество компонентов для тестирования  
            'gmm_auto_criterion': 'bic',  # 'bic', 'aic', 'combined' - меняем на BIC для получения большего количества кластеров
            # combined = 50% BIC + 30% balance + 20% silhouette
            
            # K-Means параметры (новые)
            'kmeans_n_clusters_range': [3, 4, 5, 6, 7, 8],
            'kmeans_init': 'k-means++',
            'kmeans_n_init': 10,
            'kmeans_max_iter': 300,
            'kmeans_random_state': self.RANDOM_STATE,
        }
        
        # Preprocessing parameters
        self.PREPROCESSING = {
            'correlation_threshold': 0.8,
            'pca_variance_ratio': 0.95,
            'scaler_type': 'robust',  # or 'standard'
            'power_transform': True,
            'random_state': self.RANDOM_STATE  # Добавляем для PCA и других алгоритмов
        }
        
        # Grid search optimization
        self.OPTIMIZATION = {
            'max_combinations': 100,  # Ограничиваем для скорости
            'early_stopping': True,
            'early_stopping_patience': 20,
            'scoring_weights': {
                'silhouette': 1.0,
                'noise_penalty': -1.5,
                'cluster_bonus': 0.5,
                'davies_bouldin_penalty': -0.3
            }
        }
        
    def update_random_state(self, new_seed: int):
        """
        Обновление random state для всех компонентов
        
        Args:
            new_seed: Новое значение seed
        """
        self.RANDOM_STATE = new_seed
        
        # Обновляем все random_state параметры
        self.CLUSTERING_PARAMS['random_state'] = new_seed
        self.CLUSTERING_PARAMS['gmm_random_state'] = new_seed
        self.CLUSTERING_PARAMS['kmeans_random_state'] = new_seed
        self.PREPROCESSING['random_state'] = new_seed
        
        print(f"Configuration updated: RANDOM_STATE = {new_seed}")
        
    def setup_feature_params(self):
        """Параметры для feature engineering"""
        self.FEATURE_CONFIG = {
            'expected_columns': [
                'transaction_id', 'transaction_timestamp', 'card_id', 'expiry_date', 
                'issuer_bank_name', 'merchant_id', 'merchant_mcc', 'merchant_city', 
                'transaction_type', 'transaction_amount_kzt', 'original_amount', 
                'transaction_currency', 'acquirer_country_iso', 'pos_entry_mode', 'wallet_type'
            ],
            'time_features': {
                'weekend_days': [5, 6],
                'holiday_months': [11, 12, 1],
                'time_periods': {
                    'night': (0, 6),
                    'morning': (6, 12),
                    'afternoon': (12, 18),
                    'evening': (18, 24)
                }
            },
            'behavioral_indicators': {
                'digital_wallet_column': 'wallet_type',
                'currency_column': 'transaction_currency',
                'base_currency': 'KZT',
                'contactless_indicator': 'Contactless'
            }
        }
        
    def get_total_combinations(self):
        """Подсчет общего количества комбинаций для grid search"""
        params = self.CLUSTERING_PARAMS
        total = (len(params['min_cluster_size_range']) * 
                len(params['min_samples_range']) * 
                len(params['cluster_selection_epsilon_range']) * 
                len(params['cluster_selection_method']) * 
                len(params['metric']))
        return min(total, self.OPTIMIZATION['max_combinations'])
    
    def print_config_summary(self):
        """Выводим краткую инфу о конфигурации"""
        print("CONFIGURATION SUMMARY:")
        print(f"   Файл данных: {self.DATA_FILE}")
        print(f"   Random state: {self.RANDOM_STATE}")
        print(f"   Grid search комбинации: {self.get_total_combinations()}")
        print(f"   Параллельные задачи: {self.CLUSTERING_PARAMS['n_jobs']}")
        print(f"   PCA variance: {self.PREPROCESSING['pca_variance_ratio']}")


# Global configuration instance
config = Config()

# Utility functions for easy access
def get_config():
    """Получить глобальную конфигурацию"""
    return config

def get_clustering_params():
    """Получить параметры кластеризации"""
    return config.CLUSTERING_PARAMS

def get_preprocessing_params():
    """Получить параметры preprocessing"""
    return config.PREPROCESSING

def get_paths():
    """Получить пути к файлам"""
    return {
        'data_file': config.DATA_FILE,
        'segments_file': config.SEGMENTS_FILE,
        'segments_csv': config.SEGMENTS_CSV,
        'summary_file': config.SUMMARY_FILE,
        'results_json': config.RESULTS_JSON,
        'insights_json': config.INSIGHTS_JSON
    } 