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
        """Загрузка переменных окружения"""
        try:
            from dotenv import load_dotenv
            load_dotenv(dotenv_path='../.env')
            print("✅ Environment variables loaded from .env file")
        except ImportError:
            print("⚠️ python-dotenv not installed, using system environment variables only")
        except Exception as e:
            print(f"⚠️ Could not load .env file: {e}")
            
        # OpenAI Configuration
        self.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'your-api-key-here')
        if self.OPENAI_API_KEY == 'your-api-key-here':
            print("⚠️ OpenAI API key not found in environment variables")
            print("💡 Please add your API key to .env file: OPENAI_API_KEY=your-actual-key")
        else:
            print("✅ OpenAI API key loaded successfully")
    
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
        # Clustering Configuration
        self.CLUSTERING_PARAMS = {
            # Выбор алгоритма: 'hdbscan', 'gmm', 'kmeans'
            'algorithm': 'gmm',  # Используем GMM для сбалансированных кластеров
            
            # HDBSCAN параметры (оставляем для совместимости)
            'min_cluster_size_range': [5, 10, 15, 20, 30, 50],
            'min_samples_range': [3, 5, 10, 15],
            'cluster_selection_epsilon_range': [0.0, 0.1, 0.2],
            'cluster_selection_method': ['eom', 'leaf'],
            'metric': ['euclidean', 'manhattan'],
            'n_jobs': -1,
            'core_dist_n_jobs': -1,
            'random_state': None,  # HDBSCAN в нашей версии не поддерживает random_state
            
            # GMM параметры (новые)
            'gmm_n_components_range': [3, 4, 5, 6, 7],
            'gmm_covariance_type': ['full', 'tied', 'diag', 'spherical'],
            'gmm_init_params': 'kmeans',
            'gmm_max_iter': 100,
            'gmm_tol': 1e-3,
            'gmm_random_state': 42,
            
            # K-Means параметры (новые)
            'kmeans_n_clusters_range': [3, 4, 5, 6, 7, 8],
            'kmeans_init': 'k-means++',
            'kmeans_n_init': 10,
            'kmeans_max_iter': 300,
            'kmeans_random_state': 42,
        }
        
        # Preprocessing parameters
        self.PREPROCESSING = {
            'correlation_threshold': 0.8,
            'pca_variance_ratio': 0.95,
            'scaler_type': 'robust',  # or 'standard'
            'power_transform': True
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
        """Вывод краткой информации о конфигурации"""
        print("🔧 CONFIGURATION SUMMARY:")
        print(f"   📁 Data file: {self.DATA_FILE}")
        print(f"   🎯 Random state: {self.CLUSTERING_PARAMS['random_state']}")
        print(f"   🧬 Grid search combinations: {self.get_total_combinations()}")
        print(f"   ⚡ Parallel jobs: {self.CLUSTERING_PARAMS['n_jobs']}")
        print(f"   📊 PCA variance: {self.PREPROCESSING['pca_variance_ratio']}")


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