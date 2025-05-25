#!/usr/bin/env python3
"""
Data Processing Module
Feature Engineering + Preprocessing Pipeline
"""

import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import RobustScaler, StandardScaler, PowerTransformer
from sklearn.decomposition import PCA
from typing import Tuple, Dict, Any

from config import get_config

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Feature Engineering для customer segmentation"""
    
    def __init__(self):
        self.config = get_config()
        self.feature_config = self.config.FEATURE_CONFIG
        
    def extract_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Advanced Feature Engineering for Customer Segmentation
        
        Business Logic:
        - FREQUENCY: How often customers transact (loyalty indicator)
        - MONETARY: Transaction amounts (profitability indicator) 
        - RECENCY: Time patterns (engagement indicator)
        - BEHAVIORAL: Payment methods, timing (risk/preference indicator)
        - GEOGRAPHICAL: Location diversity (lifestyle indicator)
        """
        
        logger.info("🔧 FEATURE ENGINEERING: Creating business-driven features...")
        logger.info(f"📊 Available columns: {list(df.columns)}")
        
        # Создаем временные фичи
        df = self._create_time_features(df)
        
        # Создаем поведенческие индикаторы
        df = self._create_behavioral_indicators(df)
        
        # Агрегируем по клиентам
        features = self._aggregate_customer_features(df)
        
        # Создаем дополнительные фичи
        features = self._create_derived_features(features)
        
        # Очищаем данные
        features = features.fillna(0)
        
        logger.info(f"✅ Created {len(features.columns)-1} features with strong business rationale")
        logger.info("📈 Features categories:")
        logger.info("   • FREQUENCY: tx_count, tx_frequency, days_active")
        logger.info("   • MONETARY: avg_amount, total_amount, amount_volatility, CLV")
        logger.info("   • BEHAVIORAL: digital_wallet_ratio, contactless_ratio, time patterns")
        logger.info("   • GEOGRAPHICAL: city_diversity, country_diversity, mcc_diversity")
        logger.info("   • DERIVED: payment_sophistication, spending_consistency")
        
        return features
    
    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создание временных фичей"""
        time_config = self.feature_config['time_features']
        
        # Извлекаем компоненты времени
        df['hour'] = df['transaction_timestamp'].dt.hour
        df['day_of_week'] = df['transaction_timestamp'].dt.dayofweek
        df['month'] = df['transaction_timestamp'].dt.month
        
        # Создаем индикаторы
        df['is_weekend'] = df['day_of_week'].isin(time_config['weekend_days'])
        df['is_holiday_season'] = df['month'].isin(time_config['holiday_months'])
        
        return df
    
    def _create_behavioral_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создание поведенческих индикаторов"""
        behavioral = self.feature_config['behavioral_indicators']
        
        # Digital wallet adoption
        df['is_digital_wallet'] = df[behavioral['digital_wallet_column']].notnull()
        
        # International transactions
        df['is_international'] = df[behavioral['currency_column']] != behavioral['base_currency']
        
        # Contactless payments
        df['is_contactless'] = df['pos_entry_mode'] == behavioral['contactless_indicator']
        
        return df
    
    def _aggregate_customer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Агрегация фичей по клиентам"""
        features = df.groupby('card_id').agg({
            # FREQUENCY FEATURES
            'transaction_id': ['count'],
            
            # MONETARY FEATURES
            'transaction_amount_kzt': ['mean', 'std', 'median', 'sum', 'min', 'max'],
            
            # BEHAVIORAL FEATURES
            'is_digital_wallet': [lambda x: x.mean()],
            'is_international': [lambda x: x.mean()],
            'is_contactless': [lambda x: x.mean()],
            'transaction_type': [lambda x: x.nunique()],
            'pos_entry_mode': [lambda x: x.nunique()],
            
            # TIME PATTERNS
            'hour': [
                lambda x: ((x >= 0) & (x < 6)).mean(),      # Night
                lambda x: ((x >= 6) & (x < 12)).mean(),     # Morning
                lambda x: ((x >= 12) & (x < 18)).mean(),    # Afternoon
                lambda x: ((x >= 18) & (x < 24)).mean(),    # Evening
            ],
            'is_weekend': [lambda x: x.mean()],
            'is_holiday_season': [lambda x: x.mean()],
            
            # GEOGRAPHICAL FEATURES
            'merchant_city': ['nunique'],
            'merchant_mcc': ['nunique'],
            'acquirer_country_iso': ['nunique'],
            'issuer_bank_name': ['nunique'],
            
            # RECENCY FEATURES
            'transaction_timestamp': [
                lambda x: (x.max() - x.min()).days + 1,    # Days active
                lambda x: len(x) / ((x.max() - x.min()).days + 1),  # Frequency
            ]
        }).reset_index()
        
        # Flatten column names
        features.columns = [
            'card_id', 'tx_count', 'avg_amount', 'std_amount', 'median_amount', 
            'total_amount', 'min_amount', 'max_amount', 'digital_wallet_ratio', 
            'international_ratio', 'contactless_ratio', 'tx_type_variety', 'payment_method_variety',
            'night_ratio', 'morning_ratio', 'afternoon_ratio', 'evening_ratio',
            'weekend_ratio', 'holiday_ratio', 'city_diversity', 'mcc_diversity',
            'country_diversity', 'bank_diversity', 'days_active', 'tx_frequency'
        ]
        
        return features
    
    def _create_derived_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Создание производных фичей"""
        # Amount volatility
        features['amount_volatility'] = features['std_amount'] / features['avg_amount']
        
        # High value ratio
        features['high_value_ratio'] = (features['max_amount'] / features['avg_amount']).fillna(1)
        
        # Spending consistency
        features['spending_consistency'] = 1 / (1 + features['amount_volatility'].fillna(0))
        
        # Customer lifetime value
        features['customer_lifetime_value'] = features['total_amount'] * features['tx_frequency']
        
        # Average daily amount
        features['avg_daily_amount'] = features['total_amount'] / features['days_active']
        
        # Payment sophistication (composite score)
        features['payment_sophistication'] = (
            features['digital_wallet_ratio'] + 
            features['contactless_ratio'] + 
            features['payment_method_variety'] / 5
        ) / 3
        
        return features


class DataPreprocessor:
    """Preprocessing pipeline для кластеризации"""
    
    def __init__(self):
        self.config = get_config()
        self.preprocessing_config = self.config.PREPROCESSING
        self.scaler = None
        self.pca = None
        self.power_transformer = None
        
    def prepare_for_clustering(self, features_df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, Any]]:
        """
        Полный preprocessing pipeline:
        1. Удаление card_id
        2. RobustScaler
        3. Correlation removal
        4. PCA
        5. Power transformation
        """
        logger.info("🔬 PREPROCESSING PIPELINE: Preparing data for clustering...")
        
        # Удаляем card_id
        ml_features = features_df.drop(['card_id'], axis=1)
        logger.info(f"📊 Features shape: {ml_features.shape}")
        
        # Стандартизация
        ml_features_scaled = self._apply_scaling(ml_features)
        
        # Удаление коррелированных фичей
        ml_features_decorr = self._remove_correlations(ml_features_scaled, ml_features.columns)
        
        # PCA для снижения размерности
        ml_features_pca = self._apply_pca(ml_features_decorr)
        
        # Power transformation
        ml_features_final = self._apply_power_transform(ml_features_pca)
        
        # Сохраняем информацию о preprocessing
        preprocessing_info = {
            'original_shape': ml_features.shape,
            'scaled_shape': ml_features_scaled.shape,
            'decorr_shape': ml_features_decorr.shape,
            'pca_shape': ml_features_pca.shape,
            'final_shape': ml_features_final.shape,
            'pca_variance_explained': self.pca.explained_variance_ratio_.sum() if self.pca else None
        }
        
        logger.info("✅ Preprocessing completed")
        logger.info(f"   Original: {preprocessing_info['original_shape']}")
        logger.info(f"   Final: {preprocessing_info['final_shape']}")
        if preprocessing_info['pca_variance_explained']:
            logger.info(f"   PCA variance retained: {preprocessing_info['pca_variance_explained']:.1%}")
        
        return ml_features, ml_features_final, preprocessing_info
    
    def _apply_scaling(self, ml_features: pd.DataFrame) -> np.ndarray:
        """Применение scaling"""
        scaler_type = self.preprocessing_config['scaler_type']
        
        if scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()
        
        ml_features_scaled = self.scaler.fit_transform(ml_features)
        logger.info(f"✅ Applied {scaler_type} scaling")
        
        return ml_features_scaled
    
    def _remove_correlations(self, ml_features_scaled: np.ndarray, column_names: pd.Index) -> np.ndarray:
        """Удаление коррелированных фичей"""
        threshold = self.preprocessing_config['correlation_threshold']
        
        correlation_matrix = np.corrcoef(ml_features_scaled.T)
        to_remove = set()
        
        for i in range(len(correlation_matrix)):
            for j in range(i+1, len(correlation_matrix)):
                if abs(correlation_matrix[i][j]) > threshold:
                    to_remove.add(j)
        
        if to_remove:
            keep_indices = [i for i in range(ml_features_scaled.shape[1]) if i not in to_remove]
            ml_features_decorr = ml_features_scaled[:, keep_indices]
            logger.info(f"🎯 Removed {len(to_remove)} highly correlated features (threshold={threshold})")
        else:
            ml_features_decorr = ml_features_scaled
            logger.info("✅ No highly correlated features found")
        
        return ml_features_decorr
    
    def _apply_pca(self, ml_features_decorr: np.ndarray) -> np.ndarray:
        """Применение PCA"""
        variance_ratio = self.preprocessing_config['pca_variance_ratio']
        random_state = self.config.CLUSTERING_PARAMS['random_state']
        
        self.pca = PCA(n_components=variance_ratio, random_state=random_state)
        ml_features_pca = self.pca.fit_transform(ml_features_decorr)
        
        explained_variance = np.sum(self.pca.explained_variance_ratio_)
        logger.info(f"📊 PCA: {ml_features_pca.shape[1]} components, {explained_variance:.1%} variance retained")
        
        return ml_features_pca
    
    def _apply_power_transform(self, ml_features_pca: np.ndarray) -> np.ndarray:
        """Применение power transformation"""
        if not self.preprocessing_config['power_transform']:
            return ml_features_pca
        
        try:
            self.power_transformer = PowerTransformer(method='yeo-johnson', standardize=True)
            ml_features_transformed = self.power_transformer.fit_transform(ml_features_pca)
            logger.info("⚡ Applied Yeo-Johnson power transformation")
            return ml_features_transformed
        except Exception as e:
            logger.warning(f"⚠️ Power transformation failed: {e}, using PCA features")
            return ml_features_pca


def process_transaction_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, Any]]:
    """
    Главная функция для обработки транзакционных данных
    
    Args:
        df: DataFrame с транзакциями
    
    Returns:
        features_df: DataFrame с фичами
        ml_features_processed: Обработанные данные для ML
        processing_info: Информация о preprocessing
    """
    logger.info("🚀 Starting transaction data processing...")
    
    # Feature Engineering
    feature_engineer = FeatureEngineer()
    features_df = feature_engineer.extract_advanced_features(df)
    
    # Preprocessing
    preprocessor = DataPreprocessor()
    ml_features, ml_features_processed, processing_info = preprocessor.prepare_for_clustering(features_df)
    
    logger.info("✅ Transaction data processing completed")
    
    return features_df, ml_features_processed, processing_info 