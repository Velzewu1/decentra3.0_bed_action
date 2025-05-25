#!/usr/bin/env python3
"""
Helper Utilities Module
Загрузка данных, логирование, валидация
"""

import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
import warnings

def setup_logging(level: str = 'INFO', log_file: Optional[str] = None) -> logging.Logger:
    """
    Настройка централизованного логирования
    
    Args:
        level: Уровень логирования (DEBUG, INFO, WARNING, ERROR)
        log_file: Путь к файлу лога (опционально)
    
    Returns:
        Настроенный logger
    """
    # Создаем formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Настраиваем root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Удаляем существующие handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (если указан файл)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Подавляем лишние warnings
    warnings.filterwarnings('ignore')
    
    return root_logger

def load_transaction_data(file_path: str, parse_dates: bool = True) -> pd.DataFrame:
    """
    Загрузка транзакционных данных с валидацией (поддержка CSV и Parquet)
    
    Args:
        file_path: Путь к файлу с данными
        parse_dates: Парсить ли даты автоматически
    
    Returns:
        DataFrame с транзакциями
    """
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"📁 Loading transaction data from {file_path}...")
        
        # Проверяем существование файла
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        # Определяем формат файла по расширению
        file_extension = Path(file_path).suffix.lower()
        
        # Загружаем данные в зависимости от формата
        if file_extension == '.parquet':
            logger.info("📦 Loading Parquet file...")
            df = pd.read_parquet(file_path)
            # Для parquet файлов даты обычно уже корректно типизированы
            if parse_dates and 'transaction_timestamp' in df.columns:
                if not pd.api.types.is_datetime64_any_dtype(df['transaction_timestamp']):
                    df['transaction_timestamp'] = pd.to_datetime(df['transaction_timestamp'])
        
        elif file_extension == '.csv':
            logger.info("📄 Loading CSV file...")
            if parse_dates:
                df = pd.read_csv(file_path, parse_dates=['transaction_timestamp'])
            else:
                df = pd.read_csv(file_path)
        
        else:
            raise ValueError(f"Unsupported file format: {file_extension}. Use .csv or .parquet")
        
        logger.info(f"✅ Loaded {len(df):,} transactions for {df['card_id'].nunique():,} customers")
        
        return df
        
    except Exception as e:
        logger.error(f"❌ Error loading data: {e}")
        raise

def validate_dataframe_schema(df: pd.DataFrame, expected_columns: List[str]) -> bool:
    """
    Валидация схемы DataFrame
    
    Args:
        df: DataFrame для проверки
        expected_columns: Ожидаемые колонки
    
    Returns:
        True если схема корректна
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"🔍 Validating dataframe schema...")
    logger.info(f"📊 Dataset columns ({len(df.columns)}): {list(df.columns)}")
    
    # Проверяем наличие ключевых колонок
    missing_columns = set(expected_columns) - set(df.columns)
    if missing_columns:
        logger.warning(f"⚠️ Missing columns: {missing_columns}")
        return False
    
    # Проверяем типы данных
    if 'transaction_timestamp' in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df['transaction_timestamp']):
            logger.warning("⚠️ transaction_timestamp is not datetime type")
            return False
    
    if 'card_id' in df.columns:
        if df['card_id'].isnull().any():
            logger.warning("⚠️ Found null values in card_id")
            return False
    
    if 'transaction_amount_kzt' in df.columns:
        if not pd.api.types.is_numeric_dtype(df['transaction_amount_kzt']):
            logger.warning("⚠️ transaction_amount_kzt is not numeric")
            return False
    
    logger.info(f"✅ Schema validation passed - {len(expected_columns)} expected fields found")
    return True

def validate_features_dataframe(features_df: pd.DataFrame) -> bool:
    """
    Валидация DataFrame с фичами
    
    Args:
        features_df: DataFrame с feature engineering результатами
    
    Returns:
        True если валидация прошла успешно
    """
    logger = logging.getLogger(__name__)
    
    logger.info("🔍 Validating features dataframe...")
    
    # Проверяем обязательные колонки
    required_cols = ['card_id']
    missing_required = set(required_cols) - set(features_df.columns)
    if missing_required:
        logger.error(f"❌ Missing required columns: {missing_required}")
        return False
    
    # Проверяем на NaN значения
    nan_cols = features_df.columns[features_df.isnull().any()].tolist()
    if nan_cols:
        logger.warning(f"⚠️ Found NaN values in columns: {nan_cols}")
        logger.info("💡 NaN values will be filled with 0")
    
    # Проверяем на infinite значения
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    inf_cols = []
    for col in numeric_cols:
        if np.isinf(features_df[col]).any():
            inf_cols.append(col)
    
    if inf_cols:
        logger.warning(f"⚠️ Found infinite values in columns: {inf_cols}")
        return False
    
    # Проверяем размерность
    if len(features_df) == 0:
        logger.error("❌ Features dataframe is empty")
        return False
    
    logger.info(f"✅ Features validation passed - {len(features_df)} customers, {len(features_df.columns)-1} features")
    return True

def clean_features_dataframe(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Очистка DataFrame с фичами
    
    Args:
        features_df: Исходный DataFrame
    
    Returns:
        Очищенный DataFrame
    """
    logger = logging.getLogger(__name__)
    
    logger.info("🧹 Cleaning features dataframe...")
    
    # Создаем копию
    cleaned_df = features_df.copy()
    
    # Заполняем NaN значения нулями
    nan_count_before = cleaned_df.isnull().sum().sum()
    cleaned_df = cleaned_df.fillna(0)
    
    if nan_count_before > 0:
        logger.info(f"✅ Filled {nan_count_before} NaN values with 0")
    
    # Заменяем infinite значения
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    inf_count = 0
    
    for col in numeric_cols:
        inf_mask = np.isinf(cleaned_df[col])
        if inf_mask.any():
            inf_count += inf_mask.sum()
            # Заменяем +inf на максимальное конечное значение
            max_finite = cleaned_df[col][~inf_mask].max()
            cleaned_df.loc[inf_mask & (cleaned_df[col] > 0), col] = max_finite
            # Заменяем -inf на минимальное конечное значение
            min_finite = cleaned_df[col][~inf_mask].min()
            cleaned_df.loc[inf_mask & (cleaned_df[col] < 0), col] = min_finite
    
    if inf_count > 0:
        logger.info(f"✅ Replaced {inf_count} infinite values")
    
    logger.info(f"✅ Features cleaned - shape: {cleaned_df.shape}")
    return cleaned_df

def save_dataframe(df: pd.DataFrame, file_path: str, format: str = 'parquet') -> bool:
    """
    Сохранение DataFrame в файл
    
    Args:
        df: DataFrame для сохранения
        file_path: Путь к файлу
        format: Формат файла ('parquet', 'csv', 'json')
    
    Returns:
        True если сохранение успешно
    """
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"💾 Saving dataframe to {file_path} ({format} format)...")
        
        if format.lower() == 'parquet':
            df.to_parquet(file_path, index=False)
        elif format.lower() == 'csv':
            df.to_csv(file_path, index=False)
        elif format.lower() == 'json':
            df.to_json(file_path, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        file_size = Path(file_path).stat().st_size / (1024 * 1024)  # MB
        logger.info(f"✅ Saved successfully - {file_size:.2f} MB")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error saving dataframe: {e}")
        return False

def save_json(data: Dict[str, Any], file_path: str) -> bool:
    """
    Сохранение данных в JSON файл
    
    Args:
        data: Данные для сохранения
        file_path: Путь к файлу
    
    Returns:
        True если сохранение успешно
    """
    logger = logging.getLogger(__name__)
    
    try:
        import json
        
        logger.info(f"💾 Saving JSON to {file_path}...")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"✅ JSON saved successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error saving JSON: {e}")
        return False

def print_dataframe_info(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """
    Вывод краткой информации о DataFrame
    
    Args:
        df: DataFrame для анализа
        name: Название DataFrame
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"📊 {name} Info:")
    logger.info(f"   Shape: {df.shape}")
    logger.info(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    
    # Информация о типах данных
    dtype_counts = df.dtypes.value_counts()
    logger.info(f"   Data types: {dict(dtype_counts)}")
    
    # Информация о missing values
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        logger.info(f"   Missing values: {missing_count}")
    
    # Дубликаты
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        logger.info(f"   Duplicates: {duplicates}")

def set_random_state(seed: int = 42) -> None:
    """
    Фиксация random state для воспроизводимости
    
    Args:
        seed: Значение seed
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"🎲 Setting random state to {seed} for reproducibility...")
    
    import random
    import numpy as np
    
    # Python random
    random.seed(seed)
    
    # NumPy random
    np.random.seed(seed)
    
    # Для scikit-learn будем передавать random_state в параметрах
    logger.info("✅ Random state fixed for reproducibility")

# Инициализируем логирование при импорте модуля
setup_logging() 