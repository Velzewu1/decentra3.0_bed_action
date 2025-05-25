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
    Настройка логирования
    
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
    Загрузка транзакционных данных с автоопределением формата
    
    Args:
        file_path: Путь к файлу с данными
        parse_dates: Парсить ли даты автоматически
    
    Returns:
        DataFrame с транзакциями
    """
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Читаем данные из {file_path}")
        
        # Проверяем существование файла
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Файл не найден: {file_path}")
        
        # Определяем формат файла по расширению
        file_extension = Path(file_path).suffix.lower()
        
        # Загружаем в зависимости от формата
        if file_extension == '.parquet':
            logger.info("Читаем Parquet файл")
            df = pd.read_parquet(file_path)
            # Для parquet файлов даты обычно уже корректно типизированы
            if parse_dates and 'transaction_timestamp' in df.columns:
                if not pd.api.types.is_datetime64_any_dtype(df['transaction_timestamp']):
                    df['transaction_timestamp'] = pd.to_datetime(df['transaction_timestamp'])
        
        elif file_extension == '.csv':
            logger.info("Читаем CSV файл")
            if parse_dates:
                df = pd.read_csv(file_path, parse_dates=['transaction_timestamp'])
            else:
                df = pd.read_csv(file_path)
        
        else:
            raise ValueError(f"Не поддерживаемый формат: {file_extension}. Используйте .csv или .parquet")
        
        logger.info(f"Загружено {len(df):,} транзакций для {df['card_id'].nunique():,} клиентов")
        
        return df
        
    except Exception as e:
        logger.error(f"Ошибка загрузки данных: {e}")
        raise

def validate_dataframe_schema(df: pd.DataFrame, expected_columns: List[str]) -> bool:
    """
    Проверяем схему DataFrame
    
    Args:
        df: DataFrame для проверки
        expected_columns: Ожидаемые колонки
    
    Returns:
        True если схема корректна
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"Проверяем данные")
    logger.info(f"В dataset {len(df.columns)} колонок: {list(df.columns)}")
    
    # Проверяем наличие ключевых колонок
    missing_columns = set(expected_columns) - set(df.columns)
    if missing_columns:
        logger.warning(f"Отсутствуют колонки: {missing_columns}")
        return False
    
    # Проверяем типы данных
    if 'transaction_timestamp' in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df['transaction_timestamp']):
            logger.warning("transaction_timestamp не является datetime типом")
            return False
    
    if 'card_id' in df.columns:
        if df['card_id'].isnull().any():
            logger.warning("Найдены пустые значения в card_id")
            return False
    
    if 'transaction_amount_kzt' in df.columns:
        if not pd.api.types.is_numeric_dtype(df['transaction_amount_kzt']):
            logger.warning("transaction_amount_kzt не является числовым типом")
            return False
    
    logger.info(f"Проверка пройдена - найдено {len(expected_columns)} нужных полей")
    return True

def validate_features_dataframe(features_df: pd.DataFrame) -> bool:
    """
    Проверяем DataFrame с фичами
    
    Args:
        features_df: DataFrame с результатами feature engineering
    
    Returns:
        True если валидация прошла успешно
    """
    logger = logging.getLogger(__name__)
    
    logger.info("Проверяем сгенерированные фичи")
    
    # Проверяем обязательные колонки
    required_cols = ['card_id']
    missing_required = set(required_cols) - set(features_df.columns)
    if missing_required:
        logger.error(f"Отсутствуют обязательные колонки: {missing_required}")
        return False
    
    # Проверяем на NaN значения
    nan_cols = features_df.columns[features_df.isnull().any()].tolist()
    if nan_cols:
        logger.warning(f"Найдены NaN значения в колонках: {nan_cols}")
        logger.info("NaN будут заполнены нулями")
    
    # Проверяем на infinite значения
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    inf_cols = []
    for col in numeric_cols:
        if np.isinf(features_df[col]).any():
            inf_cols.append(col)
    
    if inf_cols:
        logger.warning(f"Найдены бесконечные значения в колонках: {inf_cols}")
        return False
    
    # Проверяем размерность
    if len(features_df) == 0:
        logger.error("DataFrame с фичами пуст")
        return False
    
    logger.info(f"Валидация ОК - {len(features_df)} клиентов, {len(features_df.columns)-1} фичей")
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
    
    logger.info("Чистим данные")
    
    # Создаем копию
    cleaned_df = features_df.copy()
    
    # Заполняем NaN значения нулями
    nan_count_before = cleaned_df.isnull().sum().sum()
    cleaned_df = cleaned_df.fillna(0)
    
    if nan_count_before > 0:
        logger.info(f"Заполнили {nan_count_before} NaN значений нулями")
    
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
        logger.info(f"Заменили {inf_count} бесконечных значений")
    
    logger.info(f"Готово - размерность: {cleaned_df.shape}")
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
        logger.info(f"Сохраняем в {file_path} ({format})")
        
        if format.lower() == 'parquet':
            df.to_parquet(file_path, index=False)
        elif format.lower() == 'csv':
            df.to_csv(file_path, index=False)
        elif format.lower() == 'json':
            df.to_json(file_path, indent=2)
        else:
            raise ValueError(f"Не поддерживаемый формат: {format}")
        
        file_size = Path(file_path).stat().st_size / (1024 * 1024)  # MB
        logger.info(f"Сохранено - {file_size:.2f} MB")
        
        return True
        
    except Exception as e:
        logger.error(f"Ошибка сохранения: {e}")
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
        
        logger.info(f"Сохраняем JSON в {file_path}")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"JSON готов")
        return True
        
    except Exception as e:
        logger.error(f"Ошибка сохранения JSON: {e}")
        return False

def print_dataframe_info(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """
    Выводим краткую информацию о DataFrame
    
    Args:
        df: DataFrame для анализа
        name: Название DataFrame
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"Инфо по {name}:")
    logger.info(f"   Размерность: {df.shape}")
    logger.info(f"   Память: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    
    # Информация о типах данных
    dtype_counts = df.dtypes.value_counts()
    logger.info(f"   Типы данных: {dict(dtype_counts)}")
    
    # Информация о missing values
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        logger.info(f"   Пропуски: {missing_count}")
    
    # Дубликаты
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        logger.info(f"   Дубликаты: {duplicates}")

def set_random_state(seed: int = 42) -> None:
    """
    Фиксируем random state для воспроизводимости результатов
    
    Args:
        seed: Значение seed для воспроизводимости
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"Устанавливаем seed = {seed}")
    
    import random
    import numpy as np
    import os
    
    # Python random
    random.seed(seed)
    
    # NumPy random
    np.random.seed(seed)
    
    # TensorFlow (если используется)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        logger.info("   TensorFlow готов")
    except ImportError:
        pass
    
    # PyTorch (если используется)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        logger.info("   PyTorch готов")
    except ImportError:
        pass
    
    # Scikit-learn использует NumPy random, но добавляем переменную окружения
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Для HDBSCAN и других библиотек, которые могут использовать системное время
    try:
        import hdbscan
        # HDBSCAN не имеет глобального random_state, устанавливается через параметры
        logger.info("   HDBSCAN готов")
    except ImportError:
        pass
    
    # Для UMAP (если используется)
    try:
        import umap
        logger.info("   UMAP готов")
    except ImportError:
        pass
    
    logger.info(f"Seed {seed} установлен везде")

# Инициализируем логирование при импорте модуля
setup_logging() 