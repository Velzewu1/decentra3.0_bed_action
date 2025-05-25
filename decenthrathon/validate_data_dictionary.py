#!/usr/bin/env python3
"""
Валидация Data Dictionary против реальных данных
"""

import pandas as pd
from core.data_processing import process_transaction_data

def validate_data_dictionary():
    """Проверка соответствия data dictionary и реальных данных"""
    
    print("🔍 ВАЛИДАЦИЯ DATA DICTIONARY")
    print("="*50)
    
    # Читаем dictionary
    try:
        dict_df = pd.read_csv('data_dictionary.csv', comment='#')
        dict_fields = set(dict_df['Название'].tolist())
        print(f"📄 В data_dictionary.csv: {len(dict_fields)} полей")
    except Exception as e:
        print(f"❌ Ошибка чтения data_dictionary.csv: {e}")
        return False
    
    # Проверяем реальные данные
    try:
        print("📊 Загружаем выборку транзакций...")
        df = pd.read_parquet('DECENTRATHON_3.0.parquet')
        df_sample = df.sample(1000, random_state=42)
        
        print("⚙️ Обрабатываем через feature engineering...")
        features, _, _ = process_transaction_data(df_sample)
        
        # Добавляем поле segment (создается после кластеризации)
        engineered_fields = set(features.columns.tolist() + ['segment'])
        print(f"🎯 Полей после feature engineering: {len(engineered_fields)}")
        
    except Exception as e:
        print(f"❌ Ошибка обработки данных: {e}")
        return False
    
    # Анализируем по этапам
    print("\n🔍 АНАЛИЗ ПО ЭТАПАМ ОБРАБОТКИ:")
    print("-" * 40)
    
    # 1. Исходные поля транзакций
    original_fields = set(df.columns.tolist())
    dict_original = set(dict_df[dict_df['Категория'] == 'Исходные']['Название'].tolist())
    
    print(f"1. ИСХОДНЫЕ ТРАНЗАКЦИИ:")
    print(f"   В парquet файле: {len(original_fields)}")
    print(f"   В словаре: {len(dict_original)}")
    
    missing_original = original_fields - dict_original
    if missing_original:
        print(f"   ⚠️  Не задокументированы: {sorted(missing_original)}")
    
    extra_original = dict_original - original_fields
    if extra_original:
        print(f"   ⚠️  В словаре лишние: {sorted(extra_original)}")
    
    if not missing_original and not extra_original:
        print("   ✅ Исходные поля полностью задокументированы")
    
    # 2. Признаки после feature engineering
    dict_engineered = set(dict_df[dict_df['Категория'] != 'Исходные']['Название'].tolist())
    
    # card_id остается из исходных данных в финальном датасете
    dict_engineered.add('card_id')
    
    print(f"\n2. ПОСЛЕ FEATURE ENGINEERING:")
    print(f"   Созданных полей: {len(engineered_fields)}")
    print(f"   В словаре (включая card_id): {len(dict_engineered)}")
    
    missing_engineered = engineered_fields - dict_engineered
    if missing_engineered:
        print(f"   ⚠️  Не задокументированы: {sorted(missing_engineered)}")
    
    extra_engineered = dict_engineered - engineered_fields
    if extra_engineered:
        print(f"   ⚠️  В словаре лишние: {sorted(extra_engineered)}")
    
    if not missing_engineered and not extra_engineered:
        print("   ✅ Созданные признаки полностью задокументированы")
    
    # 3. Общая статистика по категориям
    print(f"\n3. СТАТИСТИКА ПО КАТЕГОРИЯМ:")
    categories = dict_df['Категория'].value_counts()
    for cat, count in categories.items():
        print(f"   {cat}: {count} полей")
    
    # Итоговая оценка
    print(f"\n🎯 ИТОГОВАЯ ОЦЕНКА:")
    print("-" * 20)
    
    total_issues = len(missing_original) + len(extra_original) + len(missing_engineered) + len(extra_engineered)
    
    if total_issues == 0:
        print("✅ Data Dictionary полностью корректен!")
        print("   - Все исходные поля задокументированы")
        print("   - Все созданные признаки задокументированы") 
        print("   - Структура соответствует pipeline обработки")
        return True
    else:
        print(f"⚠️  Найдено {total_issues} несоответствий")
        if missing_original or extra_original:
            print("   - Проблемы с исходными полями")
        if missing_engineered or extra_engineered:
            print("   - Проблемы с созданными признаками")
        return False

if __name__ == "__main__":
    validate_data_dictionary() 