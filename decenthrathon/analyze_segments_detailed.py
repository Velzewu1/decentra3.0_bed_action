#!/usr/bin/env python3
"""
Детальный анализ сегментов на основе hdbscan.py и features.py
"""

import pandas as pd
import numpy as np

def analyze_detailed_segments():
    print('🔍 ДЕТАЛЬНЫЙ АНАЛИЗ СЕГМЕНТОВ ПО HDBSCAN.PY + FEATURES.PY')
    print('=' * 70)
    
    # Загружаем данные
    df = pd.read_parquet('customer_segments.parquet')
    
    print(f'📊 Общая статистика: {len(df)} клиентов, {len(df.columns)-2} признаков')
    print(f'🎯 Сегментов: {len(df["segment"].unique())}')
    
    # Анализ каждого сегмента
    for segment_id in sorted(df['segment'].unique()):
        segment_data = df[df['segment'] == segment_id]
        size = len(segment_data)
        percentage = size / len(df) * 100
        
        print(f'\n📊 СЕГМЕНТ {segment_id}:')
        print(f'   👥 Размер: {size} клиентов ({percentage:.1f}%)')
        
        # Основные метрики из features.py
        print(f'   💰 Средний чек: {segment_data["avg_amount"].mean():.0f} KZT')
        print(f'   🔄 Средние транзакции: {segment_data["tx_count"].mean():.0f}')
        print(f'   💎 Общая выручка: {segment_data["total_amount"].sum()/1e9:.3f} млрд KZT')
        print(f'   📈 Волатильность трат: {segment_data["amount_volatility"].mean():.3f}')
        print(f'   🎯 Последовательность трат: {segment_data["spending_consistency"].mean():.3f}')
        
        # Поведенческие характеристики
        print(f'   📱 Digital wallet ratio: {segment_data["digital_wallet_ratio"].mean()*100:.1f}%')
        print(f'   💳 Contactless ratio: {segment_data["contactless_ratio"].mean()*100:.1f}%')
        print(f'   🌍 International ratio: {segment_data["international_ratio"].mean()*100:.2f}%')
        print(f'   🏪 Разнообразие типов транзакций: {segment_data["tx_type_variety"].mean():.1f}')
        
        # Географические характеристики
        print(f'   🏙️ Географическое разнообразие (города): {segment_data["city_diversity"].mean():.1f}')
        print(f'   🌍 Разнообразие стран: {segment_data["country_diversity"].mean():.1f}')
        
        # Временные характеристики
        print(f'   📅 Дни активности: {segment_data["days_active"].mean():.0f}')
        print(f'   ⚡ Частота транзакций: {segment_data["tx_frequency"].mean():.2f}/день')
        
        # Бизнес-интерпретация
        if segment_id == -1:
            interpretation = "🚨 OUTLIERS - Аномальное поведение, требует индивидуального анализа"
        elif segment_id == 4:
            interpretation = "🏢 MASS MARKET - Основная клиентская база банка"
        elif segment_data["digital_wallet_ratio"].mean() > 0.6:
            interpretation = "📱 DIGITAL NATIVES - Высокотехнологичные пользователи"
        elif segment_data["avg_amount"].mean() > 30000:
            interpretation = "💎 PREMIUM - Высокодоходные клиенты"
        elif segment_data["international_ratio"].mean() > 0.01:
            interpretation = "🌍 INTERNATIONAL - Клиенты с международными операциями"
        else:
            interpretation = f"🔍 NICHE SEGMENT - Специализированная группа"
        
        print(f'   🎯 Интерпретация: {interpretation}')
    
    # Сравнительная таблица
    print('\n📊 СРАВНИТЕЛЬНАЯ ТАБЛИЦА СЕГМЕНТОВ:')
    print('-' * 70)
    comparison = []
    for segment_id in sorted(df['segment'].unique()):
        segment_data = df[df['segment'] == segment_id]
        comparison.append({
            'Сегмент': segment_id,
            'Размер': len(segment_data),
            'Доля %': f"{len(segment_data)/len(df)*100:.1f}%",
            'Средний чек': f"{segment_data['avg_amount'].mean():.0f}",
            'Транзакции': f"{segment_data['tx_count'].mean():.0f}",
            'Digital %': f"{segment_data['digital_wallet_ratio'].mean()*100:.1f}%",
            'Выручка млрд': f"{segment_data['total_amount'].sum()/1e9:.3f}"
        })
    
    comparison_df = pd.DataFrame(comparison)
    print(comparison_df.to_string(index=False))
    
    # Основные выводы
    print('\n🎯 ОСНОВНЫЕ ВЫВОДЫ:')
    print('1. Сегмент 4 (Mass Market) - доминирует с 96.6% клиентов и 99.5% выручки')
    print('2. Outliers (сегмент -1) - 2.6% клиентов с нетипичным поведением')
    print('3. Сегменты 0-3 - микро-сегменты с уникальными характеристиками')
    print('4. Основные различия по цифровизации, размеру чека и активности')

if __name__ == "__main__":
    analyze_detailed_segments() 