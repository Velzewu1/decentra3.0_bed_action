#!/usr/bin/env python3

import pandas as pd
import numpy as np

# Загружаем данные
print("📊 АНАЛИЗ ТРЕХ КЛАСТЕРОВ\n")

# Summary
segments_df = pd.read_parquet('segment_summary.parquet')
print("🎯 КРАТКАЯ СВОДКА ПО КЛАСТЕРАМ:")
print(segments_df[['segment', 'segment_name', 'size', 'percentage', 'avg_amount', 'avg_transactions']].to_string())

# Детальный анализ
df = pd.read_csv('customer_segments.csv')
print(f"\n📈 РАСПРЕДЕЛЕНИЕ КЛИЕНТОВ:")
print(df['segment'].value_counts().sort_index())

# Анализ каждого кластера
print(f"\n🔍 ДЕТАЛЬНЫЙ АНАЛИЗ КЛАСТЕРОВ:")

for cluster_id in sorted(df['segment'].unique()):
    cluster_data = df[df['segment'] == cluster_id]
    
    if cluster_id == -1:
        cluster_name = "OUTLIERS (Шум)"
    else:
        cluster_name = f"CLUSTER {cluster_id}"
    
    print(f"\n{'='*50}")
    print(f"🎯 {cluster_name}")
    print(f"{'='*50}")
    print(f"👥 Размер: {len(cluster_data):,} клиентов ({len(cluster_data)/len(df)*100:.1f}%)")
    
    # Основные метрики
    print(f"💰 Средняя сумма транзакции: {cluster_data['avg_amount'].mean():,.2f} тенге")
    print(f"📊 Среднее количество транзакций: {cluster_data['tx_count'].mean():.1f}")
    print(f"💳 Общий доход: {cluster_data['total_amount'].sum():,.0f} тенге")
    
    # Поведенческие характеристики
    print(f"📱 Digital Wallet использование: {cluster_data['digital_wallet_ratio'].mean():.1%}")
    print(f"💎 Contactless платежи: {cluster_data['contactless_ratio'].mean():.1%}")
    print(f"🌍 Международные операции: {cluster_data['international_ratio'].mean():.2%}")
    
    # Географическое разнообразие
    print(f"🏙️  Разнообразие городов: {cluster_data['city_diversity'].mean():.1f}")
    print(f"🌐 Разнообразие стран: {cluster_data['country_diversity'].mean():.1f}")
    
    # Продвинутые метрики
    print(f"🎓 Изощренность платежей: {cluster_data['payment_sophistication'].mean():.3f}")
    print(f"💼 Customer Lifetime Value: {cluster_data['customer_lifetime_value'].mean():,.0f} тенге")
    
    # Временные паттерны
    if 'weekend_ratio' in cluster_data.columns:
        print(f"🎉 Активность на выходных: {cluster_data['weekend_ratio'].mean():.1%}")

print(f"\n🔄 ПОЧЕМУ 3 КЛАСТЕРА ВМЕСТО 5?")
print("="*50)
print("1. Более оптимальные параметры HDBSCAN")
print("2. Улучшенная предобработка данных")
print("3. Более строгие критерии качества кластеризации")
print("4. Ensemble refinement оптимизировал количество кластеров")
print("5. Данные естественным образом группируются в 3 основных сегмента") 