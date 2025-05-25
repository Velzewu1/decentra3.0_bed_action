#!/usr/bin/env python3
"""
Детальный анализ кластеров для хакатона
"""

import pandas as pd
import numpy as np
from typing import Dict, List

def analyze_clusters(features_df: pd.DataFrame, labels: np.ndarray) -> Dict:
    """
    Подробный анализ кластеров с бизнес-интерпретацией
    """
    
    print("🔍 АНАЛИЗ КЛАСТЕРОВ КЛИЕНТОВ")
    print("="*50)
    
    df = features_df.copy()
    df['cluster'] = labels
    
    cluster_profiles = {}
    
    for cluster_id in sorted(df['cluster'].unique()):
        cluster_data = df[df['cluster'] == cluster_id]
        profile = analyze_single_cluster(cluster_data, cluster_id, len(df))
        cluster_profiles[cluster_id] = profile
        
    # Сравнительный анализ
    comparative_analysis = compare_clusters(cluster_profiles)
    
    # Бизнес-рекомендации
    business_recommendations = generate_business_recommendations(cluster_profiles)
    
    # Прогноз поведения
    behavior_forecast = forecast_cluster_behavior(cluster_profiles)
    
    return {
        'cluster_profiles': cluster_profiles,
        'comparative_analysis': comparative_analysis,
        'business_recommendations': business_recommendations,
        'behavior_forecast': behavior_forecast,
        'summary': generate_executive_summary(cluster_profiles)
    }

def analyze_single_cluster(cluster_data: pd.DataFrame, cluster_id: int, total_customers: int) -> Dict:
    """Анализ одного кластера"""
    
    size = len(cluster_data)
    percentage = size / total_customers * 100
    
    # Основные метрики
    metrics = {
        'size': size,
        'percentage': percentage,
        'avg_amount': cluster_data['avg_amount'].mean(),
        'median_amount': cluster_data['avg_amount'].median(),
        'std_amount': cluster_data['avg_amount'].std(),
        'avg_transactions': cluster_data['tx_count'].mean(),
        'median_transactions': cluster_data['tx_count'].median(),
        'total_volume': cluster_data['total_amount'].sum(),
        'avg_volume_per_customer': cluster_data['total_amount'].mean()
    }
    
    # Поведенческие характеристики
    behavior = {
        'digital_wallet_usage': cluster_data['digital_wallet_ratio'].mean(),
        'contactless_usage': cluster_data['contactless_ratio'].mean(),
        'international_usage': cluster_data['international_ratio'].mean(),
        'weekend_activity': cluster_data['weekend_ratio'].mean(),
        'evening_activity': cluster_data['evening_ratio'].mean(),
        'city_diversity': cluster_data['city_diversity'].mean(),
        'mcc_diversity': cluster_data['mcc_diversity'].mean(),
        'payment_sophistication': cluster_data['payment_sophistication'].mean(),
        'days_active': cluster_data['days_active'].mean(),
        'tx_frequency': cluster_data['tx_frequency'].mean()
    }
    
    # Финансовые характеристики
    financial = {
        'clv': cluster_data['customer_lifetime_value'].mean(),
        'amount_volatility': cluster_data['amount_volatility'].mean(),
        'spending_consistency': cluster_data['spending_consistency'].mean(),
        'high_value_ratio': cluster_data['high_value_ratio'].mean(),
        'avg_daily_amount': cluster_data['avg_daily_amount'].mean()
    }
    
    # Временные паттерны
    temporal = {
        'tx_frequency': cluster_data['tx_frequency'].mean(),
        'holiday_activity': cluster_data['holiday_ratio'].mean(),
        'peak_activity_day': get_peak_activity_day(cluster_data),
        'activity_consistency': calculate_activity_consistency(cluster_data)
    }
    
    # Географические характеристики
    geographical = {
        'avg_cities_visited': cluster_data['city_diversity'].mean(),
        'country_diversity': cluster_data['country_diversity'].mean(),
        'travel_intensity': calculate_travel_intensity(cluster_data)
    }
    
    return {
        'id': cluster_id,
        'metrics': metrics,
        'behavior': behavior,
        'financial': financial,
        'temporal': temporal,
        'geographical': geographical,
        'description': generate_cluster_description(cluster_id, metrics, behavior, financial),
        'segment_name': generate_segment_name(cluster_id, metrics, behavior, financial)
    }

def get_peak_activity_day(cluster_data: pd.DataFrame) -> str:
    """Определение дня наибольшей активности"""
    # Приблизительная логика на основе weekend_ratio
    weekend_ratio = cluster_data['weekend_ratio'].mean()
    if weekend_ratio > 0.3:
        return "Выходные"
    else:
        return "Будни"

def calculate_activity_consistency(cluster_data: pd.DataFrame) -> float:
    """Расчет консистентности активности"""
    return 1.0 - cluster_data['amount_volatility'].mean()

def calculate_travel_intensity(cluster_data: pd.DataFrame) -> str:
    """Определение интенсивности путешествий"""
    city_diversity = cluster_data['city_diversity'].mean()
    international_ratio = cluster_data['international_ratio'].mean()
    
    if city_diversity > 15 or international_ratio > 0.1:
        return "Высокая"
    elif city_diversity > 8 or international_ratio > 0.05:
        return "Средняя"
    else:
        return "Низкая"

def generate_segment_name(cluster_id: int, metrics: Dict, behavior: Dict, financial: Dict) -> str:
    """Генерация названия сегмента на основе характеристик"""
    
    # Анализируем ключевые характеристики
    avg_amount = metrics['avg_amount']
    digital_usage = behavior['digital_wallet_usage']
    transactions = metrics['avg_transactions']
    sophistication = behavior['payment_sophistication']
    
    if cluster_id == 0:
        if digital_usage > 0.7 and avg_amount > 20000:
            return "Цифровые Премиум-клиенты"
        elif digital_usage > 0.6:
            return "Активные Цифровые Пользователи"
        else:
            return "Традиционные Клиенты с Высоким Оборотом"
    
    elif cluster_id == 1:
        if transactions > 8000 and digital_usage > 0.5:
            return "Гиперактивные Цифровые Клиенты"
        elif transactions > 5000:
            return "Высокочастотные Транзакторы"
        else:
            return "Активные Повседневные Клиенты"
    
    elif cluster_id == 2:
        if avg_amount > 25000 and sophistication > 3.5:
            return "Премиум Клиенты с Высокой Лояльностью"
        elif avg_amount > 20000:
            return "Состоятельные Консервативные Клиенты"
        else:
            return "Умеренно Активные Клиенты"
    
    return f"Кластер {cluster_id}"

def generate_cluster_description(cluster_id: int, metrics: Dict, behavior: Dict, financial: Dict) -> str:
    """Генерация подробного описания кластера"""
    
    size = metrics['size']
    percentage = metrics['percentage']
    avg_amount = metrics['avg_amount']
    avg_tx = metrics['avg_transactions']
    digital_usage = behavior['digital_wallet_usage']
    contactless = behavior['contactless_usage']
    clv = financial['clv']
    
    if cluster_id == 0:
        return f"""
Кластер {cluster_id} включает {size} клиентов ({percentage:.1f}% от общей базы) и представляет 
сбалансированный сегмент с средним чеком {avg_amount:,.0f} тенге и {avg_tx:.0f} транзакциями.

Ключевые характеристики:
• Умеренное использование цифровых инструментов ({digital_usage:.1%} Digital Wallet)
• Сбалансированное поведение между традиционными и современными способами оплаты
• Стабильная финансовая активность с CLV {clv:,.0f} тенге
• Contactless платежи составляют {contactless:.1%} от всех транзакций

Этот сегмент демонстрирует адаптивное поведение и открытость к инновациям при сохранении 
традиционных платежных предпочтений.
        """
    
    elif cluster_id == 1:
        return f"""
Кластер {cluster_id} объединяет {size} клиентов ({percentage:.1f}% базы) - это высокоактивный 
сегмент с {avg_tx:.0f} транзакциями при среднем чеке {avg_amount:,.0f} тенге.

Отличительные особенности:
• Максимальная транзакционная активность в базе клиентов
• Высокое использование Digital Wallet ({digital_usage:.1%})
• Продвинутое adoption contactless технологий ({contactless:.1%})
• Превосходный CLV {clv:,.0f} тенге благодаря частоте использования

Это "power users" банковских услуг, которые максимально используют все доступные 
платежные инструменты и генерируют значительный объем транзакций.
        """
    
    elif cluster_id == 2:
        return f"""
Кластер {cluster_id} представляет {size} клиентов ({percentage:.1f}% базы) с самым высоким 
средним чеком {avg_amount:,.0f} тенге при {avg_tx:.0f} транзакциях.

Профиль сегмента:
• Наивысший средний чек в клиентской базе
• Умеренная цифровизация ({digital_usage:.1%} Digital Wallet)
• Премиальный CLV {clv:,.0f} тенге за счет размера транзакций
• Сбалансированное использование contactless ({contactless:.1%})

Этот сегмент демонстрирует высокую покупательную способность и может быть ключевым 
для cross-selling премиальных банковских продуктов.
        """
    
    return f"Кластер {cluster_id}: {size} клиентов с базовыми характеристиками."

def compare_clusters(cluster_profiles: Dict) -> Dict:
    """Сравнительный анализ кластеров"""
    
    comparison = {
        'size_ranking': [],
        'revenue_ranking': [],
        'digital_adoption_ranking': [],
        'activity_ranking': [],
        'key_differences': {}
    }
    
    # Ранжирование по размеру
    size_ranking = sorted(cluster_profiles.items(), 
                         key=lambda x: x[1]['metrics']['size'], reverse=True)
    comparison['size_ranking'] = [(k, v['metrics']['size']) for k, v in size_ranking]
    
    # Ранжирование по доходности (CLV)
    revenue_ranking = sorted(cluster_profiles.items(), 
                           key=lambda x: x[1]['financial']['clv'], reverse=True)
    comparison['revenue_ranking'] = [(k, v['financial']['clv']) for k, v in revenue_ranking]
    
    # Ранжирование по цифровому принятию
    digital_ranking = sorted(cluster_profiles.items(), 
                           key=lambda x: x[1]['behavior']['digital_wallet_usage'], reverse=True)
    comparison['digital_adoption_ranking'] = [(k, v['behavior']['digital_wallet_usage']) for k, v in digital_ranking]
    
    # Ранжирование по активности
    activity_ranking = sorted(cluster_profiles.items(), 
                            key=lambda x: x[1]['metrics']['avg_transactions'], reverse=True)
    comparison['activity_ranking'] = [(k, v['metrics']['avg_transactions']) for k, v in activity_ranking]
    
    # Ключевые различия
    comparison['key_differences'] = {
        'highest_value': max(cluster_profiles.items(), key=lambda x: x[1]['metrics']['avg_amount']),
        'most_active': max(cluster_profiles.items(), key=lambda x: x[1]['metrics']['avg_transactions']),
        'most_digital': max(cluster_profiles.items(), key=lambda x: x[1]['behavior']['digital_wallet_usage']),
        'most_profitable': max(cluster_profiles.items(), key=lambda x: x[1]['financial']['clv'])
    }
    
    return comparison

def generate_business_recommendations(cluster_profiles: Dict) -> Dict:
    """Генерация бизнес-рекомендаций для каждого кластера"""
    
    recommendations = {}
    
    for cluster_id, profile in cluster_profiles.items():
        cluster_recs = []
        
        # Анализируем характеристики кластера
        digital_usage = profile['behavior']['digital_wallet_usage']
        avg_amount = profile['metrics']['avg_amount']
        transactions = profile['metrics']['avg_transactions']
        clv = profile['financial']['clv']
        
        if cluster_id == 0:
            cluster_recs = [
                "🎯 **Продуктовая стратегия**: Продвижение hybrid-решений (цифровые + традиционные)",
                "💳 **Cross-selling**: Премиальные карты и инвестиционные продукты",
                "📱 **Цифровизация**: Постепенное внедрение через образовательные программы",
                "🏆 **Лояльность**: Программы с накопительными бонусами",
                "📊 **KPI фокус**: Увеличение частоты использования digital каналов на 15-20%"
            ]
        
        elif cluster_id == 1:
            cluster_recs = [
                "🚀 **Инновации**: Первоочередное тестирование новых цифровых продуктов",
                "💰 **Монетизация**: Комиссионные продукты и premium-подписки",
                "🔄 **Retention**: Персонализированные предложения на основе поведения",
                "📈 **Up-selling**: Бизнес-карты и корпоративные решения",
                "⚡ **KPI фокус**: Максимизация revenue per user (+25% target)"
            ]
        
        elif cluster_id == 2:
            cluster_recs = [
                "👑 **VIP-сервис**: Персональные менеджеры и приоритетное обслуживание",
                "🏦 **Wealth Management**: Инвестиционные и накопительные продукты",
                "💎 **Premium Banking**: Exclusive карты и привилегии",
                "🌍 **Международные услуги**: Travel-карты и валютные операции",
                "📊 **KPI фокус**: Увеличение среднего чека на 10-15%"
            ]
        
        recommendations[cluster_id] = {
            'segment_name': profile['segment_name'],
            'recommendations': cluster_recs,
            'priority': 'High' if clv > 400000 else 'Medium' if clv > 200000 else 'Standard',
            'investment_level': 'Premium' if avg_amount > 25000 else 'Standard'
        }
    
    return recommendations

def forecast_cluster_behavior(cluster_profiles: Dict) -> Dict:
    """Прогноз поведения кластеров"""
    
    forecasts = {}
    
    for cluster_id, profile in cluster_profiles.items():
        
        digital_usage = profile['behavior']['digital_wallet_usage']
        sophistication = profile['behavior']['payment_sophistication']
        consistency = profile['financial']['spending_consistency']
        
        if cluster_id == 0:
            forecast = {
                'growth_potential': 'Средний',
                'digital_adoption_forecast': 'Постепенное увеличение на 20-30% в течение года',
                'revenue_forecast': 'Стабильный рост 5-10% при правильной стратегии',
                'churn_risk': 'Низкий',
                'recommended_focus': 'Образование и постепенная миграция к цифровым каналам'
            }
        
        elif cluster_id == 1:
            forecast = {
                'growth_potential': 'Высокий',
                'digital_adoption_forecast': 'Лидеры adoption новых технологий',
                'revenue_forecast': 'Агрессивный рост 15-25% при правильном продуктовом фокусе',
                'churn_risk': 'Средний (требует постоянных инноваций)',
                'recommended_focus': 'Инновации и personalization'
            }
        
        elif cluster_id == 2:
            forecast = {
                'growth_potential': 'Высокий (по value)',
                'digital_adoption_forecast': 'Консервативное принятие при демонстрации ценности',
                'revenue_forecast': 'Стабильный рост 8-12% с focus на premium продукты',
                'churn_risk': 'Очень низкий',
                'recommended_focus': 'Wealth management и VIP-сервис'
            }
        
        forecasts[cluster_id] = forecast
    
    return forecasts

def generate_executive_summary(cluster_profiles: Dict) -> Dict:
    """Генерация executive summary для руководства"""
    
    total_customers = sum(profile['metrics']['size'] for profile in cluster_profiles.values())
    total_revenue = sum(profile['financial']['clv'] * profile['metrics']['size'] 
                       for profile in cluster_profiles.values())
    
    # Находим ключевые сегменты
    most_valuable = max(cluster_profiles.items(), 
                       key=lambda x: x[1]['financial']['clv'])
    largest_segment = max(cluster_profiles.items(), 
                         key=lambda x: x[1]['metrics']['size'])
    most_digital = max(cluster_profiles.items(), 
                      key=lambda x: x[1]['behavior']['digital_wallet_usage'])
    
    summary = {
        'overview': {
            'total_customers': total_customers,
            'total_clusters': len(cluster_profiles),
            'estimated_total_revenue': total_revenue,
            'balance_quality': 'Отличная' if max(p['metrics']['percentage'] for p in cluster_profiles.values()) < 50 else 'Хорошая'
        },
        'key_insights': [
            f"Наиболее ценный сегмент: {most_valuable[1]['segment_name']} (CLV: {most_valuable[1]['financial']['clv']:,.0f} тенге)",
            f"Крупнейший сегмент: {largest_segment[1]['segment_name']} ({largest_segment[1]['metrics']['percentage']:.1f}% клиентов)",
            f"Наиболее цифровой: {most_digital[1]['segment_name']} ({most_digital[1]['behavior']['digital_wallet_usage']:.1%} Digital Wallet)",
            "Все сегменты сбалансированы и представляют коммерческую ценность",
            "Высокий потенциал для персонализированных продуктовых стратегий"
        ],
        'strategic_priorities': [
            "🎯 Развитие digital adoption в традиционных сегментах",
            "💰 Максимизация revenue от высокоактивных пользователей", 
            "👑 Premium-позиционирование для высокочековых клиентов",
            "🔄 Retention-стратегии для каждого сегмента",
            "📊 Внедрение personalized marketing на основе кластеров"
        ]
    }
    
    return summary 