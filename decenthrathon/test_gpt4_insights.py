#!/usr/bin/env python3
"""
Тест исправленной GPT-4 интеграции для insights
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
from openai import OpenAI
import os
from dotenv import load_dotenv

def test_gpt4_insights():
    print("🧠 ТЕСТ ИСПРАВЛЕННОЙ GPT-4 ИНТЕГРАЦИИ")
    print("=" * 50)
    
    # Загружаем переменные окружения
    load_dotenv(dotenv_path='../.env')
    api_key = os.getenv('OPENAI_API_KEY')
    
    print(f"🔑 API Key: {api_key[:20]}...")
    
    try:
        client = OpenAI(api_key=api_key)
        
        # Моковые данные сегментов
        segment_stats = {
            "Segment_0": {
                "size": 5,
                "avg_transactions": 44.2,
                "avg_amount": 14914.0,
                "total_revenue": 5100000.0,
                "digital_wallet_ratio": 0.614,
                "international_ratio": 0.0,
                "contactless_ratio": 0.192
            },
            "Segment_4": {
                "size": 1932,
                "avg_transactions": 5962.3,
                "avg_amount": 23034.0,
                "total_revenue": 231385900000.0,
                "digital_wallet_ratio": 0.505,
                "international_ratio": 0.001,
                "contactless_ratio": 0.179
            },
            "Outliers": {
                "size": 52,
                "avg_transactions": 270.1,
                "avg_amount": 31080.0,
                "total_revenue": 1172600000.0,
                "digital_wallet_ratio": 0.325,
                "international_ratio": 0.002,
                "contactless_ratio": 0.202
            }
        }
        
        prompt = f"""
You are a senior banking analyst. Analyze these customer segments and provide:

1. MEANINGFUL NAMES for each segment based on behavior
2. KEY CHARACTERISTICS of each segment  
3. BUSINESS RECOMMENDATIONS (marketing, products, retention)
4. REVENUE OPPORTUNITIES for each segment

Algorithm Used: HDBSCAN_UltraOptimized
Quality Metrics: {{'n_clusters': 5, 'silhouette_score': 0.528, 'noise_ratio': 0.026}}

Segment Statistics:
{json.dumps(segment_stats, indent=2)}

Return structured JSON with format:
{{
    "segment_name": {{
        "business_name": "Meaningful name",
        "description": "Key characteristics", 
        "recommendations": ["action1", "action2", "action3"],
        "revenue_opportunity": "How to monetize"
    }}
}}
"""
        
        print("🔗 Отправка запроса к GPT-4...")
        
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are an expert banking analyst specializing in customer segmentation. Return only valid JSON without markdown formatting."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        print("📥 Получен ответ от GPT-4")
        
        # ИСПРАВЛЕННАЯ ОБРАБОТКА ОТВЕТА
        response_content = response.choices[0].message.content.strip()
        
        print(f"🔍 Raw response length: {len(response_content)} chars")
        print(f"🔍 Raw response starts with: {response_content[:100]}...")
        
        # Remove markdown code blocks if present
        if response_content.startswith('```json'):
            response_content = response_content[7:]  # Remove ```json
        if response_content.startswith('```'):
            response_content = response_content[3:]   # Remove ```
        if response_content.endswith('```'):
            response_content = response_content[:-3]  # Remove trailing ```
        
        response_content = response_content.strip()
        
        print(f"🔍 Cleaned response length: {len(response_content)} chars")
        print(f"🔍 Cleaned response starts with: {response_content[:100]}...")
        
        insights = json.loads(response_content)
        
        print("✅ JSON парсинг успешен!")
        print(f"📊 Найдено сегментов: {len(insights)}")
        
        for segment_name, details in insights.items():
            print(f"\n🎯 {segment_name}:")
            print(f"   📝 Название: {details.get('business_name', 'N/A')}")
            print(f"   📖 Описание: {details.get('description', 'N/A')[:100]}...")
            print(f"   💡 Рекомендации: {len(details.get('recommendations', []))} пунктов")
            
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON ошибка: {e}")
        print(f"🔍 Проблемная часть: {response_content[max(0, e.pos-50):e.pos+50]}")
        return False
        
    except Exception as e:
        print(f"❌ Общая ошибка: {e}")
        return False

if __name__ == "__main__":
    success = test_gpt4_insights()
    
    if success:
        print("\n🎯 РЕЗУЛЬТАТ: GPT-4 insights работают корректно!")
    else:
        print("\n⚠️ РЕЗУЛЬТАТ: Требуется дополнительная отладка") 