import json
import numpy as np
import pandas as pd
from openai import OpenAI

def generate_gpt4_insights(features_df, labels, chosen_algorithm, quality_metrics, OPENAI_API_KEY):
    """
    Generate business insights using GPT-4 integration
    """
    print("\n🧠 GENERATING GPT-4 INSIGHTS...")
    
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Prepare segment statistics
        features_with_labels = features_df.copy()
        features_with_labels['segment'] = labels
        
        segment_stats = {}
        for segment_id in sorted(set(labels)):
            if segment_id == -1:
                segment_name = "Outliers"
            else:
                segment_name = f"Segment_{segment_id}"
            
            segment_data = features_with_labels[features_with_labels['segment'] == segment_id]
            
            segment_stats[segment_name] = {
                'size': len(segment_data),
                'avg_transactions': segment_data['tx_count'].mean(),
                'avg_amount': segment_data['avg_amount'].mean(),
                'total_revenue': segment_data['total_amount'].sum(),
                'digital_wallet_ratio': segment_data['digital_wallet_ratio'].mean(),
                'international_ratio': segment_data['international_ratio'].mean(),
                'contactless_ratio': segment_data['contactless_ratio'].mean(),
                'weekend_ratio': segment_data['weekend_ratio'].mean(),
                'city_diversity': segment_data['city_diversity'].mean(),
                'country_diversity': segment_data['country_diversity'].mean(),
                'payment_sophistication': segment_data['payment_sophistication'].mean()
            }
        
        # Generate insights with GPT-4
        prompt = f"""
You are a senior banking analyst. Analyze these customer segments and provide:

1. MEANINGFUL NAMES for each segment based on behavior
2. KEY CHARACTERISTICS of each segment  
3. BUSINESS RECOMMENDATIONS (marketing, products, retention)
4. REVENUE OPPORTUNITIES for each segment

Algorithm Used: {chosen_algorithm}
Quality Metrics: {quality_metrics}

Segment Statistics:
{json.dumps(segment_stats, indent=2, default=str)}

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
        
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are an expert banking analyst specializing in customer segmentation. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        insights = json.loads(response.choices[0].message.content)
        print("✅ GPT-4 insights generated successfully")
        return insights
        
    except Exception as e:
        print(f"⚠️ GPT-4 integration failed: {e}")
        print("💡 Using fallback segment analysis...")
        
        # Fallback analysis
        fallback_insights = {}
        for segment_name, stats in segment_stats.items():
            fallback_insights[segment_name] = {
                "business_name": segment_name,
                "description": f"Segment with {stats['size']} customers",
                "recommendations": ["Analyze behavior", "Develop targeted offers", "Monitor engagement"],
                "revenue_opportunity": "Personalized banking products"
            }
        
        return fallback_insights
