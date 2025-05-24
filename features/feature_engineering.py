import pandas as pd
import numpy as np

def extract_advanced_features(df):
    """
    Advanced Feature Engineering for Customer Segmentation
    
    Business Logic for Feature Selection (using ONLY real dataset fields):
    - FREQUENCY features: How often customers transact (loyalty indicator)
    - MONETARY features: Transaction amounts (profitability indicator) 
    - RECENCY features: Time patterns (engagement indicator)
    - BEHAVIORAL features: Payment methods, timing (risk/preference indicator)
    - GEOGRAPHICAL features: Location diversity (lifestyle indicator)
    
    REAL DATASET FIELDS (15 total):
    transaction_id, transaction_timestamp, card_id, expiry_date, issuer_bank_name,
    merchant_id, merchant_mcc, merchant_city, transaction_type, transaction_amount_kzt,
    original_amount, transaction_currency, acquirer_country_iso, pos_entry_mode, wallet_type
    """
    
    print("🔧 FEATURE ENGINEERING: Creating business-driven features...")
    print(f"📊 Available columns: {list(df.columns)}")
    
    # Time-based features from transaction_timestamp
    df['hour'] = df['transaction_timestamp'].dt.hour
    df['day_of_week'] = df['transaction_timestamp'].dt.dayofweek
    df['month'] = df['transaction_timestamp'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6])
    df['is_holiday_season'] = df['month'].isin([11, 12, 1])
    
    # Digital adoption indicator
    df['is_digital_wallet'] = df['wallet_type'].notnull()
    
    # International transaction indicator
    df['is_international'] = df['transaction_currency'] != 'KZT'
    
    # Contactless payment indicator
    df['is_contactless'] = df['pos_entry_mode'] == 'Contactless'
    
    features = df.groupby('card_id').agg({
        # FREQUENCY FEATURES (Customer Activity)
        'transaction_id': ['count'],  # Total transactions
        
        # MONETARY FEATURES (Customer Value)
        'transaction_amount_kzt': ['mean', 'std', 'median', 'sum', 'min', 'max'],
        
        # BEHAVIORAL FEATURES (Customer Preferences)
        'is_digital_wallet': [lambda x: x.mean()],  # Digital wallet adoption
        'is_international': [lambda x: x.mean()],   # International transactions ratio
        'is_contactless': [lambda x: x.mean()],     # Contactless payment ratio
        'transaction_type': [lambda x: x.nunique()], # Variety of transaction types
        'pos_entry_mode': [lambda x: x.nunique()],  # Variety of payment methods
        
        'hour': [
            lambda x: ((x >= 0) & (x < 6)).mean(),      # Night owl behavior
            lambda x: ((x >= 6) & (x < 12)).mean(),     # Morning person
            lambda x: ((x >= 12) & (x < 18)).mean(),    # Afternoon activity
            lambda x: ((x >= 18) & (x < 24)).mean(),    # Evening activity
        ],
        'is_weekend': [lambda x: x.mean()],             # Weekend vs weekday preference
        'is_holiday_season': [lambda x: x.mean()],      # Holiday shopping behavior
        
        # GEOGRAPHICAL FEATURES (Customer Mobility)
        'merchant_city': ['nunique'],                   # Geographic diversity
        'merchant_mcc': ['nunique'],                    # Merchant category diversity
        'acquirer_country_iso': ['nunique'],            # Country diversity
        'issuer_bank_name': ['nunique'],                # Bank diversity (multiple cards)
        
        # RECENCY FEATURES (Customer Engagement)
        'transaction_timestamp': [
            lambda x: (x.max() - x.min()).days + 1,    # Days active
            lambda x: len(x) / ((x.max() - x.min()).days + 1),  # Transaction frequency
        ]
    }).reset_index()
    
    # Flatten column names
    features.columns = ['card_id', 'tx_count', 'avg_amount', 'std_amount', 'median_amount', 
                       'total_amount', 'min_amount', 'max_amount', 'digital_wallet_ratio', 
                       'international_ratio', 'contactless_ratio', 'tx_type_variety', 'payment_method_variety',
                       'night_ratio', 'morning_ratio', 'afternoon_ratio', 'evening_ratio',
                       'weekend_ratio', 'holiday_ratio', 'city_diversity', 'mcc_diversity',
                       'country_diversity', 'bank_diversity', 'days_active', 'tx_frequency']
    
    # Additional derived features
    features['amount_volatility'] = features['std_amount'] / features['avg_amount']
    features['high_value_ratio'] = (features['max_amount'] / features['avg_amount']).fillna(1)
    features['spending_consistency'] = 1 / (1 + features['amount_volatility'].fillna(0))
    features['customer_lifetime_value'] = features['total_amount'] * features['tx_frequency']
    features['avg_daily_amount'] = features['total_amount'] / features['days_active']
    features['payment_sophistication'] = (features['digital_wallet_ratio'] + 
                                        features['contactless_ratio'] + 
                                        features['payment_method_variety'] / 5) / 3
    
    # Fill NaN values
    features = features.fillna(0)
    
    print(f"✅ Created {len(features.columns)-1} features with strong business rationale")
    print(f"📈 Features based on real dataset fields:")
    print(f"   • FREQUENCY: tx_count, tx_frequency, days_active")
    print(f"   • MONETARY: avg_amount, total_amount, amount_volatility, CLV")
    print(f"   • BEHAVIORAL: digital_wallet_ratio, contactless_ratio, time patterns")
    print(f"   • GEOGRAPHICAL: city_diversity, country_diversity, mcc_diversity")
    print(f"   • DERIVED: payment_sophistication, spending_consistency")
    
    return features
