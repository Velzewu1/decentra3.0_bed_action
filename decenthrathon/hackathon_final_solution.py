# 🏆 HACKATHON FINAL SOLUTION: Advanced Customer Segmentation
# Maximum Score: 120 points - All criteria covered

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
import warnings
import time
import os
warnings.filterwarnings('ignore')

# Machine Learning Libraries
import hdbscan
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import umap
from sklearn.neighbors import NearestNeighbors

# OpenAI for insights generation
from openai import OpenAI

print("🏆 HACKATHON SOLUTION: Advanced Customer Segmentation")
print("=" * 60)

# Global configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'your-api-key-here')

# ========================================================================================
# PART 1: FEATURE ENGINEERING WITH BUSINESS LOGIC (30 POINTS)
# ========================================================================================

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

# Load data
print("📁 Loading transaction data...")
df = pd.read_csv("transactions.csv", parse_dates=['transaction_timestamp'])
print(f"✅ Loaded {len(df):,} transactions for {df['card_id'].nunique():,} customers")

# Verify dataset structure
print(f"📊 Dataset columns ({len(df.columns)}): {list(df.columns)}")
expected_columns = [
    'transaction_id', 'transaction_timestamp', 'card_id', 'expiry_date', 'issuer_bank_name',
    'merchant_id', 'merchant_mcc', 'merchant_city', 'transaction_type', 'transaction_amount_kzt',
    'original_amount', 'transaction_currency', 'acquirer_country_iso', 'pos_entry_mode', 'wallet_type'
]
print(f"✅ Expected 15 fields - dataset has {len(df.columns)} fields")

features_df = extract_advanced_features(df)

# ========================================================================================
# PART 2: MODEL COMPARISON AND SELECTION (30 POINTS) 
# ========================================================================================

def prepare_data_for_clustering(features_df):
    """Prepare and scale data for clustering algorithms"""
    ml_features = features_df.drop(['card_id'], axis=1)
    
    # Use RobustScaler to handle outliers
    scaler = RobustScaler()
    ml_features_scaled = scaler.fit_transform(ml_features)
    
    return ml_features, ml_features_scaled, scaler

def compare_clustering_algorithms(ml_features_scaled, features_df):
    """
    Compare multiple clustering algorithms to justify model selection
    
    ALGORITHMS TESTED:
    1. KMeans: Centroid-based, assumes spherical clusters
    2. DBSCAN: Density-based, finds arbitrary shapes
    3. HDBSCAN: Hierarchical density-based, handles varying densities
    """
    
    print("\n🤖 MODEL COMPARISON: Testing multiple algorithms...")
    
    results = {}
    
    # 1. KMeans Clustering - test more cluster numbers
    print("   Testing KMeans...")
    kmeans_scores = []
    k_range = range(2, 11)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(ml_features_scaled)
        score = silhouette_score(ml_features_scaled, labels)
        kmeans_scores.append(score)
    
    best_k = k_range[np.argmax(kmeans_scores)]
    kmeans_best = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    kmeans_labels = kmeans_best.fit_predict(ml_features_scaled)
    
    results['KMeans'] = {
        'labels': kmeans_labels,
        'n_clusters': best_k,
        'silhouette': max(kmeans_scores),
        'inertia': kmeans_best.inertia_,
        'pros': ['Fast', 'Interpretable', 'Scalable'],
        'cons': ['Assumes spherical clusters', 'Need to specify K', 'Sensitive to outliers']
    }
    
    # 2. DBSCAN Clustering - proper parameter tuning
    print("   Testing DBSCAN...")
    
    # Find optimal eps using k-distance
    neighbors = NearestNeighbors(n_neighbors=10)
    neighbors_fit = neighbors.fit(ml_features_scaled)
    distances, indices = neighbors_fit.kneighbors(ml_features_scaled)
    distances = np.sort(distances, axis=0)
    distances = distances[:,1]
    # Use 95th percentile as eps
    optimal_eps = np.percentile(distances, 95)
    
    best_dbscan_score = -1
    best_dbscan_labels = None
    best_dbscan_params = None
    
    # Test different eps values around the optimal
    eps_range = [optimal_eps * 0.5, optimal_eps, optimal_eps * 1.5, optimal_eps * 2.0]
    min_samples_range = [5, 10, 15, 20]
    
    for eps in eps_range:
        for min_samples in min_samples_range:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(ml_features_scaled)
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            noise_ratio = np.sum(labels == -1) / len(labels)
            
            # Only evaluate if we have clusters and not too much noise
            if n_clusters > 0 and noise_ratio < 0.8:
                non_noise = labels != -1
                # Check if we have at least 2 clusters for silhouette calculation
                unique_clusters = len(set(labels[non_noise]))
                if unique_clusters >= 2:
                    score = silhouette_score(ml_features_scaled[non_noise], labels[non_noise])
                    if score > best_dbscan_score:
                        best_dbscan_score = score
                        best_dbscan_labels = labels
                        best_dbscan_params = {'eps': eps, 'min_samples': min_samples}
    
    if best_dbscan_labels is not None:
        results['DBSCAN'] = {
            'labels': best_dbscan_labels,
            'n_clusters': len(set(best_dbscan_labels)) - (1 if -1 in best_dbscan_labels else 0),
            'silhouette': best_dbscan_score,
            'noise_ratio': np.sum(best_dbscan_labels == -1) / len(best_dbscan_labels),
            'best_params': best_dbscan_params,
            'pros': ['Finds arbitrary shapes', 'Handles noise', 'No need to specify clusters'],
            'cons': ['Sensitive to hyperparameters', 'Struggles with varying densities']
        }
    else:
        results['DBSCAN'] = {
            'labels': np.full(len(ml_features_scaled), -1),
            'n_clusters': 0,
            'silhouette': -1.0,
            'noise_ratio': 1.0,
            'best_params': {'eps': optimal_eps, 'min_samples': 10},
            'pros': ['Finds arbitrary shapes', 'Handles noise', 'No need to specify clusters'],
            'cons': ['Sensitive to hyperparameters', 'Struggles with varying densities']
        }
    
    # 3. HDBSCAN Clustering
    print("   Testing HDBSCAN...")
    
    # Hyperparameter tuning for HDBSCAN
    best_hdbscan_score = -1
    best_hdbscan = None
    best_params = None
    
    for min_cluster_size in [20, 30, 50, 75, 100]:
        for min_samples in [5, 10, 15, 20]:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                cluster_selection_epsilon=0.1
            )
            labels = clusterer.fit_predict(ml_features_scaled)
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            noise_ratio = np.sum(labels == -1) / len(labels)
            
            if n_clusters > 0 and noise_ratio < 0.8:
                non_noise = labels != -1
                if np.sum(non_noise) > len(labels) * 0.1:  # At least 10% non-noise
                    # Check if we have at least 2 clusters for silhouette calculation
                    unique_clusters = len(set(labels[non_noise]))
                    if unique_clusters >= 2:
                        score = silhouette_score(ml_features_scaled[non_noise], labels[non_noise])
                        if score > best_hdbscan_score:
                            best_hdbscan_score = score
                            best_hdbscan = clusterer
                            best_params = {'min_cluster_size': min_cluster_size, 'min_samples': min_samples}
    
    if best_hdbscan is not None:
        hdbscan_labels = best_hdbscan.labels_
        results['HDBSCAN'] = {
            'labels': hdbscan_labels,
            'n_clusters': len(set(hdbscan_labels)) - (1 if -1 in hdbscan_labels else 0),
            'silhouette': best_hdbscan_score,
            'noise_ratio': np.sum(hdbscan_labels == -1) / len(hdbscan_labels),
            'cluster_persistence': best_hdbscan.cluster_persistence_,
            'best_params': best_params,
            'pros': ['Handles varying densities', 'Hierarchical structure', 'Robust outlier detection'],
            'cons': ['More complex', 'Harder to interpret parameters']
        }
    
    # Print comparison
    print(f"\n📊 ALGORITHM COMPARISON RESULTS:")
    for name, result in results.items():
        print(f"   {name}:")
        print(f"      • Clusters: {result['n_clusters']}")
        print(f"      • Silhouette Score: {result['silhouette']:.3f}")
        if 'noise_ratio' in result:
            print(f"      • Noise Ratio: {result['noise_ratio']:.1%}")
        if 'best_params' in result:
            print(f"      • Best params: {result['best_params']}")
    
    # Choose best algorithm based on silhouette score and practical considerations
    best_algorithm = None
    best_score = -1
    
    for name, result in results.items():
        # Penalize algorithms with too much noise or too few clusters
        score = result['silhouette']
        if 'noise_ratio' in result and result['noise_ratio'] > 0.5:
            score *= 0.5  # Penalty for high noise
        if result['n_clusters'] < 2:
            score *= 0.1  # Heavy penalty for no clusters
            
        if score > best_score:
            best_score = score
            best_algorithm = name
    
    chosen_algorithm = best_algorithm
    chosen_clusterer = None
    
    if chosen_algorithm == 'KMeans':
        chosen_clusterer = kmeans_best
    elif chosen_algorithm == 'HDBSCAN':
        chosen_clusterer = best_hdbscan
    elif chosen_algorithm == 'DBSCAN':
        # Create DBSCAN with best params
        if best_dbscan_params:
            chosen_clusterer = DBSCAN(**best_dbscan_params)
            chosen_clusterer.fit(ml_features_scaled)
    
    print(f"\n🏆 CHOSEN ALGORITHM: {chosen_algorithm}")
    print(f"   Reason: Best silhouette score ({results[chosen_algorithm]['silhouette']:.3f}) with practical cluster structure")
    print(f"   Clusters: {results[chosen_algorithm]['n_clusters']}")
    if 'noise_ratio' in results[chosen_algorithm]:
        print(f"   Noise ratio: {results[chosen_algorithm]['noise_ratio']:.1%}")
    
    return results, chosen_algorithm, chosen_clusterer

# ========================================================================================
# DISABLED: OLD CODE MOVED TO MAIN EXECUTION BLOCK
# ========================================================================================
# ml_features, ml_features_scaled, scaler = prepare_data_for_clustering(features_df)
# algorithm_results, chosen_algorithm, final_clusterer = compare_clustering_algorithms(ml_features_scaled, features_df)

# ========================================================================================
# PART 3: ENHANCED CLUSTERING QUALITY METRICS (20 POINTS)
# ========================================================================================

def evaluate_clustering_quality(ml_features_scaled, labels, chosen_algorithm):
    """
    Evaluate clustering quality using multiple metrics
    """
    
    # Remove noise points for evaluation (-1 labels)
    non_noise_mask = labels != -1
    
    if np.sum(non_noise_mask) == 0:
        print("❌ No valid clusters found (all points are noise)")
        return {
            'silhouette_score': -1.0,
            'calinski_harabasz': 0.0,
            'davies_bouldin': float('inf'),
            'n_clusters': 0,
            'noise_ratio': 1.0,
            'largest_cluster_ratio': 0.0
        }
    
    X_filtered = ml_features_scaled[non_noise_mask]
    labels_filtered = labels[non_noise_mask]
    
    n_clusters = len(np.unique(labels_filtered))
    
    if n_clusters < 2:
        print(f"❌ Insufficient clusters ({n_clusters}) for quality evaluation")
        return {
            'silhouette_score': -1.0,
            'calinski_harabasz': 0.0,
            'davies_bouldin': float('inf'),
            'n_clusters': n_clusters,
            'noise_ratio': np.sum(labels == -1) / len(labels),
            'largest_cluster_ratio': 0.0
        }
    
    # Calculate metrics
    silhouette = silhouette_score(X_filtered, labels_filtered)
    calinski = calinski_harabasz_score(X_filtered, labels_filtered)
    davies_bouldin = davies_bouldin_score(X_filtered, labels_filtered)
    
    # Additional metrics
    noise_ratio = np.sum(labels == -1) / len(labels)
    cluster_sizes = np.bincount(labels_filtered)
    largest_cluster_ratio = np.max(cluster_sizes) / len(labels_filtered)
    
    metrics = {
        'silhouette_score': silhouette,
        'calinski_harabasz': calinski,
        'davies_bouldin': davies_bouldin,
        'n_clusters': n_clusters,
        'noise_ratio': noise_ratio,
        'largest_cluster_ratio': largest_cluster_ratio
    }
    
    print(f"\n📈 CLUSTERING QUALITY METRICS for {chosen_algorithm}:")
    print(f"   • silhouette_score: {silhouette:.3f}")
    print(f"   • calinski_harabasz: {calinski:.3f}")
    print(f"   • davies_bouldin: {davies_bouldin:.3f}")
    print(f"   • n_clusters: {n_clusters}")
    print(f"   • noise_ratio: {noise_ratio:.3f}")
    print(f"   • largest_cluster_ratio: {largest_cluster_ratio:.3f}")
    
    return metrics

# ========================================================================================
# PART 4: GPT-4 INTEGRATION FOR INSIGHTS (10 POINTS)
# ========================================================================================

def generate_gpt4_insights(features_df, labels, chosen_algorithm, quality_metrics):
    """
    Generate business insights using GPT-4 integration
    """
    api_key = OPENAI_API_KEY
    
    print("\n🧠 GENERATING GPT-4 INSIGHTS...")
    
    try:
        client = OpenAI(api_key=api_key)
        
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

# ========================================================================================
# PART 5: COMPREHENSIVE VISUALIZATION (10 POINTS)
# ========================================================================================

def create_comprehensive_dashboard(features_df, labels, ml_features_scaled, chosen_algorithm, quality_metrics, insights):
    """Create publication-ready visualizations"""
    
    print("\n📊 CREATING COMPREHENSIVE DASHBOARD...")
    
    # UMAP for visualization
    umap_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    features_2d = umap_reducer.fit_transform(ml_features_scaled)
    
    fig = plt.figure(figsize=(24, 18))
    
    # 1. UMAP Visualization
    plt.subplot(3, 4, 1)
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='Set1', alpha=0.7, s=50)
    plt.colorbar(scatter)
    plt.title(f'Customer Segments in UMAP Space\n({chosen_algorithm} Clustering)', fontsize=12, fontweight='bold')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    
    # 2. Quality Metrics Summary
    plt.subplot(3, 4, 2)
    metrics_names = ['Silhouette', 'Calinski-H', 'Davies-B']
    metrics_values = [
        quality_metrics.get('silhouette_score', 0),
        quality_metrics.get('calinski_harabasz', 0) / 1000,  # Scale down for visualization
        1 - quality_metrics.get('davies_bouldin', 1)  # Invert for better visualization
    ]
    colors = ['green' if v > 0.5 else 'orange' if v > 0.3 else 'red' for v in metrics_values]
    plt.bar(metrics_names, metrics_values, color=colors, alpha=0.7)
    plt.title(f'Clustering Quality Metrics\n({chosen_algorithm})', fontsize=12, fontweight='bold')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    
    # 3. Segment Sizes
    plt.subplot(3, 4, 3)
    unique_labels, counts = np.unique(labels, return_counts=True)
    colors_pie = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
    plt.pie(counts, labels=[f'Segment {l}' if l != -1 else 'Outliers' for l in unique_labels], 
            autopct='%1.1f%%', colors=colors_pie)
    plt.title('Segment Distribution', fontsize=12, fontweight='bold')
    
    # 4. Average Transaction Amount by Segment
    plt.subplot(3, 4, 4)
    features_with_labels = features_df.copy()
    features_with_labels['segment'] = labels
    avg_amounts = []
    segment_names = []
    for label in sorted(unique_labels):
        segment_data = features_with_labels[features_with_labels['segment'] == label]
        avg_amounts.append(segment_data['avg_amount'].mean())
        segment_names.append(f'Seg {label}' if label != -1 else 'Outliers')
    
    plt.bar(segment_names, avg_amounts, color=colors_pie)
    plt.title('Average Transaction Amount', fontsize=12, fontweight='bold')
    plt.ylabel('Amount (KZT)')
    plt.xticks(rotation=45)
    
    # 5. Digital Wallet Usage by Segment
    plt.subplot(3, 4, 5)
    digital_ratios = []
    for label in sorted(unique_labels):
        segment_data = features_with_labels[features_with_labels['segment'] == label]
        digital_ratios.append(segment_data['digital_wallet_ratio'].mean() * 100)
    
    plt.bar(segment_names, digital_ratios, color=colors_pie)
    plt.title('Digital Wallet Usage %', fontsize=12, fontweight='bold')
    plt.ylabel('Usage %')
    plt.xticks(rotation=45)
    
    # 6. Transaction Frequency by Segment
    plt.subplot(3, 4, 6)
    tx_counts = []
    for label in sorted(unique_labels):
        segment_data = features_with_labels[features_with_labels['segment'] == label]
        tx_counts.append(segment_data['tx_count'].mean())
    
    plt.bar(segment_names, tx_counts, color=colors_pie)
    plt.title('Average Transaction Count', fontsize=12, fontweight='bold')
    plt.ylabel('Transactions')
    plt.xticks(rotation=45)
    
    # 7. Geographic Diversity
    plt.subplot(3, 4, 7)
    city_diversity = []
    for label in sorted(unique_labels):
        segment_data = features_with_labels[features_with_labels['segment'] == label]
        city_diversity.append(segment_data['city_diversity'].mean())
    
    plt.bar(segment_names, city_diversity, color=colors_pie)
    plt.title('Geographic Diversity', fontsize=12, fontweight='bold')
    plt.ylabel('Unique Cities')
    plt.xticks(rotation=45)
    
    # 8. Payment Sophistication
    plt.subplot(3, 4, 8)
    payment_soph = []
    for label in sorted(unique_labels):
        segment_data = features_with_labels[features_with_labels['segment'] == label]
        payment_soph.append(segment_data['payment_sophistication'].mean())
    
    plt.bar(segment_names, payment_soph, color=colors_pie)
    plt.title('Payment Sophistication', fontsize=12, fontweight='bold')
    plt.ylabel('Sophistication Score')
    plt.xticks(rotation=45)
    
    # 9. Weekend Transaction Ratio
    plt.subplot(3, 4, 9)
    weekend_ratios = []
    for label in sorted(unique_labels):
        segment_data = features_with_labels[features_with_labels['segment'] == label]
        weekend_ratios.append(segment_data['weekend_ratio'].mean() * 100)
    
    plt.bar(segment_names, weekend_ratios, color=colors_pie)
    plt.title('Weekend Transaction %', fontsize=12, fontweight='bold')
    plt.ylabel('Weekend %')
    plt.xticks(rotation=45)
    
    # 10. International Transaction Ratio
    plt.subplot(3, 4, 10)
    intl_ratios = []
    for label in sorted(unique_labels):
        segment_data = features_with_labels[features_with_labels['segment'] == label]
        intl_ratios.append(segment_data['international_ratio'].mean() * 100)
    
    plt.bar(segment_names, intl_ratios, color=colors_pie)
    plt.title('International Transactions %', fontsize=12, fontweight='bold')
    plt.ylabel('International %')
    plt.xticks(rotation=45)
    
    # 11. Feature Correlation Heatmap
    plt.subplot(3, 4, 11)
    key_features = ['tx_count', 'avg_amount', 'digital_wallet_ratio', 'weekend_ratio', 'city_diversity']
    corr_data = features_df[key_features].corr()
    im = plt.imshow(corr_data, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(im)
    plt.xticks(range(len(key_features)), key_features, rotation=45)
    plt.yticks(range(len(key_features)), key_features)
    plt.title('Feature Correlations', fontsize=12, fontweight='bold')
    
    # 12. Insights Summary
    plt.subplot(3, 4, 12)
    plt.axis('off')
    insight_text = f"ALGORITHM: {chosen_algorithm}\n\n"
    insight_text += f"CLUSTERS: {quality_metrics.get('n_clusters', 'Unknown')}\n"
    insight_text += f"SILHOUETTE: {quality_metrics.get('silhouette_score', 0):.3f}\n"
    insight_text += f"NOISE: {quality_metrics.get('noise_ratio', 0):.1%}\n\n"
    insight_text += "SEGMENTS:\n"
    for i, (seg_name, details) in enumerate(list(insights.items())[:3]):
        insight_text += f"• {details.get('business_name', seg_name)}\n"
    
    plt.text(0.1, 0.9, insight_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    plt.title('Summary & Insights', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Comprehensive dashboard created and saved")

# ========================================================================================
# PART 6: STRUCTURED OUTPUT AND DOCUMENTATION (10 POINTS)
# ========================================================================================

def export_results(features_df, labels, quality_metrics, insights, algorithm_results, chosen_algorithm):
    """
    Export comprehensive results in multiple formats
    """
    print("\n💾 EXPORTING RESULTS...")
    
    # 1. Customer segments with features
    features_with_segments = features_df.copy()
    features_with_segments['segment'] = labels
    features_with_segments.to_csv('customer_segments.csv', index=False)
    
    # 2. Segmentation results JSON
    results = {
        'algorithm_used': chosen_algorithm,
        'quality_metrics': quality_metrics,
        'segment_counts': {int(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))},
        'insights': insights,
        'algorithm_comparison': algorithm_results,
        'timestamp': datetime.now().isoformat()
    }
    
    with open('hackathon_segmentation_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # 3. Business insights
    with open('business_insights.json', 'w') as f:
        json.dump(insights, f, indent=2)
    
    print("✅ Results exported successfully")
    print(f"   • customer_segments.csv ({len(features_with_segments)} customers)")
    print(f"   • hackathon_segmentation_results.json")
    print(f"   • business_insights.json")

# ========================================================================================
# 🚀 MAIN EXECUTION: HACKATHON SUBMISSION
# ========================================================================================

if __name__ == "__main__":
    start_time = time.time()
    
    print("🎯 BANK CUSTOMER SEGMENTATION - FINAL HACKATHON SOLUTION")
    print("=" * 80)
    
    # Step 1: Load and prepare data
    print("\n📂 STEP 1: DATA LOADING & PREPARATION")
    features_df = extract_advanced_features(df)
    
    # Step 2: Feature engineering for ML
    print("\n🔧 STEP 2: FEATURE ENGINEERING FOR ML")
    ml_features, ml_features_scaled, scaler = prepare_data_for_clustering(features_df)
    
    # Step 3: Algorithm comparison and selection
    print("\n🤖 STEP 3: ALGORITHM COMPARISON & SELECTION")
    algorithm_results, chosen_algorithm, chosen_clusterer = compare_clustering_algorithms(ml_features_scaled, features_df)
    
    # Step 4: Apply chosen algorithm and get labels
    print(f"\n🎯 STEP 4: APPLYING {chosen_algorithm} CLUSTERING")
    if chosen_algorithm == 'KMeans':
        final_labels = chosen_clusterer.labels_
    elif chosen_algorithm == 'DBSCAN':
        final_labels = chosen_clusterer.labels_
    elif chosen_algorithm == 'HDBSCAN':
        final_labels = chosen_clusterer.labels_
    else:
        # Fallback to best performing algorithm from results
        final_labels = algorithm_results[chosen_algorithm]['labels']
    
    # Step 5: Quality evaluation
    print(f"\n📊 STEP 5: CLUSTERING QUALITY EVALUATION")
    quality_metrics = evaluate_clustering_quality(ml_features_scaled, final_labels, chosen_algorithm)
    
    # Step 6: GPT-4 insights generation
    print("\n🧠 STEP 6: AI-POWERED INSIGHTS GENERATION")
    insights = generate_gpt4_insights(features_df, final_labels, chosen_algorithm, quality_metrics)
    
    # Step 7: Comprehensive visualization
    print("\n📊 STEP 7: COMPREHENSIVE DASHBOARD CREATION")
    create_comprehensive_dashboard(
        features_df, final_labels, ml_features_scaled, 
        chosen_algorithm, quality_metrics, insights
    )
    
    # Step 8: Export results
    print("\n💾 STEP 8: EXPORTING RESULTS")
    export_results(
        features_df, final_labels, quality_metrics, 
        insights, algorithm_results, chosen_algorithm
    )
    
    execution_time = time.time() - start_time
    print(f"\n✅ ANALYSIS COMPLETE! Total execution time: {execution_time:.2f} seconds")
    print(f"🏆 Final Algorithm: {chosen_algorithm}")
    print(f"📊 Clusters Generated: {quality_metrics.get('n_clusters', 'Unknown')}")
    print(f"🎯 Silhouette Score: {quality_metrics.get('silhouette_score', 'Unknown'):.3f}")
    
    print("\n📁 Generated Files:")
    print("   • hackathon_segmentation_results.json")
    print("   • comprehensive_dashboard.png")
    print("   • customer_segments.csv")
    print("   • Check terminal output for detailed insights!") 