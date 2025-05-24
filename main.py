# 🏆 HACKATHON FINAL SOLUTION: Advanced Customer Segmentation with Optimized HDBSCAN
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

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path='../.env')  # Load from parent directory
    print("✅ Environment variables loaded from .env file")
except ImportError:
    print("⚠️ python-dotenv not installed, using system environment variables only")
except Exception as e:
    print(f"⚠️ Could not load .env file: {e}")

# Machine Learning Libraries
import hdbscan
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import umap
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import PowerTransformer

# OpenAI for insights generation
from openai import OpenAI

from features.feature_engineering import extract_advanced_features
from clustering.hdbscan_optimizer import optimize_hdbscan_clustering
from clustering.quality_metrics import evaluate_clustering_quality
from insights.gpt_insights import generate_gpt4_insights
from insights.visualization import create_comprehensive_dashboard
from utils.export_results import export_results

print("🏆 HACKATHON SOLUTION: Advanced Customer Segmentation")
print("=" * 60)

# Global configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'your-api-key-here')
if OPENAI_API_KEY == 'your-api-key-here':
    print("⚠️ OpenAI API key not found in environment variables")
    print("💡 Please add your API key to .env file: OPENAI_API_KEY=your-actual-key")
else:
    print("✅ OpenAI API key loaded successfully")

if __name__ == "__main__":
    start_time = time.time()
    
    print("🎯 BANK CUSTOMER SEGMENTATION - FINAL HACKATHON SOLUTION")
    print("=" * 80)
    
    # Step 1: Load and prepare data
    print("\n📂 STEP 1: DATA LOADING & PREPARATION")
    df = pd.read_csv("data/transactions.csv", parse_dates=['transaction_timestamp'])
    features_df = extract_advanced_features(df)
    
    # Step 2: Feature engineering for ML
    print("\n🔧 STEP 2: FEATURE ENGINEERING FOR ML")
    ml_features = features_df.drop(['card_id'], axis=1)
    scaler = RobustScaler()
    ml_features_scaled = scaler.fit_transform(ml_features)
    
    # Step 3: Ultra-optimized HDBSCAN clustering
    print("\n🚀 STEP 3: ULTRA-OPTIMIZED HDBSCAN CLUSTERING")
    algorithm_results, chosen_algorithm, final_clusterer = optimize_hdbscan_clustering(ml_features_scaled, features_df)
    
    # Step 4: Get final labels from ultra-optimized HDBSCAN
    print(f"\n🎯 STEP 4: EXTRACTING {chosen_algorithm} RESULTS")
    final_labels = algorithm_results[chosen_algorithm]['labels']
    
    print(f"   ✅ Final clusters: {algorithm_results[chosen_algorithm]['n_clusters']}")
    print(f"   ✅ Noise ratio: {algorithm_results[chosen_algorithm]['noise_ratio']:.1%}")
    print(f"   ✅ Silhouette score: {algorithm_results[chosen_algorithm]['silhouette']:.3f}")
    if 'feature_set_used' in algorithm_results[chosen_algorithm]:
        print(f"   ✅ Feature set used: {algorithm_results[chosen_algorithm]['feature_set_used']}")
    
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
