import json
import pandas as pd
from datetime import datetime
import numpy as np

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
