import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap

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
