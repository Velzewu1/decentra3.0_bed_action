#!/usr/bin/env python3
"""
Модуль визуализации кластеров для хакатона
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Настройка стиля
sns.set_theme()
sns.set_palette("husl")

def create_cluster_visualizations(features_df: pd.DataFrame, ml_features_processed: np.ndarray, 
                                labels: np.ndarray, output_dir: str = "."):
    """
    Создание комплексной визуализации кластеров для хакатона
    """
    
    print("🎨 СОЗДАНИЕ ВИЗУАЛИЗАЦИИ КЛАСТЕРОВ...")
    print("="*50)
    
    # Добавляем сегменты к данным
    features_with_clusters = features_df.copy()
    features_with_clusters['cluster'] = labels
    
    # 1. Обзорная визуализация кластеров
    create_cluster_overview(features_with_clusters, output_dir)
    
    # 2. PCA визуализация в 2D и 3D
    create_pca_visualization(ml_features_processed, labels, output_dir)
    
    # 3. t-SNE визуализация
    create_tsne_visualization(ml_features_processed, labels, output_dir)
    
    # 4. Бизнес-метрики по кластерам
    create_business_metrics_plots(features_with_clusters, output_dir)
    
    # 5. Детальные характеристики кластеров
    create_cluster_characteristics(features_with_clusters, output_dir)
    
    # 6. Интерактивная визуализация (Plotly)
    create_interactive_plots(features_with_clusters, ml_features_processed, labels, output_dir)
    
    print("✅ Все визуализации созданы успешно!")
    return features_with_clusters

def create_cluster_overview(df: pd.DataFrame, output_dir: str):
    """Обзорная визуализация размеров и характеристик кластеров"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('🎯 Обзор кластеров клиентов', fontsize=16, fontweight='bold')
    
    # 1. Размеры кластеров
    cluster_sizes = df['cluster'].value_counts().sort_index()
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    axes[0,0].pie(cluster_sizes.values, labels=[f'Кластер {i}' for i in cluster_sizes.index], 
                  autopct='%1.1f%%', colors=colors, startangle=90)
    axes[0,0].set_title('Распределение клиентов по кластерам')
    
    # 2. Количество клиентов
    cluster_sizes.plot(kind='bar', ax=axes[0,1], color=colors)
    axes[0,1].set_title('Количество клиентов в кластерах')
    axes[0,1].set_xlabel('Кластер')
    axes[0,1].set_ylabel('Количество клиентов')
    
    # 3. Средний чек по кластерам
    avg_amount = df.groupby('cluster')['avg_amount'].mean()
    avg_amount.plot(kind='bar', ax=axes[1,0], color=colors)
    axes[1,0].set_title('Средний чек по кластерам')
    axes[1,0].set_xlabel('Кластер')
    axes[1,0].set_ylabel('Средний чек (тенге)')
    
    # 4. Количество транзакций
    avg_tx = df.groupby('cluster')['tx_count'].mean()
    avg_tx.plot(kind='bar', ax=axes[1,1], color=colors)
    axes[1,1].set_title('Среднее количество транзакций')
    axes[1,1].set_xlabel('Кластер')
    axes[1,1].set_ylabel('Количество транзакций')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/cluster_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Обзорная визуализация создана")

def create_pca_visualization(X: np.ndarray, labels: np.ndarray, output_dir: str):
    """PCA визуализация в 2D и 3D"""
    
    # PCA 2D
    pca_2d = PCA(n_components=2, random_state=42)
    X_pca_2d = pca_2d.fit_transform(X)
    
    # PCA 3D
    pca_3d = PCA(n_components=3, random_state=42)
    X_pca_3d = pca_3d.fit_transform(X)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('📊 PCA Визуализация кластеров', fontsize=14, fontweight='bold')
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    # 2D PCA
    for i, color in enumerate(colors):
        mask = labels == i
        axes[0].scatter(X_pca_2d[mask, 0], X_pca_2d[mask, 1], 
                       c=color, label=f'Кластер {i}', alpha=0.7, s=50)
    
    axes[0].set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%} variance)')
    axes[0].set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%} variance)')
    axes[0].set_title('PCA 2D')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Объясненная дисперсия
    cumsum_var = np.cumsum(pca_3d.explained_variance_ratio_)
    axes[1].plot(range(1, len(cumsum_var)+1), cumsum_var, 'bo-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Количество компонент')
    axes[1].set_ylabel('Объясненная дисперсия')
    axes[1].set_title('Объясненная дисперсия PCA')
    axes[1].grid(True, alpha=0.3)
    
    # Добавляем аннотации
    for i, var in enumerate(cumsum_var):
        axes[1].annotate(f'{var:.1%}', (i+1, var), textcoords="offset points", 
                        xytext=(0,10), ha='center')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/pca_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ PCA визуализация создана")

def create_tsne_visualization(X: np.ndarray, labels: np.ndarray, output_dir: str):
    """t-SNE визуализация для лучшего разделения кластеров"""
    
    print("   🔄 Выполняется t-SNE (может занять время)...")
    
    # t-SNE с разными параметрами
    tsne_30 = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne_30 = tsne_30.fit_transform(X)
    
    tsne_50 = TSNE(n_components=2, perplexity=50, random_state=42)
    X_tsne_50 = tsne_50.fit_transform(X)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('🧠 t-SNE Визуализация кластеров', fontsize=14, fontweight='bold')
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    # t-SNE perplexity=30
    for i, color in enumerate(colors):
        mask = labels == i
        axes[0].scatter(X_tsne_30[mask, 0], X_tsne_30[mask, 1], 
                       c=color, label=f'Кластер {i}', alpha=0.7, s=50)
    
    axes[0].set_title('t-SNE (perplexity=30)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # t-SNE perplexity=50
    for i, color in enumerate(colors):
        mask = labels == i
        axes[1].scatter(X_tsne_50[mask, 0], X_tsne_50[mask, 1], 
                       c=color, label=f'Кластер {i}', alpha=0.7, s=50)
    
    axes[1].set_title('t-SNE (perplexity=50)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/tsne_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ t-SNE визуализация создана")

def create_business_metrics_plots(df: pd.DataFrame, output_dir: str):
    """Бизнес-метрики по кластерам"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('💼 Бизнес-характеристики кластеров', fontsize=16, fontweight='bold')
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    # 1. Digital Wallet использование
    digital_wallet = df.groupby('cluster')['digital_wallet_ratio'].mean()
    digital_wallet.plot(kind='bar', ax=axes[0,0], color=colors)
    axes[0,0].set_title('Использование Digital Wallet')
    axes[0,0].set_ylabel('Доля (%)')
    axes[0,0].tick_params(axis='x', rotation=0)
    
    # 2. Contactless платежи
    contactless = df.groupby('cluster')['contactless_ratio'].mean()
    contactless.plot(kind='bar', ax=axes[0,1], color=colors)
    axes[0,1].set_title('Contactless платежи')
    axes[0,1].set_ylabel('Доля (%)')
    axes[0,1].tick_params(axis='x', rotation=0)
    
    # 3. Международные транзакции
    international = df.groupby('cluster')['international_ratio'].mean()
    international.plot(kind='bar', ax=axes[0,2], color=colors)
    axes[0,2].set_title('Международные транзакции')
    axes[0,2].set_ylabel('Доля (%)')
    axes[0,2].tick_params(axis='x', rotation=0)
    
    # 4. Географическое разнообразие
    city_diversity = df.groupby('cluster')['city_diversity'].mean()
    city_diversity.plot(kind='bar', ax=axes[1,0], color=colors)
    axes[1,0].set_title('Географическое разнообразие')
    axes[1,0].set_ylabel('Количество городов')
    axes[1,0].tick_params(axis='x', rotation=0)
    
    # 5. Платежная изощренность
    payment_soph = df.groupby('cluster')['payment_sophistication'].mean()
    payment_soph.plot(kind='bar', ax=axes[1,1], color=colors)
    axes[1,1].set_title('Платежная изощренность')
    axes[1,1].set_ylabel('Индекс')
    axes[1,1].tick_params(axis='x', rotation=0)
    
    # 6. Активность по дням
    days_active = df.groupby('cluster')['days_active'].mean()
    days_active.plot(kind='bar', ax=axes[1,2], color=colors)
    axes[1,2].set_title('Активные дни')
    axes[1,2].set_ylabel('Количество дней')
    axes[1,2].tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/business_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Бизнес-метрики созданы")

def create_cluster_characteristics(df: pd.DataFrame, output_dir: str):
    """Детальные характеристики кластеров"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('📈 Детальные характеристики кластеров', fontsize=16, fontweight='bold')
    
    # 1. Boxplot средних чеков
    sns.boxplot(data=df, x='cluster', y='avg_amount', ax=axes[0,0])
    axes[0,0].set_title('Распределение средних чеков')
    axes[0,0].set_ylabel('Средний чек (тенге)')
    
    # 2. Boxplot количества транзакций
    sns.boxplot(data=df, x='cluster', y='tx_count', ax=axes[0,1])
    axes[0,1].set_title('Распределение количества транзакций')
    axes[0,1].set_ylabel('Количество транзакций')
    
    # 3. Heatmap корреляций ключевых метрик
    key_metrics = ['avg_amount', 'tx_count', 'digital_wallet_ratio', 
                   'contactless_ratio', 'city_diversity', 'payment_sophistication']
    
    correlation_data = []
    for cluster in sorted(df['cluster'].unique()):
        cluster_data = df[df['cluster'] == cluster][key_metrics].mean()
        correlation_data.append(cluster_data.values)
    
    correlation_df = pd.DataFrame(correlation_data, 
                                 columns=key_metrics,
                                 index=[f'Кластер {i}' for i in sorted(df['cluster'].unique())])
    
    sns.heatmap(correlation_df.T, annot=True, fmt='.3f', ax=axes[1,0], cmap='RdYlBu_r')
    axes[1,0].set_title('Средние значения метрик по кластерам')
    
    # 4. Violin plot для digital wallet
    sns.violinplot(data=df, x='cluster', y='digital_wallet_ratio', ax=axes[1,1])
    axes[1,1].set_title('Распределение Digital Wallet по кластерам')
    axes[1,1].set_ylabel('Digital Wallet Ratio')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/cluster_characteristics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Детальные характеристики созданы")

def create_interactive_plots(df: pd.DataFrame, X: np.ndarray, labels: np.ndarray, output_dir: str):
    """Интерактивные графики с Plotly (отключены для упрощения)"""
    # Отключено для упрощения - создаем только PNG файлы
    print("✅ Интерактивные графики пропущены (создаются только PNG)")

def generate_cluster_summary_table(df: pd.DataFrame, output_dir: str):
    """Генерация сводной таблицы характеристик кластеров (упрощенная)"""
    # Возвращаем только DataFrame без сохранения в CSV
    summary_stats = []
    
    for cluster in sorted(df['cluster'].unique()):
        cluster_data = df[df['cluster'] == cluster]
        
        stats = {
            'Кластер': f'Кластер {cluster}',
            'Размер': len(cluster_data),
            'Доля (%)': f"{len(cluster_data) / len(df) * 100:.1f}%",
            'Средний чек (тенге)': f"{cluster_data['avg_amount'].mean():,.0f}",
            'Медианный чек (тенге)': f"{cluster_data['avg_amount'].median():,.0f}",
            'Ср. количество транзакций': f"{cluster_data['tx_count'].mean():.0f}",
            'Digital Wallet (%)': f"{cluster_data['digital_wallet_ratio'].mean():.1%}",
            'Contactless (%)': f"{cluster_data['contactless_ratio'].mean():.1%}",
            'Международные (%)': f"{cluster_data['international_ratio'].mean():.1%}",
            'Активные дни': f"{cluster_data['days_active'].mean():.0f}",
            'Географическое разнообразие': f"{cluster_data['city_diversity'].mean():.1f}",
            'Платежная изощренность': f"{cluster_data['payment_sophistication'].mean():.2f}"
        }
        
        summary_stats.append(stats)
    
    summary_df = pd.DataFrame(summary_stats)
    print("✅ Сводная таблица создана (только в памяти)")
    return summary_df 