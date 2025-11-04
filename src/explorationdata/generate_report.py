#!/usr/bin/env python
"""
Generate a comprehensive markdown report from pipeline results
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd


def load_summary(output_dir: Path) -> dict:
    """Load pipeline summary"""
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            return json.load(f)
    return {}


def load_statistics(output_dir: Path) -> pd.DataFrame:
    """Load image statistics"""
    stats_path = output_dir / "tables" / "image_stats.csv"
    if stats_path.exists():
        return pd.read_csv(stats_path)
    return pd.DataFrame()


def load_clusters(output_dir: Path) -> pd.DataFrame:
    """Load clustering results"""
    clusters_path = output_dir / "tables" / "clusters.csv"
    if clusters_path.exists():
        return pd.read_csv(clusters_path)
    return pd.DataFrame()


def generate_report(output_dir: Path, output_file: Path):
    """Generate comprehensive markdown report"""

    summary = load_summary(output_dir)
    df_stats = load_statistics(output_dir)
    df_clusters = load_clusters(output_dir)

    # Start report
    report = []
    report.append("# Comprehensive EDA Report - COVID-19 Radiography Dataset")
    report.append("")
    report.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"**Output Directory**: `{output_dir}`")
    report.append("")
    report.append("---")
    report.append("")

    # Executive Summary
    report.append("## Executive Summary")
    report.append("")
    report.append("This report presents a comprehensive exploratory data analysis of the COVID-19 radiography dataset, "
                  "including statistical analysis, deep learning embeddings, dimensionality reduction, and clustering.")
    report.append("")

    # Dataset Overview
    report.append("## 1. Dataset Overview")
    report.append("")

    if summary:
        report.append(f"- **Total Images**: {summary.get('total_images', 'N/A')}")
        report.append(f"- **Corrupted Images**: {summary.get('corrupted_images', 0)}")
        report.append(f"- **Classes**: {', '.join(summary.get('classes', []))}")
        report.append(f"- **Random Seed**: {summary.get('seed', 'N/A')}")
        report.append(f"- **Device Used**: {summary.get('device', 'N/A')}")
        report.append("")

    # Class Distribution
    if not df_stats.empty:
        report.append("### Class Distribution")
        report.append("")
        class_counts = df_stats['class'].value_counts()
        report.append("| Class | Count | Percentage |")
        report.append("|-------|-------|------------|")
        total = len(df_stats)
        for cls, count in class_counts.items():
            pct = (count / total) * 100
            report.append(f"| {cls} | {count} | {pct:.2f}% |")
        report.append("")

    # Image Statistics
    report.append("## 2. Image Statistics")
    report.append("")

    if not df_stats.empty:
        report.append("### Dimensions")
        report.append("")
        report.append(f"- **Average Width**: {df_stats['width'].mean():.1f} px (±{df_stats['width'].std():.1f})")
        report.append(f"- **Average Height**: {df_stats['height'].mean():.1f} px (±{df_stats['height'].std():.1f})")
        report.append(f"- **Width Range**: {df_stats['width'].min():.0f} - {df_stats['width'].max():.0f} px")
        report.append(f"- **Height Range**: {df_stats['height'].min():.0f} - {df_stats['height'].max():.0f} px")
        report.append("")

        report.append("### Intensity Statistics")
        report.append("")
        report.append(f"- **Average Mean Intensity**: {df_stats['mean'].mean():.2f} (±{df_stats['mean'].std():.2f})")
        report.append(f"- **Average Std Intensity**: {df_stats['std'].mean():.2f} (±{df_stats['std'].std():.2f})")
        report.append("")

        # Per-class statistics
        report.append("### Statistics by Class")
        report.append("")
        report.append("| Class | Avg Width | Avg Height | Avg Mean Intensity | Avg Std Intensity |")
        report.append("|-------|-----------|------------|-------------------|-------------------|")
        for cls in sorted(df_stats['class'].unique()):
            cls_df = df_stats[df_stats['class'] == cls]
            report.append(f"| {cls} | {cls_df['width'].mean():.1f} | {cls_df['height'].mean():.1f} | "
                         f"{cls_df['mean'].mean():.2f} | {cls_df['std'].mean():.2f} |")
        report.append("")

    # Embeddings
    report.append("## 3. Deep Learning Embeddings")
    report.append("")
    if summary and 'embedding_shape' in summary:
        shape = summary['embedding_shape']
        report.append(f"- **Model**: ResNet50 (pre-trained on ImageNet)")
        report.append(f"- **Embedding Shape**: {shape[0]} images × {shape[1]} features")
        report.append(f"- **Device**: {summary.get('device', 'N/A')}")
        report.append("")

    # Dimensionality Reduction
    report.append("## 4. Dimensionality Reduction")
    report.append("")

    pca_path = output_dir / "tables" / "pca_top10.csv"
    if pca_path.exists():
        df_pca = pd.read_csv(pca_path)
        report.append("### PCA (Principal Component Analysis)")
        report.append("")
        report.append(f"- **Components**: 50")
        report.append(f"- **Variance explained by top 10**: {df_pca['cumulative_variance'].iloc[9]:.2%}")
        report.append("")

    report.append("### UMAP")
    report.append("")
    report.append("- **Parameters**: n_neighbors=15, min_dist=0.1")
    report.append("- **Output**: 2D projection for visualization")
    report.append("")

    report.append("### t-SNE")
    report.append("")
    report.append("- **Parameters**: perplexity=30")
    report.append("- **Sample Size**: max 5000 images (for computational efficiency)")
    report.append("")

    # Clustering
    report.append("## 5. Clustering Analysis")
    report.append("")

    if summary and 'kmeans_metrics' in summary:
        km = summary['kmeans_metrics']
        report.append("### KMeans Clustering")
        report.append("")
        report.append(f"- **Number of Clusters**: {len(df_stats['class'].unique()) if not df_stats.empty else 'N/A'}")
        report.append(f"- **Adjusted Rand Index (ARI)**: {km.get('ari', 0):.3f}")
        report.append(f"- **Normalized Mutual Information (NMI)**: {km.get('nmi', 0):.3f}")
        report.append("")

    if summary and 'dbscan_metrics' in summary:
        db = summary['dbscan_metrics']
        report.append("### DBSCAN Clustering")
        report.append("")
        report.append(f"- **Parameters**: eps=0.5, min_samples=5")
        report.append(f"- **Adjusted Rand Index (ARI)**: {db.get('ari', 0):.3f}")
        report.append(f"- **Normalized Mutual Information (NMI)**: {db.get('nmi', 0):.3f}")
        report.append("")

    # Key Observations
    report.append("## 6. Key Observations")
    report.append("")

    observations = []

    # Data quality
    if summary:
        corrupted = summary.get('corrupted_images', 0)
        total = summary.get('total_images', 0)
        if corrupted > 0:
            pct = (corrupted / total) * 100 if total > 0 else 0
            observations.append(f"- **Data Quality**: Found {corrupted} corrupted images ({pct:.2f}% of dataset)")
        else:
            observations.append("- **Data Quality**: All images loaded successfully, no corrupted files detected")

    # Class balance
    if not df_stats.empty:
        class_counts = df_stats['class'].value_counts()
        max_count = class_counts.max()
        min_count = class_counts.min()
        imbalance_ratio = max_count / min_count if min_count > 0 else 0
        if imbalance_ratio > 2:
            observations.append(f"- **Class Imbalance**: Dataset shows class imbalance (ratio: {imbalance_ratio:.2f}:1)")
        else:
            observations.append("- **Class Balance**: Dataset is relatively well-balanced across classes")

    # Clustering performance
    if summary and 'kmeans_metrics' in summary:
        ari = summary['kmeans_metrics'].get('ari', 0)
        if ari > 0.5:
            observations.append(f"- **Clustering**: Strong clustering performance (ARI={ari:.3f}), suggesting distinct visual patterns per class")
        elif ari > 0.3:
            observations.append(f"- **Clustering**: Moderate clustering performance (ARI={ari:.3f}), some overlap between classes")
        else:
            observations.append(f"- **Clustering**: Weak clustering performance (ARI={ari:.3f}), significant overlap between classes")

    # Dimensionality
    if pca_path.exists():
        df_pca = pd.read_csv(pca_path)
        var_10 = df_pca['cumulative_variance'].iloc[9]
        if var_10 > 0.8:
            observations.append(f"- **Dimensionality**: High variance captured by top 10 PCs ({var_10:.2%}), suggesting strong latent structure")
        else:
            observations.append(f"- **Dimensionality**: Moderate variance by top 10 PCs ({var_10:.2%}), complex feature space")

    for obs in observations:
        report.append(obs)
    report.append("")

    # Recommendations
    report.append("## 7. Recommendations for Further Analysis")
    report.append("")
    report.append("1. **Feature Engineering**: Explore additional features from masked regions")
    report.append("2. **Data Augmentation**: Consider augmentation strategies to balance classes")
    report.append("3. **Deep Learning**: Train CNN classifiers using the extracted embeddings")
    report.append("4. **Attention Mechanisms**: Investigate Grad-CAM to understand model focus")
    report.append("5. **Cross-validation**: Validate clustering results with k-fold cross-validation")
    report.append("6. **Ensemble Methods**: Combine multiple clustering algorithms for robust results")
    report.append("")

    # Visualizations
    report.append("## 8. Generated Visualizations")
    report.append("")

    figures_dir = output_dir / "figures"
    if figures_dir.exists():
        figures = sorted(figures_dir.glob("*.png"))
        if figures:
            report.append("The following visualizations have been generated:")
            report.append("")
            for fig in figures:
                report.append(f"- `{fig.name}`")
            report.append("")

    # Performance
    report.append("## 9. Pipeline Performance")
    report.append("")
    if summary and 'total_time_seconds' in summary:
        total_time = summary['total_time_seconds']
        report.append(f"- **Total Execution Time**: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

        if 'timings' in summary:
            report.append("")
            report.append("### Step Timings")
            report.append("")
            report.append("| Step | Time (seconds) |")
            report.append("|------|----------------|")
            for step, time_s in summary['timings'].items():
                report.append(f"| {step} | {time_s:.2f} |")
        report.append("")

    # Conclusion
    report.append("## 10. Conclusion")
    report.append("")
    report.append("This comprehensive EDA provides valuable insights into the COVID-19 radiography dataset. "
                  "The analysis reveals patterns in image characteristics, class distributions, and relationships "
                  "between different pathology classes. The embeddings and dimensionality reductions can be used "
                  "for downstream tasks such as classification, anomaly detection, and model training.")
    report.append("")
    report.append("---")
    report.append("")
    report.append(f"*Report generated by EDA Pipeline v1.0.0*")

    # Write report
    report_text = "\n".join(report)
    with open(output_file, 'w') as f:
        f.write(report_text)

    print(f"Report generated: {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Generate comprehensive report from EDA pipeline results"
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Path to pipeline output directory'
    )

    parser.add_argument(
        '--report-file',
        type=str,
        default=None,
        help='Output report file path (default: output_dir/report.md)'
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"Error: Output directory does not exist: {output_dir}")
        return 1

    if args.report_file:
        report_file = Path(args.report_file)
    else:
        report_file = output_dir / "report.md"

    generate_report(output_dir, report_file)
    return 0


if __name__ == "__main__":
    exit(main())
