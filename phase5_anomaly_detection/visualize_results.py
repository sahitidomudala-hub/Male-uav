"""
Phase 5: Visualization and Threshold Sanity Check
==================================================

Creates visualizations of reconstruction error distributions and 
anomaly detection results using existing Phase 5 classifications.

NO data modification. NO retraining. NO threshold changes.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
import pipeline_config as _pc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def create_distribution_comparison(val_errors_path: str,
                                   test_errors_path: str,
                                   threshold: float,
                                   output_dir: str):
    """
    Create histogram comparing validation and test error distributions.
    
    Args:
        val_errors_path: Path to validation errors CSV
        test_errors_path: Path to test errors CSV
        threshold: Threshold value (0.146577)
        output_dir: Output directory for plots
    """
    # Load data
    val_df = pd.read_csv(val_errors_path)
    test_df = pd.read_csv(test_errors_path)
    
    val_errors = val_df['reconstruction_error_mse'].values
    test_errors = test_df['reconstruction_error_mse'].values
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Overlaid histograms
    ax1 = axes[0]
    
    # Use same bins for both
    max_error = max(np.percentile(val_errors, 99), np.percentile(test_errors, 99))
    bins = np.linspace(0, max_error, 100)
    
    ax1.hist(val_errors, bins=bins, alpha=0.6, label='Validation', 
             color='blue', edgecolor='black', density=True)
    ax1.hist(test_errors, bins=bins, alpha=0.6, label='Test', 
             color='orange', edgecolor='black', density=True)
    
    # Threshold line
    ax1.axvline(threshold, color='red', linestyle='--', linewidth=2, 
                label=f'Threshold ({_pc.THRESHOLD_PERCENTILE:g}th %ile) = {threshold:.6f}')
    
    ax1.set_xlabel('Reconstruction Error (MSE)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax1.set_title('Validation vs Test Reconstruction Error Distributions', 
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Box plots
    ax2 = axes[1]
    
    data_to_plot = [val_errors, test_errors]
    positions = [1, 2]
    
    bp = ax2.boxplot(data_to_plot, positions=positions, widths=0.6,
                     patch_artist=True, showmeans=True,
                     meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
    
    # Color boxes
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightsalmon')
    
    # Threshold line
    ax2.axhline(threshold, color='red', linestyle='--', linewidth=2,
                label=f'Threshold = {threshold:.6f}')
    
    ax2.set_xticks(positions)
    ax2.set_xticklabels(['Validation', 'Test'], fontsize=12, fontweight='bold')
    ax2.set_ylabel('Reconstruction Error (MSE)', fontsize=12, fontweight='bold')
    ax2.set_title('Reconstruction Error Distribution Comparison (Box Plots)', 
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add statistics
    stats_text = f"""Validation: n={len(val_errors)}, mean={val_errors.mean():.6f}, median={np.median(val_errors):.6f}
Test: n={len(test_errors)}, mean={test_errors.mean():.6f}, median={np.median(test_errors):.6f}"""
    
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'error_distribution_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")
    
    return val_errors, test_errors


def create_anomaly_timeline(results_path: str,
                            threshold: float,
                            output_dir: str):
    """
    Create timeline plot of test reconstruction errors with anomaly markers.
    
    Args:
        results_path: Path to anomaly detection results CSV
        threshold: Threshold value
        output_dir: Output directory
    """
    # Load results
    results_df = pd.read_csv(results_path)
    
    # Separate normal and anomaly
    normal_df = results_df[results_df['classification'] == 'Normal']
    anomaly_df = results_df[results_df['classification'] == 'Anomaly']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Plot normal sequences
    ax.scatter(normal_df['sequence_index'], normal_df['reconstruction_error'],
               c='blue', alpha=0.5, s=20, label=f'Normal (n={len(normal_df)})',
               edgecolors='none')
    
    # Plot anomaly sequences
    ax.scatter(anomaly_df['sequence_index'], anomaly_df['reconstruction_error'],
               c='red', alpha=0.7, s=30, label=f'Anomaly (n={len(anomaly_df)})',
               edgecolors='darkred', linewidths=0.5)
    
    # Threshold line
    ax.axhline(threshold, color='darkred', linestyle='--', linewidth=2,
               label=f'Threshold = {threshold:.6f}', zorder=100)
    
    ax.set_xlabel('Test Sequence Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Reconstruction Error (MSE)', fontsize=12, fontweight='bold')
    ax.set_title('Test Sequence Reconstruction Errors: Normal vs Anomaly Classification',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Add anomaly percentage
    anomaly_pct = (len(anomaly_df) / len(results_df)) * 100
    ax.text(0.02, 0.98, f'Anomaly Rate: {anomaly_pct:.2f}%\n({len(anomaly_df)}/{len(results_df)} sequences)',
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'anomaly_timeline.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")


def create_summary_visualization(results_path: str, output_dir: str):
    """
    Create simple bar chart summarizing classification results.
    
    Args:
        results_path: Path to anomaly detection results CSV
        output_dir: Output directory
    """
    # Load results
    results_df = pd.read_csv(results_path)
    
    # Count classifications
    counts = results_df['classification'].value_counts()
    normal_count = counts.get('Normal', 0)
    anomaly_count = counts.get('Anomaly', 0)
    total = len(results_df)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    categories = ['Normal', 'Anomaly']
    values = [normal_count, anomaly_count]
    colors = ['#2E86AB', '#A23B72']
    
    bars = ax.bar(categories, values, color=colors, edgecolor='black', linewidth=2)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        percentage = (value / total) * 100
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value}\n({percentage:.2f}%)',
                ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    ax.set_ylabel('Number of Sequences', fontsize=12, fontweight='bold')
    ax.set_title(f'Anomaly Detection Results Summary\n(Total: {total} test sequences)',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Set y-axis to start at 0
    ax.set_ylim(0, max(values) * 1.15)
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'classification_summary.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")


def perform_sanity_check(val_errors: np.ndarray,
                        test_errors: np.ndarray,
                        threshold: float) -> dict:
    """
    Perform threshold sanity check by analyzing distributions.
    
    Args:
        val_errors: Validation reconstruction errors
        test_errors: Test reconstruction errors
        threshold: Current threshold value
        
    Returns:
        Dictionary with sanity check findings
    """
    # Calculate statistics
    val_stats = {
        'mean': val_errors.mean(),
        'median': np.median(val_errors),
        'std': val_errors.std(),
        'min': val_errors.min(),
        'max': val_errors.max(),
        'p95': np.percentile(val_errors, 95),
        'p99': np.percentile(val_errors, 99)
    }
    
    test_stats = {
        'mean': test_errors.mean(),
        'median': np.median(test_errors),
        'std': test_errors.std(),
        'min': test_errors.min(),
        'max': test_errors.max(),
        'p95': np.percentile(test_errors, 95),
        'p99': np.percentile(test_errors, 99)
    }
    
    # Anomaly rates
    val_anomaly_rate = (val_errors > threshold).sum() / len(val_errors) * 100
    test_anomaly_rate = (test_errors > threshold).sum() / len(test_errors) * 100
    
    # Distribution similarity (coefficient of variation)
    val_cv = val_stats['std'] / val_stats['mean']
    test_cv = test_stats['std'] / test_stats['mean']
    
    # Check for substantial differences
    mean_ratio = test_stats['mean'] / val_stats['mean']
    median_ratio = test_stats['median'] / val_stats['median']
    
    findings = {
        'validation_statistics': val_stats,
        'test_statistics': test_stats,
        'threshold_percentile_in_validation': _pc.THRESHOLD_PERCENTILE,
        'validation_anomaly_rate_pct': val_anomaly_rate,
        'test_anomaly_rate_pct': test_anomaly_rate,
        'distribution_comparison': {
            'val_mean': val_stats['mean'],
            'test_mean': test_stats['mean'],
            'mean_ratio_test_vs_val': mean_ratio,
            'val_median': val_stats['median'],
            'test_median': test_stats['median'],
            'median_ratio_test_vs_val': median_ratio,
            'val_cv': val_cv,
            'test_cv': test_cv
        },
        'observations': []
    }
    
    # Generate observations
    if mean_ratio > 2.0:
        findings['observations'].append(
            f"Test mean ({test_stats['mean']:.6f}) is {mean_ratio:.2f}x validation mean ({val_stats['mean']:.6f}) - SUBSTANTIAL DIFFERENCE"
        )
    elif mean_ratio > 1.5:
        findings['observations'].append(
            f"Test mean ({test_stats['mean']:.6f}) is {mean_ratio:.2f}x validation mean ({val_stats['mean']:.6f}) - Moderate difference"
        )
    else:
        findings['observations'].append(
            f"Test mean ({test_stats['mean']:.6f}) similar to validation mean ({val_stats['mean']:.6f}) - ratio {mean_ratio:.2f}"
        )
    
    if test_stats['max'] > val_stats['max'] * 10:
        findings['observations'].append(
            f"Test max error ({test_stats['max']:.2f}) is {test_stats['max']/val_stats['max']:.1f}x validation max ({val_stats['max']:.2f}) - Very high outliers in test"
        )
    
    findings['observations'].append(
        f"Threshold ({threshold:.6f}) is at {val_anomaly_rate:.2f}% of validation (by design) and {test_anomaly_rate:.2f}% of test"
    )
    
    if 15.0 <= test_anomaly_rate <= 25.0:
        findings['observations'].append(
            f"Test anomaly rate ({test_anomaly_rate:.2f}%) is in reasonable range for investigation"
        )
    elif test_anomaly_rate > 50:
        findings['observations'].append(
            f"WARNING: Test anomaly rate ({test_anomaly_rate:.2f}%) is very high - may indicate distribution shift or inappropriate threshold"
        )
    
    return findings


def main():
    base_dir = Path('.')
    output_dir = base_dir / 'outputs'
    
    # File paths
    val_errors_path = output_dir / 'reconstruction_errors_val.csv'
    test_errors_path = output_dir / 'reconstruction_errors_test.csv'
    results_path = output_dir / 'anomaly_detection_results.csv'
    
    # Load threshold from report
    import json
    report_path = output_dir / 'anomaly_detection_report.json'
    with open(report_path, 'r') as f:
        report = json.load(f)
    threshold = report['threshold_calculation']['threshold_value']
    
    print("="*70)
    print("PHASE 5: VISUALIZATION + THRESHOLD SANITY CHECK")
    print("="*70)
    print(f"\nUsing existing threshold: {threshold:.6f}")
    print(f"NO modifications will be made to data or classifications.\n")
    
    # Create visualizations
    print("[1/3] Creating distribution comparison...")
    val_errors, test_errors = create_distribution_comparison(
        val_errors_path, test_errors_path, threshold, output_dir
    )
    
    print("\n[2/3] Creating anomaly timeline...")
    create_anomaly_timeline(results_path, threshold, output_dir)
    
    print("\n[3/3] Creating summary visualization...")
    create_summary_visualization(results_path, output_dir)
    
    # Perform sanity check
    print("\n" + "="*70)
    print("THRESHOLD SANITY CHECK")
    print("="*70)
    
    findings = perform_sanity_check(val_errors, test_errors, threshold)
    
    print("\nDistribution Statistics:")
    print(f"\nValidation (n={len(val_errors)}):")
    print(f"  Mean:   {findings['validation_statistics']['mean']:.6f}")
    print(f"  Median: {findings['validation_statistics']['median']:.6f}")
    print(f"  Std:    {findings['validation_statistics']['std']:.6f}")
    print(f"  Range:  [{findings['validation_statistics']['min']:.6f}, {findings['validation_statistics']['max']:.6f}]")
    
    print(f"\nTest (n={len(test_errors)}):")
    print(f"  Mean:   {findings['test_statistics']['mean']:.6f}")
    print(f"  Median: {findings['test_statistics']['median']:.6f}")
    print(f"  Std:    {findings['test_statistics']['std']:.6f}")
    print(f"  Range:  [{findings['test_statistics']['min']:.6f}, {findings['test_statistics']['max']:.6f}]")
    
    print("\nObservations:")
    for i, obs in enumerate(findings['observations'], 1):
        print(f"  {i}. {obs}")
    
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)
    
    # Generate recommendation
    mean_ratio = findings['distribution_comparison']['mean_ratio_test_vs_val']
    test_anomaly_rate = findings['test_anomaly_rate_pct']
    
    if mean_ratio > 3.0 or test_anomaly_rate > 40:
        print("⚠ CAUTION: Substantial difference between validation and test distributions.")
        print("   Consider investigating:")
        print("   - Whether test period contains different operating conditions")
        print("   - Whether validation set is representative of normal operation")
        print("   - Whether a different threshold percentile is appropriate")
    elif mean_ratio > 1.8:
        print("ℹ MODERATE: Test distribution shows higher errors than validation.")
        print("   This is not unusual but worth monitoring.")
        print(f"   The {test_anomaly_rate:.2f}% anomaly rate appears reasonable for investigation.")
    else:
        print("✓ REASONABLE: Distributions are similar.")
        print(f"   The {test_anomaly_rate:.2f}% anomaly rate appears appropriate.")
        print(f"   {_pc.THRESHOLD_PERCENTILE:g}th percentile threshold seems suitable for this dataset.")
    
    print("\nNOTE: Without ground-truth fault labels, we cannot validate whether")
    print("      detected anomalies correspond to actual faults.")
    
    # Save findings
    findings_path = output_dir / 'threshold_sanity_check.json'
    with open(findings_path, 'w') as f:
        json.dump(findings, f, indent=2)
    print(f"\n  Detailed findings saved: {findings_path}")
    
    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print("="*70)

if __name__ == '__main__':
    main()
