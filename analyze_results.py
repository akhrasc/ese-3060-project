#!/usr/bin/env python3
"""
ESE 3060 Final Project - Part 1
Warmup Ratio Experiment Analysis

This script analyzes the results from the warmup ratio experiments,
creates comparison tables and visualizations.

Usage:
    python analyze_results.py
"""

import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def load_results(logs_dir='logs'):
    """Load all experiment results from logs directory."""
    results = {}
    
    for folder in os.listdir(logs_dir):
        if folder.startswith('warmup_'):
            log_path = os.path.join(logs_dir, folder, 'log.pt')
            if os.path.exists(log_path):
                data = torch.load(log_path)
                warmup_ratio = data['warmup_ratio']
                results[warmup_ratio] = {
                    'accs': data['accs'].numpy(),
                    'times': data['times'].numpy(),
                    'mean_acc': data['mean_acc'],
                    'std_acc': data['std_acc'],
                    'mean_time': data['mean_time'],
                    'std_time': data['std_time'],
                    'num_runs': data['num_runs'],
                }
    
    return dict(sorted(results.items()))

def print_results_table(results):
    """Print a formatted results table."""
    print("\n" + "="*90)
    print("EXPERIMENT RESULTS: Warmup Ratio Ablation Study")
    print("="*90)
    print(f"{'Warmup Ratio':^15} | {'Mean Acc (%)':^15} | {'Std Acc':^10} | {'Mean Time (s)':^15} | {'Std Time':^10} | {'Runs':^6}")
    print("-"*90)
    
    baseline_acc = results.get(0.23, {}).get('mean_acc', None)
    
    for warmup, data in results.items():
        acc_diff = ""
        if baseline_acc and warmup != 0.23:
            diff = (data['mean_acc'] - baseline_acc) * 100
            acc_diff = f" ({'+' if diff >= 0 else ''}{diff:.2f}%)"
        
        marker = " *" if warmup == 0.23 else ""
        print(f"{warmup:^15.2f} | {data['mean_acc']*100:^15.2f} | {data['std_acc']*100:^10.2f} | {data['mean_time']:^15.2f} | {data['std_time']:^10.2f} | {data['num_runs']:^6}{marker}")
    
    print("-"*90)
    print("* = baseline (original warmup ratio)")
    print()

def statistical_comparison(results, baseline_ratio=0.23):
    """Perform statistical tests comparing each variant to baseline."""
    if baseline_ratio not in results:
        print("Baseline not found in results!")
        return
    
    baseline = results[baseline_ratio]
    
    print("\n" + "="*70)
    print("STATISTICAL ANALYSIS (vs Baseline)")
    print("="*70)
    print(f"Baseline: warmup_ratio = {baseline_ratio}")
    print(f"{'Warmup':^10} | {'Acc Diff':^12} | {'t-stat':^10} | {'p-value':^12} | {'Significant?':^12}")
    print("-"*70)
    
    for warmup, data in results.items():
        if warmup == baseline_ratio:
            continue
        
        # Two-sample t-test for accuracy
        t_stat, p_value = stats.ttest_ind(data['accs'], baseline['accs'])
        acc_diff = (data['mean_acc'] - baseline['mean_acc']) * 100
        significant = "YES" if p_value < 0.05 else "NO"
        
        print(f"{warmup:^10.2f} | {acc_diff:^+12.3f}% | {t_stat:^10.3f} | {p_value:^12.4f} | {significant:^12}")
    
    print("-"*70)
    print("Significance level: p < 0.05")
    print()

def create_figures(results, output_dir='figures'):
    """Create visualization figures."""
    os.makedirs(output_dir, exist_ok=True)
    
    warmup_ratios = list(results.keys())
    mean_accs = [results[w]['mean_acc'] * 100 for w in warmup_ratios]
    std_accs = [results[w]['std_acc'] * 100 for w in warmup_ratios]
    mean_times = [results[w]['mean_time'] for w in warmup_ratios]
    std_times = [results[w]['std_time'] for w in warmup_ratios]
    
    # Figure 1: Accuracy vs Warmup Ratio
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(warmup_ratios)), mean_accs, yerr=std_accs, 
                  capsize=5, color='steelblue', edgecolor='black')
    
    # Highlight baseline
    if 0.23 in warmup_ratios:
        baseline_idx = warmup_ratios.index(0.23)
        bars[baseline_idx].set_color('darkorange')
    
    ax.set_xlabel('Warmup Ratio', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Effect of LR Warmup Ratio on CIFAR-10 Accuracy', fontsize=14)
    ax.set_xticks(range(len(warmup_ratios)))
    ax.set_xticklabels([f'{w:.2f}' for w in warmup_ratios])
    ax.axhline(y=mean_accs[warmup_ratios.index(0.23)] if 0.23 in warmup_ratios else mean_accs[0], 
               color='red', linestyle='--', alpha=0.5, label='Baseline')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_vs_warmup.png'), dpi=150)
    plt.savefig(os.path.join(output_dir, 'accuracy_vs_warmup.pdf'))
    print(f"Saved: {output_dir}/accuracy_vs_warmup.png")
    
    # Figure 2: Training Time vs Warmup Ratio  
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(warmup_ratios)), mean_times, yerr=std_times,
                  capsize=5, color='seagreen', edgecolor='black')
    
    # Highlight baseline
    if 0.23 in warmup_ratios:
        baseline_idx = warmup_ratios.index(0.23)
        bars[baseline_idx].set_color('darkorange')
    
    ax.set_xlabel('Warmup Ratio', fontsize=12)
    ax.set_ylabel('Training Time (seconds)', fontsize=12)
    ax.set_title('Effect of LR Warmup Ratio on Training Time', fontsize=14)
    ax.set_xticks(range(len(warmup_ratios)))
    ax.set_xticklabels([f'{w:.2f}' for w in warmup_ratios])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'time_vs_warmup.png'), dpi=150)
    plt.savefig(os.path.join(output_dir, 'time_vs_warmup.pdf'))
    print(f"Saved: {output_dir}/time_vs_warmup.png")
    
    # Figure 3: Combined plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy subplot
    bars1 = ax1.bar(range(len(warmup_ratios)), mean_accs, yerr=std_accs,
                    capsize=4, color='steelblue', edgecolor='black')
    if 0.23 in warmup_ratios:
        bars1[warmup_ratios.index(0.23)].set_color('darkorange')
    ax1.set_xlabel('Warmup Ratio')
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.set_title('Accuracy vs Warmup Ratio')
    ax1.set_xticks(range(len(warmup_ratios)))
    ax1.set_xticklabels([f'{w:.2f}' for w in warmup_ratios])
    ax1.grid(axis='y', alpha=0.3)
    
    # Time subplot
    bars2 = ax2.bar(range(len(warmup_ratios)), mean_times, yerr=std_times,
                    capsize=4, color='seagreen', edgecolor='black')
    if 0.23 in warmup_ratios:
        bars2[warmup_ratios.index(0.23)].set_color('darkorange')
    ax2.set_xlabel('Warmup Ratio')
    ax2.set_ylabel('Training Time (s)')
    ax2.set_title('Training Time vs Warmup Ratio')
    ax2.set_xticks(range(len(warmup_ratios)))
    ax2.set_xticklabels([f'{w:.2f}' for w in warmup_ratios])
    ax2.grid(axis='y', alpha=0.3)
    
    plt.suptitle('ESE 3060 Part 1: LR Warmup Ratio Ablation Study', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_results.png'), dpi=150)
    plt.savefig(os.path.join(output_dir, 'combined_results.pdf'))
    print(f"Saved: {output_dir}/combined_results.png")
    
    plt.close('all')

def export_csv(results, output_path='results.csv'):
    """Export results to CSV for easy spreadsheet use."""
    with open(output_path, 'w') as f:
        f.write("warmup_ratio,mean_acc,std_acc,mean_time,std_time,num_runs\n")
        for warmup, data in results.items():
            f.write(f"{warmup},{data['mean_acc']},{data['std_acc']},{data['mean_time']},{data['std_time']},{data['num_runs']}\n")
    print(f"Exported results to: {output_path}")

def main():
    print("ESE 3060 Final Project - Part 1: Results Analysis")
    print("="*50)
    
    # Load results
    results = load_results()
    
    if not results:
        print("No results found in logs/ directory!")
        print("Run 'python run_experiments.py' first.")
        return
    
    print(f"\nFound results for {len(results)} warmup ratios: {list(results.keys())}")
    
    # Print results table
    print_results_table(results)
    
    # Statistical analysis
    statistical_comparison(results)
    
    # Create figures
    create_figures(results)
    
    # Export to CSV
    export_csv(results)
    
    print("\nAnalysis complete!")
    print("Check the 'figures/' directory for visualizations.")

if __name__ == "__main__":
    main()
