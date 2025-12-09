#!/usr/bin/env python3
"""
ESE 3060 Final Project - Part 1
Activation Function Experiment Analysis

Analyzes results, creates tables and figures.
"""

import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Display names for activations
ACTIVATION_NAMES = {
    'gelu': 'GELU',
    'relu': 'ReLU',
    'relu_squared': 'ReLU²',
    'swish': 'Swish',
}

# Colors for plotting
ACTIVATION_COLORS = {
    'gelu': 'darkorange',      # Baseline - highlight
    'relu': 'steelblue',
    'relu_squared': 'forestgreen',  # Main experiment - highlight
    'swish': 'mediumpurple',
}

def load_results(logs_dir='logs'):
    # Load all activation experiment results from logs directory
    results = {}
    
    for folder in os.listdir(logs_dir):
        if folder.startswith('activation_'):
            log_path = os.path.join(logs_dir, folder, 'log.pt')
            if os.path.exists(log_path):
                data = torch.load(log_path)
                activation = data['activation']
                results[activation] = {
                    'accs': data['accs'].numpy(),
                    'times': data['times'].numpy(),
                    'mean_acc': data['mean_acc'],
                    'std_acc': data['std_acc'],
                    'mean_time': data['mean_time'],
                    'std_time': data['std_time'],
                    'num_runs': data['num_runs'],
                }
    
    # Sort by a consistent order
    order = ['gelu', 'relu_squared', 'relu', 'swish']
    return {k: results[k] for k in order if k in results}

def print_results_table(results):
    print("\n" + "="*100)
    print("RESULTS: Activation Function Ablation Study")
    print("="*100)
    print(f"{'Activation':^15} | {'Mean Acc (%)':^15} | {'Std Acc':^10} | {'Mean Time (s)':^15} | {'Std Time':^10} | {'Runs':^6}")
    print("-"*100)
    
    baseline_acc = results.get('gelu', {}).get('mean_acc', None)
    baseline_time = results.get('gelu', {}).get('mean_time', None)
    
    for activation, data in results.items():
        marker = " *" if activation == 'gelu' else ""
        name = ACTIVATION_NAMES.get(activation, activation)
        print(f"{name:^15} | {data['mean_acc']*100:^15.2f} | {data['std_acc']*100:^10.2f} | {data['mean_time']:^15.2f} | {data['std_time']:^10.2f} | {data['num_runs']:^6}{marker}")
    
    print("-"*100)
    print("* = baseline (GELU)")
    print()

def statistical_comparison(results, baseline='gelu'):
    """Perform statistical tests comparing each variant to baseline."""
    if baseline not in results:
        print(f"Baseline '{baseline}' not found in results!")
        return {}
    
    baseline_data = results[baseline]
    comparisons = {}
    
    print("\n" + "="*90)
    print("STATISTICAL ANALYSIS (vs GELU Baseline)")
    print("="*90)
    print(f"{'Activation':^15} | {'Acc Diff':^12} | {'Time Diff':^12} | {'t-stat (acc)':^12} | {'p-value':^12} | {'Significant?':^12}")
    print("-"*90)
    
    for activation, data in results.items():
        if activation == baseline:
            continue
        
        # Two-sample t-test for accuracy
        t_stat_acc, p_value_acc = stats.ttest_ind(data['accs'], baseline_data['accs'])
        
        # Two-sample t-test for time
        t_stat_time, p_value_time = stats.ttest_ind(data['times'], baseline_data['times'])
        
        acc_diff = (data['mean_acc'] - baseline_data['mean_acc']) * 100
        time_diff = data['mean_time'] - baseline_data['mean_time']
        time_diff_pct = (time_diff / baseline_data['mean_time']) * 100
        
        significant = "YES" if p_value_acc < 0.05 or p_value_time < 0.05 else "NO"
        
        name = ACTIVATION_NAMES.get(activation, activation)
        print(f"{name:^15} | {acc_diff:^+12.3f}% | {time_diff:^+12.3f}s | {t_stat_acc:^12.3f} | {p_value_acc:^12.4f} | {significant:^12}")
        
        comparisons[activation] = {
            'acc_diff': acc_diff,
            'time_diff': time_diff,
            'time_diff_pct': time_diff_pct,
            't_stat_acc': t_stat_acc,
            'p_value_acc': p_value_acc,
            't_stat_time': t_stat_time,
            'p_value_time': p_value_time,
            'significant': p_value_acc < 0.05 or p_value_time < 0.05,
        }
    
    print("-"*90)
    print("Significance level: p < 0.05")
    print()
    
    return comparisons

def create_figures(results, output_dir='figures'):
    os.makedirs(output_dir, exist_ok=True)
    
    activations = list(results.keys())
    names = [ACTIVATION_NAMES.get(a, a) for a in activations]
    mean_accs = [results[a]['mean_acc'] * 100 for a in activations]
    std_accs = [results[a]['std_acc'] * 100 for a in activations]
    mean_times = [results[a]['mean_time'] for a in activations]
    std_times = [results[a]['std_time'] for a in activations]
    colors = [ACTIVATION_COLORS.get(a, 'gray') for a in activations]
    
    # Figure 1: Accuracy vs Activation
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(activations)), mean_accs, yerr=std_accs, 
                  capsize=5, color=colors, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Activation Function', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Effect of Activation Function on CIFAR-10 Accuracy', fontsize=14)
    ax.set_xticks(range(len(activations)))
    ax.set_xticklabels(names)
    
    # Add baseline reference line
    if 'gelu' in results:
        ax.axhline(y=results['gelu']['mean_acc']*100, color='red', linestyle='--', 
                   alpha=0.7, label='GELU Baseline')
        ax.legend()
    
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(bottom=min(mean_accs) - 0.5, top=max(mean_accs) + 0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'activation_accuracy.png'), dpi=150)
    plt.savefig(os.path.join(output_dir, 'activation_accuracy.pdf'))
    print(f"Saved: {output_dir}/activation_accuracy.png")
    
    # Figure 2: Training Time vs Activation
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(activations)), mean_times, yerr=std_times,
                  capsize=5, color=colors, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Activation Function', fontsize=12)
    ax.set_ylabel('Training Time (seconds)', fontsize=12)
    ax.set_title('Effect of Activation Function on Training Time', fontsize=14)
    ax.set_xticks(range(len(activations)))
    ax.set_xticklabels(names)
    
    # Add baseline reference line
    if 'gelu' in results:
        ax.axhline(y=results['gelu']['mean_time'], color='red', linestyle='--', 
                   alpha=0.7, label='GELU Baseline')
        ax.legend()
    
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'activation_time.png'), dpi=150)
    plt.savefig(os.path.join(output_dir, 'activation_time.pdf'))
    print(f"Saved: {output_dir}/activation_time.png")
    
    # Figure 3: Combined plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy subplot
    bars1 = ax1.bar(range(len(activations)), mean_accs, yerr=std_accs,
                    capsize=4, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Activation Function', fontsize=11)
    ax1.set_ylabel('Test Accuracy (%)', fontsize=11)
    ax1.set_title('Accuracy vs Activation', fontsize=12)
    ax1.set_xticks(range(len(activations)))
    ax1.set_xticklabels(names)
    ax1.grid(axis='y', alpha=0.3)
    if 'gelu' in results:
        ax1.axhline(y=results['gelu']['mean_acc']*100, color='red', linestyle='--', alpha=0.5)
    
    # Time subplot
    bars2 = ax2.bar(range(len(activations)), mean_times, yerr=std_times,
                    capsize=4, color=colors, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Activation Function', fontsize=11)
    ax2.set_ylabel('Training Time (s)', fontsize=11)
    ax2.set_title('Training Time vs Activation', fontsize=12)
    ax2.set_xticks(range(len(activations)))
    ax2.set_xticklabels(names)
    ax2.grid(axis='y', alpha=0.3)
    if 'gelu' in results:
        ax2.axhline(y=results['gelu']['mean_time'], color='red', linestyle='--', alpha=0.5)
    
    plt.suptitle('ESE 3060 Part 1: Activation Function Ablation Study', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'activation_combined.png'), dpi=150)
    plt.savefig(os.path.join(output_dir, 'activation_combined.pdf'))
    print(f"Saved: {output_dir}/activation_combined.png")
    
    # Figure 4: Speedup comparison (percentage change from baseline)
    if 'gelu' in results:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        baseline_time = results['gelu']['mean_time']
        speedups = [(baseline_time - results[a]['mean_time']) / baseline_time * 100 
                    for a in activations]
        
        bar_colors = ['green' if s > 0 else 'red' for s in speedups]
        bars = ax.bar(range(len(activations)), speedups, color=bar_colors, 
                      edgecolor='black', linewidth=1.5, alpha=0.7)
        
        ax.axhline(y=0, color='black', linewidth=1)
        ax.set_xlabel('Activation Function', fontsize=12)
        ax.set_ylabel('Speedup vs GELU (%)', fontsize=12)
        ax.set_title('Training Speedup Relative to GELU Baseline', fontsize=14)
        ax.set_xticks(range(len(activations)))
        ax.set_xticklabels(names)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, speedup) in enumerate(zip(bars, speedups)):
            height = bar.get_height()
            ax.annotate(f'{speedup:+.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3 if height >= 0 else -15),
                        textcoords="offset points",
                        ha='center', va='bottom' if height >= 0 else 'top',
                        fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'activation_speedup.png'), dpi=150)
        plt.savefig(os.path.join(output_dir, 'activation_speedup.pdf'))
        print(f"Saved: {output_dir}/activation_speedup.png")
    
    plt.close('all')

def export_csv(results, output_path='activation_results.csv'):
    with open(output_path, 'w') as f:
        f.write("activation,mean_acc,std_acc,mean_time,std_time,num_runs\n")
        for activation, data in results.items():
            f.write(f"{activation},{data['mean_acc']},{data['std_acc']},{data['mean_time']},{data['std_time']},{data['num_runs']}\n")
    print(f"Exported results to: {output_path}")

def print_latex_table(results, comparisons):
    print("\n" + "="*70)
    print("LATEX TABLE")
    print("="*70)
    print(r"""
\begin{table}[h]
\centering
\caption{Activation Function Experiment Results}
\begin{tabular}{lcccc}
\toprule
Activation & Mean Acc (\%) & Std (\%) & Mean Time (s) & Speedup \\
\midrule""")
    
    baseline_time = results.get('gelu', {}).get('mean_time', 1)
    
    for activation, data in results.items():
        name = ACTIVATION_NAMES.get(activation, activation)
        speedup = (baseline_time - data['mean_time']) / baseline_time * 100
        
        if activation == 'gelu':
            print(f"\\textbf{{{name}}} & \\textbf{{{data['mean_acc']*100:.2f}}} & \\textbf{{{data['std_acc']*100:.2f}}} & \\textbf{{{data['mean_time']:.2f}}} & baseline \\\\")
        else:
            sig = comparisons.get(activation, {}).get('significant', False)
            sig_marker = "$^*$" if sig else ""
            print(f"{name} & {data['mean_acc']*100:.2f} & {data['std_acc']*100:.2f} & {data['mean_time']:.2f} & {speedup:+.1f}\\%{sig_marker} \\\\")
    
    print(r"""\bottomrule
\end{tabular}
\end{table}
""")
    print("="*70)

def main():
    print("ESE 3060 Final Project - Part 1: Activation Results Analysis")
    print("="*60)
    
    # Load results
    results = load_results()
    
    if not results:
        print("No activation results found in logs/ directory!")
        print("Run 'python run_activation_experiments.py' first.")
        return
    
    print(f"\nFound results for {len(results)} activations: {list(results.keys())}")
    
    # Print results table
    print_results_table(results)
    
    # Statistical analysis
    comparisons = statistical_comparison(results)
    
    # Create figures
    create_figures(results)
    
    # Export to CSV
    export_csv(results)
    
    # Print LaTeX table
    print_latex_table(results, comparisons)
    
    print("\nAnalysis complete!")
    print("Check the 'figures/' directory for visualizations.")

if __name__ == "__main__":
    main()


