#!/usr/bin/env python3
"""
ESE 3060 Final Project - Part 1
Activation Function Ablation Experiments

Runs experiments comparing GELU, ReLU, ReLU^2, and Swish.
"""

import subprocess
import sys
import os
import time

# Activation functions to test
ACTIVATIONS = [
    'gelu',          # Baseline (default)
    'relu_squared',  # Main experiment
    'relu',          # Ablation
    'swish',         # Ablation (control, similar to GELU)
]

# Number of runs per activation (25 for statistical significance)
DEFAULT_NUM_RUNS = 25

def run_experiment(activation, num_runs):
    print(f"\n{'='*70}")
    print(f"  RUNNING: {activation.upper()}")
    print(f"  Runs: {num_runs}")
    print(f"{'='*70}\n")
    
    start_time = time.time()
    
    cmd = [
        sys.executable, 'airbench94_activation.py',
        '--activation', activation,
        '--num_runs', str(num_runs)
    ]
    
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    
    elapsed = time.time() - start_time
    print(f"\n[{activation}] Completed in {elapsed:.1f}s (exit code: {result.returncode})")
    
    return result.returncode == 0

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run activation function experiments')
    parser.add_argument('--num_runs', type=int, default=DEFAULT_NUM_RUNS,
                        help=f'Number of runs per activation (default: {DEFAULT_NUM_RUNS})')
    parser.add_argument('--activations', nargs='+', default=ACTIVATIONS,
                        choices=['gelu', 'relu', 'relu_squared', 'swish'],
                        help='Activation functions to test')
    args = parser.parse_args()
    
    print("="*70)
    print("  ESE 3060 Final Project - Activation Function Experiments")
    print("="*70)
    print(f"\nActivations to test: {args.activations}")
    print(f"Runs per activation: {args.num_runs}")
    print(f"Estimated time: ~{len(args.activations) * args.num_runs * 5 / 60:.1f} minutes")
    print()
    
    results = {}
    total_start = time.time()
    
    for activation in args.activations:
        success = run_experiment(activation, args.num_runs)
        results[activation] = 'SUCCESS' if success else 'FAILED'
    
    total_elapsed = time.time() - total_start
    
    print("\n" + "="*70)
    print("  EXPERIMENT SUMMARY")
    print("="*70)
    for activation, status in results.items():
        print(f"  {activation:15} : {status}")
    print(f"\n  Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} minutes)")
    print("="*70)
    
    print("\nNext steps:")
    print("  1. Run: python analyze_activation_results.py")
    print("  2. Check figures/ directory for visualizations")
    print("  3. Update report_part1.tex with results")

if __name__ == "__main__":
    main()


