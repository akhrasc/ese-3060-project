#!/usr/bin/env python3
"""
ESE 3060 Final Project - Part 1
Warmup Ratio Experiment Runner

Runs airbench94.py with different warmup ratios.
"""

import subprocess
import sys

# Warmup ratios to test (including baseline at 0.23)
WARMUP_RATIOS = [0.05, 0.10, 0.15, 0.20, 0.23, 0.30, 0.35]
NUM_RUNS = 25  # Number of runs per variant

def run_experiment(warmup_ratio, num_runs):
    print(f"\n{'#'*70}")
    print(f"# Running: warmup_ratio = {warmup_ratio}")
    print(f"{'#'*70}\n")
    
    cmd = [
        sys.executable, 
        "airbench94.py",
        "--warmup_ratio", str(warmup_ratio),
        "--num_runs", str(num_runs)
    ]
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print(f"ERROR: Experiment with warmup_ratio={warmup_ratio} failed!")
        return False
    return True

def main():
    print("="*70)
    print("ESE 3060 Final Project - Part 1: Warmup Ratio Experiment")
    print("="*70)
    print(f"\nWarmup ratios to test: {WARMUP_RATIOS}")
    print(f"Runs per variant: {NUM_RUNS}")
    print(f"Total runs: {len(WARMUP_RATIOS) * NUM_RUNS}")
    print("\nStarting experiments...\n")
    
    successful = []
    failed = []
    
    for warmup_ratio in WARMUP_RATIOS:
        if run_experiment(warmup_ratio, NUM_RUNS):
            successful.append(warmup_ratio)
        else:
            failed.append(warmup_ratio)
    
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    print(f"Successful: {successful}")
    if failed:
        print(f"Failed: {failed}")
    print("\nLogs saved in ./logs/warmup_<ratio>/")
    print("Run 'python analyze_results.py' to analyze results.")

if __name__ == "__main__":
    main()


