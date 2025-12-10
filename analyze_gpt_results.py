import os
import glob
import re
import matplotlib.pyplot as plt
import numpy as np

# CONFIG: Update these paths to where you saved your logs
LOG_DIR = 'logs' 
FIGURE_DIR = 'figures'
os.makedirs(FIGURE_DIR, exist_ok=True)

def parse_log(filepath):
    """Parses a single log file to extract steps, val_loss, and train_time."""
    steps = []
    val_losses = []
    times = []
    
    with open(filepath, 'r') as f:
        for line in f:
            # Look for lines with val_loss
            # Example: step:5100/5100 val_loss:3.2927 train_time:812253ms step_avg:159.58ms
            if 'val_loss' in line:
                match = re.search(r'step:(\d+)/\d+ val_loss:([\d\.]+) train_time:(\d+)ms', line)
                if match:
                    step = int(match.group(1))
                    val_loss = float(match.group(2))
                    time_ms = float(match.group(3))
                    
                    steps.append(step)
                    val_losses.append(val_loss)
                    times.append(time_ms / 1000.0) # Convert to seconds

    return steps, val_losses, times

def main():
    # 1. Find all log files
    log_files = glob.glob(os.path.join(LOG_DIR, "*.txt"))
    # Filter for the specific files we care about (baseline and swiglu)
    log_files = [f for f in log_files if 'baseline' in os.path.basename(f) or 'swiglu' in os.path.basename(f)]
    
    if not log_files:
        print(f"No baseline/swiglu log files found in {LOG_DIR}. Please check naming!")
        return

    print(f"Found {len(log_files)} logs: {[os.path.basename(f) for f in log_files]}")

    experiments = {}
    
    for log_file in log_files:
        filename = os.path.basename(log_file)
        
        if 'swiglu' in filename.lower():
            label = 'SwiGLU'
        else:
            label = 'Baseline (ReLU^2)'
            
        steps, val_losses, times = parse_log(log_file)
        
        if not steps:
            print(f"Warning: No data found in {filename}")
            continue

        if label not in experiments:
            experiments[label] = {'steps': [], 'val': [], 'time': []}
        
        experiments[label]['steps'].append(steps)
        experiments[label]['val'].append(val_losses)
        experiments[label]['time'].append(times)

    # 3. Plotting
    # Create two subplots: one full view, one zoomed view
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    colors = {'Baseline (ReLU^2)': 'blue', 'SwiGLU': 'orange'}
    
    # Plot 1: Full View (skipping first few steps to avoid extreme scaling)
    for label, data in experiments.items():
        if not data['steps']:
            continue
        for i, (run_steps, run_val) in enumerate(zip(data['steps'], data['val'])):
            # Skip first 5 data points to avoid the initial spike (loss ~16)
            start_idx = 5 if len(run_steps) > 5 else 0
            if i == 0:
                ax1.plot(run_steps[start_idx:], run_val[start_idx:], color=colors.get(label, 'gray'), alpha=0.7, linestyle='-', label=label)
            else:
                ax1.plot(run_steps[start_idx:], run_val[start_idx:], color=colors.get(label, 'gray'), alpha=0.7, linestyle='--')

    ax1.set_xlabel('Step')
    ax1.set_ylabel('Validation Loss')
    ax1.set_title('Validation Loss: Full Training (ignoring init spike)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Zoomed View (Last 2000 steps)
    for label, data in experiments.items():
        if not data['steps']:
            continue
        for i, (run_steps, run_val) in enumerate(zip(data['steps'], data['val'])):
            # Zoom in on steps > 2000
            zoom_steps = []
            zoom_val = []
            for s, v in zip(run_steps, run_val):
                if s > 2000:
                    zoom_steps.append(s)
                    zoom_val.append(v)
            
            if i == 0:
                ax2.plot(zoom_steps, zoom_val, color=colors.get(label, 'gray'), alpha=0.7, linestyle='-', label=label)
            else:
                ax2.plot(zoom_steps, zoom_val, color=colors.get(label, 'gray'), alpha=0.7, linestyle='--')

    ax2.set_xlabel('Step')
    ax2.set_ylabel('Validation Loss')
    ax2.set_title('Zoomed View (Steps 2000-5100)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    # Force Y-axis to show the difference clearly
    ax2.set_ylim(3.25, 3.6)
    
    plt.tight_layout()
    output_path = os.path.join(FIGURE_DIR, 'gpt_comparison.png')
    plt.savefig(output_path, dpi=300)
    print(f"Saved figure to {output_path}")

if __name__ == "__main__":
    main()
