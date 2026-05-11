'''
This is a file to define the function: sample_in_block().
The function will sample stimulus amplitude along Hard-A or Hard-B distributions
AND manage to save plots during the last trial in each block.
'''

## Two types of sampling distribution:
## 1) Hard-A
## 2) Hard-B
## Inputs: 
#  - distribution_type: 0-Uniform, 1-Hard A, 2-Hard B
#  - num_block: block number
#  - num_trials: number of trials in this block
## Outputs: amplitude, boundary, logical_distance, true category
## Params to change: decay_rate_magnitude_


import pandas as pd
import os
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from sample import main_logic

def sample_in_block(id, session, distribution_type, num_block, num_trials, phys_boundary=0.5, amp_scale=0.45):
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # get current path of this script
    RESULT_PATH = os.path.join(SCRIPT_DIR, "results")
    GRAPH_PATH = os.path.join(RESULT_PATH, "graphs")
    
    if not os.path.exists(GRAPH_PATH):
        os.makedirs(GRAPH_PATH)

    if distribution_type == 1:
        condition_name = 'Hard-A'
    elif distribution_type == 2:
        condition_name = 'Hard-B'
    else:
        condition_name = 'Uniform'

    params = {
        'left_edge_value': -1.0,
        'boundary_value': 0.0,
        'right_edge_value': 1.0,
        'P_central_region': 0,
        'chosen_side': 'auto',
        'plot_legend': True,
        'num_simulations': 1,
        'pause_duration': 0.0,
        'plot_distribution': True,
        'range_percentage_left': 100,
        'range_percentage_right': 100,
    }

    if condition_name == 'Hard-A':
        params.update({
            'left_dist_type': 'Exponential',
            'left_exp_peak_at_boundary_flag': True,
            'decay_rate_magnitude_left': 2.0,
            'right_dist_type': 'Uniform'
        })
    elif condition_name == 'Hard-B':
        params.update({
            'left_dist_type': 'Uniform',
            'right_dist_type': 'Exponential',
            'right_exp_peak_at_boundary_flag': True,
            'decay_rate_magnitude_right': 2.0
        })
    else:
        params.update({'left_dist_type': 'Uniform', 'right_dist_type': 'Uniform'})

    params['Mode'] = 'generate_single_sample'
    block_results = []
    all_logical_values = []
    print(f"Starting Block: {condition_name}")

    for i in range(num_trials):
        logical_val, _, _ = main_logic(params, P_left_derived=0.5, 
                                       overall_left_edge=-1.0, overall_right_edge=1.0,
                                       main_boundary_actual=0.0, 
                                       min_central_sampling_range=-1.0, max_central_sampling_range=1.0,
                                       active_left_dist_start=-1.0, active_right_dist_end=1.0)
        # avoid sample on the boundary
        if abs(logical_val) < 1e-3:
            logical_val = 1e-3 if np.random.rand() > 0.5 else -1e-3
        
        all_logical_values.append(logical_val)

        physical_amplitude = phys_boundary + (logical_val * amp_scale)
        side = 0 if logical_val < 0 else 1
        
        block_results.append({
            'Block_Condition': condition_name,
            'Trial_In_Block': i + 1,
            'Logical_Value': round(logical_val, 4),
            'Physical_Amplitude': round(physical_amplitude, 4),
            'Physical_Boundary': phys_boundary,
            'Side': side,
            'Distance_to_Boundary': round(abs(logical_val), 4)
        })
    if num_trials < 5:
        n_bins = 3
    elif num_trials < 30:
        n_bins = 5
    elif num_trials < 60:
        n_bins = 8
    else:
        n_bins = 15
    fig, ax = plt.subplots(figsize=(10, 6))
    counts, bins, patches = ax.hist(all_logical_values, bins=n_bins, density=False, 
                                     alpha=0.7, color='steelblue', edgecolor='black')
    ## theoretical distribution
    if condition_name == 'Hard-A':
        x_vals_left = np.linspace(-1, 0, 100)
        decay_rate = 2.0
        theo_pdf_left = decay_rate * np.exp(decay_rate * x_vals_left)
        bin_width = bins[1] - bins[0]
        theo_counts_left = theo_pdf_left * len(all_logical_values) * bin_width
        ax.plot(x_vals_left, theo_counts_left, 'r-', linewidth=2, label='Theoretical: Exponential')
       
        x_vals_right = np.linspace(0, 1, 100)
        uniform_density = 1.0 / 1.0
        theo_counts_right = np.ones_like(x_vals_right) * uniform_density * len(all_logical_values) * bin_width
        ax.plot(x_vals_right, theo_counts_right, 'g--', linewidth=2, label='Theoretical: Uniform')
        
    elif condition_name == 'Hard-B':
        x_vals_left = np.linspace(-1, 0, 100)
        uniform_density = 1.0 / 1.0
        bin_width = bins[1] - bins[0]
        theo_counts_left = np.ones_like(x_vals_left) * uniform_density * len(all_logical_values) * bin_width
        ax.plot(x_vals_left, theo_counts_left, 'g--', linewidth=2, label='Theoretical: Uniform')

        x_vals_right = np.linspace(0, 1, 100)
        decay_rate = 2.0
        theo_pdf_right = decay_rate * np.exp(-decay_rate * x_vals_right)
        theo_counts_right = theo_pdf_right * len(all_logical_values) * bin_width
        ax.plot(x_vals_right, theo_counts_right, 'r-', linewidth=2, label='Theoretical: Exponential')
    
    ax.set_xlabel('Logical Value')
    ax.set_ylabel('Frequency')
    ax.axvline(x=0, color='red', linestyle='--', label='Boundary')
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_path = os.path.join(GRAPH_PATH, f"ID{id}_S{session}_Block{num_block}-{condition_name}.png")
    
    ax.set_title(f"Final Dist: {condition_name} (n={num_trials}) (amp_scale={amp_scale})")
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

    return pd.DataFrame(block_results)