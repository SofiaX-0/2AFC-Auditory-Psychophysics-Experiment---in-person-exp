'''
This is a file to define the function: sample_in_block().
The function will sample stimulus amplitude along Hard-A or Hard-B distributions
AND save distribution plots for each block.
'''

## 3 types of sampling distribution
## Inputs: 
#  - distribution_type: 0-Uniform, 1-Hard A, 2-Hard B
#  - num_block: block number
#  - num_trials: number of trials in this block
## Output:
# pandas DataFrame containing sampled stimulus information
## Params to change: decay_rate_magnitude_


import pandas as pd
import os
import numpy as np
import matplotlib
matplotlib.use('Agg') # Agg backend
import matplotlib.pyplot as plt
from sample import main_logic, get_pdf_general

def sample_in_block(id, session, distribution_type, num_block, num_trials, physical_min=46, physical_max=80):
    ###### MIN/MAX default should be changed after calibration, they are measured in dba
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

    decay_rate = 2.153

    params = {
        'left_edge_value': -1.0,
        'boundary_value': 0.0,
        'right_edge_value': 1.0,
        'P_central_region': 0,
        'chosen_side': 'auto',
        'plot_legend': True,
        'range_percentage_left': 100,
        'range_percentage_right': 100,
    }

    if condition_name == 'Hard-A':
        params.update({
            'left_dist_type': 'Exponential',
            'left_exp_peak_at_boundary_flag': True,
            'decay_rate_magnitude_left': decay_rate,
            'right_dist_type': 'Uniform'
        })
    elif condition_name == 'Hard-B':
        params.update({
            'left_dist_type': 'Uniform',
            'right_exp_peak_at_boundary_flag': True,
            'decay_rate_magnitude_right': decay_rate,
            'right_dist_type': 'Exponential',
        })
    else:
        params.update({'left_dist_type': 'Uniform', 'right_dist_type': 'Uniform'})

    params['Mode'] = 'generate_single_sample'
    block_results = [] # to store block trial info of this block
    all_logical_values = [] # to store logical values of the sample

    for i in range(num_trials):
        '''
        from sample.py -> main_logic(params, P_left_derived, overall_left_edge, overall_right_edge, 
               main_boundary_actual, min_central_sampling_range, max_central_sampling_range,
               active_left_dist_start, active_right_dist_end):
            return samples_history_out, None, {}
        '''
        logical_val, _, _ = main_logic(params, P_left_derived=0.5, 
                                       overall_left_edge=-1.0, overall_right_edge=1.0,
                                       main_boundary_actual=0.0, 
                                       min_central_sampling_range=-1.0, max_central_sampling_range=1.0,
                                       active_left_dist_start=-1.0, active_right_dist_end=1.0)
        
        all_logical_values.append(logical_val) # logical_val is in [-1, 1] scale
        # logical values are already defined in logarithmic stimulus space
        # linearly map logical values onto the calibrated stimulus range

        physical_boundary = (physical_min + physical_max) / 2 # eg, 60 dba
        physical_scale = (physical_max - physical_min) / 2 # eg, 15 dba
        easy_thereshold = 0.5 * physical_scale # eg, 7.5 dba
        Target_dba = (physical_boundary + logical_val * physical_scale) # current stimulus dba
        distance_toB_in_dba = abs(Target_dba - physical_boundary)
        side = 0 if logical_val < 0 else 1
        
        block_results.append({
            'Block_Condition': condition_name,
            'Trial_In_Block': i + 1,
            'Logical_Value': round(logical_val, 4),
            'Target_dba': round(Target_dba, 4),
            'Physical_Boundary': round(physical_boundary, 4),
            'Side': side,
            'Distance_toB_in_dba': round(distance_toB_in_dba, 4),
            'Easy_threshold': round(easy_thereshold, 2)
        })
    if num_trials < 5:
        n_bins = 3
    elif num_trials <= 30:
        n_bins = 5
    else:
        n_bins = 12
    fig, ax = plt.subplots(figsize=(10, 6))
    _, bins, _ = ax.hist(
        all_logical_values,
        bins=n_bins,
        range=(-1, 1),
        density=False,
        alpha=0.7,
        color='steelblue',
        edgecolor='black'
    )
    bin_width = bins[1] - bins[0]
    n_samples = len(all_logical_values)

    ## theoretical distribution
    '''
    from sample.py -> def get_pdf_general(x, dist_type_full_name, current_params, min_val, max_val, 
                    is_exp_peak_at_boundary_flag, is_hn_peak_at_boundary_flag, 
                    is_sin_peak_at_boundary_flag, current_main_boundary):
    '''
    common_params = {
        'normal_mean': 0,
        'normal_std_dev': 0.1,
        'half_normal_std_dev': 0.1,
        'sinusoidal_amplitude_factor': 0,
        'sinusoidal_frequency_factor': 0
    }

    if condition_name == 'Hard-A':

        # LEFT: Exponential
        x_vals_left = np.linspace(-1, 0, 200) # [-1, 0], 200 evenly distributed points

        left_params = {
            **common_params, # unfold dict
            'lambda_actual': decay_rate
        }

        theo_pdf_left = get_pdf_general(
            x_vals_left,
            'Exponential',
            left_params,
            -1.0,
            0.0,
            True,   # peak at boundary
            False,
            False,
            0.0
        )

        theo_counts_left = theo_pdf_left * n_samples * bin_width

        ax.plot(
            x_vals_left,
            theo_counts_left,
            'r-',
            linewidth=2,
            label='Theoretical: Exponential'
        )

        # RIGHT: Uniform
        x_vals_right = np.linspace(0, 1, 200)

        right_params = {
            **common_params,
            'lambda_actual': 0
        }

        theo_pdf_right = get_pdf_general(
            x_vals_right,
            'Uniform',
            right_params,
            0.0,
            1.0,
            False,
            False,
            False,
            0.0
        )

        theo_counts_right = theo_pdf_right * n_samples * bin_width

        ax.plot(
            x_vals_right,
            theo_counts_right,
            'g--',
            linewidth=2,
            label='Theoretical: Uniform'
        )

    elif condition_name == 'Hard-B':

        # LEFT: Uniform
        x_vals_left = np.linspace(-1, 0, 200)

        left_params = {
            **common_params,
            'lambda_actual': 0
        }

        theo_pdf_left = get_pdf_general(
            x_vals_left,
            'Uniform',
            left_params,
            -1.0,
            0.0,
            False,
            False,
            False,
            0.0
        )
        theo_counts_left = theo_pdf_left * n_samples * bin_width

        ax.plot(
            x_vals_left,
            theo_counts_left,
            'g--',
            linewidth=2,
            label='Theoretical: Uniform'
        )

        # RIGHT: Exponential
        x_vals_right = np.linspace(0, 1, 200)

        right_params = {
            **common_params,
            'lambda_actual': decay_rate
        }

        theo_pdf_right = get_pdf_general(
            x_vals_right,
            'Exponential',
            right_params,
            0.0,
            1.0,
            True,   # peak at boundary
            False,
            False,
            0.0
        )
        theo_counts_right = theo_pdf_right * n_samples * bin_width
        ax.plot(
            x_vals_right,
            theo_counts_right,
            'r-',
            linewidth=2,
            label='Theoretical: Exponential'
        )

    else:

        # Uniform both sides
        x_vals = np.linspace(-1, 1, 400)

        uniform_params = {
            **common_params,
            'lambda_actual': 0
        }

        theo_pdf = get_pdf_general(
            x_vals,
            'Uniform',
            uniform_params,
            -1.0,
            1.0,
            False,
            False,
            False,
            0.0
        )
        theo_counts = theo_pdf * n_samples * bin_width

        ax.plot(
            x_vals,
            theo_counts,
            'g--',
            linewidth=2,
            label='Theoretical: Uniform'
        )
    
    ax.set_xlabel('Logical Value')
    ax.set_ylabel('Count')
    ax.axvline(x=0, color='red', linestyle='--')
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_path = os.path.join(GRAPH_PATH, f"ID{id}_S{session}_Block{num_block}-{condition_name}.png")
    
    ax.set_title(f"Sampled Logical Stimulus Distribution: {condition_name} (n={num_trials}) (dBA range={physical_min:.2f}-{physical_max:.2f})")
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

    return pd.DataFrame(block_results)