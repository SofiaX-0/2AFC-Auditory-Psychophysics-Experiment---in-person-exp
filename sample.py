############################################################################
## This file contains the following:
## CreateSamples_from_Distribution(varargin)
## This function generates random samples from a customizable bimodal 
## distribution and can optionally plot the distribution, a history histogram, and dynamic current picks.
## The overall distribution spans a user-defined range [left_edge_value, right_edge_value]
## with a user-defined boundary at 'boundary_value'.
## The actual distribution (PDF) for each side is generated over a percentage
## of the range between the overall edge and the boundary.
## A central sampling bias mechanism is added to draw more picks near the boundary without altering the PDF shape.
#############################################################################
"""
Parameters (as keyword arguments):
    1)mode (str): 'run_simulation' (default), 'generate_single_sample','initialize_plot', or 'update_plot'.
    2)left_probability (float): Relative probability for the left side.
    3)right_probability (float): Relative probability for the right side.
    4)chosen_side (str, optional): Only for 'generate_single_sample' mode. 
        ='left', 'right', or 'auto' (default). If 'left' or 'right', the sample is forced from that side.

    5)left_edge_value (float): Absolute value for the leftmost boundary of the plot range.
    6)boundary_value (float): Absolute value for the central boundary between left and right distributions.
    7)right_edge_value (float): Absolute value for the rightmost boundary of the plot range.

    8)left_dist_type (str): Type for left side distribution. Options: 'Uniform',
        'Exponential', 'Anti_Exponential', 'Normal', 'Half Normal',
        'Anti Half Normal', 'Sinusoidal', 'Anti_Sinusoidal'.
    9)decay_rate_magnitude_left (float): Positive magnitude for 'Exponential' types.
    10)normal_mean_left (float): Mean for 'Normal' distribution.
    11)normal_std_dev_left (float): Standard deviation for 'Normal' distribution.
    12)half_normal_std_dev_left (float): Standard deviation for 'Half Normal' types.
    13)sinusoidal_amplitude_factor_left (float): Amplitude (0 to 1) for 'Sinusoidal' types.
    14)sinusoidal_frequency_factor_left (float): Frequency for 'Sinusoidal' types.
    15)range_percentage_left (float): Percentage (0 to 100) of [left_edge_value, boundary_value]
        that the left distribution is active over (e.g., 70 for inner 70%).

    16)right_dist_type (str): Type for right side distribution (same options as left_dist_type).
    17)decay_rate_magnitude_right (float): Positive magnitude for 'Exponential' types.
    18)normal_mean_right (float): Mean for 'Normal' distribution.
    19)normal_std_dev_right (float): Standard deviation for 'Normal' distribution.
    20)half_normal_std_dev_right (float): Standard deviation for 'Half Normal' types.
    21)sinusoidal_amplitude_factor_right (float): Amplitude (0 to 1) for 'Sinusoidal' types.
    22)sinusoidal_frequency_factor_right (float): Frequency for 'Sinusoidal' types.
    23)range_percentage_right (float): Percentage (0 to 100) of [boundary_value, right_edge_value]
        that the right distribution is active over (e.g., 70 for inner 70%).

    24)P_central_region (float): Probability (0 to 1) of applying central sampling bias.
    25)central_region_width (float): Width of the central zone for biased sampling.

    26)num_simulations (int): Number of random picks to simulate (only in 'run_simulation' mode).
    27)pause_duration (float): Pause duration between updates in seconds (only in 'run_simulation' mode).

    For 'initialize_plot' mode:
        ax_handle (matplotlib.axes.Axes, optional): Axes to plot on. If not provided, a new figure/axes is created.

    For 'update_plot' mode:
        ax_handle (matplotlib.axes.Axes): Axes to plot on.
        current_stimulus (float): Most recent stimulus value to plot.
        samples_history_in (array-like): Current history of all samples.
        plot_handles_in (dict): Dictionary containing handles of existing plot elements (from initialize_static_plot).
        plot_histogram (bool): True to plot/update histogram.
        plot_chosen_stimuli (bool): True to plot/update current stimulus marker.
        plot_distribution (bool): True to plot/show the PDF curve.
        plot_legend (bool): True to plot/show the legends.

Returns:
    samples_history_out (array-like): Array of generated samples. Single value if 'generate_single_sample' mode.
    h_fig_out (matplotlib.figure.Figure): Handle to the plot figure.
    plot_handles_out (dict): Dictionary containing handles of plot elements.
"""
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import matplotlib.pyplot as plt

params = {
    'Mode': 'run_simulation', 
    
    # Side Probabilities
    'left_probability': 0.5,
    'right_probability': 0.5,
    'chosen_side': 'auto',

    # Custom Boundary and Edge Values
    'left_edge_value': -1.0,
    'boundary_value': 0.0,
    'right_edge_value': 1.0,

    # Left Side Parameters
    'left_dist_type': 'Half Normal',
    'decay_rate_magnitude_left': 2.153,
    'normal_mean_left': -0.5,
    'normal_std_dev_left': 0.15,
    'half_normal_std_dev_left': 0.2,
    'sinusoidal_amplitude_factor_left': 0.8,
    'sinusoidal_frequency_factor_left': 1.0,
    'range_percentage_left': 100.0,

    # Right Side Parameters
    'right_dist_type': 'Sinusoidal',
    'decay_rate_magnitude_right': 2.153,
    'normal_mean_right': 0.5,
    'normal_std_dev_right': 0.15,
    'half_normal_std_dev_right': 0.2,
    'sinusoidal_amplitude_factor_right': 0.8,
    'sinusoidal_frequency_factor_right': 1.0,
    'range_percentage_right': 100.0,

    # Central Sampling Bias Parameters
    'P_central_region': 0.0,
    'central_region_width': 0.0,

    # Simulation Parameters
    'num_simulations': 200,
    'pause_duration': 0.05,

    # Plot Update Specific Parameters
    'ax_handle': None,
    'current_stimulus': np.nan, # no figure for now
    'samples_history_in': [],
    'plot_handles_in': {},
    'plot_histogram': True,
    'plot_chosen_stimuli': True,
    'plot_distribution': True,
    'plot_legend': True
}

# --- Derived Parameters and Input Validation ---
# Overall edges and main boundary
overall_left_edge = params['left_edge_value']
overall_right_edge = params['right_edge_value']
main_boundary_actual = params['boundary_value']
# Validate custom edge and boundary values
if overall_left_edge >= main_boundary_actual:
    raise ValueError('left_edge_value must be less than boundary_value.')

if main_boundary_actual >= overall_right_edge:
    raise ValueError('boundary_value must be less than right_edge_value.')

total_range = overall_right_edge - overall_left_edge
if params['central_region_width'] > total_range:
    raise ValueError('central_region_width cannot exceed the total range.')
# Calculate P_left_derived from left_probability and right_probability
total_prob = params['left_probability'] + params['right_probability']
if total_prob == 0:
    raise ValueError('left_probability and right_probability cannot both be zero.')

P_left_derived = params['left_probability'] / total_prob # normalized probability of left side

# --- Determine Peak Behavior Flags for each distribution type ---
def check_dist(dist_str, target, is_anti=False):
    '''
    Check if the target distribution is in the distribution string (name of the distribution).
    '''
    target_in = target.lower() in dist_str.lower() # Is target in the distribution string?
    anti_in = 'anti' in dist_str.lower() # Is 'anti' in the distribution string?
    if is_anti:
        return target_in and anti_in
    return target_in and not anti_in

# For Exponential: true means peak at main_boundary_actual, false means peak at respective edge
params['left_exp_peak_at_boundary_flag'] = check_dist(params['left_dist_type'], 'Exponential')
params['right_exp_peak_at_boundary_flag'] = check_dist(params['right_dist_type'], 'Exponential')
# For Half Normal: true means peak at main_boundary_actual, false means peak at respective edge
params['left_hn_peak_at_boundary_flag'] = check_dist(params['left_dist_type'], 'Half Normal')
params['right_hn_peak_at_boundary_flag'] = check_dist(params['right_dist_type'], 'Half Normal')
# For Sinusoidal: true means peak at main_boundary_actual, false means peak at respective edge
params['left_sin_peak_at_boundary_flag'] = check_dist(params['left_dist_type'], 'Sinusoidal')
params['right_sin_peak_at_boundary_flag'] = check_dist(params['right_dist_type'], 'Sinusoidal')

# Determine actual lambda values for exponential 
params['lambda_left_actual'] = params['decay_rate_magnitude_left']
if 'anti exponential' in params['left_dist_type'].lower():
    params['lambda_left_actual'] = -params['decay_rate_magnitude_left']
params['lambda_right_actual'] = params['decay_rate_magnitude_right']
if 'anti exponential' in params['right_dist_type'].lower():
    params['lambda_right_actual'] = -params['decay_rate_magnitude_right']

# Calculate Active Ranges and Sampling Zones
active_left_dist_start = overall_left_edge + (main_boundary_actual - overall_left_edge) * (1 - params['range_percentage_left'] / 100)
active_right_dist_end = main_boundary_actual + (overall_right_edge - main_boundary_actual) * (params['range_percentage_right'] / 100)
min_central_sampling_range = main_boundary_actual - params['central_region_width'] / 2
max_central_sampling_range = main_boundary_actual + params['central_region_width'] / 2

# --- Helper Functions (Nested for parameter access) ---
def get_pdf_general(x, dist_type_full_name, current_params, min_val, max_val, 
                    is_exp_peak_at_boundary_flag, is_hn_peak_at_boundary_flag, 
                    is_sin_peak_at_boundary_flag, current_main_boundary):
    """
    Calculate the Probability Density Function (PDF) for various truncated distributions.

    This function supports Uniform, Exponential, Normal, Half-Normal, and Sinusoidal 
    distributions. It ensures the resulting PDF is correctly normalized so that the 
    total area under the curve within the range [min_val, max_val] equals 1.0.

    Parameters:
        x (array_like): Input values where the PDF is evaluated.
        dist_type_full_name (str): Type of distribution (e.g., 'normal', 'exponential').
        current_params (dict): Dictionary containing distribution parameters (mu, sigma, lambda, etc.).
        min_val (float): Lower bound of the truncation range of this distribution.
        max_val (float): Upper bound of the truncation range of this distribution.
        is_exp_peak_at_boundary_flag (bool): If True, peaks the exponential dist at the main boundary.
        is_hn_peak_at_boundary_flag (bool): If True, peaks the half-normal dist at the main boundary.
        is_sin_peak_at_boundary_flag (bool): If True, peaks the sinusoidal dist at the main boundary.
        current_main_boundary (float): Reference point used to determine the direction/peak of the distribution.

    Returns:
        ndarray: An array of the same shape as x containing the calculated PDF values. 
                 Values outside [min_val, max_val] are set to 0.0.
    """
    x = np.atleast_1d(x) # => ndarray
    pdf_val = np.zeros_like(x, dtype=float) # [0.0,0.0,...0.0]
    idx_in_range = (x >= min_val) & (x <= max_val)
    x_in_range = x[idx_in_range] # => only numbers in the range
    range_length = max_val - min_val
    if range_length <= 0 or x_in_range.size == 0:
        return pdf_val
    dist_name = dist_type_full_name.lower()
    # --- Case: Uniform ---
    if dist_name == 'uniform':
        pdf_val[idx_in_range] = 1.0 / range_length
    # --- Case: Exponential / Anti Exponential ---
    elif 'exponential' in dist_name:
        lam = current_params['lambda_actual']
        if abs(lam) < 1e-6:
            pdf_val[idx_in_range] = 1.0 / range_length # => uniform
        else:
            if is_exp_peak_at_boundary_flag:
                lambda_eff = abs(lam) if min_val < current_main_boundary else -abs(lam)
            else:
                lambda_eff = -abs(lam) if min_val < current_main_boundary else abs(lam)
            ## Normalization
            denominator = np.exp(lambda_eff * max_val) - np.exp(lambda_eff * min_val) # denominator of the normalization factor
            if denominator == 0:
                pdf_val[idx_in_range] = 0
            else:
                normalization_factor = lambda_eff / denominator
                pdf_val[idx_in_range] = normalization_factor * np.exp(lambda_eff * x_in_range)
    # --- Case: Normal ---
    elif dist_name == 'normal':
        mu = current_params['normal_mean']
        sigma = current_params['normal_std_dev']
        cdf_min = norm.cdf(min_val, mu, sigma)
        cdf_max = norm.cdf(max_val, mu, sigma)
        trunc_prob = cdf_max - cdf_min
        if trunc_prob <= 0:
            pdf_val[idx_in_range] = 0
        else:
            pdf_val[idx_in_range] = norm.pdf(x_in_range, mu, sigma) / trunc_prob
    # --- Case: Half Normal / Anti Half Normal ---
    elif 'half normal' in dist_name:
        sigma = current_params['half_normal_std_dev']
        
        if is_hn_peak_at_boundary_flag:
            mu_half_normal = current_main_boundary
        else:
            if max_val == current_main_boundary: # Left side
                mu_half_normal = min_val
            else: # Right side
                mu_half_normal = max_val
            
        integral_over_range = norm.cdf(max_val, mu_half_normal, sigma) - norm.cdf(min_val, mu_half_normal, sigma)
        if integral_over_range <= 0:
            pdf_val[idx_in_range] = 0
        else:
            pdf_val[idx_in_range] = 2 * norm.pdf(x_in_range, mu_half_normal, sigma) / integral_over_range
    # --- Case: Sinusoidal / Anti Sinusoidal ---
    elif 'sinusoidal' in dist_name:
        amplitude = current_params['sinusoidal_amplitude_factor']
        frequency = current_params['sinusoidal_frequency_factor']
        
        if is_sin_peak_at_boundary_flag:
            target_x_peak = max_val if (min_val < current_main_boundary and max_val == current_main_boundary) else min_val
        else:
            target_x_peak = min_val if (min_val < current_main_boundary and max_val == current_main_boundary) else max_val
            
        calculated_phase = np.pi/2 - frequency * np.pi * (target_x_peak - min_val) / range_length
        ## Normalization
        def unnormalized_func(val):
            '''
            Unnormalized function of the sinusoidal distribution.
            '''
            return 1 + amplitude * np.sin(frequency * np.pi * (val - min_val) / range_length + calculated_phase)
        
        if frequency == 0:
            integral_val = (1 + amplitude * np.sin(calculated_phase)) * range_length
        else:
            k_norm = frequency * np.pi / range_length
            integral_val = (range_length * (1 + amplitude * np.sin(calculated_phase)) - 
                           (amplitude / k_norm) * (np.cos(frequency * np.pi + calculated_phase) - np.cos(calculated_phase)))
        
        normalization_constant = 1 / integral_val if integral_val > 0 else 0 # if integral_val is 0, then the normalization constant is 0
        pdf_val[idx_in_range] = normalization_constant * unnormalized_func(x_in_range)

    return pdf_val # => array of pdf values for the input x values


def generate_rand_general(dist_type_full_name, current_params, min_val, max_val, 
                          is_exp_peak_at_boundary_flag, is_hn_peak_at_boundary_flag, 
                          is_sin_peak_at_boundary_flag, current_main_boundary):
    
    range_length = max_val - min_val
    if range_length <= 0:
        return min_val

    #  Nested helper to encapsulate the generation logic for one attempt
    def generate_single_attempt():
        """
        Generate a single random sample from the specified truncated distribution.

        This function uses Inverse Transform Sampling to convert a uniform random 
        number [0, 1) into a value that follows the shape of the chosen 
        distribution (Uniform, Exponential, Normal, Half-Normal, or Sinusoidal) 
        within the bounds [min_val, max_val].

        Returns:
            float: A single random value sampled from the target distribution.
            Defaults to a uniform sample if calculation fails or parameters are invalid.
        """
        u_val_used = np.random.rand() # uniformly sample from [0,1)
        dist_name = dist_type_full_name.lower()
        val = min_val + u_val_used * range_length # val is the uniform sample from [min_val, max_val)

        # --- Case: Uniform ---
        if dist_name == 'uniform':
            val = min_val + u_val_used * range_length

        # --- Case: Exponential / Anti Exponential ---
        elif 'exponential' in dist_name:
            lam = current_params['lambda_actual']
            if abs(lam) < 1e-6:
                val = min_val + u_val_used * range_length
            else:
                if is_exp_peak_at_boundary_flag:
                    lambda_eff = abs(lam) if min_val < current_main_boundary else -abs(lam)
                else:
                    lambda_eff = -abs(lam) if min_val < current_main_boundary else abs(lam)
                
                # e^{lambda x} = term_inside_log
                term_inside_log = u_val_used * (np.exp(lambda_eff * max_val) - np.exp(lambda_eff * min_val)) + np.exp(lambda_eff * min_val)
                if term_inside_log <= 0:
                    val = min_val + u_val_used * range_length
                else:
                    val = (1.0 / lambda_eff) * np.log(term_inside_log) # => val is the sample from the exponential distribution

        # --- Case: Normal ---
        elif dist_name == 'normal':
            mu = current_params['normal_mean']
            sigma = current_params['normal_std_dev']
            cdf_min = norm.cdf(min_val, loc=mu, scale=sigma)
            cdf_max = norm.cdf(max_val, loc=mu, scale=sigma)
            val = norm.ppf(cdf_min + u_val_used * (cdf_max - cdf_min), loc=mu, scale=sigma)

        # --- Case: Half Normal / Anti Half Normal ---
        elif 'half normal' in dist_name:
            sigma = current_params['half_normal_std_dev']
            if is_hn_peak_at_boundary_flag:
                mu_hn = current_main_boundary
            else:
                mu_hn = min_val if max_val == current_main_boundary else max_val
            
            cdf_min_trunc = norm.cdf(min_val, loc=mu_hn, scale=sigma)
            cdf_max_trunc = norm.cdf(max_val, loc=mu_hn, scale=sigma)
            rand_candidate = norm.ppf(cdf_min_trunc + u_val_used * (cdf_max_trunc - cdf_min_trunc), loc=mu_hn, scale=sigma)
            
            if is_hn_peak_at_boundary_flag:
                if min_val < current_main_boundary:
                    val = current_main_boundary - abs(rand_candidate - current_main_boundary)
                else:
                    val = current_main_boundary + abs(rand_candidate - current_main_boundary)
            else:
                if max_val == current_main_boundary:
                    val = mu_hn + abs(rand_candidate - mu_hn)
                else:
                    val = mu_hn - abs(rand_candidate - mu_hn)
            val = np.clip(val, min_val, max_val)

        # --- Case: Sinusoidal / Anti Sinusoidal ---
        elif 'sinusoidal' in dist_name:
            amp = current_params['sinusoidal_amplitude_factor']
            freq = current_params['sinusoidal_frequency_factor']
            
            if is_sin_peak_at_boundary_flag:
                target_x_peak = max_val if (min_val < current_main_boundary and max_val == current_main_boundary) else min_val
            else:
                target_x_peak = min_val if (min_val < current_main_boundary and max_val == current_main_boundary) else max_val
            
            range_len = max_val - min_val
            calc_phase = np.pi/2 - freq * np.pi * (target_x_peak - min_val) / range_len
            
            if freq == 0:
                integral_val = (1 + amp * np.sin(calc_phase)) * range_len
            else:
                k_norm = freq * np.pi / range_len
                integral_val = (range_len * (1 + amp * np.sin(calc_phase)) - 
                               (amp / k_norm) * (np.cos(freq * np.pi + calc_phase) - np.cos(calc_phase)))
            
            if integral_val <= 0:
                val = min_val + u_val_used * range_len
            else:
                norm_const = 1.0 / integral_val
                # define cdf
                def cdf_func_sinusoidal(x_v):
                    k = freq * np.pi / range_len
                    return norm_const * (
                        (x_v - min_val) + 
                        (amp / k) * (-np.cos(k * (x_v - min_val) + calc_phase) + np.cos(calc_phase))
                    ) - u_val_used
                
                try:
                    val = brentq(cdf_func_sinusoidal, min_val, max_val)
                except:
                    val = min_val + u_val_used * range_len
        return val # => val is the sample from the sinusoidal distribution

    # Initial generation attempt
    rand_val = generate_single_attempt() # actually a sample from the distribution

    # Ensure rand_val is strictly between min_val and max_val
    while rand_val <= min_val or rand_val >= max_val:
        rand_val = generate_single_attempt()
        
    return rand_val

# --- Plot Static Elements Helper Function ---
def initialize_static_plot(ax, total_pdf_vals, x_plot, overall_left_edge, overall_right_edge, 
                           main_boundary_actual, min_central_sampling_range, 
                           max_central_sampling_range, active_left_dist_start, 
                           active_right_dist_end, plot_legend):
    '''
    Plot the static elements of the plot.
    '''
    ax.clear() # clear the axes
    
    # Plot the Bimodal Probability Density Function (PDF)
    h_pdf, = ax.plot(x_plot, total_pdf_vals, 'b-', linewidth=2, label='Overall PDF') # theoretical PDF(1st element of the list)
    pdf_max = np.max(total_pdf_vals) if len(total_pdf_vals) > 0 else 1.0 # pdf_max is the highest point of pdf function
    y_limit = pdf_max * 1.1

    # Add a dashed red line at the main boundary (x=0) for clarity
    h_main_boundary_line = ax.axvline(x=main_boundary_actual, color='r', linestyle='--', 
                                      linewidth=1.5, label='Main Boundary')

    # Plot the central sampling region boundaries
    h_central_left_line = ax.axvline(x=min_central_sampling_range, color='g', linestyle=':', linewidth=1)
    h_central_right_line = ax.axvline(x=max_central_sampling_range, color='g', linestyle=':', 
                                      linewidth=1, label='Central Bias Zone Boundary')
    
    # Boundary text
    # h_central_text_left = ax.text(min_central_sampling_range, y_limit * 1.05, f'CB Start ({min_central_sampling_range:.2f})', 
    #         color='g', fontsize=8, ha='center', va='bottom')
    # h_central_text_right = ax.text(max_central_sampling_range, y_limit * 1.05, f'CB End ({max_central_sampling_range:.2f})', 
    #         color='g', fontsize=8, ha='center', va='bottom')

    # Plot the overall range boundaries (user-defined edges)
    h_overall_left_edge_line = ax.axvline(x=overall_left_edge, color='k', linestyle='--', linewidth=1)
    h_overall_right_edge_line = ax.axvline(x=overall_right_edge, color='k', linestyle='--', 
                                           linewidth=1, label='Overall Range Edge')
    
    # h_overall_left_edge_text = ax.text(overall_left_edge, y_limit * 1.05, f'Left Edge ({overall_left_edge:.2f})', 
    #        color='k', fontsize=8, ha='center', va='bottom')
    # h_overall_right_edge_text = ax.text(overall_right_edge, y_limit * 1.05, f'Right Edge ({overall_right_edge:.2f})', 
    #        color='k', fontsize=8, ha='center', va='bottom')

    # Plot the active distribution range boundaries (derived from percentage)
    h_active_left_dist_line = ax.axvline(x=active_left_dist_start, color='m', linestyle=':', linewidth=1)
    h_active_right_dist_line = ax.axvline(x=active_right_dist_end, color='m', linestyle=':', 
                                          linewidth=1, label='Active Dist. Range')
    
    h_active_left_dist_text = ax.text(active_left_dist_start, y_limit, f'Left Active ({active_left_dist_start:.2f})', 
            color='m', fontsize=8, ha='center', va='top')
    h_active_right_dist_text = ax.text(active_right_dist_end, y_limit, f'Right Active ({active_right_dist_end:.2f})', 
            color='m', fontsize=8, ha='center', va='top')

    # Customize plot appearance (static elements)
    ax.set_title('Bimodal PDF with Central Sampling Bias')
    ax.set_xlabel('Value (x)')
    ax.set_ylabel('Probability Density')
    ax.grid(True, linestyle=':', alpha=0.6)
    
    x_range = overall_right_edge - overall_left_edge
    ax.set_xlim(overall_left_edge - 0.1 * x_range, overall_right_edge + 0.1 * x_range) 
    ax.set_ylim(0, y_limit * 1.2)

    plot_handles = {
        'h_pdf': h_pdf,
        'h_main_boundary_line': h_main_boundary_line,
        'h_central_left_line': h_central_left_line,
        'h_central_right_line': h_central_right_line,
        # 'h_central_text_left': h_central_text_left,
        # 'h_central_text_right': h_central_text_right,
        'h_overall_left_edge_line': h_overall_left_edge_line,
        'h_overall_right_edge_line': h_overall_right_edge_line,
        # 'h_overall_left_edge_text': h_overall_left_edge_text,
        # 'h_overall_right_edge_text': h_overall_right_edge_text,
        'h_active_left_dist_line': h_active_left_dist_line,
        'h_active_right_dist_line': h_active_right_dist_line,
        'h_active_left_dist_text': h_active_left_dist_text,
        'h_active_right_dist_text': h_active_right_dist_text
    }

    # Dynamically build legend entries
    if plot_legend:
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        
        legend_elements = [
            h_pdf, 
            h_main_boundary_line, 
            h_central_right_line, 
            h_overall_right_edge_line, 
            h_active_right_dist_line,
            Patch(facecolor='gray', edgecolor='gray', alpha=0.5, label='Sample History Histogram'),
            Line2D([0], [0], marker='o', color='w', label='Current Pick', 
                   markerfacecolor='g', markersize=10)
        ]
        plot_handles['h_legend'] = ax.legend(handles=legend_elements, loc='best', fontsize='small')

    return plot_handles

# -- Update Dynamic Plot Elements Helper Function ---
def update_dynamic_plot(ax, current_stimulus, samples_history_in, plot_handles_in, 
                        plot_histogram_flag, plot_chosen_stimuli_flag, 
                        plot_legend_flag, plot_distribution_flag):
    '''
    Updates sample history, histogram, pick marker/label, legend visibility, and static distribution artists on the axes for the current trial.
    '''
    
    samples_history_out = list(samples_history_in)
    if not np.isnan(current_stimulus):
        samples_history_out.append(current_stimulus) # make a copy
    
    plot_handles_out = plot_handles_in
    # Update Histogram
    if plot_histogram_flag:
        if 'h_hist' in plot_handles_out and plot_handles_out['h_hist'] is not None:
            for patch in plot_handles_out['h_hist']:
                patch.remove()

        if len(samples_history_out) > 0:
            _, _, patches = ax.hist(samples_history_out, bins=50, density=True, 
                                     facecolor=(0.7, 0.7, 0.7), edgecolor=(0.5, 0.5, 0.5), 
                                     alpha=0.5, zorder=1)
            plot_handles_out['h_hist'] = patches
    else:
        if 'h_hist' in plot_handles_out and plot_handles_out['h_hist'] is not None:
            for patch in plot_handles_out['h_hist']:
                patch.remove()
            plot_handles_out['h_hist'] = None

    # Update Current Stimulus Marker
    if plot_chosen_stimuli_flag and not np.isnan(current_stimulus):
        if 'h_pick_marker' in plot_handles_out and plot_handles_out['h_pick_marker'] is not None:
            plot_handles_out['h_pick_marker'].set_data([current_stimulus], [0])
            plot_handles_out['h_pick_marker'].set_visible(True)
        else:
            line, = ax.plot(current_stimulus, 0, 'go', markersize=10, markerfacecolor='g', zorder=5)
            plot_handles_out['h_pick_marker'] = line
            
        # Update Legend
        pick_str = f"Picked: {current_stimulus:.4f}"
        y_pos = plot_handles_out.get('total_pdf_vals_max', 1.0) * 0.05
        if 'h_pick_text' in plot_handles_out and plot_handles_out['h_pick_text'] is not None:
            plot_handles_out['h_pick_text'].set_position((current_stimulus, y_pos))
            plot_handles_out['h_pick_text'].set_text(pick_str)
            plot_handles_out['h_pick_text'].set_visible(True)
        else:
            txt = ax.text(current_stimulus, y_pos, pick_str, color='g', 
                          fontweight='bold', ha='center', va='bottom', zorder=6)
            plot_handles_out['h_pick_text'] = txt
    else:
        # hide
        if 'h_pick_marker' in plot_handles_out and plot_handles_out['h_pick_marker']:
            plot_handles_out['h_pick_marker'].set_visible(False)
        if 'h_pick_text' in plot_handles_out and plot_handles_out['h_pick_text']:
            plot_handles_out['h_pick_text'].set_visible(False)

    # Legend
    if 'h_legend' in plot_handles_out and plot_handles_out['h_legend']:
        plot_handles_out['h_legend'].set_visible(plot_legend_flag)

    static_visibility = plot_distribution_flag
    static_handle_names = [
        'h_pdf', 'h_main_boundary_line', 'h_central_left_line', 'h_central_right_line', 
        'h_central_text_left', 'h_central_text_right', 'h_overall_left_edge_line', 
        'h_overall_left_edge_text', 'h_overall_right_edge_line', 'h_overall_right_edge_text', 
        'h_active_left_dist_line', 'h_active_left_dist_text', 'h_active_right_dist_line', 
        'h_active_right_dist_text'
    ]
    
    for name in static_handle_names:
        if name in plot_handles_out and plot_handles_out[name] is not None:
            plot_handles_out[name].set_visible(static_visibility)

    return samples_history_out, plot_handles_out

# --- Main Function Logic ---
def main_logic(params, P_left_derived, overall_left_edge, overall_right_edge, 
               main_boundary_actual, min_central_sampling_range, max_central_sampling_range,
               active_left_dist_start, active_right_dist_end):
    for side in ['left', 'right']:
        dist_type = params.get(f'{side}_dist_type', '').lower()
        if 'exponential' in dist_type:
            mag = params.get(f'decay_rate_magnitude_{side}', 0) # mag is 
            params[f'lambda_{side}_actual'] = -mag if 'anti' in dist_type else mag
        else:
            params[f'lambda_{side}_actual'] = 0
    
    for side in ['left', 'right']:
        for prefix in ['exp', 'hn', 'sin']:
            key = f'{side}_{prefix}_peak_at_boundary_flag'
            if key not in params:
                params[key] = False
    
    def get_side_params(side):
        return {
            'lambda_actual': params.get(f'lambda_{side}_actual', 0),
            'normal_mean': params.get(f'normal_mean_{side}', 0),
            'normal_std_dev': params.get(f'normal_std_dev_{side}', 0.1),
            'half_normal_std_dev': params.get(f'half_normal_std_dev_{side}', 0.1),
            'sinusoidal_amplitude_factor': params.get(f'sinusoidal_amplitude_factor_{side}', 0),
            'sinusoidal_frequency_factor': params.get(f'sinusoidal_frequency_factor_{side}', 0)
        }

    current_params_left = get_side_params('left')
    current_params_right = get_side_params('right')

    samples_history_out = []
    plot_handles_out = {}

    # --- Mode: generate_single_sample ---
    if params['Mode'] == 'generate_single_sample':
        actual_side = params['chosen_side'].lower()
        if actual_side == 'auto':
            actual_side = 'left' if np.random.rand() < P_left_derived else 'right'

        if np.random.rand() < params['P_central_region']:
            current_pick_found = False
            while not current_pick_found:
                if actual_side == 'left':
                    candidate = generate_rand_general(params['left_dist_type'], current_params_left, 
                                                      overall_left_edge, main_boundary_actual,
                                                      params['left_exp_peak_at_boundary_flag'],
                                                      params['left_hn_peak_at_boundary_flag'],
                                                      params['left_sin_peak_at_boundary_flag'], main_boundary_actual)
                else:
                    candidate = generate_rand_general(params['right_dist_type'], current_params_right, 
                                                      main_boundary_actual, overall_right_edge,
                                                      params['right_exp_peak_at_boundary_flag'],
                                                      params['right_hn_peak_at_boundary_flag'],
                                                      params['right_sin_peak_at_boundary_flag'], main_boundary_actual)
                
                if min_central_sampling_range <= candidate <= max_central_sampling_range:
                    samples_history_out = candidate
                    current_pick_found = True
        else:
            # non-bias sampling
            if actual_side == 'left':
                samples_history_out = generate_rand_general(params['left_dist_type'], current_params_left, 
                                                            active_left_dist_start, main_boundary_actual,
                                                            params['left_exp_peak_at_boundary_flag'],
                                                            params['left_hn_peak_at_boundary_flag'],
                                                            params['left_sin_peak_at_boundary_flag'], main_boundary_actual)
            else:
                samples_history_out = generate_rand_general(params['right_dist_type'], current_params_right, 
                                                            main_boundary_actual, active_right_dist_end,
                                                            params['right_exp_peak_at_boundary_flag'],
                                                            params['right_hn_peak_at_boundary_flag'],
                                                            params['right_sin_peak_at_boundary_flag'], main_boundary_actual)
        return samples_history_out, None, {}
    
    # --- Mode: initialize_plot ---
    if params['Mode'] == 'initialize_plot':
        plt.ion()
        fig, ax = plt.subplots(figsize=(10, 6))
        x_plot = np.linspace(overall_left_edge, overall_right_edge, 500)

        pdf_l = get_pdf_general(x_plot, params['left_dist_type'], current_params_left, 
                                active_left_dist_start, main_boundary_actual,
                                params['left_exp_peak_at_boundary_flag'], 
                                params['left_hn_peak_at_boundary_flag'], 
                                params['left_sin_peak_at_boundary_flag'], main_boundary_actual)
        
        pdf_r = get_pdf_general(x_plot, params['right_dist_type'], current_params_right, 
                                main_boundary_actual, active_right_dist_end,
                                params['right_exp_peak_at_boundary_flag'], 
                                params['right_hn_peak_at_boundary_flag'], 
                                params['right_sin_peak_at_boundary_flag'], main_boundary_actual)

        total_pdf_vals = P_left_derived * pdf_l + (1 - P_left_derived) * pdf_r
        
        plot_handles_out = initialize_static_plot(ax, total_pdf_vals, x_plot, overall_left_edge, 
                                                 overall_right_edge, main_boundary_actual, 
                                                 min_central_sampling_range, max_central_sampling_range, 
                                                 active_left_dist_start, active_right_dist_end, params['plot_legend'])
        
        plt.show()
        plt.pause(0.1)
        return None, fig, plot_handles_out
    

    # --- Mode: run_simulation ---
    elif params['Mode'] == 'run_simulation':
        plt.ion()
        fig, ax = plt.subplots(figsize=(10, 6))
        x_plot = np.linspace(overall_left_edge, overall_right_edge, 500)

        pdf_l = get_pdf_general(x_plot, params['left_dist_type'], current_params_left, 
                                active_left_dist_start, main_boundary_actual,
                                params['left_exp_peak_at_boundary_flag'], 
                                params['left_hn_peak_at_boundary_flag'], 
                                params['left_sin_peak_at_boundary_flag'], main_boundary_actual)
        
        pdf_r = get_pdf_general(x_plot, params['right_dist_type'], current_params_right, 
                                main_boundary_actual, active_right_dist_end,
                                params['right_exp_peak_at_boundary_flag'], 
                                params['right_hn_peak_at_boundary_flag'], 
                                params['right_sin_peak_at_boundary_flag'], main_boundary_actual)

        total_pdf_vals = P_left_derived * pdf_l + (1 - P_left_derived) * pdf_r
        
        plot_handles_out = initialize_static_plot(ax, total_pdf_vals, x_plot, overall_left_edge, 
                                                 overall_right_edge, main_boundary_actual, 
                                                 min_central_sampling_range, max_central_sampling_range, 
                                                 active_left_dist_start, active_right_dist_end, params['plot_legend'])
        
        plot_handles_out['total_pdf_vals_max'] = np.max(total_pdf_vals)

        print(f"Starting simulation of {params['num_simulations']} random picks...")

        for k in range(1, params['num_simulations'] + 1):
            if np.random.rand() < params['P_central_region']:
                found = False
                while not found:
                    side = 'left' if np.random.rand() < P_left_derived else 'right'
                    if side == 'left':
                        candidate = generate_rand_general(params['left_dist_type'], current_params_left, 
                                                         overall_left_edge, main_boundary_actual,
                                                         params['left_exp_peak_at_boundary_flag'],
                                                         params['left_hn_peak_at_boundary_flag'],
                                                         params['left_sin_peak_at_boundary_flag'], main_boundary_actual)
                    else:
                        candidate = generate_rand_general(params['right_dist_type'], current_params_right, 
                                                         main_boundary_actual, overall_right_edge,
                                                         params['right_exp_peak_at_boundary_flag'],
                                                         params['right_hn_peak_at_boundary_flag'],
                                                         params['right_sin_peak_at_boundary_flag'], main_boundary_actual)
                    
                    if min_central_sampling_range <= candidate <= max_central_sampling_range:
                        current_pick = candidate
                        found = True
            else:
                # non-bias sampling
                side = 'left' if np.random.rand() < P_left_derived else 'right'
                if side == 'left':
                    current_pick = generate_rand_general(params['left_dist_type'], current_params_left, 
                                                        active_left_dist_start, main_boundary_actual,
                                                        params['left_exp_peak_at_boundary_flag'],
                                                        params['left_hn_peak_at_boundary_flag'],
                                                        params['left_sin_peak_at_boundary_flag'], main_boundary_actual)
                else:
                    current_pick = generate_rand_general(params['right_dist_type'], current_params_right, 
                                                        main_boundary_actual, active_right_dist_end,
                                                        params['right_exp_peak_at_boundary_flag'],
                                                        params['right_hn_peak_at_boundary_flag'],
                                                        params['right_sin_peak_at_boundary_flag'], main_boundary_actual)

            # update plot
            samples_history_out, plot_handles_out = update_dynamic_plot(ax, current_pick, samples_history_out, 
                                                                        plot_handles_out, True, True, True, True)
            
            plt.draw()
            plt.pause(params['pause_duration'])
            
        plt.ioff()
        print("Simulation complete.")
        return samples_history_out, fig, plot_handles_out

    return samples_history_out, None, plot_handles_out # None: align different modes