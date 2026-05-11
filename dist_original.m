function [samples_history_out, h_fig_out, plot_handles_out] = CreateSamples_from_Distribution(varargin)

% CreateSamples_from_Distribution
%
% This function is specifically called within StimulusSection to perform
% the actions

% This function generates random samples from a customizable bimodal
% distribution and can optionally plot the distribution, a history histogram,
% and dynamic current picks.
%
% The overall distribution spans a user-defined range [left_edge_value, right_edge_value]
% with a user-defined boundary at 'boundary_value'.
% The actual distribution (PDF) for each side is generated over a percentage
% of the range between the overall edge and the boundary.
% A central sampling bias mechanism is added to draw more picks near the
% boundary without altering the PDF shape.
%
% Inputs (Name-Value Pairs):
%   'Mode'                        - (char) 'run_simulation' (default), 'generate_single_sample', 'initialize_plot', or 'update_plot'.
%   'left_probability'            - (double) Relative probability for the left side.
%   'right_probability'           - (double) Relative probability for the right side.
%   'chosen_side'                 - (char, optional) For 'generate_single_sample' mode: 'left', 'right', or 'auto' (default).
%                                   If 'left' or 'right', the sample is forced from that side.
%
%   'left_edge_value'             - (double) The absolute value for the leftmost boundary of the overall plot range.
%   'boundary_value'              - (double) The absolute value for the central boundary between left and right distributions.
%   'right_edge_value'            - (double) The absolute value for the rightmost boundary of the overall plot range.
%
%   'left_dist_type'              - (char) Type for left side: 'Uniform', 'Exponential', 'Anti_Exponential',
%                                   'Normal', 'Half Normal', 'Anti Half Normal', 'Sinusoidal', 'Anti_Sinusoidal'.
%   'decay_rate_magnitude_left'   - (double) Positive magnitude for 'Exponential' types.
%   'normal_mean_left'            - (double) Mean for 'Normal' distribution.
%   'normal_std_dev_left'         - (double) Standard deviation for 'Normal' distribution.
%   'half_normal_std_dev_left'    - (double) Standard deviation for 'Half Normal' types.
%   'sinusoidal_amplitude_factor_left' - (double) Amplitude (0 to 1) for 'Sinusoidal' types.
%   'sinusoidal_frequency_factor_left' - (double) Frequency for 'Sinusoidal' types.
%   'range_percentage_left'       - (double) Percentage (0 to 100) of the [left_edge_value, boundary_value] range
%                                   that the left distribution is active over (e.g., 70 for inner 70%).
%
%   'right_dist_type'             - (char) Type for right side (same options as left_dist_type).
%   'decay_rate_magnitude_right'  - (double) Positive magnitude for 'Exponential' types.
%   'normal_mean_right'           - (double) Mean for 'Normal' distribution.
%   'normal_std_dev_right'        - (double) Standard deviation for 'Normal' distribution.
%   'half_normal_std_dev_right'   - (double) Standard deviation for 'Half Normal' types.
%   'sinusoidal_amplitude_factor_right' - (double) Amplitude (0 to 1) for 'Sinusoidal' types.
%   'sinusoidal_frequency_factor_right' - (double) Frequency for 'Sinusoidal' types.
%   'range_percentage_right'      - (double) Percentage (0 to 100) of the [boundary_value, right_edge_value] range
%                                   that the right distribution is active over (e.g., 70 for inner 70%).
%
%   'P_central_region'            - (double) Probability (0 to 1) of applying central sampling bias.
%   'central_region_width'        - (double) Width of the central zone for biased sampling.
%
%   'num_simulations'             - (int) Number of random picks to simulate (only in 'run_simulation' mode).
%   'pause_duration'              - (double) Pause duration between updates in seconds (only in 'run_simulation' mode).
%
%   For 'initialize_plot' mode:
%   'ax_handle'                   - (axes handle, optional) Handle to the axes to plot on. If not provided, a new figure/axes is created.
%
%   For 'update_plot' mode:
%   'ax_handle'                   - (axes handle) Handle to the axes to plot on.
%   'current_stimulus'            - (double) The most recent stimulus value to plot.
%   'samples_history_in'          - (double array) The current history of all samples.
%   'plot_handles_in'             - (struct) Struct containing handles of existing plot elements (from initialize_static_plot).
%   'plot_histogram'              - (logical) True to plot/update histogram.
%   'plot_chosen_stimuli'         - (logical) True to plot/update current stimulus marker.
%   'plot_distribution'           - (logical) True to plot/show the PDF curve.
%   'plot_legend'                 - (logical) True to plot/show the legends.
%
% Outputs:
%   samples_history_out           - (double array) Array of generated samples.
%                                   If 'generate_single_sample' mode, it's a single value.
%   h_fig_out                     - (figure handle) Handle to the plot figure.
%   plot_handles_out              - (struct) Struct containing handles of plot elements.

% --- Input Parsing ---
p = inputParser;
p.KeepUnmatched = true; % Allow unmatched for future flexibility

% Define parameters and their default values
addParameter(p, 'Mode', 'run_simulation', @(x) ismember(x, {'run_simulation', 'generate_single_sample', 'initialize_plot', 'update_plot'}));

% Side Probabilities
addParameter(p, 'left_probability', 0.5, @(x) isscalar(x) && x >= 0);
addParameter(p, 'right_probability', 0.5, @(x) isscalar(x) && x >= 0);
addParameter(p, 'chosen_side', 'auto', @(x) ismember(lower(x), {'left', 'right', 'auto'})); % For single sample mode

% Custom Boundary and Edge Values
addParameter(p, 'left_edge_value', -1.0, @(x) isscalar(x) && isnumeric(x));
addParameter(p, 'boundary_value', 0.0, @(x) isscalar(x) && isnumeric(x));
addParameter(p, 'right_edge_value', 1.0, @(x) isscalar(x) && isnumeric(x));

% Left Side Parameters
addParameter(p, 'left_dist_type', 'Half Normal', @ischar);
addParameter(p, 'decay_rate_magnitude_left', 2.153, @(x) isscalar(x) && x >= 0);
addParameter(p, 'normal_mean_left', -0.5, @isnumeric);
addParameter(p, 'normal_std_dev_left', 0.15, @(x) isscalar(x) && x > 0);
addParameter(p, 'half_normal_std_dev_left', 0.2, @(x) isscalar(x) && x > 0);
addParameter(p, 'sinusoidal_amplitude_factor_left', 0.8, @(x) isscalar(x) && x >= 0 && x <= 1);
addParameter(p, 'sinusoidal_frequency_factor_left', 1, @(x) isscalar(x) && x >= 0);
% Removed sinusoidal_peak_location_percentage_left as it's now derived
addParameter(p, 'range_percentage_left', 100, @(x) isscalar(x) && x >= 0 && x <= 100);

% Right Side Parameters
addParameter(p, 'right_dist_type', 'Sinusoidal', @ischar);
addParameter(p, 'decay_rate_magnitude_right', 5, @(x) isscalar(x) && x >= 0);
addParameter(p, 'normal_mean_right', 0.5, @isnumeric);
addParameter(p, 'normal_std_dev_right', 0.15, @(x) isscalar(x) && x > 0);
addParameter(p, 'half_normal_std_dev_right', 0.2, @(x) isscalar(x) && x > 0);
addParameter(p, 'sinusoidal_amplitude_factor_right', 0.8, @(x) isscalar(x) && x >= 0 && x <= 1);
addParameter(p, 'sinusoidal_frequency_factor_right', 1, @(x) isscalar(x) && x >= 0);
% Removed sinusoidal_peak_location_percentage_right as it's now derived
addParameter(p, 'range_percentage_right', 100, @(x) isscalar(x) && x >= 0 && x <= 100);

% Central Sampling Bias Parameters
addParameter(p, 'P_central_region', 0.3, @(x) isscalar(x) && x >= 0 && x <= 1);
addParameter(p, 'central_region_width', 0.2, @(x) isscalar(x) && x > 0);

% Simulation Parameters
addParameter(p, 'num_simulations', 200, @(x) isscalar(x) && x > 0 && mod(x,1)==0);
addParameter(p, 'pause_duration', 0.05, @(x) isscalar(x) && x >= 0);

% Plot Update Specific Parameters
addParameter(p, 'ax_handle', [], @(x) isempty(x) || isgraphics(x, 'axes'));
addParameter(p, 'current_stimulus', NaN, @isnumeric);
addParameter(p, 'samples_history_in', [], @isnumeric);
addParameter(p, 'plot_handles_in', struct(), @isstruct);
addParameter(p, 'plot_histogram', true, @islogical);
addParameter(p, 'plot_chosen_stimuli', true, @islogical);
addParameter(p, 'plot_distribution', true, @islogical);
addParameter(p, 'plot_legend', true, @islogical);


% Parse inputs
parse(p, varargin{:});
params = p.Results;

% --- Derived Parameters and Input Validation ---

% Overall edges and main boundary
overall_left_edge = params.left_edge_value;
overall_right_edge = params.right_edge_value;
main_boundary_actual = params.boundary_value;

% Validate custom edge and boundary values
if overall_left_edge >= main_boundary_actual
    error('left_edge_value must be less than boundary_value.');
end
if main_boundary_actual >= overall_right_edge
    error('boundary_value must be less than right_edge_value.');
end
if params.central_region_width > (overall_right_edge - overall_left_edge)
    error('central_region_width cannot exceed the total range (right_edge_value - left_edge_value).');
end

% Calculate P_left_derived from left_probability and right_probability
total_prob = params.left_probability + params.right_probability;
if total_prob == 0
    error('left_probability and right_probability cannot both be zero.');
end
P_left_derived = params.left_probability / total_prob;

% --- Determine Peak Behavior Flags for each distribution type ---

% For Exponential: true means peak at main_boundary_actual, false means peak at respective edge
params.left_exp_peak_at_boundary_flag = contains(params.left_dist_type, 'Exponential', 'IgnoreCase', true) && ~contains(params.left_dist_type, 'Anti', 'IgnoreCase', true);
params.right_exp_peak_at_boundary_flag = contains(params.right_dist_type, 'Exponential', 'IgnoreCase', true) && ~contains(params.right_dist_type, 'Anti', 'IgnoreCase', true);

% For Half Normal: true means peak at main_boundary_actual, false means peak at respective edge
params.left_hn_peak_at_boundary_flag = contains(params.left_dist_type, 'Half Normal', 'IgnoreCase', true) && ~contains(params.left_dist_type, 'Anti', 'IgnoreCase', true);
params.right_hn_peak_at_boundary_flag = contains(params.right_dist_type, 'Half Normal', 'IgnoreCase', true) && ~contains(params.right_dist_type, 'Anti', 'IgnoreCase', true);

% For Sinusoidal: true means peak at main_boundary_actual, false means peak at respective edge
params.left_sin_peak_at_boundary_flag = contains(params.left_dist_type, 'Sinusoidal', 'IgnoreCase', true) && ~contains(params.left_dist_type, 'Anti', 'IgnoreCase', true);
params.right_sin_peak_at_boundary_flag = contains(params.right_dist_type, 'Sinusoidal', 'IgnoreCase', true) && ~contains(params.right_dist_type, 'Anti', 'IgnoreCase', true);


% Determine actual lambda values for exponential based on derived flags
params.lambda_left_actual = params.decay_rate_magnitude_left;
if contains(params.left_dist_type, 'Anti Exponential', 'IgnoreCase', true)
    params.lambda_left_actual = -params.decay_rate_magnitude_left;
end

params.lambda_right_actual = params.decay_rate_magnitude_right;
if contains(params.right_dist_type, 'Anti Exponential', 'IgnoreCase', true)
    params.lambda_right_actual = -params.decay_rate_magnitude_right;
end

% These are the *active* ranges for the PDF definition and unbiased sampling
active_left_dist_start = overall_left_edge + (main_boundary_actual - overall_left_edge) * (1 - params.range_percentage_left / 100);
active_right_dist_end = main_boundary_actual + (overall_right_edge - main_boundary_actual) * (params.range_percentage_right / 100);

% These are the boundaries for the central sampling zone
min_central_sampling_range = main_boundary_actual - params.central_region_width / 2;
max_central_sampling_range = main_boundary_actual + params.central_region_width / 2;


% --- Helper Functions (Nested for parameter access) ---

function pdf_val = get_pdf_general(x, dist_type_full_name, current_params, min_val, max_val, is_exp_peak_at_boundary_flag, is_hn_peak_at_boundary_flag, is_sin_peak_at_boundary_flag, current_main_boundary)
    % current_params: struct with lambda_actual, normal_mean, normal_std_dev, etc.
    % current_main_boundary: The main boundary value (e.g., 0.0 or user-defined)
    
    pdf_val = zeros(size(x));
    idx_in_range = (x >= min_val & x <= max_val);
    x_in_range = x(idx_in_range);
    range_length = max_val - min_val;

    if range_length <= 0 || isempty(x_in_range)
        return;
    end

    switch lower(dist_type_full_name)
        case 'uniform'
            pdf_val(idx_in_range) = 1 / range_length;

        case {'exponential', 'anti exponential'}
            lambda = current_params.lambda_actual;
            if abs(lambda) < 1e-6
                pdf_val(idx_in_range) = 1 / range_length;
            else
                % Adjust lambda sign based on peak_at_boundary_flag for consistency
                if is_exp_peak_at_boundary_flag % Peak at current_main_boundary
                    if min_val < current_main_boundary % Left side
                        lambda_eff = abs(lambda); % Increase towards main_boundary_actual
                    else % Right side
                        lambda_eff = -abs(lambda); % Decrease towards main_boundary_actual
                    end
                else % Peak at edge (min_val or max_val)
                    if min_val < current_main_boundary % Left side
                        lambda_eff = -abs(lambda); % Decrease towards main_boundary_actual (peak at min_val)
                    else % Right side
                        lambda_eff = abs(lambda); % Increase towards max_val (peak at max_val)
                    end
                end

                denominator = (exp(lambda_eff * max_val) - exp(lambda_eff * min_val));
                if denominator == 0
                    pdf_val(idx_in_range) = 0;
                else
                    normalization_factor = lambda_eff / denominator;
                    pdf_val(idx_in_range) = normalization_factor .* exp(lambda_eff .* x_in_range);
                end
            end

        case 'normal'
            mu = current_params.normal_mean;
            sigma = current_params.normal_std_dev;
            cdf_min = normcdf(min_val, mu, sigma);
            cdf_max = normcdf(max_val, mu, sigma);
            trunc_prob = cdf_max - cdf_min;
            if trunc_prob <= 0
                pdf_val(idx_in_range) = 0;
            else
                pdf_val(idx_in_range) = normpdf(x_in_range, mu, sigma) / trunc_prob;
            end

        case {'half normal', 'anti half normal'}
            sigma = current_params.half_normal_std_dev;
            
            % Determine mu_half_normal based on is_hn_peak_at_boundary_flag and range
            if is_hn_peak_at_boundary_flag % Peak at the main boundary
                mu_half_normal = current_main_boundary;
            else % Peak at edge of the current sub-range (min_val or max_val)
                if max_val == current_main_boundary % Left side distribution (peaks at min_val)
                    mu_half_normal = min_val;
                else % Right side distribution (peaks at max_val)
                    mu_half_normal = max_val;
                end
            end

            integral_over_range = normcdf(max_val, mu_half_normal, sigma) - normcdf(min_val, mu_half_normal, sigma);
            if integral_over_range <= 0
                pdf_val(idx_in_range) = 0;
            else
                pdf_val(idx_in_range) = 2 * normpdf(x_in_range, mu_half_normal, sigma) / integral_over_range;
            end

        case {'sinusoidal', 'anti sinusoidal'}
            amplitude = current_params.sinusoidal_amplitude_factor;
            frequency = current_params.sinusoidal_frequency_factor;
            
            % Determine target x_peak based on is_sin_peak_at_boundary_flag
            if is_sin_peak_at_boundary_flag % Regular Sinusoidal: peaks at boundary
                if min_val < current_main_boundary && max_val == current_main_boundary % Left side
                    target_x_peak = max_val; % Peak at boundary_value
                else % Right side
                    target_x_peak = min_val; % Peak at boundary_value
                end
            else % Anti_Sinusoidal: peaks at outer edge
                if min_val < current_main_boundary && max_val == current_main_boundary % Left side
                    target_x_peak = min_val; % Peak at active_left_dist_start
                else % Right side
                    target_x_peak = max_val; % Peak at active_right_dist_end
                end
            end

            % Calculate phase for this target_x_peak
            % We want (frequency * pi * (target_x_peak - min_val) / range_length + calculated_phase) = pi/2 (for a positive peak)
            calculated_phase = pi/2 - frequency * pi * (target_x_peak - min_val) / range_length;

            unnormalized_func = @(val) (1 + amplitude * sin(frequency * pi * (val - min_val) / range_length + calculated_phase));
            
            if frequency == 0
                integral_val = (1 + amplitude * sin(calculated_phase)) * range_length;
            else
                k_norm = frequency * pi / range_length;
                integral_val = (range_length * (1 + amplitude * sin(calculated_phase)) - ...
                                (amplitude / k_norm) * (cos(frequency * pi + calculated_phase) - cos(calculated_phase)));
            end
            
            if integral_val <= 0
                normalization_constant = 0;
            else
                normalization_constant = 1 / integral_val;
            end
            pdf_val(idx_in_range) = normalization_constant .* unnormalized_func(x_in_range);
    end
end

function rand_val = generate_rand_general(dist_type_full_name, current_params, min_val, max_val, is_exp_peak_at_boundary_flag, is_hn_peak_at_boundary_flag, is_sin_peak_at_boundary_flag, current_main_boundary)
    % current_params: struct with lambda_actual, normal_mean, normal_std_dev, etc.
    % current_main_boundary: The main boundary value (e.g., 0.0 or user-defined)
    
    range_length = max_val - min_val;
    if range_length <= 0
        rand_val = min_val; % Fallback for zero or negative range
        return;
    end

    % Nested helper to encapsulate the generation logic for one attempt
    function [val, U_val_used] = generate_single_attempt(dist_type_full_name, current_params, min_val, max_val, is_exp_peak_at_boundary_flag, is_hn_peak_at_boundary_flag, is_sin_peak_at_boundary_flag, current_main_boundary,range_length)
        U_val_used = rand(); % Generate a uniform random number in (0,1)

        switch lower(dist_type_full_name)
            case 'uniform'
                val = min_val + U_val_used * range_length;
            case {'exponential', 'anti exponential'}
                lambda = current_params.lambda_actual;
                if abs(lambda) < 1e-6
                    val = min_val + U_val_used * range_length;
                else
                    % Adjust lambda_eff based on peak_at_boundary_flag
                    if is_exp_peak_at_boundary_flag
                        if min_val < current_main_boundary
                            lambda_eff = abs(lambda);
                        else
                            lambda_eff = -abs(lambda);
                        end
                    else
                        if min_val < current_main_boundary
                            lambda_eff = -abs(lambda);
                        else
                            lambda_eff = abs(lambda);
                        end
                    end
                    term_inside_log = U_val_used * (exp(lambda_eff * max_val) - exp(lambda_eff * min_val)) + exp(lambda_eff * min_val);
                    if term_inside_log <= 0
                        val = min_val + U_val_used * range_length; % Fallback
                    else
                        val = (1/lambda_eff) * log(term_inside_log);
                    end
                end
           
            case 'normal'
                mu = current_params.normal_mean;
                sigma = current_params.normal_std_dev;
                cdf_min = normcdf(min_val, mu, sigma);
                cdf_max = normcdf(max_val, mu, sigma);
                val = norminv(cdf_min + U_val_used * (cdf_max - cdf_min), mu, sigma);
            
            case {'half normal', 'anti half normal'}
                sigma = current_params.half_normal_std_dev;
                if is_hn_peak_at_boundary_flag
                    mu_half_normal = current_main_boundary;
                else
                    if max_val == current_main_boundary
                        mu_half_normal = min_val;
                    else
                        mu_half_normal = max_val;
                    end
                end
                cdf_min_trunc = normcdf(min_val, mu_half_normal, sigma);
                cdf_max_trunc = normcdf(max_val, mu_half_normal, sigma);
                rand_val_candidate = norminv(cdf_min_trunc + U_val_used * (cdf_max_trunc - cdf_min_trunc), mu_half_normal, sigma);
                if is_hn_peak_at_boundary_flag
                    if min_val < current_main_boundary
                        val = current_main_boundary - abs(rand_val_candidate - current_main_boundary);
                    else
                        val = current_main_boundary + abs(rand_val_candidate - current_main_boundary);
                    end
                else
                    if max_val == current_main_boundary
                        val = mu_half_normal + abs(rand_val_candidate - mu_half_normal);
                    else
                        val = mu_half_normal - abs(rand_val_candidate - mu_half_normal);
                    end
                end
                % Apply clamping to ensure it's within [min_val, max_val] before the strict check
                val = max(min_val, min(max_val, val)); 
            
            case {'sinusoidal', 'anti sinusoidal'}
                amplitude = current_params.sinusoidal_amplitude_factor;
                frequency = current_params.sinusoidal_frequency_factor;
                if is_sin_peak_at_boundary_flag
                    if min_val < current_main_boundary && max_val == current_main_boundary
                        target_x_peak = max_val;
                    else
                        target_x_peak = min_val;
                    end
                else
                    if min_val < current_main_boundary && max_val == current_main_boundary
                        target_x_peak = min_val;
                    else
                        target_x_peak = max_val;
                    end
                end
                calculated_phase = pi/2 - frequency * pi * (target_x_peak - min_val) / range_length;
                if frequency == 0
                    integral_val = (1 + amplitude * sin(calculated_phase)) * range_length;
                else
                    k_norm = frequency * pi / range_length;
                    integral_val = (range_length * (1 + amplitude * sin(calculated_phase)) - ...
                                    (amplitude / k_norm) * (cos(frequency * pi + calculated_phase) - cos(calculated_phase)));
                end
                if integral_val <= 0
                    val = min_val + U_val_used * range_length; % Fallback
                else
                    normalization_constant = 1 / integral_val;
                    cdf_func_sinusoidal = @(x_val) (...
                        normalization_constant * (...
                            (x_val - min_val) + ...
                            (amplitude / (frequency * pi / range_length)) * (...
                                -cos(frequency * pi * (x_val - min_val) / range_length + calculated_phase) + ...
                                cos(calculated_phase) ...
                            )...
                        )...
                    );
                    func_to_solve = @(x_val) cdf_func_sinusoidal(x_val) - U_val_used;
                    try
                        val = fzero(func_to_solve, [min_val, max_val]);
                    catch ME
                        val = min_val + U_val_used * range_length; % Fallback
                    end
                end
        end
    end

    % Initial generation attempt
    rand_val = generate_single_attempt(dist_type_full_name, current_params, min_val, max_val, is_exp_peak_at_boundary_flag, is_hn_peak_at_boundary_flag, is_sin_peak_at_boundary_flag, current_main_boundary,range_length); 

    % Ensure rand_val is strictly between min_val and max_val
    % Re-sample if it falls exactly on an edge due to floating point precision
    while rand_val <= min_val || rand_val >= max_val
        rand_val = generate_single_attempt(dist_type_full_name, current_params, min_val, max_val, is_exp_peak_at_boundary_flag, is_hn_peak_at_boundary_flag, is_sin_peak_at_boundary_flag, current_main_boundary,range_length); % Re-generate
    end
end

% --- Plot Static Elements Helper Function ---
    function plot_handles = initialize_static_plot(ax, total_pdf_vals, x_plot, overall_left_edge, overall_right_edge, main_boundary_actual, min_central_sampling_range, max_central_sampling_range, active_left_dist_start, active_right_dist_end,plot_legend)
    
    axes(ax);
    cla(ax); % Clear axes content if already exists
    hold on;

    % Plot the Bimodal Probability Density Function (PDF)
    plot_handles.h_pdf = plot(x_plot, total_pdf_vals, 'LineWidth', 2, 'Color', 'b');
    plot_handles.total_pdf_vals_max = max(total_pdf_vals);

    % Add a dashed red line at the main boundary (x=0) for clarity
    plot_handles.h_main_boundary_line = plot([main_boundary_actual main_boundary_actual], [0 max(total_pdf_vals)*1.1], 'r--', 'LineWidth', 1.5);

    % Plot the central sampling region boundaries
    plot_handles.h_central_left_line = plot([min_central_sampling_range min_central_sampling_range], [0 max(total_pdf_vals)*1.1], 'g:', 'LineWidth', 1);
    plot_handles.h_central_right_line = plot([max_central_sampling_range max_central_sampling_range], [0 max(total_pdf_vals)*1.1], 'g:', 'LineWidth', 1);
    plot_handles.h_central_text_left = text(min_central_sampling_range, max(total_pdf_vals)*1.15, sprintf('CB Start (%.2f)', min_central_sampling_range), ...
                               'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center', 'Color', 'g', 'FontSize', 8);
    plot_handles.h_central_text_right = text(max_central_sampling_range, max(total_pdf_vals)*1.15, sprintf('CB End (%.2f)', max_central_sampling_range), ...
                                'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center', 'Color', 'g', 'FontSize', 8);

    % Plot the overall range boundaries (user-defined edges)
    plot_handles.h_overall_left_edge_line = plot([overall_left_edge overall_left_edge], [0 max(total_pdf_vals)*1.1], 'k--', 'LineWidth', 1);
    plot_handles.h_overall_left_edge_text = text(overall_left_edge, max(total_pdf_vals)*1.15, sprintf('Overall Left Edge (%.2f)', overall_left_edge), ...
         'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center', 'Color', 'k', 'FontSize', 8);
    plot_handles.h_overall_right_edge_line = plot([overall_right_edge overall_right_edge], [0 max(total_pdf_vals)*1.1], 'k--', 'LineWidth', 1);
    plot_handles.h_overall_right_edge_text = text(overall_right_edge, max(total_pdf_vals)*1.15, sprintf('Overall Right Edge (%.2f)', overall_right_edge), ...
         'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center', 'Color', 'k', 'FontSize', 8);
    
    % Plot the active distribution range boundaries (derived from percentage)
    plot_handles.h_active_left_dist_line = plot([active_left_dist_start active_left_dist_start], [0 max(total_pdf_vals)*1.1], 'm:', 'LineWidth', 1);
    plot_handles.h_active_left_dist_text = text(active_left_dist_start, max(total_pdf_vals)*1.10, sprintf('Left Active Start (%.2f)', active_left_dist_start), ...
         'VerticalAlignment', 'top', 'HorizontalAlignment', 'center', 'Color', 'm', 'FontSize', 8);
    plot_handles.h_active_right_dist_line = plot([active_right_dist_end active_right_dist_end], [0 max(total_pdf_vals)*1.1], 'm:', 'LineWidth', 1);
    plot_handles.h_active_right_dist_text = text(active_right_dist_end, max(total_pdf_vals)*1.10, sprintf('Right Active End (%.2f)', active_right_dist_end), ...
         'VerticalAlignment', 'top', 'HorizontalAlignment', 'center', 'Color', 'm', 'FontSize', 8);

    % Customize plot appearance (static elements)
    title(ax, 'Bimodal Probability Density Function with Central Sampling Bias and Sample History');
    xlabel(ax, 'Value (x)');
    ylabel(ax, 'Probability Density');
    grid(ax, 'on');
    xlim(ax, [overall_left_edge - 0.1*(overall_right_edge - overall_left_edge), overall_right_edge + 0.1*(overall_right_edge - overall_left_edge)]);
    ylim(ax, [0 max(total_pdf_vals)*1.2]); % Set initial Y-limit based on PDF

    % Dynamically build legend entries
    % Create dynamic plot objects here to get their handles for the legend
    plot_handles.h_hist = histogram(ax, [], 'Normalization', 'pdf', 'BinLimits', [overall_left_edge, overall_right_edge], 'NumBins', 50, ...
                                   'FaceColor', [0.7 0.7 0.7], 'EdgeColor', [0.5 0.5 0.5], 'FaceAlpha', 0.5, 'Visible', 'off'); % Initially off
    plot_handles.h_pick_marker = plot(ax, NaN, NaN, 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g', 'Visible', 'off'); % Initially off
    plot_handles.h_pick_text = text(ax, NaN, NaN, '', 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center', ...
                                   'Color', 'g', 'FontWeight', 'bold', 'Visible', 'off'); % Initially off

    legend_handles = [plot_handles.h_pdf, plot_handles.h_main_boundary_line, ...
                      plot_handles.h_central_left_line, plot_handles.h_overall_left_edge_line, ...
                      plot_handles.h_active_left_dist_line, ...
                      plot_handles.h_hist, plot_handles.h_pick_marker]; % Use actual handles
    legend_strings = {'Overall PDF', 'Main Boundary', 'Central Bias Zone Boundary', 'Overall Range Edge', 'Active Dist. Range', 'Sample History Histogram', 'Current Pick'};

    plot_handles.h_legend = legend(ax, legend_handles, legend_strings, 'Location', 'best', 'AutoUpdate', 'off');

    if plot_legend
        set(plot_handles.h_legend,'Visible','on')
    else
        set(plot_handles.h_legend,'Visible','off')
    end
    
    hold off;
end

% --- Update Dynamic Plot Elements Helper Function ---
function [samples_history_out, plot_handles_out] = update_dynamic_plot(ax, current_stimulus, samples_history_in, plot_handles_in, plot_histogram_flag, plot_chosen_stimuli_flag, plot_legend_flag,plot_distribution_flag)
    % params_passed_for_limits: contains overall_left_edge, overall_right_edge for histogram bins, and other static params
    
    samples_history_out = samples_history_in; % Initialize output with input history
    plot_handles_out = plot_handles_in; % Initialize output with input handles (including static ones)

    % Set current axes
    axes(ax);
    hold on;

    % Add new stimulus to history if valid
    if ~isnan(current_stimulus)
        samples_history_out = [samples_history_in, current_stimulus]; %#ok<AGROW>
    end

    % Update histogram
    if plot_histogram_flag
        set(plot_handles_out.h_hist, 'Data', samples_history_out, 'Visible', 'on');
    else
        set(plot_handles_out.h_hist, 'Visible', 'off');
    end

    % Update current stimulus marker
    if plot_chosen_stimuli_flag && ~isnan(current_stimulus)
        set(plot_handles_out.h_pick_marker, 'XData', current_stimulus, 'YData', 0, 'Visible', 'on');
        set(plot_handles_out.h_pick_text, 'Position', [current_stimulus, plot_handles_out.total_pdf_vals_max*0.05, 0], ...
                                 'String', sprintf('Picked: %.4f', current_stimulus), 'Visible', 'on');
    else
        set(plot_handles_out.h_pick_marker, 'Visible', 'off');
        set(plot_handles_out.h_pick_text, 'Visible', 'off');
    end

    % Show/hide legend
    if plot_legend_flag
        set(plot_handles_out.h_legend,'Visible','on')
    else
        set(plot_handles_out.h_legend,'Visible','off')
    end

    % Update static distribution elements visibility based on plot_distribution_flag
    % (assuming plot_handles_out already contains these from initialize_static_plot)
    static_elements_visibility = 'off';
    if plot_distribution_flag
        static_elements_visibility = 'on';
    end

    % Iterate through known static handles and set visibility
    static_handle_names = {'h_pdf', 'h_main_boundary_line', 'h_central_left_line', 'h_central_right_line', ...
                           'h_central_text_left', 'h_central_text_right', 'h_overall_left_edge_line', ...
                           'h_overall_left_edge_text', 'h_overall_right_edge_line', 'h_overall_right_edge_text', ...
                           'h_active_left_dist_line', 'h_active_left_dist_text', 'h_active_right_dist_line', ...
                           'h_active_right_dist_text'};
    
    for i = 1:numel(static_handle_names)
        handle_name = static_handle_names{i};
        if isfield(plot_handles_out, handle_name) && isvalid(plot_handles_out.(handle_name))
            set(plot_handles_out.(handle_name), 'Visible', static_elements_visibility);
        end
    end
    
    hold off;
end


% --- Main Function Logic ---
samples_history_out = []; % Initialize output
h_fig_out = []; % Initialize output
plot_handles_out = struct(); % Initialize output

% Prepare parameters for helper functions (structs for left and right sides)
current_params_left.lambda_actual = params.lambda_left_actual;
current_params_left.normal_mean = params.normal_mean_left;
current_params_left.normal_std_dev = params.normal_std_dev_left;
current_params_left.half_normal_std_dev = params.half_normal_std_dev_left;
current_params_left.sinusoidal_amplitude_factor = params.sinusoidal_amplitude_factor_left;
current_params_left.sinusoidal_frequency_factor = params.sinusoidal_frequency_factor_left;

current_params_right.lambda_actual = params.lambda_right_actual;
current_params_right.normal_mean = params.normal_mean_right;
current_params_right.normal_std_dev = params.normal_std_dev_right;
current_params_right.half_normal_std_dev = params.half_normal_std_dev_right;
current_params_right.sinusoidal_amplitude_factor = params.sinusoidal_amplitude_factor_right;
current_params_right.sinusoidal_frequency_factor = params.sinusoidal_frequency_factor_right;


% Handle different modes of operation
if strcmp(params.Mode, 'generate_single_sample')
    % Generate a single sample and return
    
    % Determine the side for this specific sample
    actual_side_for_sample = lower(params.chosen_side);
    if strcmp(actual_side_for_sample, 'auto')
        U_side_choice = rand();
        if U_side_choice < P_left_derived
            actual_side_for_sample = 'left';
        else
            actual_side_for_sample = 'right';
        end
    end

    U_bias_decision = rand();
    if U_bias_decision < params.P_central_region
        % Biased Sampling: Try to get a sample from the central zone
        current_pick_found = false;
        while ~current_pick_found
            if strcmp(actual_side_for_sample, 'left')
                % Sample from the full left overall range
                candidate_pick = generate_rand_general(params.left_dist_type, current_params_left, overall_left_edge, main_boundary_actual, ...
                                                    params.left_exp_peak_at_boundary_flag, params.left_hn_peak_at_boundary_flag, params.left_sin_peak_at_boundary_flag, main_boundary_actual);
            else % 'right'
                % Sample from the full right overall range
                candidate_pick = generate_rand_general(params.right_dist_type, current_params_right, main_boundary_actual, overall_right_edge, ...
                                                    params.right_exp_peak_at_boundary_flag, params.right_hn_peak_at_boundary_flag, params.right_sin_peak_at_boundary_flag, main_boundary_actual);
            end
            if candidate_pick >= min_central_sampling_range && candidate_pick <= max_central_sampling_range
                samples_history_out = candidate_pick;
                current_pick_found = true;
            end
        end
    else
        % Unbiased Sampling: Pick from the determined side's ACTIVE range
        if strcmp(actual_side_for_sample, 'left')
            samples_history_out = generate_rand_general(params.left_dist_type, current_params_left, active_left_dist_start, main_boundary_actual, ...
                                                 params.left_exp_peak_at_boundary_flag, params.left_hn_peak_at_boundary_flag, params.left_sin_peak_at_boundary_flag, main_boundary_actual);
        else % 'right'
            samples_history_out = generate_rand_general(params.right_dist_type, current_params_right, main_boundary_actual, active_right_dist_end, ...
                                                 params.right_exp_peak_at_boundary_flag, params.right_hn_peak_at_boundary_flag, params.right_sin_peak_at_boundary_flag, main_boundary_actual);
        end
    end
    
    % Outputs for 'generate_single_sample' mode
    h_fig_out = []; 
    plot_handles_out = struct();

elseif strcmp(params.Mode, 'run_simulation')
    % --- Plot Initialization (Static) ---
    h_fig_out = figure;
    ax = gca; % Get current axes handle
    
    % Generate a fine grid of x values for smooth plotting across the overall range
    x_plot = linspace(overall_left_edge, overall_right_edge, 500);

    % Calculate PDF values for the left and right components. These define the overall PDF shape.
    pdf_left_component = get_pdf_general(x_plot, params.left_dist_type, current_params_left, active_left_dist_start, main_boundary_actual, ...
                                        params.left_exp_peak_at_boundary_flag, params.left_hn_peak_at_boundary_flag, params.left_sin_peak_at_boundary_flag, main_boundary_actual);
    pdf_right_component = get_pdf_general(x_plot, params.right_dist_type, current_params_right, main_boundary_actual, active_right_dist_end, ...
                                         params.right_exp_peak_at_boundary_flag, params.right_hn_peak_at_boundary_flag, params.right_sin_peak_at_boundary_flag, main_boundary_actual);

    % Combine the components to form the total bimodal PDF (this is the theoretical distribution)
    total_pdf_vals = P_left_derived * pdf_left_component + (1 - P_left_derived) * pdf_right_component;

    % Initialize static plot elements
    plot_handles_out = initialize_static_plot(ax, total_pdf_vals, x_plot, overall_left_edge, overall_right_edge, main_boundary_actual, min_central_sampling_range, max_central_sampling_range, active_left_dist_start, active_right_dist_end,params.plot_legend);
    
    % Initialize dynamic plot handles (will be updated by update_dynamic_plot)
    plot_handles_out.h_hist = []; % Initialize as empty/null
    plot_handles_out.h_pick_marker = [];
    plot_handles_out.h_pick_text = [];

    fprintf('Starting simulation of %d random picks...\n', params.num_simulations);

    for k = 1:params.num_simulations
        % Generate a uniform random number to decide if we apply central sampling bias
        U_bias_decision = rand();
        
        current_pick_side_str = ''; % To store 'LEFT' or 'RIGHT' for fprintf
        
        if U_bias_decision < params.P_central_region
            % --- Biased Sampling: Try to get a sample from the central zone ---
            current_pick_found = false;
            while ~current_pick_found
                % Decide if the underlying pick is from the left or right distribution
                U_side_choice = rand();
                if U_side_choice < P_left_derived
                    % Sample from the full left overall range
                    candidate_pick = generate_rand_general(params.left_dist_type, current_params_left, overall_left_edge, main_boundary_actual, ...
                                                        params.left_exp_peak_at_boundary_flag, params.left_hn_peak_at_boundary_flag, params.left_sin_peak_at_boundary_flag, main_boundary_actual);
                    current_pick_side_str = 'LEFT';
                else
                    % Sample from the full right overall range
                    candidate_pick = generate_rand_general(params.right_dist_type, current_params_right, main_boundary_actual, overall_right_edge, ...
                                                        params.right_exp_peak_at_boundary_flag, params.right_hn_peak_at_boundary_flag, params.right_sin_peak_at_boundary_flag, main_boundary_actual);
                    current_pick_side_str = 'RIGHT';
                end

                % Check if the candidate falls within the central sampling zone
                if candidate_pick >= min_central_sampling_range && candidate_pick <= max_central_sampling_range
                    current_pick = candidate_pick;
                    current_pick_found = true;
                    fprintf('Random pick (BIASED) from %s distribution: %.4f\n', current_pick_side_str, current_pick);
                end
                % If not found, the loop continues to re-sample
            end
        else
            % --- Unbiased Sampling: Pick from the overall bimodal distribution's ACTIVE range ---
            U_side_choice = rand(); 
            if U_side_choice < P_left_derived
                current_pick = generate_rand_general(params.left_dist_type, current_params_left, active_left_dist_start, main_boundary_actual, ...
                                                     params.left_exp_peak_at_boundary_flag, params.left_hn_peak_at_boundary_flag, params.left_sin_peak_at_boundary_flag, main_boundary_actual);
                current_pick_side_str = 'LEFT';
                fprintf('Random pick (UNBIASED) from LEFT side (active range [%.2f, %.2f]): %.4f\n', active_left_dist_start, main_boundary_actual, current_pick);
            else
                current_pick = generate_rand_general(params.right_dist_type, current_params_right, main_boundary_actual, active_right_dist_end, ...
                                                     params.right_exp_peak_at_boundary_flag, params.right_hn_peak_at_boundary_flag, params.right_sin_peak_at_boundary_flag, main_boundary_actual);
                current_pick_side_str = 'RIGHT';
                fprintf('Random pick (UNBIASED) from RIGHT side (active range [%.2f, %.2f]): %.4f\n', main_boundary_actual, active_right_dist_end, current_pick);
            end
        end
        
        % Update the plot elements
        [samples_history_out, plot_handles_out] = update_dynamic_plot(ax, current_pick, samples_history_out, plot_handles_out, ...
                                                                     true, true, true,true);% Always show all during simulation
        
        drawnow;
        pause(params.pause_duration);
        
        fprintf('Pick %d: %.4f\n', k, current_pick);
    end

    fprintf('Simulation complete. Total picks: %d\n', params.num_simulations);

elseif strcmp(params.Mode, 'initialize_plot')
    % --- Initialize Static Plot ---
    % Get axes handle, create if not provided
    if isempty(params.ax_handle) || ~isgraphics(params.ax_handle, 'axes')
        h_fig_out = figure;
        ax = gca;
    else
        ax = params.ax_handle;
        h_fig_out = get(ax, 'Parent'); % Get figure handle from axes
    end

    % Generate a fine grid of x values for smooth plotting across the overall range
    x_plot = linspace(overall_left_edge, overall_right_edge, 500);

    % Calculate PDF values for the left and right components. These define the overall PDF shape.
    pdf_left_component = get_pdf_general(x_plot, params.left_dist_type, current_params_left, active_left_dist_start, main_boundary_actual, ...
                                        params.left_exp_peak_at_boundary_flag, params.left_hn_peak_at_boundary_flag, params.left_sin_peak_at_boundary_flag, main_boundary_actual);
    pdf_right_component = get_pdf_general(x_plot, params.right_dist_type, current_params_right, main_boundary_actual, active_right_dist_end, ...
                                         params.right_exp_peak_at_boundary_flag, params.right_hn_peak_at_boundary_flag, params.right_sin_peak_at_boundary_flag, main_boundary_actual);

    % Combine the components to form the total bimodal PDF (this is the theoretical distribution)
    total_pdf_vals = P_left_derived * pdf_left_component + (1 - P_left_derived) * pdf_right_component;

    % Call helper to initialize static plot elements
    plot_handles_out = initialize_static_plot(ax, total_pdf_vals, x_plot, overall_left_edge, overall_right_edge, main_boundary_actual, min_central_sampling_range, max_central_sampling_range, active_left_dist_start, active_right_dist_end,params.plot_legend);
    
    % No samples history returned for this mode
    samples_history_out = [];

elseif strcmp(params.Mode, 'update_plot')
    % --- Update existing plot elements ---
    if isempty(params.ax_handle) || ~isgraphics(params.ax_handle, 'axes')
        % error('In ''update_plot'' mode, an existing axes handle (''ax_handle'') must be provided.');
        return;
    end
    
    % % Re-calculate PDF values as they are needed for scaling and text placement
    % % (This is necessary because total_pdf_vals is used for text positioning, etc.)
    % x_plot = linspace(overall_left_edge, overall_right_edge, 500);
    % pdf_left_component = get_pdf_general(x_plot, params.left_dist_type, current_params_left, active_left_dist_start, main_boundary_actual, ...
    %                                     params.left_exp_peak_at_boundary_flag, params.left_hn_peak_at_boundary_flag, params.left_sin_peak_at_boundary_flag, main_boundary_actual);
    % pdf_right_component = get_pdf_general(x_plot, params.right_dist_type, current_params_right, main_boundary_actual, active_right_dist_end, ...
    %                                      params.right_exp_peak_at_boundary_flag, params.right_hn_peak_at_boundary_flag, params.right_sin_peak_at_boundary_flag, main_boundary_actual);
    % total_pdf_vals = P_left_derived * pdf_left_component + (1 - P_left_derived) * pdf_right_component;

    [samples_history_out, plot_handles_out] = update_dynamic_plot(params.ax_handle, params.current_stimulus, params.samples_history_in, params.plot_handles_in, ...
                                                                 params.plot_histogram, params.plot_chosen_stimuli, params.plot_legend,params.plot_distribution);
    
    h_fig_out = get(params.ax_handle, 'Parent'); % Return figure handle of the updated axes
    drawnow;
else
    return;% error('Invalid Mode specified. Use ''run_simulation'', ''generate_single_sample'', ''initialize_plot'', or ''update_plot''.');
end

end % End of main function
