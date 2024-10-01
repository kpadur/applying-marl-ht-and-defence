# %% [markdown]
# ## Applying Multi-Agent Reinforcement Learning to Study Hybrid Threats and Defensive Countermeasures
# Chapter 4\ 
# Sensitivity analysis for experiment 1\
# Generate parameter sets for the sensitivity analysis using Sobol's method
# %% [markdown]
# Import libraries
import numpy as np
from SALib.sample import sobol
import os
import datetime
from SALib import ProblemSpec
# %% [markdown]
# Setup date, chapter, and experiment
date = datetime.datetime.now().strftime("%Y-%m-%d")
chapter = 4
experiment = 1

# %% [markdown]
# Define the output directory and ensure it exists
save_data_path = os.path.join("sensitivity_analysis", "param_values")
if not os.path.exists(save_data_path):
    os.makedirs(save_data_path)

# %% [markdown]
# Define the problem for SALib
problem = ProblemSpec({
    'names': ['kappa', 'rho', 'center_up_to_down', 'center_down_to_up', 'end_up_to_down', 'end_down_to_up', 
              'cost', 'direct_exp_weight', 'satisfaction_threshold', 'forgetting_factor'],
    'bounds': [
        [2, 10], # kappa
        [0.01, 0.99], # rho
        [0.01, 0.99], # center_up_to_down
        [0.01, 0.99], # center_down_to_up
        [0.01, 0.99], # end_up_to_down
        [0.01, 0.99], # end_down_to_up
        [0.01, 0.99], # cost
        [0.01, 0.99], # direct_exp_weight
        [0.01, 0.99], # satisfaction_threshold
        [0.01, 0.99] # forgetting_factor
    ],
    'outputs': ['total_rewards', 'action_rewards', 'opinion_rewards', 'sp0_social_trust', 'sp1_social_trust', 'sp2_social_trust',
                'sp0_request_rate', 'sp1_request_rate', 'sp2_request_rate','op0_expression_rate', 'op1_expression_rate', 'op2_expression_rate',
                'op3_expression_rate', 'op4_expression_rate', 'op5_expression_rate', 'sp0_availability', 'sp1_availability', 'sp2_availability']
})

# %% [markdown]
# Generate parameter sets
N = 512  # Base sample size
param_values = sobol.sample(problem, N, calc_second_order=False)

# Ensure kappa is an integer by rounding and clipping to the original bounds
param_values[:, 0] = np.round(param_values[:, 0])
param_values[:, 0] = np.clip(param_values[:, 0], 2, 10).astype(int)

# Verify the total number of samples generated
print(f'Total number of samples: {param_values.shape[0]}')  # Should print 3072

# %% [markdown]
# Save parameter sets to the specified file
output_file = os.path.join(save_data_path, f'ch{chapter}-exp{experiment}-sa-params-{date}.csv')
np.savetxt(output_file, param_values, delimiter=',')

# %% [markdown]
# Split parameters into multiple files for different machines
num_machines = 512
split_params = np.array_split(param_values, num_machines)

for i, params in enumerate(split_params):
    machine_output_file = os.path.join(save_data_path, f'param_values_machine_{i}.csv')
    np.savetxt(machine_output_file, params, delimiter=',')
