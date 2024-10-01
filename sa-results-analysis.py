# %% [markdown]
# ## Applying Multi-Agent Reinforcement Learning to Study Hybrid Threats and Defensive Countermeasures
# Chapter 4\ 
# Sensitivity analysis for experiment 1\
# Analyse the results of the sensitivity analysis using Sobol's method
# %% [markdown]
# Import libraries
import pandas as pd
import numpy as np
from SALib.sample import sobol
from SALib.analyze import sobol as sobol_analyze
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import datetime

# Setup date, chapter, and experiment
date = datetime.datetime.now().strftime("%Y-%m-%d")
chapter = 4
experiment = 1

# %% [markdown]
# Load data
data = pd.read_csv('sensitivity_analysis/exp1-results/ch4-exp1-sa-results.csv')
print(data.shape[0], data.shape[1]) # 2200 rows 28 columns
# Process columns with eval and extract first elements
for col in ['center_up_to_down', 'center_down_to_up', 'end_up_to_down', 'end_down_to_up', 'cost']:
    data[col] = data[col].apply(eval).apply(lambda x: x[0])
data.head()

# %% [markdown]
# Separate inputs (parameters) and outputs
inputs = data.iloc[:,:10] 
outputs = data.iloc[:,10:]
print(inputs.head())
print(outputs.head())

# %% [markdown]
# Get information from data
# Define the total number of rows in the data
n = data.shape[0]
# Number of parameters (dimensions) in the Sobol sequence
d = inputs.shape[1] 
print("Number of rows in the data", n, " and number of parameters", d) 

# Define whether to calculate second-order indices
calc_second_order = False  # Set to True if second-order indices are needed

# Calculate initial N based on whether second-order indices are needed
if calc_second_order:
    N = int(n / (2 * d + 2))
else:
    N = int(n / (d + 2))
print("Initial number of samples for sensitivity analysis:", N)

adjusted_N = 2 ** int(np.floor(np.log2(N)))
print("Adjusted number of samples for sensitivity analysis (power of 2):", adjusted_N)

# Ensure total evaluations match the available data size
if calc_second_order:
    total_evaluations = adjusted_N * (2 * d + 2)
else:
    total_evaluations = adjusted_N * (d + 2)

# Ensure that the total number of rows in the dataset is correct
if n != total_evaluations:
    print(f"Adjusting data size from {n} to {total_evaluations}")
    data = data.iloc[:total_evaluations, :]
    inputs = data.iloc[:, :10]
    outputs = data.iloc[:, 10:]
    n = total_evaluations

print("Final number of rows in the data:", n)

# %% [markdown]
# Generate Sobol sequence for X
problem = {
    'num_vars': d,
    'names': inputs.columns.tolist(),
    'bounds': [
        (2, 10), # kappa (number of sn connections)
        (0.01, 0.99),  # rho
        (0.01, 0.99),  # center_up_to_down
        (0.01, 0.99),  # center_down_to_up
        (0.01, 0.99),  # end_up_to_down
        (0.01, 0.99),  # end_down_to_up
        (0.01, 0.99),  # cost (example of integer range)
        (0.01, 0.99),  # direct_exp_weight
        (0.01, 0.99),  # satisfaction_threshold
        (0.01, 0.99)   # forgetting_factor
    ]
}
# %% [markdown]
# Generate samples using Saltelli's method
param_values = sobol.sample(problem, adjusted_N, calc_second_order=calc_second_order)

# %% [markdown]
# Perform Sobol sensitivity analysis for each output variable
results = []
for i in range(outputs.shape[1]):
    Y = outputs.iloc[:, i].to_numpy()
    # Perform Sobol sensitivity analysis for each output variable
    sobol_result = sobol_analyze.analyze(problem, Y, calc_second_order=calc_second_order, print_to_console=False)
    results.append(sobol_result)

#%%
# Define Latex names
latex_names = {
    'kappa': r'$\kappa$',
    'rho': r'$\rho$',
    'center_up_to_down': r'$\Psi$',
    'center_down_to_up': r'$\psi$',
    'end_up_to_down': r'$\Lambda$',
    'end_down_to_up': r'$\lambda$',
    'cost': r'$\zeta$',
    'direct_exp_weight': r'$w$',
    'satisfaction_threshold': r'$\tau$',
    'forgetting_factor': r'$\eta$'
}
# %% [markdown]
# Collect results for each output variable
table_data = []
for output_name, result in zip(outputs.columns, results):
    for name, s1, s1_conf, st, st_conf in zip(problem['names'], result['S1'], result['S1_conf'], result['ST'], result['ST_conf']):
        table_data.append({
            'Output': output_name,
            'Parameter': latex_names[name],
            'First-order Index (S1)': s1,
            'S1 Confidence Interval': s1_conf,
            'Total-order Index (ST)': st,
            'ST Confidence Interval': st_conf
        })

# Convert to DataFrame
df_table = pd.DataFrame(table_data)
df_table
# # Save the table to a CSV file
df_table.to_csv(f'sensitivity_analysis/exp1-results/ch{chapter}-exp{experiment}-sa-sobol-results.csv', index=False)


# %% [markdown]
# Visualisation function
def plot_sobol_indices(sobol_result, parameter_names, output_name): # for all outputs

    Si = sobol_result

    indices = pd.DataFrame({
        'Parameter': parameter_names,
        'First-order': np.maximum(Si['S1'], 0),  # Clipping negative values to zero
        'Total-order': np.maximum(Si['ST'], 0),  # Clipping negative values to zero
        'First-order (conf)': Si['S1_conf'],
        'Total-order (conf)': Si['ST_conf']
    })
    
    # Apply LaTeX formatting to parameter names
    indices['Parameter'] = indices['Parameter'].map(latex_names)
    
    # Set up the matplotlib figure
    plt.figure(figsize=(12, 6))
    plt.title(f'Sobol Sensitivity Analysis for {output_name}')
    
    # First-order indices
    bar_width = 0.35
    r1 = np.arange(len(indices))
    r2 = [x + bar_width for x in r1]
    
    plt.bar(r1, indices['First-order'], color='skyblue', width=bar_width, edgecolor='grey', label='First-order')
    plt.errorbar(r1, indices['First-order'], yerr=indices['First-order (conf)'], fmt='o', color='blue')
    
    # Total-order indices
    plt.bar(r2, indices['Total-order'], color='salmon', width=bar_width, edgecolor='grey', label='Total-order')
    plt.errorbar(r2, indices['Total-order'], yerr=indices['Total-order (conf)'], fmt='o', color='red')
    
    # Improve x-axis labels
    plt.xlabel('Parameter', fontweight='bold')
    plt.xticks([r + bar_width/2 for r in range(len(indices))], indices['Parameter'], rotation=0, ha='right')
    
    # Adding labels and legend
    plt.ylabel('Sensitivity index')
    plt.legend()
    plt.show()

# %% [markdown]
# Print all results
for i in range(outputs.shape[1]):
    plot_sobol_indices(results[i], inputs.columns.tolist(), outputs.columns[i])
   

# %% [markdown]
# Regression analysis based on sobol's indices (using original data)
X = pd.DataFrame({
    #'kappa': inputs['kappa'], # specify column value
    #'rho': inputs['rho'],
    #'Psi': inputs['center_up_to_down'], # Psi
    #'psi': inputs['center_down_to_up'], # psi
    'Lambda': inputs['end_up_to_down'], # Lambda
    #'lambda': inputs['end_down_to_up'], # lambda
    #'zeta': inputs['cost'], # zeta
    #'w': inputs['direct_exp_weight'], # w
    #'tau': inputs['satisfaction_threshold'], # tau
    #'eta': inputs['forgetting_factor'] # eta
})
Y = outputs["sp2_availability"]

# Adding a constant for the intercept term
X = sm.add_constant(X)

# Fit the regression model
model = sm.OLS(Y, X).fit()

# Summary of the regression model
#print(model.summary())