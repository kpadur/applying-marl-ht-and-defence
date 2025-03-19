# Code for "Applying multi-agent reinforcement learning to study hybrid threats and defensive countermeasures"
## Introduction
The threat landscape continues to evolve, with attackers using more advanced tools to launch large-scale, sophisticated attacks at lower costs. Attackers are employing AI and ML techniques to enhance both cyberattacks and disinformation campaigns, paving the way for autonomous hybrid attack campaigns that adapt dynamically to defensive countermeasures. The potential of such autonomous adversaries raises concerns for the future of defence. While AI and ML algorithms effectively detect intrusions and anomalies, the decision to respond lies with humans, so the costs and response times can be high. To address this problem, we need autonomous defence systems that can make real-time response decisions whilst adapting to adversarial behaviours and not themselves providing a means by which attacks can be amplified.
## Project structure
```
applying-marl-ht-and-defence/
├── train_regular_agents.py          # Experiment for training regular agents (experiment 1)
├── train_marl.py                    # Experiment for training attackers and defenders (experiment 2)
├── environment_exp1.py              # Environment setup for experiment 1
├── environment_exp2.py              # Environment setup for experiment 2
├── a2c_agent.py                     # Regular agents' behaviour in the environment
├── a2c_def_agent.py                 # Defenders' behaviour in the environment
├── a2c_mal_agent.py                 # Attackers' behaviour in the environment
├── nns.py                           # Architecture of deep neural networks
├── data_analysis_exp1.py            # Functions to analyse training data (experiment 1)
├── data_analysis_exp2.py            # Functions to analyse training data (experiment 2)
├── other_functions.py               # Functions to visualise training results
├── exp1-hp-tuning.py                # Hyperparameter tuning for experiment 1
├── exp2-hp-tuning.py                # Hyperparameter tuning for experiment 2
├── exp1-sa.py                       # Sensitivity analysis for experiment 1
├── sa-param-values.py               # Generate parameter sets for the sensitivity analysis using Sobol's method
├── sa-results-analysis.py           # Analyse the results of the sensitivity analysis using Sobol's method
├── hyperparameters.csv              # (Tuned) hyperparameter values
├── parameters.csv                   # Parameter values
├── regagent-parameters              # Folder containing trained regular agents neural networks
├── hyperparameter-tuning            # Folder containing data and R scripts for hyperparameter tuning
    ├── ch4-exp1-hp-tuning.xlsx      # Data on hyperparameter tuning for experiment 1
    ├── ch4-exp2-hp-tuning.xlsx      # Data on hyperparameter tuning for experiment 2
    └── ch4-hp-tuning-vis.R          # R script for visualising hyperparameter tuning results (both experiments)
├── results                          # Folder containing R scripts for data analysis and visualisation
    ├── exp1-results                 # Data collected from experiment 1
    ├── exp2-results                 # Data collected from experiment 2
    ├── figures                      # Visualisation of the environment and agents' behaviour during training
    └── plots                        # Figures generated with R
├── R-scripts                        # Folder containing R scripts for data analysis and visualisation
    ├── ch4-exp1-analysis.R          # R script for analysing and visualising experiment 1 data
    ├── ch4-exp2-attack-analysis.R   # R script for analysing and visualising attacker behaviour
    └── ch4-exp2-defence-analysis.R  # R script for analysing and visualising defender behaviour
├── LICENSE.md                       # License
└── README.md                        # Project documentation
```
## License
MIT
## Prerequisites
```
Python 3.10 or higher version is required.

The following Python libraries are required:
- numpy (version 1.24.2 or higher)
- pandas (version 1.5.3 or higher)
- torch (version 1.13.1 or higher)
- matplotlib (version 3.7.0 or higher)
- gymnasium (version 0.29.1 or higher)
- networkx (version 3.0 or higher)
- pettingzoo (version 1.24.1 or higher)
- optuna (version 3.6.1 or higher)
- SALib (version 1.5.0 or higher)
- statsmodels (version 0.14.2 or higher)

R version 4.3.1 or higher version is required for data analysis and visualisation.

The following R libraries are required:
- readr (version 2.1.4 or higher)
- ggplot2 (version 3.4.4 or higher)
- cowplot (version 1.1.2 or higher)
- zoo (version 1.8 or higher)
- dplyr (version 1.1.3 or higher)
- tidyr (version 1.3.0 or higher)
```
