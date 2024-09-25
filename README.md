# Code for "Applying multi-agent reinforcement learning to study hybrid threats and defensive countermeasures"
## Introduction
The threat landscape continues to evolve, with attackers using more advanced tools to launch large-scale, sophisticated attacks at lower costs. 
By continuously integrating AI and ML techniques, attackers can become capable of launching autonomous hybrid attack campaigns that synchronise 
cyberattacks with disinformation campaigns whilst adapting to defensive countermeasures. The potential of such autonomous adversaries raises 
concerns for the future of defence. While AI and ML algorithms effectively detect intrusions and anomalies, the decision to respond lies with 
humans, so the costs and response times can be high. To address this problem, autonomous defence systems that can make real-time response 
decisions whilst adapting to adversarial behaviours are needed. We propose a novel MARL approach in which both attackers and defenders are DRL 
agents. They can autonomously conduct multi-domain attacks and defend against them. We develop a custom environment representing a 
Cyber-Physical-Social System and train MARL agents to devise offensive and defensive strategies.
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
├── data_analysis.py                 # Functions to analyse training data
├── other_functions.py               # Functions to visualise training results
├── exp1-hp-tuning.py                # Hyperparameter tuning for experiment 1
├── exp2-hp-tuning.py                # Hyperparameter tuning for experiment 2
├── exp1-sa.py                       # Sensitivity analysis for experiment 1
├── hyperparameters.csv              # (Tuned) hyperparameter values
├── parameters.csv                   # Parameter values
├── regagent-parameters              # Folder containing trained regular agents neural networks
├── results                          # Folder containing R scripts for data analysis and visualisation
    ├── exp1-results                 # Data collected from experiment 1
    ├── exp2-results                 # Data collected from experiment 2
    └── plots                        # Figures
├── R-scripts                        # Folder containing R scripts for data analysis and visualisation
    ├── ch4-exp1-analysis.R          # R script for analysing and visualising experiment 1 data
    ├── ch4-exp2-attack-analysis.R   # R script for analysing and visualising attacker behaviour
    └── ch4-exp2-defence-analysis.R  # R script for analysing and visualising defender behaviour
├── LICENSE.md                       # License
└── README.md                        # Project documentation
```
