# %% [markdown]
# ## Applying Multi-Agent Reinforcement Learning to Study Hybrid Threats and Defensive Countermeasures
# Chapter 4\ 
# Sensitivity analysis for experiment 1

# %% [markdown]
# Import libraries
from environment_exp1 import Environment
from a2c_agent import A2CRegAgent
from data_analysis import process_regagent_rewards, process_regagent_states_average, process_regagent_actions, process_service_provider_availability
import numpy as np
import torch
import random
import pandas as pd
import os
import datetime
import csv

# %% [markdown]
# Setup device, date, chapter, and experiment
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # currently cpu
date = datetime.datetime.now().strftime("%Y-%m-%d")
chapter = 4
experiment = 1

# %% [markdown]
# Specify output directory
save_data_path = os.path.join("sensitivity_analysis", "exp1-results")
save_figures_path = os.path.join("sensitivity_analysis", "exp1-figures")
get_nn_path = os.path.join("regagent-parameters")
param_values_path = os.path.join("sensitivity_analysis", "param_values")

# %% [markdown]
# Sensitivity analysis function
def sensitivity_analysis(trial_params):
    # Set hyperparameters (use trained agents)
    df = pd.read_csv("hyperparameters.csv")
    hyperparameters = dict(zip(df['hyperparameter'], df['value']))

    alpha_rnn1 = hyperparameters['alpha_1']
    alpha_rnn2 = hyperparameters['alpha_2']

    # Initialise seed for reproducibility
    seed = 5282
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Define training time
    n_steps = 500 # number of steps per episode
    number_of_episodes = 500

    # Create environment
    nProviders = 3
    nRegAgents = 110
    nMalAgents = 0

    # Parameters
    # order: 'kappa', 'rho', 'center_up_to_down', 'center_down_to_up', 'end_up_to_down', 'end_down_to_up', 'cost', 'direct_exp_weight', 'satisfaction_threshold', 'forgetting_factor'
    kappa = int(trial_params[0])
    rho = trial_params[1]
    center_up_to_down = [trial_params[2]] * nProviders # Psi: prob (1 to -1)
    center_down_to_up = [trial_params[3]] * nProviders # psi: prob (-1 to 1)
    end_up_to_down = [trial_params[4]] * nProviders # Lambda: prob (1 to -1)
    end_down_to_up = [trial_params[5]] * nProviders # lambda: prob (-1 to 1)
    cost = [trial_params[6]] * nProviders
    direct_exp_weight = trial_params[7]
    satisfaction_threshold = trial_params[8]
    forgetting_factor = trial_params[9]
    print("Running experiment with the following parameters: kappa", kappa, "rho", rho, "center_up_to_down", center_up_to_down, "center_down_to_up", center_down_to_up, "end_up_to_down",
          end_up_to_down, "end_down_to_up", end_down_to_up, "cost", cost, "direct_exp_weight", direct_exp_weight, "satisfaction_threshold", satisfaction_threshold, "forgetting_factor", forgetting_factor)
   
    # Create environment
    env = Environment(nRegAgents, nMalAgents, nProviders, 
                        kappa, rho, center_up_to_down, center_down_to_up, end_up_to_down, end_down_to_up, cost, 
                        direct_exp_weight, satisfaction_threshold, forgetting_factor)
    
    # Pick one agent randomly from each type of agents
    regagent_example = np.random.choice(env.regagents)

    state_shape, n_actions, n_opinions = \
        env.observation_spaces[f"regagent{regagent_example}"].shape[0], env.action_spaces[f"regagent{regagent_example}"][0].n, env.action_spaces[f"regagent{regagent_example}"][1].n

    # Reset environment state
    envstate, info = env.reset(seed=seed)

    print("Regular agents' state shape is", state_shape, ", number of actions is", n_actions, " and number of opinions is", n_opinions)

    # Define the agent with suggested hyperparameters
    # Import regular agents
    regular_agents = {f"regagent{agent}": A2CRegAgent(state_shape, n_actions, n_opinions, alpha_rnn1, alpha_rnn2, device) 
                    for agent in env.regagents}

    for agent_name, agent in regular_agents.items():
        # Load Action NN and its optimizer
        action_checkpoint = torch.load(os.path.join(get_nn_path, f'{agent_name}_checkpoint_actions_2024-06-24_5281.pth'))
        agent.action_nn.load_state_dict(action_checkpoint['actions_state_dict'])
        agent.action_opt.load_state_dict(action_checkpoint['actions_opt_state_dict'])

        # Load Opinion NN and its optimizer
        opinion_checkpoint = torch.load(os.path.join(get_nn_path, f'{agent_name}_checkpoint_opinions_2024-06-24_5281.pth'))
        agent.opinion_nn.load_state_dict(opinion_checkpoint['opinions_state_dict'])
        agent.opinion_opt.load_state_dict(opinion_checkpoint['opinions_opt_state_dict'])

    # Collect data
    total_rewards, action_rewards, opinion_rewards = np.zeros(number_of_episodes + 1),  np.zeros(number_of_episodes + 1), np.zeros(number_of_episodes + 1)
    social_trust_history = np.zeros((number_of_episodes + 1, nProviders))
    actions_history, opinions_history = np.zeros((number_of_episodes + 1, nProviders)), np.zeros((number_of_episodes + 1, nProviders*2))
    sum_sp_availability = [0, 0, 0]
    sp_availability_history = np.zeros((number_of_episodes + 1, nProviders))

    # Train and evaluate the agent
    for episode in range(1, number_of_episodes + 1):

        # Restart environment
        observations, _ = env.reset()

        # Initialise dictionaries to store rewards, observations, and actions
        all_observations = {agent_name: [observations[agent_name]] for agent_name in regular_agents.keys()}
        all_actions = {agent_name: [] for agent_name in regular_agents.keys()}
        episode_rewards = {agent_name: [] for agent_name in regular_agents.keys()}

        # Collect data for n_steps (episode length)
        for timestep in range(1, n_steps+1): # one episode

            actions = {}  # Dictionary to store the current timestep's actions
            
            for agent_name, agent in regular_agents.items():
                state = observations[agent_name]
                # Sample action and opinion
                action, opinion = agent.sample_actions(state)
                actions[agent_name] = (action, opinion) # save action and opinion
            
            # Service providers provide service
            for endpoint in env.endpoint.values():
                for provider in env.providers:
                    sum_sp_availability[provider] += endpoint[provider]

            # Perform actions, determine next state, reward, and termination
            observations, rewards, _, _, _ = env.step(actions)

            # Store the rewards, observations, and actions for each agent for the current timestep
            for agent_name in episode_rewards.keys():
                episode_rewards[agent_name].append(rewards[agent_name])
                all_observations[agent_name].append(observations[agent_name])
                all_actions[agent_name].append(actions[agent_name])

        # Process regagent rewards
        sum_regagent_rewards, service, feedback = process_regagent_rewards(episode_rewards)
        total_rewards[episode] = sum_regagent_rewards
        action_rewards[episode] = service
        opinion_rewards[episode] = feedback
        # Process social trust
        episode_states_pd = process_regagent_states_average(all_observations, env.providers, episode, n_steps)
        social_trust_history[episode][0] = episode_states_pd["trust_in_sp0"].mean()
        social_trust_history[episode][1] = episode_states_pd["trust_in_sp1"].mean()
        social_trust_history[episode][2] = episode_states_pd["trust_in_sp2"].mean()
        # Processing regagent actions and opinions
        _, regagent_action_counts, regagent_opinion_counts = process_regagent_actions(all_actions, env.regagents, episode, n_steps)
        actions_history[episode] = regagent_action_counts # add occurrences of each action (as %)
        opinions_history[episode] = regagent_opinion_counts # add occurrences of each opinion (as %)
        # Process service provider availability
        sp_availability_episode = process_service_provider_availability(nRegAgents, n_steps, sum_sp_availability)
        for provider in env.providers:
            sp_availability_history[episode][provider] = sp_availability_episode[provider]
        sum_sp_availability = [0, 0, 0]
    
    # Log the trial results to the CSV file
    log_to_csv(kappa, rho, center_up_to_down, center_down_to_up, end_up_to_down, end_down_to_up, cost, direct_exp_weight, satisfaction_threshold, forgetting_factor,
               np.mean(total_rewards), np.mean(action_rewards), np.mean(opinion_rewards), np.mean(social_trust_history[:,0]), np.mean(social_trust_history[:,1]), np.mean(social_trust_history[:,2]),
               np.mean(actions_history[:,0]), np.mean(actions_history[:,1]), np.mean(actions_history[:,2]), np.mean(opinions_history[:,0]), 
               np.mean(opinions_history[:,1]), np.mean(opinions_history[:,2]), np.mean(opinions_history[:,3]), np.mean(opinions_history[:,4]), np.mean(opinions_history[:,5]), 
               np.mean(sp_availability_history[:,0]), np.mean(sp_availability_history[:,1]), np.mean(sp_availability_history[:,2]))

# %% [markdown]
# Log trial results to csv file
def log_to_csv(kappa, rho, center_up_to_down, center_down_to_up, end_up_to_down, end_down_to_up, cost, direct_exp_weight, satisfaction_threshold, forgetting_factor,
               total_rewards, action_rewards, opinion_rewards, sp0_social_trust, sp1_social_trust, sp2_social_trust, 
               sp0_request_rate, sp1_request_rate, sp2_request_rate, op0_expression_rate, 
               op1_expression_rate, op2_expression_rate, op3_expression_rate, op4_expression_rate, op5_expression_rate, 
               sp0_availability, sp1_availability, sp2_availability):
    csv_file_path = os.path.join(save_data_path, f"ch{chapter}-exp{experiment}-{date}-sensitivity-analysis.csv")
    file_exists = os.path.isfile(csv_file_path)
    with open(csv_file_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            # Write the header if the file doesn't exist
            writer.writerow(['kappa', 'rho', 'center_up_to_down', 'center_down_to_up', 'end_up_to_down', 'end_down_to_up', 'cost', 'direct_exp_weight', 'satisfaction_threshold', 'forgetting_factor',
               'total_rewards', 'action_rewards', 'opinion_rewards', 'sp0_social_trust', 'sp1_social_trust', 'sp2_social_trust', 'sp0_request_rate', 'sp1_request_rate', 'sp2_request_rate',
                'op0_expression_rate', 'op1_expression_rate', 'op2_expression_rate', 'op3_expression_rate', 'op4_expression_rate', 'op5_expression_rate', 'sp0_availability', 'sp1_availability', 'sp2_availability'])
        # Write the trial data
        writer.writerow([kappa, rho, center_up_to_down, center_down_to_up, end_up_to_down, end_down_to_up, cost, direct_exp_weight, satisfaction_threshold, forgetting_factor,
               total_rewards, action_rewards, opinion_rewards, sp0_social_trust, sp1_social_trust, sp2_social_trust, sp0_request_rate, sp1_request_rate, sp2_request_rate,
                op0_expression_rate, op1_expression_rate, op2_expression_rate, op3_expression_rate, op4_expression_rate, op5_expression_rate, sp0_availability, sp1_availability, sp2_availability])

# %% [markdown]
# Run sensitivity analysis
# Load the parameter sets for this machine
machine_id = os.environ.get('MACHINE_ID', '1')  # Set this environment variable uniquely for each machine
# Construct the file path
file_path = os.path.join(param_values_path, f'param_values_machine_{machine_id}.csv')

# Load the CSV file
param_values = np.loadtxt(file_path, delimiter=',')

# Run sensitivity analysis
for i in range(param_values.shape[0]):
    sensitivity_analysis(param_values[i])
