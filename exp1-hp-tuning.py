# %% [markdown]
# ## Applying Multi-Agent Reinforcement Learning to Study Hybrid Threats and Defensive Countermeasures
# Chapter 4\ 
# Hyperparameter tuning for experiment 1

# %% [markdown]
# Load libraries
from environment_exp1 import Environment
from a2c_agent import A2CRegAgent
from data_analysis import process_regagent_rewards, process_regagent_actions, process_regagent_states_average
import numpy as np
import torch
import random
import pandas as pd
import os
import datetime
import csv
import optuna
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from IPython.display import clear_output

# %% [markdown]
# Setup device, date, chapter, and experiment
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # currently cpu
date = datetime.datetime.now().strftime("%Y-%m-%d")
experiment = 1
chapter = 4

# %% [markdown]
# Specify output directory
save_data_path = os.path.join("hyperparameter-tuning", "exp1-results")
save_figures_path = os.path.join("hyperparameter-tuning", "exp1-figures")

# %% [markdown]
# Define the objective function
def objective(trial):

    # Specify number of agents in the environment
    nProviders = 3
    nRegAgents = 110
    nMalAgents = 0

    # Suggest hyperparameters
    alpha_rnn1 = trial.suggest_float('alpha_rnn1', 1e-5, 5e-4)
    alpha_rnn2 = trial.suggest_float('alpha_rnn2', 1e-5, 5e-4)
    gamma_rnn1 = trial.suggest_float('gamma1', 5e-1, 9.9e-1)
    gamma_rnn2 = trial.suggest_float('gamma2', 5e-1, 9.9e-1)
    beta_start = 1
    beta_end = trial.suggest_float('beta_end', 1e-4, 1e-2)
    beta_decay = trial.suggest_int('beta_decay', 1e+2, 3e+2)

    # Initialise social network, cyber-physical system, and agent parameters
    df = pd.read_csv("data/parameters.csv")
    parameters = dict(zip(df['parameter'], df['value']))

    # Social network parameters
    kappa = int(parameters['kappa'])
    rho = parameters['rho']

    # Cyber-physical system parameters
    center_up_to_down = [parameters['center_up_to_down']] * nProviders # Psi: prob (1 to -1)
    center_down_to_up = [parameters['center_down_to_up']] * nProviders # psi: prob (-1 to 1)
    end_up_to_down = [parameters['end_up_to_down']] * nProviders # Lambda: prob (1 to -1)
    end_down_to_up = [parameters['end_down_to_up']] * nProviders # lambda: prob (-1 to 1)
    cost = [parameters['cost']] * nProviders

    # Agents' attributes (parameters)
    direct_exp_weight = parameters['direct_exp_weight']
    feedback_adj_rate = parameters['feedback_adj_rate']
    forgetting_factor = parameters['forgetting_factor']

    # Define training time
    n_steps = 500 # number of steps per episode
    number_of_episodes = 500
    vis_freq = 10
    save_fig = False

    # Seed everything
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Create environment
    env = Environment(nRegAgents, nProviders, 
                        kappa, rho, center_up_to_down, center_down_to_up, end_up_to_down, end_down_to_up, cost,
                        direct_exp_weight, feedback_adj_rate, forgetting_factor)

    # Pick one agent randomly from each type of agents
    regagent_example = np.random.choice(env.regagents)

    state_shape, n_actions, n_opinions = \
        env.observation_spaces[f"regagent{regagent_example}"].shape[0], env.action_spaces[f"regagent{regagent_example}"][0].n, env.action_spaces[f"regagent{regagent_example}"][1].n

    print("Regular agents' state shape is", state_shape, ", number of actions is", n_actions, " and number of opinions is", n_opinions)

    # Define the agent with suggested hyperparameters
    regular_agents = {f"regagent{agent}": A2CRegAgent(state_shape, n_actions, n_opinions, alpha_rnn1, alpha_rnn2, device) for agent in env.regagents}

    # Collect data
    total_rewards, action_rewards, opinion_rewards = np.zeros(number_of_episodes + 1),  np.zeros(number_of_episodes + 1), np.zeros(number_of_episodes + 1)
    social_trust_history = np.zeros((number_of_episodes + 1, nProviders))
    actions_history, opinions_history = np.zeros((number_of_episodes + 1, nProviders)), np.zeros((number_of_episodes + 1, nProviders*2))
    loss1_agents, loss2_agents = np.zeros((number_of_episodes + 1, len(env.regagents))), np.zeros((number_of_episodes + 1, len(env.regagents)))
    loss1_history, loss2_history = np.zeros(number_of_episodes + 1), np.zeros(number_of_episodes + 1)

    # Train and evaluate the agent
    for episode in range(1, number_of_episodes + 1):
        print(f"Episode {episode}/{number_of_episodes}")
        # Decay entropy coefficient
        entropy_coef = beta_start + (beta_end - beta_start) * min(episode, beta_decay) / beta_decay

        # Restart environment
        observations, _ = env.reset(seed=seed)

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

            # Perform actions, determine next state, reward, and termination
            observations, rewards, _, _, _ = env.step(actions)

            # Store the rewards, observations, and actions for each agent for the current timestep
            for agent_name in episode_rewards.keys():
                episode_rewards[agent_name].append(rewards[agent_name])
                all_observations[agent_name].append(observations[agent_name])
                all_actions[agent_name].append(actions[agent_name])

        # Compute A2C loss (and backpropagate)
        for agent_name, agent in regular_agents.items():
            loss1, loss2 = agent.compute_a2c_loss(all_observations[agent_name][:-1], all_actions[agent_name], episode_rewards[agent_name], gamma_rnn1, gamma_rnn2, entropy_coef)
            agent_id = int(agent_name.replace("regagent", ""))
            loss1_agents[episode][agent_id] = loss1.data.cpu().item()
            loss2_agents[episode][agent_id] = loss2.data.cpu().item()

        # Process regagent rewards
        sum_regagent_rewards, service, feedback = process_regagent_rewards(episode_rewards)
        total_rewards[episode] = sum_regagent_rewards
        action_rewards[episode] = service
        opinion_rewards[episode] = feedback
        # Process social trust
        mean_trust_values = process_regagent_states_average(all_observations, env.providers, n_steps)
        social_trust_history[episode] = mean_trust_values
         # Processing regagent actions and opinions
        mean_selection_rate, mean_expression_rate = process_regagent_actions(all_actions, env.providers, n_steps)
        actions_history[episode] = mean_selection_rate # add occurrences of each action (as %)
        opinions_history[episode] = mean_expression_rate # add occurrences of each opinion (as %)

        # Calculate episode mean loss over all agents
        loss1_history[episode] = np.mean(loss1_agents[episode]) # mean loss for actions
        loss2_history[episode] = np.mean(loss2_agents[episode]) # mean loss for opinions

        # Visualise data
        if episode != 0 and episode % vis_freq == 0:
            clear_output(True)

            # Plot 1: Visualise average cumulative reward
            plt.figure()
            plt.plot(total_rewards[1:episode], linewidth=0.9, color = 'mediumvioletred', label = "Cumulative reward")
            plt.plot(action_rewards[1:episode], linewidth=0.9, color = 'red', label = "Service reward")
            plt.plot(opinion_rewards[1:episode], linewidth=0.9, color = 'orange', label = "Feedback reward")
            plt.xlabel("Episode")
            plt.ylabel("Cumulative reward\nfor regular agents per episode")
            plt.grid()
            plt.legend(loc=(0.01,0.50), fontsize='x-small')
            if save_fig:
                plt.savefig(os.path.join(save_figures_path, f'ch{chapter}-hp-tuning-exp{experiment}-{date}-{seed}-score.png'))
                plt.clf()

            # Plot 2: Visualise social trust in service providers
            colors = cm.tab10(np.arange(nProviders))
            plt.figure()
            for sp in range(nProviders):
                plt.plot(social_trust_history[1:episode, sp], color=colors[sp], linewidth=0.9, label = "Provider  {}".format(sp+1))
            plt.xlabel("Episode")
            plt.ylabel("Average social trust\nin service providers per episode")
            plt.grid()
            plt.legend(loc=(0.01,0.50), fontsize='x-small')
            if save_fig:
                plt.savefig(os.path.join(save_figures_path, f'ch{chapter}-hp-tuning-exp{experiment}-{date}-{seed}-trust.png'))
                plt.clf()

            # Plot 3.1: Visualise service requests rate (%) per episode
            fig, axe = plt.subplots(nrows = 1, ncols = 2, figsize = (15,4))
            for sp in range(nProviders):
                axe[0].plot(actions_history[1:episode, sp], color=colors[sp], linewidth = 0.9,
                    label = "Providers {}".format(sp+1))
            axe[0].set_ylim(0,100)
            axe[0].set_xlabel("Episode")
            axe[0].set_ylabel("Average service request rate\nper episode")
            axe[0].grid()
            axe[0].legend(loc=(0.01, 0.50), fontsize='x-small')
            # Plot 4.2: Visualise opinion expression rate (%) per episode
            for o in range(nProviders*2):
                provider_index = o // 2  # Determine which provider this opinion corresponds to
                opinion_type = "Negative" if o % 2 == 0 else "Positive"  # Alternate between negative and positive
                linestyle = "--" if o % 2 == 0 else "-"  # Negative: dashed, Positive: solid
                axe[1].plot(opinions_history[1:episode, o], linestyle, color=colors[provider_index], linewidth=0.9,
                    label=f"{opinion_type} opinion on provider {provider_index + 1}")
            axe[1].set_ylim(0,100)
            axe[1].set_xlabel("Episode")
            axe[1].set_ylabel("Average opinion expression rate\nper episode")
            axe[1].grid()
            axe[1].legend(loc=(0.01, 0.50), fontsize='x-small')
            if save_fig:
                plt.savefig(os.path.join(save_figures_path, f'ch{chapter}-hp-tuning-exp{experiment}-{date}-{seed}-actions-opinions.png'))
                plt.clf()

            # Plot 4.1/4.2: Visualise loss/objective value per episode
            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 4))
            axs[0].plot(loss1_history[1:episode], linewidth=0.9, color = 'red')
            axs[0].set_xlabel("Episode")
            axs[0].set_ylabel("Loss (actions)")
            axs[0].grid()
            axs[1].plot(loss2_history[1:episode], linewidth=0.9, color = 'orange')
            axs[1].set_xlabel("Episode")
            axs[1].set_ylabel("Loss (opinions)")
            axs[1].grid()
            if save_fig:
                plt.savefig(os.path.join(save_figures_path, f'ch{chapter}-hp-tuning-exp{experiment}-{date}-{seed}-loss.png'))
                plt.clf()
            plt.show()

    # Calculate the mean total rewards over stabilised episodes
    stabilised_episodes = number_of_episodes - beta_decay
    mean_total_reward = np.mean(total_rewards[-stabilised_episodes:])
    mean_loss1 = np.mean(loss1_history[-stabilised_episodes:])
    mean_loss2 = np.mean(loss2_history[-stabilised_episodes:])

    # Log the trial results to the CSV file
    log_to_csv(trial.number, seed, alpha_rnn1, alpha_rnn2, gamma_rnn1, gamma_rnn2, beta_start, beta_end, beta_decay, mean_total_reward, mean_loss1, mean_loss2)

    return abs(abs(mean_loss1) + abs(mean_loss2))
            
        
# %% [markdown]
# Function to log trial results to the CSV file
def log_to_csv(trial_number, seed, alpha_rnn1, alpha_rnn2, gamma_rnn1, gamma_rnn2, beta_start, beta_end, beta_decay, mean_total_reward, mean_loss1, mean_loss2):
    csv_file_path = os.path.join(save_data_path, f"ch{chapter}-hp-tuning-exp{experiment}-{date}.csv")
    file_exists = os.path.isfile(csv_file_path)
    with open(csv_file_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            # Write the header if the file doesn't exist
            writer.writerow(['trial_number', 'seed', 'alpha_rnn1', 'alpha_rnn2', 'gamma_rnn1', 'gamma_rnn2', 'beta_start', 'beta_end', 'beta_decay', 
                             'mean_total_reward', 'mean_loss1', 'mean_loss2'])
        # Write the trial data
        writer.writerow([trial_number, seed, alpha_rnn1, alpha_rnn2, gamma_rnn1, gamma_rnn2, beta_start, beta_end, beta_decay, mean_total_reward, mean_loss1, mean_loss2])

# %% [markdown]
# Run the optimisation
if __name__ == "__main__":
    # Connect to SQLite3 database (Optuna will create this if it doesn't exist)
    storage_name = "sqlite:///db.sqlite3"
    study_name = f'ch{chapter}-hp-tuning-exp{experiment}-{date}'
    direction = 'minimize'
    study = optuna.create_study(study_name=study_name, storage=storage_name, direction=direction, load_if_exists=True)
    n_trials = 1
    study.optimize(objective, n_trials=n_trials)
