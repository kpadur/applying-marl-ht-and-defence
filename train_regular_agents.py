# %% [markdown]
# ## Applying Multi-Agent Reinforcement Learning to Study Hybrid Threats and Defensive Countermeasures
# Chapter 4\ 
# Experiment 1

# %% [markdown]
# Import libraries
from environment_exp1 import Environment
from a2c_agent import A2CRegAgent
from data_analysis import process_regagent_rewards, process_regagent_states_average, process_regagent_actions, \
    process_service_provider_availability
from save_data import read_csv_to_dict, save_data_to_csv
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from IPython.display import clear_output
import os
import datetime

# %% [markdown]
# Setup device, date, chapter, and experiment
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # currently cpu
date = datetime.datetime.now().strftime("%Y-%m-%d")
chapter = 4
experiment = 1

# %% [markdown]
# Specify output directory
save_path = os.path.join("results", "exp1-results")
nn_path = os.path.join("regagent-parameters")

# %% [markdown]
# Specify number of agents in the environment
nProviders = 3
nRegAgents = 110
nMalAgents = 0

# %% [markdown]
# Initialise (tuned) hyperparameters
hyperparameters = read_csv_to_dict("data/hyperparameters.csv")
alpha_rnn1 = hyperparameters['alpha_1']
alpha_rnn2 = hyperparameters['alpha_2']
gamma_rnn1 = hyperparameters['gamma_1']
gamma_rnn2 = hyperparameters['gamma_2']
beta_start = 1
beta_end = hyperparameters['beta_1']
beta_decay = int(hyperparameters['n_1'])

# %% [markdown]
# Initialise social network, cyber-physical system, and agent parameters
# Load parameters
parameters = read_csv_to_dict("data/parameters.csv")

# Social network parameters
kappa = int(parameters['kappa'])
rho = parameters['rho']

# Cyber-physical system parameters
center_up_to_down = [parameters['center_up_to_down']] * nProviders  # Psi: prob (1 to -1)
center_down_to_up = [parameters['center_down_to_up']] * nProviders  # psi: prob (-1 to 1)
end_up_to_down = [parameters['end_up_to_down']] * nProviders        # Lambda: prob (1 to -1)
end_down_to_up = [parameters['end_down_to_up']] * nProviders        # lambda: prob (-1 to 1)
cost = [parameters['cost']] * nProviders

# Agents' attributes (parameters)
direct_exp_weight = parameters['direct_exp_weight']
feedback_adj_rate = parameters['feedback_adj_rate']
forgetting_factor = parameters['forgetting_factor']

# %% [markdown]
# Define training time, visualisation and saving frequency
n_steps = 500 # number of steps per episode
number_of_episodes = 500 # number of episodes
vis_freq = 10
saving_freq = 10
save_fig = False # save figures
save_nns = True # save neural networks

# %% [markdown]
# Initialise seed for reproducibility
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# %% [markdown]
# Create environment
env = Environment(nRegAgents, nProviders, 
                  kappa, rho, center_up_to_down, center_down_to_up, end_up_to_down, end_down_to_up, cost,
                  direct_exp_weight, feedback_adj_rate, forgetting_factor)

# Create lists of agents, form social network of agents, and create attributes
regagents, providers = env.regagents, env.providers

# Pick one agent randomly from each type of agents
regagent_example = np.random.choice(regagents)

# Define regular agents' state space, actions, and opinions
observation_space, action_space = env.observation_spaces[f"regagent{regagent_example}"], env.action_spaces[f"regagent{regagent_example}"]

state_shape, n_actions, n_opinions = \
    env.observation_spaces[f"regagent{regagent_example}"].shape[0], env.action_spaces[f"regagent{regagent_example}"][0].n, env.action_spaces[f"regagent{regagent_example}"][1].n

# Reset environment state
envstate, info = env.reset(seed=seed)

print("Regular agents' state shape is", state_shape, ", number of actions is", n_actions, " and number of opinions is", n_opinions)

# Visualise graph
fig = env.render(graph_type='actions')
plt.show()
if save_fig:
    fig.savefig(os.path.join(save_path, f'ch{chapter}-exp{experiment}-{date}-{seed}-environment.png'))
    plt.clf()

# %% [markdown]
# Test the environment
from pettingzoo.test.parallel_test import parallel_api_test
parallel_api_test(env, num_cycles=10)

# %% [markdown]
# Create regular agents
regular_agents = {f"regagent{agent}": A2CRegAgent(state_shape, n_actions, n_opinions, alpha_rnn1, alpha_rnn2, device) 
                  for agent in env.regagents}

# %% [markdown]
# Collect data
total_rewards, action_rewards, opinion_rewards = np.zeros(number_of_episodes + 1),  np.zeros(number_of_episodes + 1), np.zeros(number_of_episodes + 1)
actions_history, opinions_history = np.zeros((number_of_episodes + 1, nProviders)), np.zeros((number_of_episodes + 1, nProviders*2))
social_trust_history = np.zeros((number_of_episodes + 1, nProviders))

loss1_agents, loss2_agents = np.zeros((number_of_episodes + 1, len(env.regagents))), np.zeros((number_of_episodes + 1, len(env.regagents)))
loss1_history, loss2_history = np.zeros(number_of_episodes + 1), np.zeros(number_of_episodes + 1)

sum_sp_availability = np.zeros(len(providers), dtype=int)
sp_availability_history = np.zeros((number_of_episodes + 1, len(providers)))

# %% [markdown]
# Train, evaluate, and visualise agents' behaviour
for episode in range(1, number_of_episodes + 1):
    print(f"Episode {episode} of {number_of_episodes} episodes")
    
    # Decay entropy coefficient for new episode
    entropy_coef = beta_start + (beta_end - beta_start) * min(episode, beta_decay) / beta_decay

    # Restart environment
    observations, _ = env.reset(seed=seed)

    # Initialise dictionaries to store rewards, observations, and actions in one episode
    all_observations = {agent_name: [observations[agent_name]] for agent_name in regular_agents.keys()}
    all_actions = {agent_name: [] for agent_name in regular_agents.keys()}
    episode_rewards = {agent_name: [] for agent_name in regular_agents.keys()}

    # Collect data for one episode (n_steps)
    for timestep in range(1, n_steps + 1): # one episode

        actions = {}  # Dictionary to store the current timestep's actions
        episode_opinion_probs = np.zeros(len(providers)*2)
        
        for agent_name, agent in regular_agents.items():
            state = observations[agent_name]
            # Sample action and opinion
            action, opinion = agent.sample_actions(state)
            actions[agent_name] = (action, opinion) # save action and opinion
        
        # Service providers service availability
        for agent_name, (action, _) in actions.items():            
            # Check if the endpoint for the requested provider is available
            if env.endpoint[agent_name][action] == 1:
                sum_sp_availability[action] += 1

        # Perform actions, determine next state, reward, and termination
        observations, rewards, terminations, truncations, infos = env.step(actions)

        # Render environment
        if episode in [1,100,500] and timestep == 500:
            clear_output(wait=True)  # Clear the previous output
            fig = env.render(graph_type='both')  # Render the graph for the current timestep
            fig.text(0.01, 0.90, f'Episode: {episode}', ha='left', fontsize=14, color='black') # Add dynamic text (episode and timestep) to the figure
            fig.text(0.01, 0.86, f'Timestep: {timestep}', ha='left', fontsize=14, color='black') # Add dynamic text (episode and timestep) to the figure
            plt.show()  # Display the new figure
            if save_fig:
                fig.savefig(os.path.join(save_path, f'ch{chapter}-exp{experiment}-{date}-{seed}-{episode}-environment.png'))
                plt.clf()

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
    mean_trust_values = process_regagent_states_average(all_observations, providers, n_steps)
    social_trust_history[episode] = mean_trust_values
    # Processing regagent actions and opinions
    mean_selection_rate, mean_expression_rate = process_regagent_actions(all_actions, providers, n_steps)
    actions_history[episode] = mean_selection_rate # add occurrences of each action (as %)
    opinions_history[episode] = mean_expression_rate # add occurrences of each opinion (as %)
    # Process regagents loss
    loss1_history[episode] = np.mean(loss1_agents[episode]) # mean loss for actions
    loss2_history[episode] = np.mean(loss2_agents[episode]) # mean loss for opinions
    # Process service provider availability
    sp_availability_episode = process_service_provider_availability(all_actions, providers, sum_sp_availability)
    sp_availability_history[episode] = sp_availability_episode
    sum_sp_availability = np.zeros(len(providers), dtype=int) # reset count to zero
        
    # Visualise data
    if episode != 1 and episode % vis_freq == 0:
        clear_output(True)

        # Plot 1: Visualise cumulative reward
        plt.figure()
        plt.plot(total_rewards[1:episode], linewidth=0.9, color = 'mediumvioletred', label = "Cumulative reward")
        plt.plot(action_rewards[1:episode], linewidth=0.9, color = 'red', label = "Service reward")
        plt.plot(opinion_rewards[1:episode], linewidth=0.9, color = 'orange', label = "Feedback reward")
        plt.xlabel("Episode")
        plt.ylabel("Cumulative reward\nfor regular agents per episode")
        plt.grid()
        plt.legend(loc=(0.01,0.50), fontsize='x-small')
        if save_fig:
            plt.savefig(os.path.join(save_path, f'ch{chapter}-exp{experiment}-{date}-{seed}-score.png'))
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
            plt.savefig(os.path.join(save_path, f'ch{chapter}-exp{experiment}-{date}-{seed}-trust.png'))
            plt.clf()

        # Plot 3: Visualise service provider availability
        plt.figure()
        for sp in range(nProviders):
            plt.plot(sp_availability_history[1:episode, sp], color=colors[sp], linewidth=0.9, label = "Provider  {}".format(sp+1))
        plt.xlabel("Episode")
        plt.ylabel("Average service provider availability\nper episode")
        plt.grid()
        plt.legend(loc=(0.01,0.50), fontsize='x-small')
        if save_fig:
            plt.savefig(os.path.join(save_path, f'ch{chapter}-exp{experiment}-{date}-{seed}-availability.png'))
            plt.clf()

        # Plot 4.1: Visualise service request rate (%) per episode
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
            plt.savefig(os.path.join(save_path, f'ch{chapter}-exp{experiment}-{date}-{seed}-actions-opinions.png'))
            plt.clf()

        # Plot 5.1/5.2: Visualise loss/objective value per episode
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
            plt.savefig(os.path.join(save_path, f'ch{chapter}-exp{experiment}-{date}-{seed}-loss.png'))
            plt.clf()
        plt.show()

# %% [markdown]
# Prepare data dictionary
data_dict = {
    'total_rewards': total_rewards, 'action_rewards': action_rewards, 'opinion_rewards': opinion_rewards,
    'sp0_social_trust': social_trust_history[:, 0], 'sp1_social_trust': social_trust_history[:, 1], 'sp2_social_trust': social_trust_history[:, 2],
    'sp0_request_rate': actions_history[:, 0], 'sp1_request_rate': actions_history[:, 1], 'sp2_request_rate': actions_history[:, 2],
    'op0_expression_rate': opinions_history[:, 0], 'op1_expression_rate': opinions_history[:, 1], 'op2_expression_rate': opinions_history[:, 2],
    'op3_expression_rate': opinions_history[:, 3], 'op4_expression_rate': opinions_history[:, 4], 'op5_expression_rate': opinions_history[:, 5],
    'mean_loss1': loss1_history, 'mean_loss2': loss2_history,
    'sp0_availability': sp_availability_history[:, 0], 'sp1_availability': sp_availability_history[:, 1], 'sp2_availability': sp_availability_history[:, 2]
}

# Save to CSV
file_name = f'ch{chapter}-exp{experiment}-{date}-{seed}-regagents-data.csv'
file_path = os.path.join(save_path, file_name)
save_data_to_csv(file_path, data_dict)

# %% [markdown]
# Save models
if save_nns:
    for agent_name, agent in regular_agents.items():
        torch.save({
            'actions_state_dict': agent.action_nn.state_dict(),
            'actions_opt_state_dict': agent.action_opt.state_dict(),
        }, f'{agent_name}_checkpoint_actions_{date}_{seed}.pth')

        torch.save({
            'opinions_state_dict': agent.opinion_nn.state_dict(),
            'opinions_opt_state_dict': agent.opinion_opt.state_dict(),
        }, f'{agent_name}_checkpoint_opinions_{date}_{seed}.pth')
