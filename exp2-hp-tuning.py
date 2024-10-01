# %% [markdown]
# ## Applying Multi-Agent Reinforcement Learning to Study Hybrid Threats and Defensive Countermeasures
# Chapter 4\ 
# Hyperparameter tuning for experiment 2

# %% [markdown]
# Load libraries
from environment_exp2 import Environment
from a2c_agent import A2CRegAgent
from a2c_def_agent import A2CServiceProvider
from a2c_mal_agent import A2CMalAgent
from data_analysis import process_attacker_rewards, process_defender_rewards, process_regagent_rewards, process_regagent_states_average
from other_functions import moving_average
import numpy as np
import torch
import random
import optuna
import os
import re
import datetime
import csv
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import clear_output

# %% [markdown]
# Setup device, date, chapter, and experiment
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # currently cpu
date = datetime.datetime.now().strftime("%Y-%m-%d")
chapter = 4
experiment = 2

# %% [markdown]
# Specify output directory
save_data_path = os.path.join("hyperparameter_tuning", "exp2-results")
save_figures_path = os.path.join("hyperparameter_tuning", "exp2-figures")
get_nn_path = os.path.join("regagent-parameters")
# %% [markdown]
# Define the objective function
def objective(trial):
    # Specify number of agents in the environment
    nProviders = 3
    nRegAgents = 100
    nMalAgents = 10

    # Suggest hyperparameters
    df = pd.read_csv("hyperparameters.csv")
    hyperparameters = dict(zip(df['hyperparameter'], df['value']))

    alpha_rnn1 = hyperparameters['alpha_1']
    alpha_rnn2 = hyperparameters['alpha_2']

    alpha_ann1 = trial.suggest_float('alpha_ann1', 1e-6, 5e-4)
    alpha_ann2 = trial.suggest_float('alpha_ann2', 1e-6, 5e-4)
    alpha_ann3 = trial.suggest_float('alpha_ann3', 1e-6, 5e-4)
    alpha_dnn1 = trial.suggest_float('alpha_dnn1', 1e-6, 5e-5)
    alpha_dnn2 = trial.suggest_float('alpha_dnn2', 1e-6, 5e-5)
    gamma_ann1 = trial.suggest_float('gamma_ann1', 1e-1, 9.9e-1)
    gamma_ann2 = trial.suggest_float('gamma_ann2', 1e-1, 9.9e-1)
    gamma_ann3 = trial.suggest_float('gamma_ann3', 1e-1, 9.9e-1)
    gamma_dnn1 = trial.suggest_float('gamma_dnn1', 1e-1, 9.9e-1)
    gamma_dnn2 = trial.suggest_float('gamma_dnn2', 1e-1, 9.9e-1)
    beta_start = 1
    beta_end = trial.suggest_float('beta_end', 1e-4, 1e-2)
    beta_decay = trial.suggest_int('beta_decay', 1e+4, 1.5e+4)

    # Initialise social network, cyber-physical system, and agent parameters
    df = pd.read_csv("parameters.csv")
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
    satisfaction_threshold = parameters['satisfaction_threshold']
    forgetting_factor = parameters['forgetting_factor']

    # Define training time
    n_steps = 100 # number of time steps per episode
    number_of_episodes = 20000 # number of episodes
    vis_freq = 1000
    save_fig = False

    # Seed everything
    seed = 5282
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Create environment
    env = Environment(nRegAgents, nMalAgents, nProviders, 
                        kappa, rho, center_up_to_down, center_down_to_up, end_up_to_down, end_down_to_up, cost,
                        direct_exp_weight, satisfaction_threshold, forgetting_factor)

    # Create lists of agents, form social network of agents, and create attributes
    agent_ids, malagents, providers, neighbours = env.agents, env.malagents, env.providers, env.neighbours

    # Pick one agent randomly from each type of agents
    regagent_example, defender_example, malagent_example = np.random.choice(env.regagents), np.random.choice(providers), np.random.choice(env.malagents)

    # Define regular agents' state space, actions, and opinions
    state_shape, n_actions, n_opinions = env.observation_spaces[f"regagent{regagent_example}"].shape[0], env.action_spaces[f"regagent{regagent_example}"][0].n, env.action_spaces[f"regagent{regagent_example}"][1].n

    # Define defenders' (service providers') state space, and actions
    state_shape_defenders, n_filter, n_answer = (len(env.observation_spaces[f"defagent{defender_example}"]), nRegAgents + nMalAgents), env.action_spaces[f"defagent{defender_example}"][0].n, env.action_spaces[f"defagent{defender_example}"][1].n

    # Define attackers' state space, and actions
    state_size_attackers, n_stage_actions, n_cyber_actions, n_misinfo_actions, stage_actions_dict, mal_contacts_dict, malagents_dict = \
        len(env.observation_spaces[f"malagent"]), env.action_spaces[f"malagent"][0].n, env.action_spaces[f"malagent"][1].n, env.action_spaces[f"malagent"][2][f"malagent{malagent_example}"].n,\
        env.stage_actions_dict , env.mcontacts_dict, env.malagents_dict

    # Reset environment state
    envstate, info = env.reset(seed=seed)

    print("Regular agents' state shape is", state_shape, ", number of actions is", n_actions, " and number of opinions is", n_opinions)
    print("Defenders' state shape is", state_shape_defenders, ", number of filter actions is", n_filter, " and number of answer actions is", n_answer)
    print("Attackers' state shape is", state_size_attackers, ", number of stage actions is", n_stage_actions, ", number of cyber actions is", n_cyber_actions, " and number of misinformation actions is", n_misinfo_actions)

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

    # Initialise service providers (defenders)
    service_providers = {f"defagent{agent}": A2CServiceProvider(state_shape_defenders, n_filter, n_answer, alpha_dnn1, alpha_dnn2, device) 
                        for agent in env.providers}

    # Initialise attackers
    attack_target = 1 # initialised with environment (identifies sp)

    malicious_agent = {f"malagent": A2CMalAgent(state_size_attackers, n_stage_actions, n_cyber_actions, n_misinfo_actions, 
                    stage_actions_dict, mal_contacts_dict, malagents_dict, malagents, neighbours, attack_target,
                    alpha_ann1, alpha_ann2, alpha_ann3, device)}

    
    # Collect data
    total_attacker_rewards, reconnaissance_rewards, cyber_attack_rewards, disinfo_rewards, termination_rewards =\
        np.zeros(number_of_episodes + 1), np.zeros(number_of_episodes + 1), np.zeros(number_of_episodes + 1), np.zeros(number_of_episodes + 1), np.zeros(number_of_episodes + 1)
    loss1_history, loss2_history, loss3_history = np.zeros(number_of_episodes + 1), np.zeros(number_of_episodes + 1), np.zeros(number_of_episodes + 1)

    total_defender_rewards, defender_filter_rewards, defender_answer_rewards =\
        np.zeros((number_of_episodes + 1, len(providers))), np.zeros((number_of_episodes + 1, len(providers))), np.zeros((number_of_episodes + 1, len(providers)))
    loss4_history = np.zeros((number_of_episodes + 1, len(providers)))
    loss5_history = np.zeros((number_of_episodes + 1, len(providers)))

    regagent_total_rewards, regagent_action_rewards, regagent_opinion_rewards = np.zeros(number_of_episodes + 1), np.zeros(number_of_episodes + 1), np.zeros(number_of_episodes + 1)
    social_trust_history = np.zeros((number_of_episodes + 1, len(providers)))

    # Train attackers and defenders
    for episode in range(1, number_of_episodes + 1):
        # Decay entropy coefficient
        entropy_coef = beta_start + (beta_end - beta_start) * min(episode, beta_decay) / beta_decay

        # Restart environment
        observations, _ = env.reset()

        # Initialise dictionaries to store rewards, observations, and actions
        all_observations = {agent_name: [observations[agent_name]] for agent_name in agent_ids if "malagent" not in agent_name}
        all_observations["malagent"] = [observations["malagent"]]
        all_actions = {agent_name: [] for agent_name in agent_ids}
        all_actions["malagent"] = []
        episode_rewards = {agent_name: [] for agent_name in agent_ids}
        episode_rewards["malagent"] = []

        # Generate trajectory over n_steps (episode length)
        for timestep in range(1, n_steps+1): # one episode
            actions = {}  # Dictionary to store the current timestep's actions

            for agent_name, agent in regular_agents.items():
                s = observations[agent_name]
                # Sample action and opinion
                action, opinion = agent.sample_actions(s)
                actions[agent_name] = (action, opinion) # save action and opinion

            for attacker_name, attacker in malicious_agent.items(): # Attackers make decisions together
                states = all_observations[attacker_name]
                # Sample actions for malicious agents
                attack_stage_action, bot_action, sn_actions_dict, botnet = attacker.sample_actions(states)
                for agent_name, provider in botnet.items():
                    actions[agent_name] = (provider, -1)

            defenders_obs = env._monitor_requests_and_sn(actions) # which agents requested service, which opinions they expressed
            observations.update(defenders_obs)

            # Update actions with attacker information
            actions["malagent"] = (attack_stage_action, bot_action, sn_actions_dict) # actions[attacker_name]

            for defender_name, defender in service_providers.items():
                state = observations[defender_name]
                # Sample action and opinion
                filters, answers = defender.sample_actions(state)
                actions[defender_name] = (filters, answers)

            # Perform actions, determine next state, reward, and termination
            observations, rewards, _, _, _ = env.step(actions)

            # Store the rewards, observations, and actions for each agent for the current timestep
            for agent_name in agent_ids:
                if agent_name != "malagent":
                    agent = int(re.findall(r'\d+', agent_name)[0])
                    if agent in providers or agent in env.regagents: # regular agents and defenders
                        all_observations[agent_name].append(observations[agent_name])
                        all_actions[agent_name].append(actions[agent_name])
                        episode_rewards[agent_name].append(rewards[agent_name])
            # attackers
            all_observations["malagent"].append(observations["malagent"])
            all_actions["malagent"].append(actions["malagent"])
            episode_rewards["malagent"].append(rewards["malagent"])
        
        # if any(episode in range(start, end) for start, end in zip(range(0, number_of_episodes, 200), range(100, number_of_episodes, 200))):
        # Attackers compute A2C loss (and backpropagate)
        for agent_name, agent in malicious_agent.items():
            loss1, loss2, loss3 = agent.compute_a2c_loss(all_observations[agent_name][1:], all_actions[agent_name], episode_rewards[agent_name], gamma_ann1, gamma_ann2, gamma_ann3, entropy_coef)
            loss1_history[episode] = loss1.data.cpu().item()
            loss2_history[episode] = loss2.data.cpu().item()
            loss3_history[episode] = loss3.data.cpu().item()
        
        # Process attacker rewards
        attack_stage_rewards, recon_rewards, term_rewards, bot_rewards, contact_rewards = process_attacker_rewards(all_actions, episode_rewards)
        total_attacker_rewards[episode] = attack_stage_rewards # add sum rewards to history
        reconnaissance_rewards[episode] = recon_rewards
        termination_rewards[episode] = term_rewards
        cyber_attack_rewards[episode] = bot_rewards
        disinfo_rewards[episode] = contact_rewards

        # if any(episode in range(start, start +99) for start in range(100, number_of_episodes, 200)):
        # Defenders compute A2C loss (and backpropagate)
        for agent_name, agent in service_providers.items():
            loss4, loss5 = agent.compute_a2c_loss(all_observations[agent_name][1:], all_actions[agent_name], episode_rewards[agent_name], gamma_dnn1, gamma_dnn2, entropy_coef)
            agent_id = int(agent_name.replace("defagent", ""))
            loss4_history[episode][agent_id] = loss4.data.cpu().item()
            loss5_history[episode][agent_id] = loss5.data.cpu().item()
        
        # Store defenders information, including rewards
        # Process defender rewards
        filter_rewards, answer_rewards = process_defender_rewards(episode_rewards, providers)
        for defender in providers:
            defender_filter_rewards[episode][defender] = filter_rewards[defender]
            defender_answer_rewards[episode][defender] = answer_rewards[defender]
            total_defender_rewards[episode][defender] = filter_rewards[defender] + answer_rewards[defender]

        # Store agents information, including states, actions, and rewards
        # Process regagent rewards
        sum_regagent_rewards, service, feedback = process_regagent_rewards(episode_rewards)
        regagent_total_rewards[episode] = sum_regagent_rewards
        regagent_action_rewards[episode] = service
        regagent_opinion_rewards[episode] = feedback
        # Process social trust
        episode_states_pd = process_regagent_states_average(all_observations, providers, episode, n_steps)
        social_trust_history[episode][0] = episode_states_pd["trust_in_sp0"].mean()
        social_trust_history[episode][1] = episode_states_pd["trust_in_sp1"].mean()
        social_trust_history[episode][2] = episode_states_pd["trust_in_sp2"].mean()
        
        if episode != 1 and episode % vis_freq == 0:
            #clear_output(True)
            print("Episode", episode, ": mean attacker reward: %.3f" % (np.mean(total_attacker_rewards[:episode])),
                " and mean defender reward: %.3f" % (np.mean(total_defender_rewards[:episode])))

            # Figure 1. Defender rewards
            plt.figure()
            for sp in providers:
                plt.plot(moving_average(total_defender_rewards[1:episode,sp]), linewidth=0.9, label=f'Provider {sp+1}')
            plt.xlabel('Episodes')
            plt.ylabel("Cumulative reward for defenders")
            plt.grid()
            plt.legend(loc=(0.01,0.50), fontsize='x-small')
            if save_fig:
                plt.savefig(os.path.join(save_figures_path, f'ch{chapter}-hp-tuning-exp{experiment}-{date}-{seed}-defender-score.png'))
                plt.clf()

            # Figure 2.1/2.2: Defender filtering and answering rewards
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 4))
            for sp in providers:
                axes[0].plot(moving_average(defender_filter_rewards[1:episode,sp]), linewidth=0.9, label=f'Provider {sp+1}')
            axes[0].set_xlabel('Episodes')
            axes[0].set_ylabel("Cumulative reward\nfor filtering")
            axes[0].grid()
            axes[0].legend(loc=(0.01,0.50), fontsize='x-small')
            for sp in providers:
                axes[1].plot(moving_average(defender_answer_rewards[1:episode,sp]), linewidth=0.9, label=f'Provider {sp+1}')
            axes[1].set_xlabel('Episodes')
            axes[1].set_ylabel("Cumulative reward\nfor information spreading")
            axes[1].grid()
            axes[1].legend(loc=(0.01,0.50), fontsize='x-small')
            if save_fig:
                plt.savefig(os.path.join(save_figures_path, f'ch{chapter}-hp-tuning-exp{experiment}-{date}-{seed}-defender-score2.png'))
                plt.clf()

            # Figure 3.1/3.2: Defender loss
            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 4))
            for sp in providers:
                axs[0].plot(moving_average(loss4_history[1:episode,sp]), linewidth=0.9, label=f'Provider {sp+1}')
            axs[0].set_xlabel("Episodes")
            axs[0].set_ylabel("Loss (filtering)")
            axs[0].grid()
            axs[0].legend(loc=(0.01,0.50), fontsize='x-small')
            for sp in providers:
                axs[1].plot(moving_average(loss5_history[1:episode,sp]), linewidth=0.9, label=f'Provider {sp+1}')
            axs[1].set_xlabel("Episodes")
            axs[1].set_ylabel("Loss (answering)")
            axs[1].grid()
            axs[0].legend(loc=(0.01,0.50), fontsize='x-small')
            if save_fig:
                plt.savefig(os.path.join(save_figures_path, f'ch{chapter}-hp-tuning-exp{experiment}-{date}-{seed}-defender-loss.png'))
                plt.clf()

            # Attackers
            # Figure 4: Attacker rewards 
            plt.figure()
            plt.plot(moving_average(total_attacker_rewards[1:episode]), linewidth=0.9, color='mediumvioletred', label="Score")
            plt.plot(moving_average(reconnaissance_rewards[1:episode]), linewidth=0.9, color='blue', label='Reconnaissance score')
            plt.plot(moving_average(cyber_attack_rewards[1:episode]), linewidth=0.9, color='orange', label="Cyberattack score")
            plt.plot(moving_average(disinfo_rewards[1:episode]), linewidth=0.9, color='green', label="Disinformation score")
            plt.plot(moving_average(termination_rewards[1:episode]), linewidth=0.9, color='gray', label='Termination score')
            plt.xlabel("Episodes")
            plt.ylabel("Cumulative reward\nfor attackers")
            plt.grid()
            plt.legend(loc=(0.01,0.50), fontsize='x-small')
            if save_fig:
                plt.savefig(os.path.join(save_figures_path, f'ch{chapter}-hp-tuning-exp{experiment}-{date}-{seed}-attacker-score.png'))
                plt.clf()

            # Figure 5.1/5.2/5.3: Attacker loss
            fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(23, 4))
            ax[0].plot(moving_average(loss1_history[1:episode]), linewidth=0.9, color = 'mediumvioletred')
            ax[0].set_xlabel("Episodes")
            ax[0].set_ylabel("Loss (attack stage)")
            ax[0].grid()
            ax[1].plot(moving_average(loss2_history[1:episode]), linewidth=0.9, color = 'orange')
            ax[1].set_xlabel("Episodes")
            ax[1].set_ylabel("Loss (cyberattack)")
            ax[1].grid()
            ax[2].plot(moving_average(loss3_history[1:episode]), linewidth=0.9, color = 'green')
            ax[2].set_xlabel("Episodes")
            ax[2].set_ylabel("Loss (disinfo)")
            ax[2].grid()
            if save_fig:
                plt.savefig(os.path.join(save_figures_path, f'ch{chapter}-hp-tuning-exp{experiment}-{date}-{seed}-attacker-loss.png'))
                plt.clf()

            # Figure 6: Regular agents rewards
            plt.figure()
            plt.plot(moving_average(regagent_total_rewards[1:episode]), linewidth=0.9, color = 'mediumvioletred', label = "Mean score")
            plt.plot(moving_average(regagent_action_rewards[1:episode]), linewidth=0.9, color = 'red', label = "Mean service score")
            plt.plot(moving_average(regagent_opinion_rewards[1:episode]), linewidth=0.9, color = 'orange', label = "Mean feedback score")
            plt.xlabel("Episodes")
            plt.ylabel("Cumulative reward\nfor regular agents")
            plt.grid()
            plt.legend(loc=(0.01,0.50), fontsize='x-small')
            if save_fig:
                plt.savefig(os.path.join(save_figures_path, f'ch{chapter}-hp-tuning-exp{experiment}-{date}-{seed}-regagent-score.png'))
                plt.clf()

            # Figure 7: Visualise social trust in service providers
            plt.figure()
            for sp in range(nProviders):
                plt.plot(moving_average(social_trust_history[1:episode, sp]), linewidth=0.9, label = "Provider  {}".format(sp+1))
            plt.xlabel("Episode")
            plt.ylabel("Average social trust\nin service providers per episode")
            plt.grid()
            plt.legend(loc=(0.01,0.50), fontsize='x-small')
            if save_fig:
                plt.savefig(os.path.join(save_figures_path, f'ch{chapter}-hp-tuning-exp{experiment}-{date}-{seed}-trust.png'))
                plt.clf()
            #plt.show()
            
    # Calculate the mean of the stabilised rewards/loss for attackers
    stabilised_episodes = number_of_episodes - beta_decay
    mean_attacker_reward = np.mean(total_attacker_rewards[-stabilised_episodes:])
    mean_recon_reward = np.mean(reconnaissance_rewards[-stabilised_episodes:])
    mean_cyber_reward = np.mean(cyber_attack_rewards[-stabilised_episodes:])
    mean_disinfo_reward = np.mean(disinfo_rewards[-stabilised_episodes:])
    mean_term_reward = np.mean(termination_rewards[-stabilised_episodes:])
    mean_loss1 = np.mean(loss1_history[-stabilised_episodes:])
    mean_loss2 = np.mean(loss2_history[-stabilised_episodes:])
    mean_loss3 = np.mean(loss3_history[-stabilised_episodes:])
    # Calculate the mean of the stabilised rewards/loss for defenders
    mean_sp0_reward = np.mean(total_defender_rewards[-stabilised_episodes:][0])
    mean_sp1_reward = np.mean(total_defender_rewards[-stabilised_episodes:][1])
    mean_sp2_reward = np.mean(total_defender_rewards[-stabilised_episodes:][2])
    # Compute mean loss
    mean_loss4_sp0 = np.mean(loss4_history[-stabilised_episodes:][0])
    mean_loss4_sp1 = np.mean(loss4_history[-stabilised_episodes:][1])
    mean_loss4_sp2 = np.mean(loss4_history[-stabilised_episodes:][2])
    mean_loss5_sp0 = np.mean(loss5_history[-stabilised_episodes:][0])
    mean_loss5_sp1 = np.mean(loss5_history[-stabilised_episodes:][1])
    mean_loss5_sp2 = np.mean(loss5_history[-stabilised_episodes:][2])
    
    # Log the trial results to the CSV file
    log_to_csv(trial.number, seed, alpha_ann1, alpha_ann2, alpha_ann3, alpha_dnn1, alpha_dnn2, gamma_ann1, gamma_ann2, gamma_ann3, \
                gamma_dnn1, gamma_dnn2, beta_start, beta_end, beta_decay, mean_attacker_reward, mean_recon_reward, mean_cyber_reward, mean_disinfo_reward, mean_term_reward, \
                mean_sp0_reward, mean_sp1_reward, mean_sp2_reward, mean_loss1, mean_loss2, mean_loss3, mean_loss4_sp0, mean_loss4_sp1, mean_loss4_sp2, mean_loss5_sp0, \
                mean_loss5_sp1, mean_loss5_sp2)
        
    return abs(abs(mean_loss1) + abs(mean_loss2) + abs(mean_loss3) + abs(mean_loss4_sp1) + abs(mean_loss5_sp1))

# %% [markdown]
# Function to log trial results to the CSV file
def log_to_csv(trial_number, seed, alpha_ann1, alpha_ann2, alpha_ann3, alpha_dnn1, alpha_dnn2, gamma_ann1, gamma_ann2, gamma_ann3, \
                gamma_dnn1, gamma_dnn2, beta_start, beta_end, beta_decay, mean_attacker_reward, mean_recon_reward, mean_cyber_reward, mean_disinfo_reward, mean_term_reward, \
                mean_sp0_reward, mean_sp1_reward, mean_sp2_reward, mean_loss1, mean_loss2, mean_loss3, mean_loss4_sp0, mean_loss4_sp1, mean_loss4_sp2, mean_loss5_sp0, \
                mean_loss5_sp1, mean_loss5_sp2):
    
    csv_file_path = os.path.join(save_data_path, f"ch{chapter}-hp-tuning-exp{experiment}-{date}.csv")
    file_exists = os.path.isfile(csv_file_path)
    with open(csv_file_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            # Write the header if the file doesn't exist
            writer.writerow(['trial.number', 'seed', 'alpha_ann1', 'alpha_ann2', 'alpha_ann3', 'alpha_dnn1', 'alpha_dnn2', 'gamma_ann1', 'gamma_ann2', 'gamma_ann3', \
                'gamma_dnn1', 'gamma_dnn2', 'beta_start', 'beta_end', 'beta_decay', 'mean_attacker_reward', 'mean_recon_reward', 'mean_cyber_reward', 'mean_disinfo_reward', 'mean_term_reward', \
                'mean_sp0_reward', 'mean_sp1_reward', 'mean_sp2_reward', 'mean_loss1', 'mean_loss2', 'mean_loss3', 'mean_loss4_sp0', 'mean_loss4_sp1', 'mean_loss4_sp2', 'mean_loss5_sp0', \
                'mean_loss5_sp1', 'mean_loss5_sp2'])
        # Write the trial data
        writer.writerow([trial_number, seed, alpha_ann1, alpha_ann2, alpha_ann3, alpha_dnn1, alpha_dnn2, gamma_ann1, gamma_ann2, gamma_ann3, \
                gamma_dnn1, gamma_dnn2, beta_start, beta_end, beta_decay, mean_attacker_reward, mean_recon_reward, mean_cyber_reward, mean_disinfo_reward, mean_term_reward, \
                mean_sp0_reward, mean_sp1_reward, mean_sp2_reward, mean_loss1, mean_loss2, mean_loss3, mean_loss4_sp0, mean_loss4_sp1, mean_loss4_sp2, mean_loss5_sp0, \
                mean_loss5_sp1, mean_loss5_sp2])


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
