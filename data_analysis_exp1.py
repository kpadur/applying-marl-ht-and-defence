import numpy as np

def process_regagent_states_average(all_observations, providers, n_steps): #
    """
    Calculate the average social trust (per episode) for every service provider.
    """
    # Only process regagent states
    all_regagent_observations = {
        agent: values for agent, values in all_observations.items() if agent.startswith('regagent')
    }

    # Initialise a matrix to hold the average trust values for each service provider at each time step
    average_trust = np.zeros((n_steps, len(providers)))

    # Calculate the average trust values for each service provider at each time step
    for t in range(n_steps):
        total_trust = np.zeros(len(providers))
        for agent in all_regagent_observations:
            total_trust += all_regagent_observations[agent][t]
        average_trust[t] = total_trust / len(all_regagent_observations)

    # Compute mean trust values across all time steps for each provider
    mean_trust_values = np.mean(average_trust, axis=0)

    return mean_trust_values

def process_regagent_actions(all_actions, providers, n_steps): #
    """
    Calculate the average service request rate (per episode) for each service provider and
    the average opinion experssing rate (per episode) for each service provider.
    """
    # Analyse action and opinion counts
    all_regagent_actions = {
        agent_name: value for agent_name, value in all_actions.items() if agent_name.startswith('regagent')
        }
    regagent_action_counts = [0] * len(providers)
    regagent_opinion_counts = [0] * len(providers)*2

    for actions_list in all_regagent_actions.values():
        for action, opinion in actions_list:
            regagent_action_counts[action] += 1
            regagent_opinion_counts[opinion] += 1

    mean_selection_rate = np.array(regagent_action_counts) * 100/ (len(all_regagent_actions)*n_steps)
    mean_expression_rate = np.array(regagent_opinion_counts) * 100/ (len(all_regagent_actions)*n_steps)

    return mean_selection_rate, mean_expression_rate

def process_regagent_rewards(episode_rewards): #
    """
    Calculate the cumulative rewards per episode for the regular agents.
    """
    # Determine total rewards
    regagent_rewards = {agent_name: value for agent_name, value in episode_rewards.items() if agent_name.startswith('regagent')}
    sum_regagent_rewards = sum([sum(r) for rewards in regagent_rewards.values() for r in rewards])
    # Determine service rewards
    service = sum([r[0] for rewards in regagent_rewards.values() for r in rewards])
    # Determine feedback rewards
    feedback = sum([r[1] for rewards in regagent_rewards.values() for r in rewards])
    return sum_regagent_rewards, service, feedback

def process_service_provider_availability(all_actions, providers, sum_sp_availability): #
    """
    Calculate average service availability rate per episode for each service provider (when requested).
    """
    # Analyse action and opinion counts
    all_regagent_actions = {
        agent_name: value for agent_name, value in all_actions.items() if agent_name.startswith('regagent')
        }
    regagent_action_counts = np.zeros(len(providers), dtype=int)

    for actions_list in all_regagent_actions.values():
        for action, _ in actions_list:
            regagent_action_counts[action] += 1
    avg_service_availability = sum_sp_availability / regagent_action_counts

    return avg_service_availability