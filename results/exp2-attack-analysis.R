# Applying multi-agent reinforcement learning to study hybrid threats and defensive countermeasures
# Exp 2 data analysis and visualisation

# Load libraries
library(readr)
library(ggplot2)
library(zoo)
library(dplyr)
library(tidyr)

#####
# Attacker rewards: Analyse the effectiveness of attackers' performance
#####
# Import data
path <- "exp2-results/"
all_files <- list.files(path, pattern = "*attacker-data.csv", full.names = TRUE)

# Calculate metrics
stabilised_episode = 11148
number_of_episodes = 20000
stabilised_period = number_of_episodes-stabilised_episode

# Rewards: Cumulative reward over stabilised episodes
extract_stabilised_rw_values <- function(file) {
  data <- read_csv(file, show_col_types = FALSE)
  rw_values <- tail(data$total_rewards, stabilised_period)
  return(rw_values)
}
rw_values <- sapply(all_files, extract_stabilised_rw_values)
rw_result <- list(
  mean = mean(rw_values),
  sd = sd(rw_values), # Convergence
  min = min(rw_values),
  max = max(rw_values)
)

print(rw_result)

# Visualise rewards
# Read 'total rewards' column from each file and combine
total_reward_data <- lapply(all_files, function(file) {
  df <- read_csv(file,show_col_types = FALSE)
  return(df$total_rewards)
})
names(total_reward_data) <- sapply(all_files, function(f) paste0("repetition_", basename(f)))
combined_total_reward_data <- as.data.frame(total_reward_data)

# Define the window size for the moving average (0 to get original results)
window_size <- 10

# Apply moving average to each column
smoothed_total_reward_data <- as.data.frame(lapply(combined_total_reward_data, function(column) {
  rollmean(column, window_size, fill = NA, align = "right")
}))

# Calculate mean, sd, and variance of the smoothed data
episode_mean_total_rewards <- apply(smoothed_total_reward_data, 1, mean, na.rm = TRUE)
episode_sd_total_rewards <- apply(smoothed_total_reward_data, 1, sd, na.rm = TRUE)
episode_variance_total_rewards <- apply(smoothed_total_reward_data, 1, var, na.rm = TRUE)

# Create summary data frame
summary_total_rewards_data <- data.frame(
  episode = 1:length(episode_mean_total_rewards),
  mean = episode_mean_total_rewards,
  sd = episode_sd_total_rewards,
  variance = episode_variance_total_rewards
)

# Read 'recon rewards' column from each file and combine
recon_reward_data <- lapply(all_files, function(file) {
  df <- read_csv(file,show_col_types = FALSE)
  return(df$recon_rewards)
})
names(recon_reward_data) <- sapply(all_files, function(f) paste0("repetition_", basename(f)))
combined_recon_reward_data <- as.data.frame(recon_reward_data)

# Apply moving average to each column
smoothed_recon_reward_data <- as.data.frame(lapply(combined_recon_reward_data, function(column) {
  rollmean(column, window_size, fill = NA, align = "right")
}))

# Calculate mean, sd, and variance of the smoothed data
episode_mean_recon_rewards <- apply(smoothed_recon_reward_data, 1, mean, na.rm = TRUE)
episode_sd_recon_rewards <- apply(smoothed_recon_reward_data, 1, sd, na.rm = TRUE)
episode_variance_recon_rewards <- apply(smoothed_recon_reward_data, 1, var, na.rm = TRUE)

summary_recon_rewards_data <- data.frame(
  episode = 1:length(episode_mean_recon_rewards),
  mean = episode_mean_recon_rewards,
  sd = episode_sd_recon_rewards,
  variance = episode_variance_recon_rewards
)

# Read 'cyber attack rewards' column from each file and combine
cyber_reward_data <- lapply(all_files, function(file) {
  df <- read_csv(file,show_col_types = FALSE)
  return(df$cyber_attack_rewards)
})
names(cyber_reward_data) <- sapply(all_files, function(f) paste0("repetition_", basename(f)))
combined_cyber_reward_data <- as.data.frame(cyber_reward_data)

# Apply moving average to each column
smoothed_cyber_reward_data <- as.data.frame(lapply(combined_cyber_reward_data, function(column) {
  rollmean(column, window_size, fill = NA, align = "right")
}))

# Calculate mean, sd, and variance of the smoothed data
episode_mean_cyber_rewards <- apply(smoothed_cyber_reward_data, 1, mean, na.rm = TRUE)
episode_sd_cyber_rewards <- apply(smoothed_cyber_reward_data, 1, sd, na.rm = TRUE)
episode_variance_cyber_rewards <- apply(smoothed_cyber_reward_data, 1, var, na.rm = TRUE)

summary_cyber_rewards_data <- data.frame(
  episode = 1:length(episode_mean_cyber_rewards),
  mean = episode_mean_cyber_rewards,
  sd = episode_sd_cyber_rewards,
  variance = episode_variance_cyber_rewards
)

# Read 'disinfo rewards' column from each file and combine
disinfo_reward_data <- lapply(all_files, function(file) {
  df <- read_csv(file,show_col_types = FALSE)
  return(df$disinfo_rewards)
})
names(disinfo_reward_data) <- sapply(all_files, function(f) paste0("repetition_", basename(f)))
combined_disinfo_reward_data <- as.data.frame(disinfo_reward_data)

# Apply moving average to each column
smoothed_disinfo_reward_data <- as.data.frame(lapply(combined_disinfo_reward_data, function(column) {
  rollmean(column, window_size, fill = NA, align = "right")
}))

# Calculate mean, sd, and variance of the smoothed data
episode_mean_disinfo_rewards <- apply(smoothed_disinfo_reward_data, 1, mean, na.rm = TRUE)
episode_sd_disinfo_rewards <- apply(smoothed_disinfo_reward_data, 1, sd, na.rm = TRUE)
episode_variance_disinfo_rewards <- apply(smoothed_disinfo_reward_data, 1, var, na.rm = TRUE)

summary_disinfo_rewards_data <- data.frame(
  episode = 1:length(episode_mean_disinfo_rewards),
  mean = episode_mean_disinfo_rewards,
  sd = episode_sd_disinfo_rewards,
  variance = episode_variance_disinfo_rewards
)

# Read 'term rewards' column from each file and combine
term_reward_data <- lapply(all_files, function(file) {
  df <- read_csv(file,show_col_types = FALSE)
  return(df$term_rewards)
})
names(term_reward_data) <- sapply(all_files, function(f) paste0("repetition_", basename(f)))
combined_term_reward_data <- as.data.frame(term_reward_data)

# Apply moving average to each column
smoothed_term_reward_data <- as.data.frame(lapply(combined_term_reward_data, function(column) {
  rollmean(column, window_size, fill = NA, align = "right")
}))

# Calculate mean, sd, and variance of the smoothed data
episode_mean_term_rewards <- apply(smoothed_term_reward_data, 1, mean, na.rm = TRUE)
episode_sd_term_rewards <- apply(smoothed_term_reward_data, 1, sd, na.rm = TRUE)
episode_variance_term_rewards <- apply(smoothed_term_reward_data, 1, var, na.rm = TRUE)

summary_term_rewards_data <- data.frame(
  episode = 1:length(episode_mean_term_rewards),
  mean = episode_mean_term_rewards,
  sd = episode_sd_term_rewards,
  variance = episode_variance_term_rewards
)


# Visualise rewards
a1 <- ggplot() +
  geom_line(data = summary_total_rewards_data, aes(x = episode, y = mean, color = "Mean score")) +
  geom_ribbon(data = summary_total_rewards_data, aes(x = episode, ymin = mean - sd, ymax = mean + sd), 
              fill = "#FF99CC", alpha = 0.2) +
  
  geom_line(data = summary_recon_rewards_data, aes(x = episode, y = mean, color = "Mean score for reconnaissance")) +
  geom_ribbon(data = summary_recon_rewards_data, aes(x = episode, ymin = mean - sd, ymax = mean + sd),
              fill = "#99CCFF", alpha = 0.2) +
  
  geom_line(data = summary_cyber_rewards_data, aes(x = episode, y = mean, color = "Mean score for a cyberattack")) +
  geom_ribbon(data = summary_cyber_rewards_data, aes(x = episode, ymin = mean - sd, ymax = mean + sd),
              fill = "#FFCC66", alpha = 0.2) +
  
  geom_line(data = summary_disinfo_rewards_data, aes(x = episode, y = mean, color = "Mean score for disinformation")) +
  geom_ribbon(data = summary_disinfo_rewards_data, aes(x = episode, ymin = mean - sd, ymax = mean + sd),
              fill = "#CCFFCC", alpha = 0.2) +
  
  geom_line(data = summary_term_rewards_data, aes(x = episode, y = mean, color = "Mean score for termination")) +
  geom_ribbon(data = summary_term_rewards_data, aes(x = episode, ymin = mean - sd, ymax = mean + sd),
              fill = "#CCCCCC", alpha = 0.2) +
  
  scale_color_manual(values = c("Mean score" = "#CC0033", 
                                "Mean score for reconnaissance" = "#0033CC",
                                "Mean score for a cyberattack" = "#FF9900",
                                "Mean score for disinformation" = "#006633",
                                "Mean score for termination" = "#666666")) +
  labs(x = "Episode", y = "Cumulative reward\nfor attackers", color = NULL) +
  theme_bw() +
  theme(
    legend.position = c(0.20, 0.70),
    plot.title = element_text(size = rel(1)),
    axis.title = element_text(size = rel(1)),
    axis.text = element_text(size = rel(1)),
    legend.title = element_text(size = rel(1)),
    legend.text = element_text(size = rel(0.80)),
    plot.margin = margin(1, 1, 1, 1, "cm"),
    axis.title.x = element_text(vjust = -1),
    axis.title.y = element_text(vjust = 1),
    legend.background = element_blank(),
    legend.key = element_blank()
  )
a1

# Saved /ch4-marl/results/plots/ch4-exp2-attacker-rewards.pdf as 5 x 7.50 (landscape)

#####
# Attack strategy: Analyse order and count of time steps in each attack stage
#####
# Read all files
order_files <- list.files(path = "exp2-results/", pattern = "*-attack-order.csv", full.names = TRUE)
count_files <- list.files(path = "exp2-results/", pattern = "*-attack-actions-count.csv", full.names = TRUE)
stabilised_episodes = 11148

# Assuming files are in the same order and there's a 1-to-1 match
combined_dfs <- lapply(1:length(order_files), function(i) {
  order_df <- read.csv(order_files[i])
  count_df <- read.csv(count_files[i])
  
  # # Remove the first 11148 rows (can remove these two lines if want to include everything)
  order_df <- tail(order_df, -stabilised_episodes) # take out everything before stabilised episodes
  count_df <- tail(count_df, -stabilised_episodes)
  
  # Remove episode column from one of the dataframes to avoid duplication
  combined_df <- cbind(order_df, count_df)
  
  return(combined_df)
})

# Combine all data into one dataframe
all_data <- do.call(rbind, combined_dfs)

colnames(all_data) <- c("reconnaissance", "cyberattack", "disinformation", 
                        "combined_attack", "cyberattack_cont", "disinfo_cont",
                        "count_reconnaissance", "count_cyberattack", "count_disinformation", 
                        "count_combined_attack", "count_cyberattack_cont", "count_disinfo_cont")

# Create strategy column
all_data$strategy <- apply(all_data[, 1:6], 1, function(row) {
  attack_labels <- c("reconnaissance", "cyberattack", "disinformation", "combined_attack", "cyberattack_cont", "disinfo_cont")
  
  # Find non-zero indices
  non_zero_indices <- which(row > 0)
  
  # Order non-zero values and get their indices
  ordered_indices <- non_zero_indices[order(row[non_zero_indices])]
  
  # Get attack labels based on ordered indices
  strategy_sequence <- attack_labels[ordered_indices]
  
  paste(strategy_sequence, collapse = "->")
})

# Calculate frequency of each attack strategy
strategy_freq <- all_data %>%
  group_by(strategy) %>%
  tally() %>%
  arrange(desc(n))

# Calculate the average time spent in each attack stage for each strategy
strategy_avg_time <- all_data %>%
  group_by(strategy) %>%
  summarise(
    reconnaissance_avg = mean(count_reconnaissance, na.rm = TRUE),
    cyberattack_avg = mean(count_cyberattack, na.rm = TRUE),
    disinformation_avg = mean(count_disinformation, na.rm = TRUE),
    combined_attack_avg = mean(count_combined_attack, na.rm = TRUE),
    cyberattack_cont_avg = mean(count_cyberattack_cont, na.rm = TRUE),
    disinfo_cont_avg = mean(count_disinfo_cont, na.rm = TRUE)
  )

# Combine the frequency and average times
final_data <- left_join(strategy_freq, strategy_avg_time, by = "strategy")
#write.csv(final_data, "attack-strategies-stabilised.csv", row.names = FALSE)
