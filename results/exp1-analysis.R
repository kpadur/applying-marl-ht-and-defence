# Applying multi-agent reinforcement learning to study hybrid threats and defensive countermeasures
# Exp 1 data analysis and visualisation
# Load libraries
library(readr)
library(ggplot2)
library(cowplot)

#####
# Read all files and combine
# Exp1
path <- "exp1-results"
all_files <- list.files(path, pattern = "*regagents-data.csv", full.names = TRUE)

#####
# Calculate metrics (Exp1)
stabilised_episode = 145
number_of_episodes = 500
stabilised_period = number_of_episodes-stabilised_episode

# Average cumulative reward over stabilised episodes
extract_stabilised_rw_values <- function(file) {
  data <- read_csv(file, show_col_types = FALSE)
  rw_values <- tail(data$total_rewards, stabilised_period)
  return(rw_values)
}
rw_values <- sapply(all_files, extract_stabilised_rw_values)
rw_result <- list(
  mean = mean(rw_values),
  sd = sd(rw_values),
  min = min(rw_values),
  max = max(rw_values)
)
print(rw_result)

# Average social trust over stabilised episodes (for every service provider)
extract_stabilised_st_values <- function(file) {
  data <- read_csv(file, show_col_types = FALSE)
  st_values <- tail(data$sp0_social_trust, stabilised_period) # change to sp0_social_trust, sp1_social_trust, sp2_social_trust
  return(st_values)
}
st_values <- sapply(all_files, extract_stabilised_st_values)
st_values <- list(
  mean = mean(st_values),
  sd = sd(st_values),
  min = min(st_values),
  max = max(st_values)
)
print(st_values)

# Service request rate (for every service provider)
extract_stabilised_sr_values <- function(file) {
  data <- read_csv(file, show_col_types = FALSE)
  sr_values <- tail(data$sp0_request_rate, stabilised_period) # change to sp0_request_rate, sp1_request_rate, sp2_request_rate
  return(sr_values)
}
sr_values <- sapply(all_files, extract_stabilised_sr_values)
sr_values <- list(
  mean = mean(sr_values),
  sd = sd(sr_values),
  min = min(sr_values),
  max = max(sr_values)
)
print(sr_values)

# Opinion expression rate (for every opinion)
extract_stabilised_ox_values <- function(file) {
  data <- read_csv(file, show_col_types = FALSE)
  ox_values <- tail(data$op0_expression_rate, stabilised_period) # change to op0_expression_rate, op1_expression_rate, op2_expression_rate, etc.
  return(ox_values)
}
ox_values <- sapply(all_files, extract_stabilised_ox_values)
ox_values <- list(
  mean = mean(ox_values),
  sd = sd(ox_values),
  min = min(ox_values),
  max = max(ox_values)
)
print(ox_values)

# Service availability rate (for every service provider)
extract_stabilised_sa_values <- function(file) {
  data <- read_csv(file, show_col_types = FALSE)
  sa_values <- tail(data$sp0_availability, stabilised_period) # change to sp0_availability, sp1_availability, sp2_availability
  sa_values <- sa_values * 100
  return(sa_values)
}
sa_values <- sapply(all_files, extract_stabilised_sa_values)
sa_values <- list(
  mean = mean(sa_values),
  sd = sd(sa_values),
  min = min(sa_values),
  max = max(sa_values)
)

print(sa_values)

# Loss
extract_stabilised_loss_values <- function(file) {
  data <- read_csv(file, show_col_types = FALSE)
  loss1_values <- tail(data$mean_loss1, stabilised_period)
  return(loss1_values)
}
loss1_values <- sapply(all_files, extract_stabilised_loss_values)
loss1_values <- list(
  mean = mean(loss1_values),
  sd = sd(loss1_values), # Convergence
  min = min(loss1_values),
  max = max(loss1_values)
)

print(loss1_values)

#####
# Visualisation
#####
# Visualisation of rewards
# Read 'total rewards' column from each file and combine
total_reward_data <- lapply(all_files, function(file) {
  df <- read_csv(file,show_col_types = FALSE)
  return(df$total_rewards)
})
names(total_reward_data) <- sapply(all_files, function(f) paste0("repetition_", basename(f)))
combined_total_reward_data <- as.data.frame(total_reward_data)
episode_mean_total_rewards <- apply(combined_total_reward_data, 1, mean, na.rm = TRUE)
episode_sd_total_rewards <- apply(combined_total_reward_data, 1, sd, na.rm = TRUE)
episode_variance_total_rewards <- apply(combined_total_reward_data, 1, var, na.rm = TRUE)

summary_total_rewards_data <- data.frame(
  episode = 1:length(episode_mean_total_rewards),
  mean = episode_mean_total_rewards,
  sd = episode_sd_total_rewards,
  variance = episode_variance_total_rewards
)

# Read 'action rewards' column from each file and combine
action_reward_data <- lapply(all_files, function(file) {
  df <- read_csv(file,show_col_types = FALSE)
  return(df$action_rewards)
})
names(action_reward_data) <- sapply(all_files, function(f) paste0("repetition_", basename(f)))
combined_action_reward_data <- as.data.frame(action_reward_data)
episode_mean_action_rewards <- apply(combined_action_reward_data, 1, mean, na.rm = TRUE)
episode_sd_action_rewards <- apply(combined_action_reward_data, 1, sd, na.rm = TRUE)
episode_variance_action_rewards <- apply(combined_action_reward_data, 1, var, na.rm = TRUE)

summary_action_rewards_data <- data.frame(
  episode = 1:length(episode_mean_action_rewards),
  mean = episode_mean_action_rewards,
  sd = episode_sd_action_rewards,
  variance = episode_variance_action_rewards
)

# Read 'opinion rewards' column from each file and combine
opinion_reward_data <- lapply(all_files, function(file) {
  df <- read_csv(file,show_col_types = FALSE)
  return(df$opinion_rewards)
})
names(opinion_reward_data) <- sapply(all_files, function(f) paste0("repetition_", basename(f)))
combined_opinion_reward_data <- as.data.frame(opinion_reward_data)
episode_mean_opinion_rewards <- apply(combined_opinion_reward_data, 1, mean, na.rm = TRUE)
episode_sd_opinion_rewards <- apply(combined_opinion_reward_data, 1, sd, na.rm = TRUE)
episode_variance_opinion_rewards <- apply(combined_opinion_reward_data, 1, var, na.rm = TRUE)

summary_opinion_rewards_data <- data.frame(
  episode = 1:length(episode_mean_opinion_rewards),
  mean = episode_mean_opinion_rewards,
  sd = episode_sd_opinion_rewards,
  variance = episode_variance_opinion_rewards
)

# Visualise rewards
p1 <- ggplot() +
  geom_line(data = summary_total_rewards_data, aes(x = episode, y = mean, color = "Average cumulative reward")) +
  geom_ribbon(data = summary_total_rewards_data, aes(x = episode, ymin = mean - sd, ymax = mean + sd), 
              fill = "#FF99CC", alpha = 0.4) +
  
  geom_line(data = summary_action_rewards_data, aes(x = episode, y = mean, color = "Average cumulative reward for actions")) +
  geom_ribbon(data = summary_action_rewards_data, aes(x = episode, ymin = mean - sd, ymax = mean + sd),
              fill = "#FFCC66", alpha = 0.4) +
  
  geom_line(data = summary_opinion_rewards_data, aes(x = episode, y = mean, color = "Average cumulative reward for opinions")) +
  geom_ribbon(data = summary_opinion_rewards_data, aes(x = episode, ymin = mean - sd, ymax = mean + sd),
              fill = "#CCFFCC", alpha = 0.4) +
  
  scale_color_manual(values = c("Average cumulative reward" = "#CC0033", 
                                "Average cumulative reward for actions" = "#FF9900", 
                                "Average cumulative reward for opinions" = "#006633")) +
  labs(x = "Episode", y = "Average cumulative reward\n(across 100 simulations)", color = NULL) +
  scale_x_continuous(limits = c(0, 500)) +  # Set explicit limits for x-axis
  scale_y_continuous(limits = c(0, 150000),  # Set y-axis limits
                     breaks = c(0, 50000, 100000, 150000),  # Define breaks
                     labels = c("0", "50000", "100000", "150000")) +  # Define labels
  theme_bw() +
  theme(
    legend.position = c(0.70, 0.65),
    plot.title = element_text(size = rel(1)),
    axis.title = element_text(size = rel(1)),
    axis.text = element_text(size = rel(1)),
    legend.title = element_text(size = rel(1)),
    legend.text = element_text(size = rel(0.80)),
    plot.margin = margin(1, 1, 1, 1, "cm"),
    axis.title.x = element_text(vjust = -1),
    axis.title.y = element_text(vjust = 1),
    legend.background = element_blank(),
    legend.key = element_blank(),
    panel.grid.major = element_line(color = "grey", linewidth = 0.5), 
    panel.grid.minor = element_line(color = "lightgrey", linewidth = 0.5)
  )
p1

# Saved /ch4-marl/results/plots/ch4-exp1-rewards.pdf as 5 x 7.50 (landscape)

# Visualise social trust
# Social trust
social_trust_sp1 <- lapply(all_files, function(file) {
  df <- read_csv(file,show_col_types = FALSE)
  df$sp0_social_trust <- df$sp0_social_trust
  return(df$sp0_social_trust)
})
names(social_trust_sp1) <- sapply(all_files, function(f) paste0("repetition_", basename(f)))
combined_social_trust_sp1 <- as.data.frame(social_trust_sp1)
episode_mean_social_trust_sp1 <- apply(combined_social_trust_sp1, 1, mean, na.rm = TRUE)
episode_sd_social_trust_sp1 <- apply(combined_social_trust_sp1, 1, sd, na.rm = TRUE)
episode_variance_social_trust_sp1 <- apply(combined_social_trust_sp1, 1, var, na.rm = TRUE)

summary_social_trust_sp1 <- data.frame(
  episode = 1:length(episode_mean_social_trust_sp1),
  mean = episode_mean_social_trust_sp1,
  sd = episode_sd_social_trust_sp1,
  variance = episode_variance_social_trust_sp1
)

social_trust_sp2 <- lapply(all_files, function(file) {
  df <- read_csv(file,show_col_types = FALSE)
  df$sp1_social_trust <- df$sp1_social_trust
  return(df$sp1_social_trust)
})
names(social_trust_sp2) <- sapply(all_files, function(f) paste0("repetition_", basename(f)))
combined_social_trust_sp2 <- as.data.frame(social_trust_sp2)
episode_mean_social_trust_sp2 <- apply(combined_social_trust_sp2, 1, mean, na.rm = TRUE)
episode_sd_social_trust_sp2 <- apply(combined_social_trust_sp2, 1, sd, na.rm = TRUE)
episode_variance_social_trust_sp2 <- apply(combined_social_trust_sp2, 1, var, na.rm = TRUE)

summary_social_trust_sp2 <- data.frame(
  episode = 1:length(episode_mean_social_trust_sp2),
  mean = episode_mean_social_trust_sp2,
  sd = episode_sd_social_trust_sp2,
  variance = episode_variance_social_trust_sp2
)

social_trust_sp3 <- lapply(all_files, function(file) {
  df <- read_csv(file,show_col_types = FALSE)
  df$sp2_social_trust <- df$sp2_social_trust
  return(df$sp2_social_trust)
})
names(social_trust_sp3) <- sapply(all_files, function(f) paste0("repetition_", basename(f)))
combined_social_trust_sp3 <- as.data.frame(social_trust_sp3)
episode_mean_social_trust_sp3 <- apply(combined_social_trust_sp3, 1, mean, na.rm = TRUE)
episode_sd_social_trust_sp3 <- apply(combined_social_trust_sp3, 1, sd, na.rm = TRUE)
episode_variance_social_trust_sp3 <- apply(combined_social_trust_sp3, 1, var, na.rm = TRUE)
summary_social_trust_sp3 <- data.frame(
  episode = 1:length(episode_mean_social_trust_sp3),
  mean = episode_mean_social_trust_sp3,
  sd = episode_sd_social_trust_sp3,
  variance = episode_variance_social_trust_sp3
)

# Visualise regular agents' service provider trust rates
p2 <- ggplot() +
  geom_line(data = summary_social_trust_sp1, aes(x = episode, y = mean, color = "Service provider 1")) +
  geom_ribbon(data = summary_social_trust_sp1, aes(x = episode, ymin = mean - sd, ymax = mean + sd), 
              fill = "#3366FF", alpha = 0.3) +
  geom_line(data = summary_social_trust_sp2, aes(x = episode, y = mean, color = "Service provider 2")) +
  geom_ribbon(data = summary_social_trust_sp2, aes(x = episode, ymin = mean - sd, ymax = mean + sd), 
              fill = "#FFCC66", alpha = 0.3) +
  geom_line(data = summary_social_trust_sp3, aes(x = episode, y = mean, color = "Service provider 3")) +
  geom_ribbon(data = summary_social_trust_sp3, aes(x = episode, ymin = mean - sd, ymax = mean + sd), 
              fill = "#339900", alpha = 0.3) +
  scale_color_manual(values = c("Service provider 1" = "#0033CC", 
                                "Service provider 2" = "#FF9900",
                                "Service provider 3" = "#006633")) +
  labs(x = "Episode", y = "Average service provider social trust rate\n(across 100 simulations)", color = NULL) +
  scale_x_continuous(limits = c(0, 500)) +  # Set explicit limits for x-axis
  scale_y_continuous(limits = c(0, 1)) +  # Set explicit limits for y-axis
  theme_bw()+
  theme(
    legend.position = c(0.80, 0.40),
    plot.title = element_text(size = rel(1)),
    axis.title = element_text(size = rel(1)),
    axis.text = element_text(size = rel(1)),
    legend.title = element_text(size = rel(1)),
    legend.text = element_text(size = rel(0.80)),
    plot.margin = margin(1, 1, 1, 1, "cm"),
    axis.title.x = element_text(vjust = -1),
    axis.title.y = element_text(vjust = 1),
    legend.background = element_blank(),
    legend.key = element_blank(),
    panel.grid.major = element_line(color = "grey", linewidth = 0.5), 
    panel.grid.minor = element_line(color = "lightgrey", linewidth = 0.5)
  )

p2

# Visualise service provider selection rate
selection_sp1 <- lapply(all_files, function(file) {
  df <- read_csv(file,show_col_types = FALSE)
  return(df$sp0_request_rate)
})
names(selection_sp1) <- sapply(all_files, function(f) paste0("repetition_", basename(f)))
combined_selection_sp1 <- as.data.frame(selection_sp1)
episode_mean_selection_sp1 <- apply(combined_selection_sp1, 1, mean, na.rm = TRUE)
episode_sd_selection_sp1 <- apply(combined_selection_sp1, 1, sd, na.rm = TRUE)
episode_variance_selection_sp1 <- apply(combined_selection_sp1, 1, var, na.rm = TRUE)
summary_selection_sp1 <- data.frame(
  episode = 1:length(episode_mean_selection_sp1),
  mean = episode_mean_selection_sp1,
  sd = episode_sd_selection_sp1,
  variance = episode_variance_selection_sp1
)

selection_sp2 <- lapply(all_files, function(file) {
  df <- read_csv(file,show_col_types = FALSE)
  return(df$sp1_request_rate)
})
names(selection_sp2) <- sapply(all_files, function(f) paste0("repetition_", basename(f)))
combined_selection_sp2 <- as.data.frame(selection_sp2)
episode_mean_selection_sp2 <- apply(combined_selection_sp2, 1, mean, na.rm = TRUE)
episode_sd_selection_sp2 <- apply(combined_selection_sp2, 1, sd, na.rm = TRUE)
episode_variance_selection_sp2 <- apply(combined_selection_sp2, 1, var, na.rm = TRUE)
summary_selection_sp2 <- data.frame(
  episode = 1:length(episode_mean_selection_sp2),
  mean = episode_mean_selection_sp2,
  sd = episode_sd_selection_sp2,
  variance = episode_variance_selection_sp2
)

selection_sp3 <- lapply(all_files, function(file) {
  df <- read_csv(file,show_col_types = FALSE)
  return(df$sp2_request_rate)
})
names(selection_sp3) <- sapply(all_files, function(f) paste0("repetition_", basename(f)))
combined_selection_sp3 <- as.data.frame(selection_sp3)
episode_mean_selection_sp3 <- apply(combined_selection_sp3, 1, mean, na.rm = TRUE)
episode_sd_selection_sp3 <- apply(combined_selection_sp3, 1, sd, na.rm = TRUE)
episode_variance_selection_sp3 <- apply(combined_selection_sp3, 1, var, na.rm = TRUE)

summary_selection_sp3 <- data.frame(
  episode = 1:length(episode_mean_selection_sp3),
  mean = episode_mean_selection_sp3,
  sd = episode_sd_selection_sp3,
  variance = episode_variance_selection_sp3
)

# Visualise regular agents' service provider selection rates
p3 <- ggplot() +
  geom_line(data = summary_selection_sp1, aes(x = episode, y = mean, color = "Service provider 1")) +
  geom_ribbon(data = summary_selection_sp1, aes(x = episode, ymin = mean - sd, ymax = mean + sd), 
              fill = "#3366FF", alpha = 0.3) +
  geom_line(data = summary_selection_sp2, aes(x = episode, y = mean, color = "Service provider 2")) +
  geom_ribbon(data = summary_selection_sp2, aes(x = episode, ymin = mean - sd, ymax = mean + sd), 
              fill = "#FFCC66", alpha = 0.3) +
  geom_line(data = summary_selection_sp3, aes(x = episode, y = mean, color = "Service provider 3")) +
  geom_ribbon(data = summary_selection_sp3, aes(x = episode, ymin = mean - sd, ymax = mean + sd), 
              fill = "#339900", alpha = 0.3) +
  scale_color_manual(values = c("Service provider 1" = "#0033CC", 
                                "Service provider 2" = "#FF9900",
                                "Service provider 3" = "#006633")) +
  labs(x = "Episode", y = "Average service request rate\n(across 100 simulations)", color = NULL) +
  scale_x_continuous(limits = c(0, 500)) +  # Set explicit limits for x-axis
  scale_y_continuous(limits = c(0, 100)) +  # Set explicit limits for y-axis
  theme_bw()+
  theme(
    legend.position = c(0.80, 0.65),
    plot.title = element_text(size = rel(1)),
    axis.title = element_text(size = rel(1)),
    axis.text = element_text(size = rel(1)),
    legend.title = element_text(size = rel(1)),
    legend.text = element_text(size = rel(0.80)),
    plot.margin = margin(1, 1, 1, 1, "cm"),
    axis.title.x = element_text(vjust = -1),
    axis.title.y = element_text(vjust = 1),
    legend.background = element_blank(),
    legend.key = element_blank(),
    panel.grid.major = element_line(color = "grey", linewidth = 0.5), 
    panel.grid.minor = element_line(color = "lightgrey", linewidth = 0.5)
  )

p3

# Visualise opinion expression rate
# Read 'expression op0' column from each file
expression_op0 <- lapply(all_files, function(file) {
  df <- read_csv(file,show_col_types = FALSE)
  return(df$op0_expression_rate)
})
names(expression_op0) <- sapply(all_files, function(f) paste0("repetition_", basename(f)))
combined_expression_op0 <- as.data.frame(expression_op0)
episode_mean_expression_op0 <- apply(combined_expression_op0, 1, mean, na.rm = TRUE)
episode_sd_expression_op0 <- apply(combined_expression_op0, 1, sd, na.rm = TRUE)
episode_variance_expression_op0 <- apply(combined_expression_op0, 1, var, na.rm = TRUE)
summary_expression_op0 <- data.frame(
  episode = 1:length(episode_mean_expression_op0),
  mean = episode_mean_expression_op0,
  sd = episode_sd_expression_op0,
  variance = episode_variance_expression_op0
)

expression_op1 <- lapply(all_files, function(file) {
  df <- read_csv(file,show_col_types = FALSE)
  return(df$op1_expression_rate)
})
names(expression_op1) <- sapply(all_files, function(f) paste0("repetition_", basename(f)))
combined_expression_op1 <- as.data.frame(expression_op1)
episode_mean_expression_op1 <- apply(combined_expression_op1, 1, mean, na.rm = TRUE)
episode_sd_expression_op1 <- apply(combined_expression_op1, 1, sd, na.rm = TRUE)
episode_variance_expression_op1 <- apply(combined_expression_op1, 1, var, na.rm = TRUE)
summary_expression_op1 <- data.frame(
  episode = 1:length(episode_mean_expression_op1),
  mean = episode_mean_expression_op1,
  sd = episode_sd_expression_op1,
  variance = episode_variance_expression_op1
)

expression_op2 <- lapply(all_files, function(file) {
  df <- read_csv(file,show_col_types = FALSE)
  return(df$op2_expression_rate)
})
names(expression_op2) <- sapply(all_files, function(f) paste0("repetition_", basename(f)))
combined_expression_op2 <- as.data.frame(expression_op2)
episode_mean_expression_op2 <- apply(combined_expression_op2, 1, mean, na.rm = TRUE)
episode_sd_expression_op2 <- apply(combined_expression_op2, 1, sd, na.rm = TRUE)
episode_variance_expression_op2 <- apply(combined_expression_op2, 1, var, na.rm = TRUE)
summary_expression_op2 <- data.frame(
  episode = 1:length(episode_mean_expression_op2),
  mean = episode_mean_expression_op2,
  sd = episode_sd_expression_op2,
  variance = episode_variance_expression_op2
)

expression_op3 <- lapply(all_files, function(file) {
  df <- read_csv(file,show_col_types = FALSE)
  return(df$op3_expression_rate)
})
names(expression_op3) <- sapply(all_files, function(f) paste0("repetition_", basename(f)))
combined_expression_op3 <- as.data.frame(expression_op3)
episode_mean_expression_op3 <- apply(combined_expression_op3, 1, mean, na.rm = TRUE)
episode_sd_expression_op3 <- apply(combined_expression_op3, 1, sd, na.rm = TRUE)
episode_variance_expression_op3 <- apply(combined_expression_op3, 1, var, na.rm = TRUE)
summary_expression_op3 <- data.frame(
  episode = 1:length(episode_mean_expression_op3),
  mean = episode_mean_expression_op3,
  sd = episode_sd_expression_op3,
  variance = episode_variance_expression_op3
) 

expression_op4 <- lapply(all_files, function(file) {
  df <- read_csv(file,show_col_types = FALSE)
  return(df$op4_expression_rate)
})
names(expression_op4) <- sapply(all_files, function(f) paste0("repetition_", basename(f)))
combined_expression_op4 <- as.data.frame(expression_op4)
episode_mean_expression_op4 <- apply(combined_expression_op4, 1, mean, na.rm = TRUE)
episode_sd_expression_op4 <- apply(combined_expression_op4, 1, sd, na.rm = TRUE)
episode_variance_expression_op4 <- apply(combined_expression_op4, 1, var, na.rm = TRUE)
summary_expression_op4 <- data.frame(
  episode = 1:length(episode_mean_expression_op4),
  mean = episode_mean_expression_op4,
  sd = episode_sd_expression_op4,
  variance = episode_variance_expression_op4
) 

expression_op5 <- lapply(all_files, function(file) {
  df <- read_csv(file,show_col_types = FALSE)
  return(df$op5_expression_rate)
})
names(expression_op5) <- sapply(all_files, function(f) paste0("repetition_", basename(f)))
combined_expression_op5 <- as.data.frame(expression_op5)
episode_mean_expression_op5 <- apply(combined_expression_op5, 1, mean, na.rm = TRUE)
episode_sd_expression_op5 <- apply(combined_expression_op5, 1, sd, na.rm = TRUE)
episode_variance_expression_op5 <- apply(combined_expression_op5, 1, var, na.rm = TRUE)
summary_expression_op5 <- data.frame(
  episode = 1:length(episode_mean_expression_op5),
  mean = episode_mean_expression_op5,
  sd = episode_sd_expression_op5,
  variance = episode_variance_expression_op5
)

# Add new variable to the data frames
summary_expression_op0$Group <- "Negative opinion on service provider 1"
summary_expression_op1$Group <- "Positive opinion on service provider 1"
summary_expression_op2$Group <- "Negative opinion on service provider 2"
summary_expression_op3$Group <- "Positive opinion on service provider 2"
summary_expression_op4$Group <- "Negative opinion on service provider 3"
summary_expression_op5$Group <- "Positive opinion on service provider 3"

# Plot mean usage with error bars for variance
p4 <- ggplot() +
  geom_line(data = summary_expression_op0, aes(x = episode, y = mean, color = Group, linetype = Group)) +
  geom_ribbon(data = summary_expression_op0, aes(x = episode, ymin = mean - sd, ymax = mean + sd, fill = Group), alpha = 0.15) +
  
  geom_line(data = summary_expression_op1, aes(x = episode, y = mean, color = Group, linetype = Group)) +
  geom_ribbon(data = summary_expression_op1, aes(x = episode, ymin = mean - sd, ymax = mean + sd, fill = Group), alpha = 0.15) +
  
  geom_line(data = summary_expression_op2, aes(x = episode, y = mean, color = Group, linetype = Group)) +
  geom_ribbon(data = summary_expression_op2, aes(x = episode, ymin = mean - sd, ymax = mean + sd, fill = Group), alpha = 0.15) +
  
  geom_line(data = summary_expression_op3, aes(x = episode, y = mean, color = Group, linetype = Group)) +
  geom_ribbon(data = summary_expression_op3, aes(x = episode, ymin = mean - sd, ymax = mean + sd, fill = Group), alpha = 0.15) +
  
  geom_line(data = summary_expression_op4, aes(x = episode, y = mean, color = Group, linetype = Group)) +
  geom_ribbon(data = summary_expression_op4, aes(x = episode, ymin = mean - sd, ymax = mean + sd, fill = Group), alpha = 0.15) +
  
  geom_line(data = summary_expression_op5, aes(x = episode, y = mean, color = Group, linetype = Group)) +
  geom_ribbon(data = summary_expression_op5, aes(x = episode, ymin = mean - sd, ymax = mean + sd, fill = Group), alpha = 0.15) +
  
  
  scale_color_manual(values = c("Negative opinion on service provider 1" = "#0033CC", "Positive opinion on service provider 1" = "#0033CC",
                                "Negative opinion on service provider 2" = "#FF9900", "Positive opinion on service provider 2" = "#FF9900",
                                "Negative opinion on service provider 3" = "#006633", "Positive opinion on service provider 3" = "#006633")) +
  scale_fill_manual(values = c("Negative opinion on service provider 1" = "#3366FF", "Positive opinion on service provider 1" = "#3366FF",
                               "Negative opinion on service provider 2" = "#FFCC66", "Positive opinion on service provider 2" = "#FFCC66",
                               "Negative opinion on service provider 3" = "#339900", "Positive opinion on service provider 3" = "#339900")) +
  scale_linetype_manual(values = c("Negative opinion on service provider 1" = "dashed", "Positive opinion on service provider 1" = "solid",
                                   "Negative opinion on service provider 2" = "dashed", "Positive opinion on service provider 2" = "solid",
                                   "Negative opinion on service provider 3" = "dashed", "Positive opinion on service provider 3" = "solid")) +
  labs(x = "Episode", y = "Average opinion expression rate\n(across 100 simulations)", color = NULL, linetype = NULL, fill = NULL) +
  scale_x_continuous(limits = c(0, 500)) +  # Set explicit limits for x-axis
  scale_y_continuous(limits = c(0, 100)) +  # Set explicit limits for y-axis
  theme_bw()+
  theme(
    legend.position = c(0.70, 0.75),
    plot.title = element_text(size = rel(1)),
    axis.title = element_text(size = rel(1)),
    axis.text = element_text(size = rel(1)),
    legend.title = element_text(size = rel(1)),
    legend.text = element_text(size = rel(0.80)),
    plot.margin = margin(1, 1, 1, 1, "cm"),
    axis.title.x = element_text(vjust = -1),
    axis.title.y = element_text(vjust = 1),
    legend.background = element_blank(),
    legend.key = element_blank(),
    panel.grid.major = element_line(color = "grey", linewidth = 0.5), 
    panel.grid.minor = element_line(color = "lightgrey", linewidth = 0.5)
  )

p4

# Visualise average service availability
availability_sp1 <- lapply(all_files, function(file) {
  df <- read_csv(file,show_col_types = FALSE)
  df$sp0_availability <- df$sp0_availability * 100
  return(df$sp0_availability)
})
names(availability_sp1) <- sapply(all_files, function(f) paste0("repetition_", basename(f)))
combined_availability_sp1 <- as.data.frame(availability_sp1)
episode_mean_availability_sp1 <- apply(combined_availability_sp1, 1, mean, na.rm = TRUE)
episode_sd_availability_sp1 <- apply(combined_availability_sp1, 1, sd, na.rm = TRUE)
episode_variance_availability_sp1 <- apply(combined_availability_sp1, 1, var, na.rm = TRUE)
summary_availability_sp1 <- data.frame(
  episode = 1:length(episode_mean_availability_sp1),
  mean = episode_mean_availability_sp1,
  sd = episode_sd_availability_sp1,
  variance = episode_variance_availability_sp1
)

availability_sp2 <- lapply(all_files, function(file) {
  df <- read_csv(file,show_col_types = FALSE)
  df$sp1_availability <- df$sp1_availability * 100
  return(df$sp1_availability)
})
names(availability_sp2) <- sapply(all_files, function(f) paste0("repetition_", basename(f)))
combined_availability_sp2 <- as.data.frame(availability_sp2)
episode_mean_availability_sp2 <- apply(combined_availability_sp2, 1, mean, na.rm = TRUE)
episode_sd_availability_sp2 <- apply(combined_availability_sp2, 1, sd, na.rm = TRUE)
episode_variance_availability_sp2 <- apply(combined_availability_sp2, 1, var, na.rm = TRUE)
summary_availability_sp2 <- data.frame(
  episode = 1:length(episode_mean_availability_sp2),
  mean = episode_mean_availability_sp2,
  sd = episode_sd_availability_sp2,
  variance = episode_variance_availability_sp2
)

availability_sp3 <- lapply(all_files, function(file) {
  df <- read_csv(file,show_col_types = FALSE)
  df$sp2_availability <- df$sp2_availability * 100
  return(df$sp2_availability)
})
names(availability_sp3) <- sapply(all_files, function(f) paste0("repetition_", basename(f)))
combined_availability_sp3 <- as.data.frame(availability_sp3)
episode_mean_availability_sp3 <- apply(combined_availability_sp3, 1, mean, na.rm = TRUE)
episode_sd_availability_sp3 <- apply(combined_availability_sp3, 1, sd, na.rm = TRUE)
episode_variance_availability_sp3 <- apply(combined_availability_sp3, 1, var, na.rm = TRUE)
summary_availability_sp3 <- data.frame(
  episode = 1:length(episode_mean_availability_sp3),
  mean = episode_mean_availability_sp3,
  sd = episode_sd_availability_sp3,
  variance = episode_variance_availability_sp3
)

# Visualise regular agents' service provider selection rates
p5 <- ggplot() +
  geom_line(data = summary_availability_sp1, aes(x = episode, y = mean, color = "Provider 1")) +
  geom_ribbon(data = summary_availability_sp1, aes(x = episode, ymin = mean - sd, ymax = mean + sd), 
              fill = "#3366FF", alpha = 0.2) +
  geom_line(data = summary_availability_sp2, aes(x = episode, y = mean, color = "Provider 2")) +
  geom_ribbon(data = summary_availability_sp2, aes(x = episode, ymin = mean - sd, ymax = mean + sd), 
              fill = "#FFCC66", alpha = 0.2) +
  geom_line(data = summary_availability_sp3, aes(x = episode, y = mean, color = "Provider 3")) +
  geom_ribbon(data = summary_availability_sp3, aes(x = episode, ymin = mean - sd, ymax = mean + sd), 
              fill = "#339900", alpha = 0.2) +
  scale_color_manual(values = c("Provider 1" = "#0033CC", 
                                "Provider 2" = "#FF9900",
                                "Provider 3" = "#006633")) +
  labs(x = "Episode", y = "Average service\navailability rate", color = NULL) +
  scale_x_continuous(limits = c(0, 500)) + 
  scale_y_continuous(limits = c(0, 100)) +
  theme_bw()+
  theme(
    legend.position = c(0.80, 0.65),
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
p5

combined_plot <- cowplot::plot_grid(p2, p3, p4, p1, labels = c("A", "B", "C", "D"))
combined_plot <- cowplot::plot_grid(p1, p3, p4, labels = c("A", "B", "C"), nrow = 1)
combined_plot
# Save the combined plot 2 as 8 x 12.75 (landscape)

# Visualise loss
loss_1 <- lapply(all_files, function(file) {
  df <- read_csv(file,show_col_types = FALSE)
  return(df$mean_loss1)
})
names(loss_1) <- sapply(all_files, function(f) paste0("repetition_", basename(f)))
combined_loss_1 <- as.data.frame(loss_1)
episode_mean_loss_1 <- apply(combined_loss_1, 1, mean, na.rm = TRUE)
episode_sd_loss_1 <- apply(combined_loss_1, 1, sd, na.rm = TRUE)
episode_variance_loss_1 <- apply(combined_loss_1, 1, var, na.rm = TRUE)
summary_loss_1 <- data.frame(
  episode = 1:length(episode_mean_loss_1),
  mean = episode_mean_loss_1,
  sd = episode_sd_loss_1,
  variance = episode_variance_loss_1
)

loss_2 <- lapply(all_files, function(file) {
  df <- read_csv(file,show_col_types = FALSE)
  return(df$mean_loss2)
})
names(loss_2) <- sapply(all_files, function(f) paste0("repetition_", basename(f)))
combined_loss_2 <- as.data.frame(loss_2)
episode_mean_loss_2 <- apply(combined_loss_2, 1, mean, na.rm = TRUE)
episode_sd_loss_2 <- apply(combined_loss_2, 1, sd, na.rm = TRUE)
episode_variance_loss_2 <- apply(combined_loss_2, 1, var, na.rm = TRUE)
summary_loss_2 <- data.frame(
  episode = 1:length(episode_mean_loss_2),
  mean = episode_mean_loss_2,
  sd = episode_sd_loss_2,
  variance = episode_variance_loss_2
)

# Visualise regular agents' service provider selection rates
p6 <- ggplot() +
  geom_line(data = summary_loss_1, aes(x = episode, y = mean, color = "Loss 1")) +
  geom_ribbon(data = summary_loss_1, aes(x = episode, ymin = mean - sd, ymax = mean + sd), 
              fill = "#FFCC66", alpha = 0.2) +
  geom_line(data = summary_loss_2, aes(x = episode, y = mean, color = "Loss 2")) +
  geom_ribbon(data = summary_loss_2, aes(x = episode, ymin = mean - sd, ymax = mean + sd), 
              fill = "#CCFFCC", alpha = 0.2) +
  scale_color_manual(values = c("Loss 1" = "#FF9900", 
                                "Loss 2" = "#006633"
                                )) +
  labs(x = "Episode", y = "Loss", color = NULL) +
  theme_bw()+
  theme(
    legend.position.inside = c(0.80, 0.65),
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
p6
