# Applying multi-agent reinforcement learning to study hybrid threats and defensive countermeasures
# Exp 2 data analysis and visualisation

# Load libraries
library(readr)
library(ggplot2)
library(cowplot)

# Import data
path <- "exp2-results/"
all_files <- list.files(path, pattern = "*defender-data.csv", full.names = TRUE)

# Calculate metrics
stabilised_episode = 11148
number_of_episodes = 20000
stabilised_period = number_of_episodes-stabilised_episode

# Rewards: Cumulative reward over stabilised episodes
# Service provider 1
total_reward_data_sp1 <- function(file) {
  data <- read_csv(file, show_col_types = FALSE)
  sp1_rw_values <- tail(data$total_rewards_sp0, stabilised_period)
  return(sp1_rw_values)
}
sp1_rw_values <- sapply(all_files, total_reward_data_sp1)
sp1_rw_result <- list(
  mean = mean(sp1_rw_values),
  sd = sd(sp1_rw_values),
  min = min(sp1_rw_values),
  max = max(sp1_rw_values)
)

print(sp1_rw_result)

# Service provider 2
total_reward_data_sp2 <- function(file) {
  data <- read_csv(file, show_col_types = FALSE)
  sp2_rw_values <- tail(data$total_rewards_sp1, stabilised_period)
  return(sp2_rw_values)
}
sp2_rw_values <- sapply(all_files, total_reward_data_sp2)
sp2_rw_result <- list(
  mean = mean(sp2_rw_values),
  sd = sd(sp2_rw_values),
  min = min(sp2_rw_values),
  max = max(sp2_rw_values)
)

print(sp2_rw_result)

# Service provider 3
total_reward_data_sp3 <- function(file) {
  data <- read_csv(file, show_col_types = FALSE)
  sp3_rw_values <- tail(data$total_rewards_sp2, stabilised_period)
  return(sp3_rw_values)
}
sp3_rw_values <- sapply(all_files, total_reward_data_sp3)
sp3_rw_result <- list(
  mean = mean(sp3_rw_values),
  sd = sd(sp3_rw_values),
  min = min(sp3_rw_values),
  max = max(sp3_rw_values)
)

print(sp3_rw_result)

# Rewards: Rewards for filtering service over stabilised episodes
# Service provider 1
filter_reward_data_sp1 <- function(file) {
  data <- read_csv(file, show_col_types = FALSE)
  sp1_filter_rw_values <- tail(data$filter_rewards_sp0, stabilised_period)
  return(sp1_filter_rw_values)
}
sp1_filter_rw_values <- sapply(all_files, filter_reward_data_sp1)
sp1_filter_rw_result <- list(
  mean = mean(sp1_filter_rw_values),
  sd = sd(sp1_filter_rw_values),
  min = min(sp1_filter_rw_values),
  max = max(sp1_filter_rw_values)
)

print(sp1_filter_rw_result)

# Service provider 2
filter_reward_data_sp2 <- function(file) {
  data <- read_csv(file, show_col_types = FALSE)
  sp2_filter_rw_values <- tail(data$filter_rewards_sp1, stabilised_period)
  return(sp2_filter_rw_values)
}
sp2_filter_rw_values <- sapply(all_files, filter_reward_data_sp2)
sp2_filter_rw_result <- list(
  mean = mean(sp2_filter_rw_values),
  sd = sd(sp2_filter_rw_values),
  min = min(sp2_filter_rw_values),
  max = max(sp2_filter_rw_values)
)

print(sp2_filter_rw_result)

# Service provider 3
filter_reward_data_sp3 <- function(file) {
  data <- read_csv(file, show_col_types = FALSE)
  sp3_filter_rw_values <- tail(data$filter_rewards_sp2, stabilised_period)
  return(sp3_filter_rw_values)
}
sp3_filter_rw_values <- sapply(all_files, filter_reward_data_sp3)
sp3_filter_rw_result <- list(
  mean = mean(sp3_filter_rw_values),
  sd = sd(sp3_filter_rw_values),
  min = min(sp3_filter_rw_values),
  max = max(sp3_filter_rw_values)
)

print(sp3_filter_rw_result)

# Rewards: Rewards for replying service over stabilised episodes
# Service provider 1
answer_reward_data_sp1 <- function(file) {
  data <- read_csv(file, show_col_types = FALSE)
  sp1_answer_rw_values <- tail(data$answer_rewards_sp0, stabilised_period)
  return(sp1_answer_rw_values)
}
sp1_answer_rw_values <- sapply(all_files, answer_reward_data_sp1)
sp1_answer_rw_result <- list(
  mean = mean(sp1_answer_rw_values),
  sd = sd(sp1_answer_rw_values), 
  min = min(sp1_answer_rw_values),
  max = max(sp1_answer_rw_values)
)

print(sp1_answer_rw_result)

# Service provider 2
answer_reward_data_sp2 <- function(file) {
  data <- read_csv(file, show_col_types = FALSE)
  sp2_answer_rw_values <- tail(data$answer_rewards_sp1, stabilised_period)
  return(sp2_answer_rw_values)
}
sp2_answer_rw_values <- sapply(all_files, answer_reward_data_sp2)
sp2_answer_rw_result <- list(
  mean = mean(sp2_answer_rw_values),
  sd = sd(sp2_answer_rw_values), 
  min = min(sp2_answer_rw_values),
  max = max(sp2_answer_rw_values)
)

print(sp2_answer_rw_result)

# Service provider 3
answer_reward_data_sp3 <- function(file) {
  data <- read_csv(file, show_col_types = FALSE)
  sp3_answer_rw_values <- tail(data$answer_rewards_sp2, stabilised_period)
  return(sp3_answer_rw_values)
}
sp3_answer_rw_values <- sapply(all_files, answer_reward_data_sp3)
sp3_answer_rw_result <- list(
  mean = mean(sp3_answer_rw_values),
  sd = sd(sp3_answer_rw_values), 
  min = min(sp3_answer_rw_values),
  max = max(sp3_answer_rw_values)
)

print(sp3_answer_rw_result)

# Visualise rewards
# Read 'total_rewards_sp0' column from each file
total_reward_data_sp1 <- lapply(all_files, function(file) {
  df <- read_csv(file,show_col_types = FALSE)
  return(df$total_rewards_sp0)
})
names(total_reward_data_sp1) <- sapply(all_files, function(f) paste0("repetition_", basename(f)))

# Combine them into one large data frame
combined_total_reward_data_sp1 <- as.data.frame(total_reward_data_sp1)
combined_total_reward_data_sp1[is.na(combined_total_reward_data_sp1)] <- 0

# Calculate sd and variance
episode_mean_total_rewards_sp1 <- apply(combined_total_reward_data_sp1, 1, mean, na.rm = TRUE)
episode_sd_total_rewards_sp1 <- apply(combined_total_reward_data_sp1, 1, sd, na.rm = TRUE)
episode_variance_total_rewards_sp1 <- apply(combined_total_reward_data_sp1, 1, var, na.rm = TRUE)

summary_total_rewards_data_sp1 <- data.frame(
  episode = 1:length(episode_mean_total_rewards_sp1),
  mean = episode_mean_total_rewards_sp1,
  sd = episode_sd_total_rewards_sp1,
  variance = episode_variance_total_rewards_sp1
)

# Read 'total_rewards_sp1' column from each file
total_reward_data_sp2 <- lapply(all_files, function(file) {
  df <- read_csv(file,show_col_types = FALSE)
  return(df$total_rewards_sp1)
})
names(total_reward_data_sp2) <- sapply(all_files, function(f) paste0("repetition_", basename(f)))

# Combine them into one large data frame
combined_total_reward_data_sp2 <- as.data.frame(total_reward_data_sp2)
combined_total_reward_data_sp2[is.na(combined_total_reward_data_sp2)] <- 0

# Calculate sd and variance
episode_mean_total_rewards_sp2 <- apply(combined_total_reward_data_sp2, 1, mean, na.rm = TRUE)
episode_sd_total_rewards_sp2 <- apply(combined_total_reward_data_sp2, 1, sd, na.rm = TRUE)
episode_variance_total_rewards_sp2 <- apply(combined_total_reward_data_sp2, 1, var, na.rm = TRUE)

summary_total_rewards_data_sp2 <- data.frame(
  episode = 1:length(episode_mean_total_rewards_sp2),
  mean = episode_mean_total_rewards_sp2,
  sd = episode_sd_total_rewards_sp2,
  variance = episode_variance_total_rewards_sp2
)

# Read 'total_rewards_sp2' column from each file
total_reward_data_sp3 <- lapply(all_files, function(file) {
  df <- read_csv(file,show_col_types = FALSE)
  return(df$total_rewards_sp2)
})
names(total_reward_data_sp3) <- sapply(all_files, function(f) paste0("repetition_", basename(f)))

# Combine them into one large data frame
combined_total_reward_data_sp3 <- as.data.frame(total_reward_data_sp3)
combined_total_reward_data_sp3[is.na(combined_total_reward_data_sp3)] <- 0

# Calculate sd and variance
episode_mean_total_rewards_sp3 <- apply(combined_total_reward_data_sp3, 1, mean, na.rm = TRUE)
episode_sd_total_rewards_sp3 <- apply(combined_total_reward_data_sp3, 1, sd, na.rm = TRUE)
episode_variance_total_rewards_sp3 <- apply(combined_total_reward_data_sp3, 1, var, na.rm = TRUE)

summary_total_rewards_data_sp3 <- data.frame(
  episode = 1:length(episode_mean_total_rewards_sp3),
  mean = episode_mean_total_rewards_sp3,
  sd = episode_sd_total_rewards_sp3,
  variance = episode_variance_total_rewards_sp3
)

# Visualise defender rewards
d1 <- ggplot() +
  geom_line(data = summary_total_rewards_data_sp1, aes(x = episode, y = mean, color = "Mean score for service provider 1")) +
  geom_ribbon(data = summary_total_rewards_data_sp1, aes(x = episode, ymin = mean - sd, ymax = mean + sd), 
              fill = "#3366FF", alpha = 0.2) +
  geom_line(data = summary_total_rewards_data_sp2, aes(x = episode, y = mean, color = "Mean score for service provider 2")) +
  geom_ribbon(data = summary_total_rewards_data_sp2, aes(x = episode, ymin = mean - sd, ymax = mean + sd), 
              fill = "#FFCC66", alpha = 0.2) +
  geom_line(data = summary_total_rewards_data_sp3, aes(x = episode, y = mean, color = "Mean score for service provider 3")) +
  geom_ribbon(data = summary_total_rewards_data_sp3, aes(x = episode, ymin = mean - sd, ymax = mean + sd), 
              fill = "#339900", alpha = 0.2) +
  scale_color_manual(values = c("Mean score for service provider 1" = "#0033CC", 
                                "Mean score for service provider 2" = "#FF9900",
                                "Mean score for service provider 3" = "#006633")) +
  labs(x = "Episode", y = "Cumulative reward\nfor defenders", color = NULL) +
  theme_bw() +
  theme(
    legend.position = c(0.80, 0.30),
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
d1

# Saved ch4-exp2-defender-rewards.pdf as 5 x 7.50 (landscape)
combined_plot_1 <- cowplot::plot_grid(a1, d1, labels = c("A", "B"), ncol = 1) 
combined_plot_1
# combined_plot_1 saved ch4-exp2-rewards as 10 x 7.50 (portrait)

#####
# Filtering rewards
#####
# Read 'rewards sp1' column from each file
filter_reward_data_sp1 <- lapply(all_files, function(file) {
  df <- read_csv(file,show_col_types = FALSE)
  return(df$filter_rewards_sp0) 
})
names(filter_reward_data_sp1) <- sapply(all_files, function(f) paste0("repetition_", basename(f)))

# Combine them into one large data frame
combined_filter_reward_data_sp1 <- as.data.frame(filter_reward_data_sp1)
combined_filter_reward_data_sp1[is.na(combined_filter_reward_data_sp1)] <- 0

# Calculate sd and variance
episode_mean_filter_rewards_sp1 <- apply(combined_filter_reward_data_sp1, 1, mean, na.rm = TRUE)
episode_sd_filter_rewards_sp1 <- apply(combined_filter_reward_data_sp1, 1, sd, na.rm = TRUE)
episode_variance_filter_rewards_sp1 <- apply(combined_filter_reward_data_sp1, 1, var, na.rm = TRUE)

summary_filter_rewards_data_sp1 <- data.frame(
  episode = 1:length(episode_mean_filter_rewards_sp1),
  mean = episode_mean_filter_rewards_sp1,
  sd = episode_sd_filter_rewards_sp1,
  variance = episode_variance_filter_rewards_sp1
)

# Read 'rewards sp2' column from each file
filter_reward_data_sp2 <- lapply(all_files, function(file) {
  df <- read_csv(file,show_col_types = FALSE)
  return(df$filter_rewards_sp1)
})
names(filter_reward_data_sp2) <- sapply(all_files, function(f) paste0("repetition_", basename(f)))

# Combine them into one large data frame
combined_filter_reward_data_sp2 <- as.data.frame(filter_reward_data_sp2)
combined_filter_reward_data_sp2[is.na(combined_filter_reward_data_sp2)] <- 0

# Calculate sd and variance
episode_mean_filter_rewards_sp2 <- apply(combined_filter_reward_data_sp2, 1, mean, na.rm = TRUE)
episode_sd_filter_rewards_sp2 <- apply(combined_filter_reward_data_sp2, 1, sd, na.rm = TRUE)
episode_variance_filter_rewards_sp2 <- apply(combined_filter_reward_data_sp2, 1, var, na.rm = TRUE)

summary_filter_rewards_data_sp2 <- data.frame(
  episode = 1:length(episode_mean_filter_rewards_sp2),
  mean = episode_mean_filter_rewards_sp2,
  sd = episode_sd_filter_rewards_sp2,
  variance = episode_variance_filter_rewards_sp2
)

# Read 'rewards sp3' column from each file
filter_reward_data_sp3 <- lapply(all_files, function(file) {
  df <- read_csv(file,show_col_types = FALSE)
  return(df$filter_rewards_sp2)
})
names(filter_reward_data_sp3) <- sapply(all_files, function(f) paste0("repetition_", basename(f)))

# Combine them into one large data frame
combined_filter_reward_data_sp3 <- as.data.frame(filter_reward_data_sp3)
combined_filter_reward_data_sp3[is.na(combined_filter_reward_data_sp3)] <- 0

# Calculate sd and variance
episode_mean_filter_rewards_sp3 <- apply(combined_filter_reward_data_sp3, 1, mean, na.rm = TRUE)
episode_sd_filter_rewards_sp3 <- apply(combined_filter_reward_data_sp3, 1, sd, na.rm = TRUE)
episode_variance_filter_rewards_sp3 <- apply(combined_filter_reward_data_sp3, 1, var, na.rm = TRUE)

summary_filter_rewards_data_sp3 <- data.frame(
  episode = 1:length(episode_mean_filter_rewards_sp3),
  mean = episode_mean_filter_rewards_sp3,
  sd = episode_sd_filter_rewards_sp3,
  variance = episode_variance_filter_rewards_sp3
)

p1 <- ggplot() +
  geom_line(data = summary_filter_rewards_data_sp1, aes(x = episode, y = mean, color = "Service provider 1")) +
  geom_ribbon(data = summary_filter_rewards_data_sp1, aes(x = episode, ymin = mean - sd, ymax = mean + sd), 
              fill = "#3366FF", alpha = 0.2) +
  geom_line(data = summary_filter_rewards_data_sp2, aes(x = episode, y = mean, color = "Service provider 2")) +
  geom_ribbon(data = summary_filter_rewards_data_sp2, aes(x = episode, ymin = mean - sd, ymax = mean + sd), 
              fill = "#FFCC66", alpha = 0.2) +
  geom_line(data = summary_filter_rewards_data_sp3, aes(x = episode, y = mean, color = "Service provider 3")) +
  geom_ribbon(data = summary_filter_rewards_data_sp3, aes(x = episode, ymin = mean - sd, ymax = mean + sd), 
              fill = "#339900", alpha = 0.2) +
  scale_color_manual(values = c("Service provider 1" = "#0033CC", 
                                "Service provider 2" = "#FF9900",
                                "Service provider 3" = "#006633")) +
  labs(x = "Episode", y = "Average cumulative reward\nfor filtering service requests\n(across 100 simulations)", color = NULL) +
  theme_bw()+
  theme(
    legend.position = c(0.80, 0.30),
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
p1

# Visualise Replying rewards
# Read 'rewards sp1' column from each file
reply_reward_data_sp1 <- lapply(all_files, function(file) {
  df <- read_csv(file,show_col_types = FALSE)
  return(df$answer_rewards_sp0) 
})
names(reply_reward_data_sp1) <- sapply(all_files, function(f) paste0("repetition_", basename(f)))

# Combine them into one large data frame
combined_reply_reward_data_sp1 <- as.data.frame(reply_reward_data_sp1)
combined_reply_reward_data_sp1[is.na(combined_reply_reward_data_sp1)] <- 0

# Calculate sd and variance
episode_mean_reply_rewards_sp1 <- apply(combined_reply_reward_data_sp1, 1, mean, na.rm = TRUE)
episode_sd_reply_rewards_sp1 <- apply(combined_reply_reward_data_sp1, 1, sd, na.rm = TRUE)
episode_variance_reply_rewards_sp1 <- apply(combined_reply_reward_data_sp1, 1, var, na.rm = TRUE)

summary_reply_rewards_data_sp1 <- data.frame(
  episode = 1:length(episode_mean_reply_rewards_sp1),
  mean = episode_mean_reply_rewards_sp1,
  sd = episode_sd_reply_rewards_sp1,
  variance = episode_variance_reply_rewards_sp1
)

# Read 'rewards sp2' column from each file
reply_reward_data_sp2 <- lapply(all_files, function(file) {
  df <- read_csv(file,show_col_types = FALSE)
  return(df$answer_rewards_sp1)
})
names(reply_reward_data_sp2) <- sapply(all_files, function(f) paste0("repetition_", basename(f)))

# Combine them into one large data frame
combined_reply_reward_data_sp2 <- as.data.frame(reply_reward_data_sp2)
combined_reply_reward_data_sp2[is.na(combined_reply_reward_data_sp2)] <- 0

# Calculate sd and variance
episode_mean_reply_rewards_sp2 <- apply(combined_reply_reward_data_sp2, 1, mean, na.rm = TRUE)
episode_sd_reply_rewards_sp2 <- apply(combined_reply_reward_data_sp2, 1, sd, na.rm = TRUE)
episode_variance_reply_rewards_sp2 <- apply(combined_reply_reward_data_sp2, 1, var, na.rm = TRUE)

summary_reply_rewards_data_sp2 <- data.frame(
  episode = 1:length(episode_mean_reply_rewards_sp2),
  mean = episode_mean_reply_rewards_sp2,
  sd = episode_sd_reply_rewards_sp2,
  variance = episode_variance_reply_rewards_sp2
)

# Read 'rewards sp3' column from each file
reply_reward_data_sp3 <- lapply(all_files, function(file) {
  df <- read_csv(file,show_col_types = FALSE)
  return(df$answer_rewards_sp2)
})
names(reply_reward_data_sp3) <- sapply(all_files, function(f) paste0("repetition_", basename(f)))

# Combine them into one large data frame
combined_reply_reward_data_sp3 <- as.data.frame(reply_reward_data_sp3)
combined_reply_reward_data_sp3[is.na(combined_reply_reward_data_sp3)] <- 0

# Calculate sd and variance
episode_mean_reply_rewards_sp3 <- apply(combined_reply_reward_data_sp3, 1, mean, na.rm = TRUE)
episode_sd_reply_rewards_sp3 <- apply(combined_reply_reward_data_sp3, 1, sd, na.rm = TRUE)
episode_variance_reply_rewards_sp3 <- apply(combined_reply_reward_data_sp3, 1, var, na.rm = TRUE)

summary_reply_rewards_data_sp3 <- data.frame(
  episode = 1:length(episode_mean_reply_rewards_sp3),
  mean = episode_mean_reply_rewards_sp3,
  sd = episode_sd_reply_rewards_sp3,
  variance = episode_variance_reply_rewards_sp3
)

p2 <- ggplot() +
  geom_line(data = summary_reply_rewards_data_sp1, aes(x = episode, y = mean, color = "Service provider 1")) +
  geom_ribbon(data = summary_reply_rewards_data_sp1, aes(x = episode, ymin = mean - sd, ymax = mean + sd), 
              fill = "#3366FF", alpha = 0.2) +
  geom_line(data = summary_reply_rewards_data_sp2, aes(x = episode, y = mean, color = "Service provider 2")) +
  geom_ribbon(data = summary_reply_rewards_data_sp2, aes(x = episode, ymin = mean - sd, ymax = mean + sd), 
              fill = "#FFCC66", alpha = 0.2) +
  geom_line(data = summary_reply_rewards_data_sp3, aes(x = episode, y = mean, color = "Service provider 3")) +
  geom_ribbon(data = summary_reply_rewards_data_sp3, aes(x = episode, ymin = mean - sd, ymax = mean + sd), 
              fill = "#339900", alpha = 0.2) + 
  scale_color_manual(values = c("Service provider 1" = "#0033CC", 
                                "Service provider 2" = "#FF9900",
                                "Service provider 3" = "#006633")) +
  labs(x = "Episode", y = "Average cumulative reward\nfor replying in social network\n(across 100 simulations)", color = NULL) +
  theme_bw() +
  theme(
    legend.position = c(0.80, 0.30),
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
    panel.grid.major = element_line(color = "grey", linewidth = 0.6), 
    panel.grid.minor = element_line(color = "lightgrey", linewidth = 0.5) 
  )

p2

combined_plot <- cowplot::plot_grid(p1, p2, labels = c("C", "D"), ncol = 1) 
combined_plot
# Saved as 10 x 7.50 (portrait)

#####
# Filtering and replying rewards on one graph (doesn't look great)
#####
# Add new variable to the data frames
summary_filter_rewards_data_sp1$Group <- "Filtering rewards provider 1"
summary_reply_rewards_data_sp1$Group <- "Answering rewards provider 1"
summary_filter_rewards_data_sp2$Group <- "Filtering rewards on provider 2"
summary_reply_rewards_data_sp2$Group <- "Answering rewards on provider 2"
summary_filter_rewards_data_sp3$Group <- "Filtering rewards on provider 3"
summary_reply_rewards_data_sp3$Group <- "Answering rewards on provider 3"

ggplot() +
  geom_line(data = summary_filter_rewards_data_sp1, aes(x = episode, y = mean, color = Group, linetype = Group)) +
  geom_ribbon(data = summary_filter_rewards_data_sp1, aes(x = episode, ymin = mean - sd, ymax = mean + sd, fill = Group), alpha = 0.1) +
  
  geom_line(data = summary_filter_rewards_data_sp2, aes(x = episode, y = mean, color = Group, linetype = Group)) +
  geom_ribbon(data = summary_filter_rewards_data_sp2, aes(x = episode, ymin = mean - sd, ymax = mean + sd, fill = Group), alpha = 0.1) +
  
  geom_line(data = summary_filter_rewards_data_sp3, aes(x = episode, y = mean, color = Group, linetype = Group)) +
  geom_ribbon(data = summary_filter_rewards_data_sp3, aes(x = episode, ymin = mean - sd, ymax = mean + sd, fill = Group), alpha = 0.1) +
  
  geom_line(data = summary_reply_rewards_data_sp1, aes(x = episode, y = mean, color = Group, linetype = Group)) +
  geom_ribbon(data = summary_reply_rewards_data_sp1, aes(x = episode, ymin = mean - sd, ymax = mean + sd, fill = Group), alpha = 0.1) +
  
  geom_line(data = summary_reply_rewards_data_sp2, aes(x = episode, y = mean, color = Group, linetype = Group)) +
  geom_ribbon(data = summary_reply_rewards_data_sp2, aes(x = episode, ymin = mean - sd, ymax = mean + sd, fill = Group), alpha = 0.1) +
  
  geom_line(data = summary_reply_rewards_data_sp3, aes(x = episode, y = mean, color = Group, linetype = Group)) +
  geom_ribbon(data = summary_reply_rewards_data_sp3, aes(x = episode, ymin = mean - sd, ymax = mean + sd, fill = Group), alpha = 0.1) +
  
  
  scale_color_manual(values = c("Filtering rewards provider 1" = "#0033CC", "Answering rewards provider 1" = "#0033CC",
                                "Filtering rewards on provider 2" = "#FF9900", "Answering rewards on provider 2" = "#FF9900",
                                "Filtering rewards on provider 3" = "#006633", "Answering rewards on provider 3" = "#006633")) +
  scale_fill_manual(values = c("Filtering rewards provider 1" = "#3366FF", "Answering rewards provider 1" = "#3366FF",
                               "Filtering rewards on provider 2" = "#FFCC66", "Answering rewards on provider 2" = "#FFCC66",
                               "Filtering rewards on provider 3" = "#339900", "Answering rewards on provider 3" = "#339900")) +
  scale_linetype_manual(values = c("Filtering rewards provider 1" = "solid", "Answering rewards provider 1" = "dashed",
                                   "Filtering rewards on provider 2" = "solid", "Answering rewards on provider 2" = "dashed",
                                   "Filtering rewards on provider 3" = "solid", "Answering rewards on provider 3" = "dashed")) +
  labs(x = "Episode", y = "Average cumulative reward\nfor filtering service requests\nand replying in the social network", color = NULL, linetype = NULL, fill = NULL) +
  expand_limits(x = c(0, 350), y = c(0, 4000)) +
  theme_minimal()+
  theme(
    legend.position = c(0.82, 0.47),
    plot.title = element_text(size = rel(1.5)),
    axis.title = element_text(size = rel(1.5)),
    axis.text = element_text(size = rel(1.5)),
    legend.title = element_text(size = rel(1.5)),
    legend.text = element_text(size = rel(1)),
    plot.margin = margin(1, 1, 1, 1, "cm"),
    axis.title.x = element_text(vjust = -1),
    axis.title.y = element_text(vjust = 1)
  )

#####
# Loss 1
#####
# Read 'loss1 sp1' column from each file
loss1_data_sp1 <- lapply(all_files, function(file) {
  df <- read_csv(file,show_col_types = FALSE)
  return(df$dloss1_history_sp1)
})
names(loss1_data_sp1) <- sapply(all_files, function(f) paste0("repetition_", basename(f)))

# Combine them into one large data frame
combined_loss1_data_sp1 <- as.data.frame(loss1_data_sp1)
combined_loss1_data_sp1[is.na(combined_loss1_data_sp1)] <- 0

# Calculate sd and variance
episode_mean_loss1_sp1 <- apply(combined_loss1_data_sp1, 1, mean, na.rm = TRUE)
episode_sd_loss1_sp1 <- apply(combined_loss1_data_sp1, 1, sd, na.rm = TRUE)
episode_variance_loss1_sp1 <- apply(combined_loss1_data_sp1, 1, var, na.rm = TRUE)

summary_loss1_data_sp1 <- data.frame(
  episode = 1:length(episode_mean_loss1_sp1),
  mean = episode_mean_loss1_sp1,
  sd = episode_sd_loss1_sp1,
  variance = episode_variance_loss1_sp1
)

# Read 'loss1 sp2' column from each file
loss1_data_sp2 <- lapply(all_files, function(file) {
  df <- read_csv(file,show_col_types = FALSE)
  return(df$dloss1_history_sp2)
})
names(loss1_data_sp2) <- sapply(all_files, function(f) paste0("repetition_", basename(f)))

# Combine them into one large data frame
combined_loss1_data_sp2 <- as.data.frame(loss1_data_sp2)
combined_loss1_data_sp2[is.na(combined_loss1_data_sp2)] <- 0

# Calculate sd and variance
episode_mean_loss1_sp2 <- apply(combined_loss1_data_sp2, 1, mean, na.rm = TRUE)
episode_sd_loss1_sp2 <- apply(combined_loss1_data_sp2, 1, sd, na.rm = TRUE)
episode_variance_loss1_sp2 <- apply(combined_loss1_data_sp2, 1, var, na.rm = TRUE)

summary_loss1_data_sp2 <- data.frame(
  episode = 1:length(episode_mean_loss1_sp2),
  mean = episode_mean_loss1_sp2,
  sd = episode_sd_loss1_sp2,
  variance = episode_variance_loss1_sp2
)

# Read 'loss1 sp3' column from each file
loss1_data_sp3 <- lapply(all_files, function(file) {
  df <- read_csv(file,show_col_types = FALSE)
  return(df$dloss1_history_sp3)
})
names(loss1_data_sp3) <- sapply(all_files, function(f) paste0("repetition_", basename(f)))

# Combine them into one large data frame
combined_loss1_data_sp3 <- as.data.frame(loss1_data_sp3)
combined_loss1_data_sp3[is.na(combined_loss1_data_sp3)] <- 0

# Calculate sd and variance
episode_mean_loss1_sp3 <- apply(combined_loss1_data_sp3, 1, mean, na.rm = TRUE)
episode_sd_loss1_sp3 <- apply(combined_loss1_data_sp3, 1, sd, na.rm = TRUE)
episode_variance_loss1_sp3 <- apply(combined_loss1_data_sp3, 1, var, na.rm = TRUE)

summary_loss1_data_sp3 <- data.frame(
  episode = 1:length(episode_mean_loss1_sp3),
  mean = episode_mean_loss1_sp3,
  sd = episode_sd_loss1_sp3,
  variance = episode_variance_loss1_sp3
)

# Visualise defender loss
ggplot() +
  geom_line(data = summary_loss1_data_sp1, aes(x = episode, y = mean, color = "Provider 1")) +
  geom_ribbon(data = summary_loss1_data_sp1, aes(x = episode, ymin = mean - sd, ymax = mean + sd), 
              fill = "#3366FF", alpha = 0.2) +
  geom_line(data = summary_loss1_data_sp2, aes(x = episode, y = mean, color = "Provider 2")) +
  geom_ribbon(data = summary_loss1_data_sp2, aes(x = episode, ymin = mean - sd, ymax = mean + sd), 
              fill = "#FFCC66", alpha = 0.2) +
  geom_line(data = summary_loss1_data_sp3, aes(x = episode, y = mean, color = "Provider 3")) +
  geom_ribbon(data = summary_loss1_data_sp3, aes(x = episode, ymin = mean - sd, ymax = mean + sd), 
              fill = "#339900", alpha = 0.2) +
  scale_color_manual(values = c("Provider 1" = "#0033CC", 
                                "Provider 2" = "#FF9900",
                                "Provider 3" = "#006633")) +
  labs(x = "Episode", y = "Average loss for filtering", color = NULL) +
  expand_limits(x = c(10, 50), y = c(0, 80)) + # Should change
  theme_minimal()+
  theme(
    legend.position = c(0.82, 0.55),
    plot.title = element_text(size = rel(1.5)),
    axis.title = element_text(size = rel(1.5)),
    axis.text = element_text(size = rel(1.5)),
    legend.title = element_text(size = rel(1.5)),
    legend.text = element_text(size = rel(1)),
    plot.margin = margin(1, 1, 1, 1, "cm"),
    axis.title.x = element_text(vjust = -1),
    axis.title.y = element_text(vjust = 1)
  )

#####
# Loss 2
#####
# Read 'loss2 sp1' column from each file
loss2_data_sp1 <- lapply(all_files, function(file) {
  df <- read_csv(file,show_col_types = FALSE)
  return(df$mean_dloss2_history_sp1)
})
names(loss2_data_sp1) <- sapply(all_files, function(f) paste0("repetition_", basename(f)))

# Combine them into one large data frame
combined_loss2_data_sp1 <- as.data.frame(loss2_data_sp1)
combined_loss2_data_sp1[is.na(combined_loss2_data_sp1)] <- 0

# Calculate sd and variance
episode_mean_loss2_sp1 <- apply(combined_loss2_data_sp1, 1, mean, na.rm = TRUE)
episode_sd_loss2_sp1 <- apply(combined_loss2_data_sp1, 1, sd, na.rm = TRUE)
episode_variance_loss2_sp1 <- apply(combined_loss2_data_sp1, 1, var, na.rm = TRUE)

summary_loss2_data_sp1 <- data.frame(
  episode = 1:length(episode_mean_loss2_sp1),
  mean = episode_mean_loss2_sp1,
  sd = episode_sd_loss2_sp1,
  variance = episode_variance_loss2_sp1
)

# Read 'loss2 sp2' column from each file
loss2_data_sp2 <- lapply(all_files, function(file) {
  df <- read_csv(file,show_col_types = FALSE)
  return(df$mean_dloss2_history_sp2)
})
names(loss2_data_sp2) <- sapply(all_files, function(f) paste0("repetition_", basename(f)))

# Combine them into one large data frame
combined_loss2_data_sp2 <- as.data.frame(loss2_data_sp2)
combined_loss2_data_sp2[is.na(combined_loss2_data_sp2)] <- 0

# Calculate sd and variance
episode_mean_loss2_sp2 <- apply(combined_loss2_data_sp2, 1, mean, na.rm = TRUE)
episode_sd_loss2_sp2 <- apply(combined_loss2_data_sp2, 1, sd, na.rm = TRUE)
episode_variance_loss2_sp2 <- apply(combined_loss2_data_sp2, 1, var, na.rm = TRUE)

summary_loss2_data_sp2 <- data.frame(
  episode = 1:length(episode_mean_loss2_sp2),
  mean = episode_mean_loss2_sp2,
  sd = episode_sd_loss2_sp2,
  variance = episode_variance_loss2_sp2
)

# Read 'loss2 sp3' column from each file
loss2_data_sp3 <- lapply(all_files, function(file) {
  df <- read_csv(file,show_col_types = FALSE)
  return(df$mean_dloss2_history_sp3)
})
names(loss2_data_sp3) <- sapply(all_files, function(f) paste0("repetition_", basename(f)))

# Combine them into one large data frame
combined_loss2_data_sp3 <- as.data.frame(loss2_data_sp3)
combined_loss2_data_sp3[is.na(combined_loss2_data_sp3)] <- 0

# Calculate sd and variance
episode_mean_loss2_sp3 <- apply(combined_loss2_data_sp3, 1, mean, na.rm = TRUE)
episode_sd_loss2_sp3 <- apply(combined_loss2_data_sp3, 1, sd, na.rm = TRUE)
episode_variance_loss2_sp3 <- apply(combined_loss2_data_sp3, 1, var, na.rm = TRUE)

summary_loss2_data_sp3 <- data.frame(
  episode = 1:length(episode_mean_loss2_sp3),
  mean = episode_mean_loss2_sp3,
  sd = episode_sd_loss2_sp3,
  variance = episode_variance_loss2_sp3
)
# Visualise defender rewards
ggplot() +
  geom_line(data = summary_loss2_data_sp1, aes(x = episode, y = mean, color = "Provider 1")) +
  geom_ribbon(data = summary_loss2_data_sp1, aes(x = episode, ymin = mean - sd, ymax = mean + sd), 
              fill = "#3366FF", alpha = 0.2) +
  geom_line(data = summary_loss2_data_sp2, aes(x = episode, y = mean, color = "Provider 2")) +
  geom_ribbon(data = summary_loss2_data_sp2, aes(x = episode, ymin = mean - sd, ymax = mean + sd), 
              fill = "#FFCC66", alpha = 0.2) +
  geom_line(data = summary_loss2_data_sp3, aes(x = episode, y = mean, color = "Provider 3")) +
  geom_ribbon(data = summary_loss2_data_sp3, aes(x = episode, ymin = mean - sd, ymax = mean + sd), 
              fill = "#339900", alpha = 0.2) +
  scale_color_manual(values = c("Provider 1" = "#0033CC", 
                                "Provider 2" = "#FF9900",
                                "Provider 3" = "#006633")) +
  labs(x = "Episode", y = "Average loss for replying", color = NULL) +
  expand_limits(x = c(0, 350), y = c(-80, 10)) +
  theme_minimal()+
  theme(
    legend.position = c(0.82, 0.55),
    plot.title = element_text(size = rel(1.5)),
    axis.title = element_text(size = rel(1.5)),
    axis.text = element_text(size = rel(1.5)),
    legend.title = element_text(size = rel(1.5)),
    legend.text = element_text(size = rel(1)),
    plot.margin = margin(1, 1, 1, 1, "cm"),
    axis.title.x = element_text(vjust = -1),
    axis.title.y = element_text(vjust = 1)
  ) 
