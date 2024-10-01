# Chapter 4: Hyperparameter Tuning
# Import libraries
library(openxlsx)
library(ggplot2)
library(paletteer)
library(psych)
library(png)

# Exp1: Regular agents' behaviour in the environment
#####
data <- read.xlsx("/Users/kartpadur/Documents/GitHub/pytorch_project/ch4-marl/hyperparameter_tuning/ch4-exp1-hp-tuning.xlsx") # Exp1
# Visualise hyperparameter tuning information
dev.off()

x <- data$trial_number
y <- data$mean_total_reward

describe(y)

# Split screen to two
my_screen_step1 <- split.screen(c(2, 1))

# Add one graph on the screen number 1 which is on top
screen(my_screen_step1[1])
plot(x,
     y, 
     pch=20, 
     xlab="Trial number",
     ylab = "Average cumulative reward",
     cex=1.0 , 
     col= "#FF9900"
     )
# Add "A" to the top-left corner
mtext("A", side=3, line=0, adj=0, cex=1.5)

# Identify the point with the maximum value in y
max_index <- which.max(y)
max_index
max_value <- y[max_index]

# Add text at the point with the maximum value, slightly offset to the right
text(x[max_index] + 0.055 * (max(x) - min(x)), max_value, labels=round(max_value, 2), cex=0.75, col="black")

# Add another graph on the screen number 2 which is on the bottom
screen(my_screen_step1[2])
hist(y, border=F , col="#FFCC66" , main="" ,  xlab="Distribution of rewards")
# Add "B" to the top-left corner
mtext("B", side=3, line=0, adj=0, cex=1.5) 

# Saved as /plots/hp-tuning-exp1.pdf with size of 8.00 x 8.50 (portrait)

#####
# Exp2: Attackers' and defenders' behaviour in the environment
data <- read.xlsx("/Users/kartpadur/Documents/GitHub/pytorch_project/ch4-marl/hyperparameter_tuning/ch4-exp2-hp-tuning.xlsx") # Exp2

# Attackers
# Visualise hyperparameter tuning information
dev.off()
x <- data$trial.number
y <- data$mean_attacker_reward

describe(y)

# Split screen to two
my_screen_step1 <- split.screen(c(2, 1))

# Add one graph on the screen number 1 which is on top
screen(my_screen_step1[1])
plot(x,
     y, 
     pch=20, 
     xlab="Trial number",
     ylab = "Average cumulative reward",
     cex=1.0 , 
     col= "#FF0000"
)

# Add "C" to the top-left corner
mtext("C", side=3, line=0, adj=0, cex=1.5)

# Identify the point with the maximum value in y
max_index <- which.max(y)
max_index
max_value <- y[max_index]

# Add text at the point with the maximum value, slightly offset to the right
text(x[max_index] + 0.05 * (max(x) - min(x)), max_value, labels=round(max_value, 2), cex=0.75, col="black")

# Add another graph on the screen number 2 which is on the bottom
screen(my_screen_step1[2])
hist(y, border=F , col="#FF0000" , main="" ,  xlab="Distribution of rewards")
# Add "D" to the top-left corner
mtext("D", side=3, line=0, adj=0, cex=1.5)

# Saved as /plots/ch4-exp2-hp-tuning-attackers.pdf with size of 8 x 8.5 (portrait)

# Defenders
# Visualise hyperparameter tuning information
dev.off()
x <- data$trial.number
y1 <- data$mean_sp0_reward
y2 <- data$mean_sp1_reward
y3 <- data$mean_sp2_reward

describe(y1)
describe(y2)
describe(y3)

# Split screen to two
my_screen_step1 <- split.screen(c(2, 3))

old_par <- par() # Save the current par settings
par(mar = c(5, 6, 4, 2) + 0.1)  # Increase the second value to increase the left margin

# Add one graph on the screen number 1 which is on top
screen(my_screen_step1[1])
plot(x,
     y1, 
     pch=20, 
     xlab="Trial number",
     ylab = "Average cumulative reward\nfor service provider 1",
     cex=1.0 , 
     col= "#0033CC"
)
mtext("E", side=3, line=0, adj=0, cex=1.5)

# Identify the point with the maximum value in y
max_index_y1 <- which.max(y1)
max_value_y1 <- y1[max_index_y1]
max_value_y1

# Add text at the point with the maximum value, slightly offset to the right
text(x[max_index_y1] + 0.08 * (max(x) - min(x)), max_value_y1, labels=round(max_value_y1, 2), cex=0.75, col="black")
par(old_par) # Reset the par settings to their original values

# Add another graph on the screen number 2 which is on the bottom
screen(my_screen_step1[4])
old_par <- par() # Save the current par settings
par(mar = c(5, 6, 4, 2) + 0.1)  # Increase the second value to increase the left margin
hist(y1, border=F , col="#0033CC" , main="" ,  xlab="Distribution of rewards")
mtext("F", side=3, line=0, adj=0, cex=1.5)
par(old_par) # Reset the par settings to their original values

# Add one graph on the screen number 1 which is on top
screen(my_screen_step1[2])

old_par <- par() # Save the current par settings
par(mar = c(5, 6, 4, 2) + 0.1)  # Increase the second value to increase the left margin

# Create the plot with the adjusted margins
plot(x,
     y2, 
     pch = 20, 
     xlab = "Trial number",
     ylab = "Average cumulative reward\nfor service provider 2",
     cex = 1.0, 
     col = "#FF9900"
)
mtext("G", side=3, line=0, adj=0, cex=1.5)


# Identify the point with the maximum value in y
max_index_y2 <- which.max(y2)
max_index_y2
max_value_y2 <- y2[max_index_y2]

# Add text at the point with the maximum value, slightly offset to the right
text(x[max_index_y2] + 0.08 * (max(x) - min(x)), max_value_y2, labels=round(max_value_y2, 2), cex=0.75, col="black")
par(old_par) # Reset the par settings to their original values

# Add another graph on the screen number 2 which is on the bottom
screen(my_screen_step1[5])
old_par <- par() # Save the current par settings
par(mar = c(5, 6, 4, 2) + 0.1)  # Increase the second value to increase the left margin
hist(y2, border=F , col="#FF9900" , main="" ,  xlab="Distribution of rewards")
mtext("H", side=3, line=0, adj=0, cex=1.5)
par(old_par) # Reset the par settings to their original values

# Add one graph on the screen number 1 which is on top
screen(my_screen_step1[3])
old_par <- par() # Save the current par settings
par(mar = c(5, 6, 4, 2) + 0.1)  # Increase the second value to increase the left margin

plot(x,
     y3, 
     pch=20, 
     xlab="Trial number",
     ylab = "Average cumulative reward\nfor service provider 3",
     cex=1.0 , 
     col= "#006633"
)
mtext("I", side=3, line=0, adj=0, cex=1.5)


# Identify the point with the maximum value in y
max_index_y3 <- which.max(y3)
max_index_y3
max_value_y3 <- y3[max_index_y3]

# Add text at the point with the maximum value, slightly offset to the right
text(x[max_index_y3] + 0.11 * (max(x) - min(x)), max_value_y3, labels=round(max_value_y3, 2), cex=0.75, col="black")
par(old_par)
# Add another graph on the screen number 2 which is on the bottom
screen(my_screen_step1[6])
old_par <- par() # Save the current par settings
par(mar = c(5, 6, 4, 2) + 0.1)  # Increase the second value to increase the left margin
hist(y3, border=F , col="#006633" , main="" ,  xlab="Distribution of rewards")
mtext("J", side=3, line=0, adj=0, cex=1.5)
par(old_par)

# Saved as /plots/ch4-exp2-hp-tuning-defenders.pdf with size of 8 x 12.75 (landscape)
