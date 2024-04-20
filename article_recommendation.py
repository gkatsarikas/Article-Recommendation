# -*- coding: utf-8 -*-


#Import the necessary libraries
import numpy as np
import random
import matplotlib.pyplot as plt

"""
Environment class

1) K denotes the number of arms
2) U denotes the number of different user classes
3) T is the horizon variable (total rounds)
"""
class Environment:
    def __init__(self,K,U,T):
        self.K = K
        self.U = U
        self.T = T
        self.click_probabilities = [
            [0.8, 0.6, 0.5, 0.4, 0.2],
            [0.2, 0.4, 0.5, 0.6, 0.8],
            [0.2, 0.4, 0.8, 0.6, 0.5],
            [0.2, 0.4, 0.8, 0.6, 0.5]
        ]
    #Click event
    def get_reward(self, article, user_class):
        return random.random() < self.click_probabilities[user_class][article]

"""
UCB Agent class
"""
class UCB:
    def __init__(self, env):
        self.env = env
        self.Q_t = np.zeros((self.env.K, self.env.U))  # Number of times an article is chosen
        self.mu = np.zeros((self.env.K, self.env.U))  # Estimated value of each article
        self.regret = 0
        self.current_round = 0

    def select_article(self, user_class):
        if self.current_round < self.env.K:
            return self.current_round  # Choose each arm once for the first K steps
        else:
            index_values = self.mu[:, user_class] + np.sqrt(np.log(self.env.T) / (self.Q_t[:, user_class]))
            return np.argmax(index_values)

    def update(self, article, reward, user_class):
        self.current_round += 1
        self.Q_t[article, user_class] += 1
        self.mu[article, user_class] += (reward - self.mu[article, user_class]) / self.Q_t[article, user_class]
        optimal_reward = max(self.env.click_probabilities[user_class])
        self.regret += optimal_reward - self.env.click_probabilities[user_class][article]

K = 5
U = 4
T = 1000

env = Environment(K, U, T)
ucb_agent = UCB(env)

# Main loop
cumulative_regret = np.zeros(T)
for t in range(T):
    user_class = int(random.uniform(0, U - 1))
    article = ucb_agent.select_article(user_class)
    reward = env.get_reward(article, user_class)
    ucb_agent.update(article, reward, user_class)
    cumulative_regret[t] = ucb_agent.regret

# Plot cumulative regret
plt.plot(cumulative_regret)
plt.title('Cumulative Regret')
plt.xlabel('Round')
plt.ylabel('Regret')
plt.show()

K = 5
U = 4
T = 10000

env = Environment(K, U, T)
ucb_agent = UCB(env)

# Main loop
cumulative_regret = np.zeros(T)
for t in range(T):
    user_class = int(random.uniform(0, U - 1))
    article = ucb_agent.select_article(user_class)
    reward = env.get_reward(article, user_class)
    ucb_agent.update(article, reward, user_class)
    cumulative_regret[t] = ucb_agent.regret

# Plot cumulative regret
plt.plot(cumulative_regret)
plt.title('Cumulative Regret')
plt.xlabel('Round')
plt.ylabel('Regret')
plt.show()

