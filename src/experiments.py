
from enviroment.frozen_lake_enviroment import FrozenLake

from algorithms.model_based_algorithms import *


import numpy as np
from helper.epsilon_greedy_algorithm import epsilon_greedy


def experiment_sarsa(env, max_episodes, eta, gamma, epsilon, optimal_value,theta,seed=None):


    random_state = np.random.RandomState(seed)
    
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    
    q = np.zeros((env.n_states, env.n_actions))
    
    for i in range(max_episodes):
        # start by reseting the environment and reset the while loop condition to False
        s = env.reset()
        isDone = False

        # get (select) the states for index i
        epsilon_selection = epsilon_greedy(epsilon[i], random_state)
        # select the action from random state using epsilon greedy
        a = epsilon_selection.selection(q[s])

        while not isDone:
            s_prime, r, isDone = env.step(a)
            a_prime = epsilon_selection.selection(q[s_prime])
            eta_i = eta[i]
            q[s, a] = q[s, a] +( eta_i* (r + (gamma * q[s_prime, a_prime]) - q[s, a]))
            # update s and a
            s = s_prime
            a = a_prime
    
        policy = q.argmax(axis=1)
        value = q.max(axis=1)
        if np.all(np.abs(optimal_value - value)< theta):
            break
    print("****************************** Sarsa converaged on episode {} **********************".format(i))
    return policy, value
    
def experiment_q_learning(env, max_episodes, eta, gamma, epsilon, optimal_value,theta,seed=None):

    random_state = np.random.RandomState(seed)
    
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    
    q = np.zeros((env.n_states, env.n_actions))
    
    for i in range(max_episodes):
        # start by reseting the environment and reset the while loop condition to False
        s = env.reset()
        isDone = False

        # get (select) the states for index i
        epsilon_selection = epsilon_greedy(epsilon[i], random_state)
        # select the action from random state using epsilon greedy
        a = epsilon_selection.selection(q[s])

        while not isDone:
            s_prime, r, isDone = env.step(a)
            a_prime = epsilon_selection.selection(q[s_prime])
            eta_i = eta[i]
            q[s, a] = q[s, a] + (eta_i * (r + (gamma * np.max(q[s_prime])) - q[s, a]))
            s = s_prime
            a = a_prime

        policy = q.argmax(axis=1)
        value = q.max(axis=1)
        if np.all(np.abs(optimal_value - value)< theta):
            break
    print("******************************Q-Learning converaged on episode {} **********************".format(i))
    return policy, value


################################## experimenting ##########################################################

seed = 0

# Small lake
small_lake =   [['&', '.', '.', '.'],
            ['.', '#', '.', '#'],
            ['.', '.', '.', '#'], 
            ['#', '.', '.', '$']]
# big lake
big_lake = [['&', '.', '.', '.', '.', '.', '.', '.'],
        ['.', '.', '.', '.', '.', '.', '.', '.'],
        ['.', '.', '.', '#', '.', '.', '.', '.'],
        ['.', '.', '.', '.', '.', '#', '.', '.'],
        ['.', '.', '.', '#', '.', '.', '.', '.'],
        ['.', '#', '#', '.', '.', '.', '#', '.'],
        ['.', '#', '.', '.', '#', '.', '#', '.'],
        ['.', '.', '.', '#', '.', '.', '.', '$']]

env_small = FrozenLake(small_lake, slip=0.1, max_steps=16, seed=seed)
env_big = FrozenLake(big_lake, slip=0.1, max_steps=16, seed=seed)
############################################################### Question 2 ############################################################### 
# print('Question 2 : How many iterations did policy iteration require to find an optimal policy for the big frozen lake? How many iterations did value iteration require? Which algorithm was faster?')
# gamma = 0.9
# theta = 0.001
# max_iterations = 100

# print('')

# policy, optimal_value = policy_iteration(env_big, gamma, theta, max_iterations,experiments=True)

# print('')

# policy, optimal_value = value_iteration(env_big, gamma, theta, max_iterations,experiments=True)

# print('')
# ############################################################### Question 3 ############################################################### 
# print('Question3: How many episodes did Sarsa control require to find an optimal policy for the small frozen lake? How many episodes did Q-learning control require?')
# max_episodes = 15000
# eta = 0.5
# epsilon = 0.5
# _, optimal_value = value_iteration(env_small, gamma, theta, max_iterations)
# theta = 0.1
# print('')

# print('## Sarsa')
# policy, value = experiment_sarsa(env_small, max_episodes, eta, gamma, epsilon, optimal_value,theta,seed=seed)
# env_small.render(policy, value)

# print('')

# print('## Q-learning')
# policy, value = experiment_q_learning(env_small, max_episodes, eta, gamma, epsilon,optimal_value ,theta,seed=seed)
# env_small.render(policy, value)

# print('')


############################################################### Question 5 ############################################################### 
print('Question 5 : Try to find an optimal policy for the big frozen lake by tweaking the parameters for Sarsa control and Q-learning control (maximum number of episodes, learning rate, and exploration factor). You must use policy evaluation to confirm that the resulting policy is optimal.')
gamma = 0.9
theta = 0.001
max_iterations = 100
max_episodes = 150000
eta = 0.5
epsilon = 0.5
_, optimal_value = value_iteration(env_big, gamma, theta, max_iterations)
theta = 0.1
print('')

print('## Sarsa')
policy, value = experiment_sarsa(env_big, max_episodes, eta, gamma, epsilon, optimal_value,theta,seed=seed)
env_big.render(policy, value)

print('')

print('## Q-learning')
policy, value = experiment_q_learning(env_big, max_episodes, eta, gamma, epsilon,optimal_value ,theta,seed=seed)
env_big.render(policy, value)

print('')