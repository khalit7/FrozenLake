
################ Tabular model-free algorithms ################
# The function sarsa receives an environment, a maximum number of episodes, 
# an initial learning rate, a discount factor, an initial exploration factor, 
# and an (optional) seed that controls the pseudorandom number generator. 
# Note that the learning rate and exploration factor decrease linearly as the number 
# of episodes increases (for instance, eta[i] contains the learning rate for episode i).
import numpy as np
from epsilon_greedy_algorithm import epsilon_greedy


def sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
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
        
    return policy, value
    
def q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):

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
    
    # print("Done")
    return policy, value
