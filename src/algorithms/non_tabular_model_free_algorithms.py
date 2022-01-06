import numpy as np
from helper.epsilon_greedy_algorithm import epsilon_greedy
################ Non-tabular model-free algorithms ################

class LinearWrapper:
    def __init__(self, env):
        self.env = env
        
        self.n_actions = self.env.n_actions
        self.n_states = self.env.n_states
        self.n_features = self.n_actions * self.n_states
        
    def encode_state(self, s):
        features = np.zeros((self.n_actions, self.n_features))
        for a in range(self.n_actions):
            i = np.ravel_multi_index((s, a), (self.n_states, self.n_actions))
            features[a, i] = 1.0
          
        return features
    
    def decode_policy(self, theta):
        policy = np.zeros(self.env.n_states, dtype=int)
        value = np.zeros(self.env.n_states)
        
        for s in range(self.n_states):
            features = self.encode_state(s)
            q = features.dot(theta)
            
            policy[s] = np.argmax(q)
            value[s] = np.max(q)
        
        return policy, value
        
    def reset(self):
        return self.encode_state(self.env.reset())
    
    def step(self, action):
        state, reward, done = self.env.step(action)
        
        return self.encode_state(state), reward, done
    
    def render(self, policy=None, value=None):
        self.env.render(policy, value)
        
def linear_sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)
    
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    
    theta = np.zeros(env.n_features)
    
    for i in range(max_episodes):
        features = env.reset()
        
        q = features.dot(theta)

                
        # get (select) the states for index i
        epsilon_selection = epsilon_greedy(epsilon[i], random_state)
        # select the action from random state using epsilon greedy
        action = epsilon_selection.selection(q)


        done = False
        while not done:
            next_features, r, done = env.step(action)
            delta = r - q[action]
            q = next_features.dot(theta)

            next_action = epsilon_selection.selection(q)
            
            delta += gamma * q[next_action]
            theta += eta[i] * delta * features[action,:]
            features = next_features
            action = next_action

    return theta
    
def linear_q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)
    
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    
    theta = np.zeros(env.n_features)
    
    for i in range(max_episodes):
        features = env.reset()
        
        # get (select) the states for index i
        epsilon_selection = epsilon_greedy(epsilon[i], random_state)
        
        q = features.dot(theta)

        done = False
        while not done:
            # select the action from random state using epsilon greedy
            action = epsilon_selection.selection(q)
                
            
            next_features, r, done = env.step(action)
            delta = r - q[action]
            q = next_features.dot(theta)
            delta += gamma * max(q)
            theta += eta[i] * delta * features[action, :]
            features = next_features

    return theta    
