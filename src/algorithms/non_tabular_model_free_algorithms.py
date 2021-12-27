import numpy as np

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

        # TODO:
        #-- Finding the lineare greedy selection
        actions = range(env.n_actions)

        if random_state.rand() < epsilon[i]:
            action = random_state.choice(actions)
        else:
            #-- finding the maximum argument randomly 
            arg = np.argsort(q[actions])[::-1]
            n_tied = sum(np.isclose(q[actions], q[actions][arg[0]]))
            action = np.random.choice(arg[0:n_tied])
            action =  actions[action]


        done = False
        while not done:
            next_features, r, done = env.step(action)
            delta = r - q[action]
            q = next_features.dot(theta)
            actions = range(env.n_actions)

            if random_state.rand() < epsilon[i]:
                next_action = random_state.choice(actions)
            else:
                #-- finding the maximum argument randomly 
                arg = np.argsort(q[actions])[::-1]
                n_tied = sum(np.isclose(q[actions], q[actions][arg[0]]))
                next_action = np.random.choice(arg[0:n_tied])
                next_action =  actions[action]
            
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
        
        # TODO:
        q = features.dot(theta)

        done = False
        while not done:
            #-- Finding the lineare greedy selection
            actions = range(env.n_actions)

            if random_state.rand() < epsilon[i]:
                action = random_state.choice(actions)
            else:
                #-- finding the maximum argument randomly 
                arg = np.argsort(q[actions])[::-1]
                n_tied = sum(np.isclose(q[actions], q[actions][arg[0]]))
                action = np.random.choice(arg[0:n_tied])
                action =  actions[action]
                
            
            next_features, r, done = env.step(action)
            delta = r - q[action]
            q = next_features.dot(theta)
            delta += gamma * max(q)
            theta += eta[i] * delta * features[action, :]
            features = next_features

    return theta    
