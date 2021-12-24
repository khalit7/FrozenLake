from frozen_lake_enviroment import *

################ Model-based algorithms ################

def policy_evaluation(env, policy, gamma, theta, max_iterations):
    value = np.zeros(env.n_states, dtype=np.float)
    identity = np.identity(env.n_actions)
    p = env.probs
    r = env.rewards

    curr_iteration = 0
    stop = False

    while curr_iteration < max_iterations and not stop:
        delta = 0

        for s in range(env.n_states):
            current_value = value[s]
            policy_action_prob = identity[policy[s]]
            value[s] = np.sum(policy_action_prob * p[:,s,:] * (r[:,s,:] + (gamma * value.reshape(-1, 1))))
            delta = max(delta, abs(current_value - value[s]))

        curr_iteration += 1
        stop = delta < theta

    return value
    
# Small lake
lake =   [['&', '.', '.', '.'],
            ['.', '#', '.', '#'],
            ['.', '.', '.', '#'],
            ['#', '.', '.', '$']]

env = FrozenLake(lake, slip=0.1, max_steps=16, seed=0)
gamma = 0.9
theta = 0.001
max_iterations = 100
policy = np.ones(env.n_states, dtype=int)
policy_evaluation(env, policy, gamma, theta, max_iterations)


# def policy_improvement(env, value, gamma):
#     policy = np.zeros(env.n_states, dtype=int)
    
#     # TODO:

#     return policy
    
# def policy_iteration(env, gamma, theta, max_iterations, policy=None):
#     if policy is None:
#         policy = np.zeros(env.n_states, dtype=int)
#     else:
#         policy = np.array(policy, dtype=int)
    
#     # TODO:
        
#     return policy, value
    
# def value_iteration(env, gamma, theta, max_iterations, value=None):
#     if value is None:
#         value = np.zeros(env.n_states)
#     else:
#         value = np.array(value, dtype=np.float)
    
#     # TODO:

#     return policy, value
