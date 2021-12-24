from frozen_lake_enviroment import *

################ Model-based algorithms ################

def policy_evaluation(env, policy, gamma, theta, max_iterations):
    value = np.zeros(env.n_states, dtype=np.float)
    identity = np.identity(env.n_actions)
    # get models of the enviroment
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
    


def policy_improvement(env, value, gamma):
    policy = np.zeros(env.n_states, dtype=int)
    # get models of the enviroment
    p = env.probs
    r = env.rewards

    for s in range(env.n_states):
        policy[s] = np.argmax(np.sum(p[:,s,:] * (r[:,s,:] + (gamma * value.reshape(-1, 1))), axis=0))


    return policy
    
def policy_iteration(env, gamma, theta, max_iterations, policy=None):
    if policy is None:
        policy = np.zeros(env.n_states, dtype=int)
    else:
        policy = np.array(policy, dtype=int)
    
    current_iteration = 0

    while current_iteration < max_iterations :
        value = policy_evaluation(env, policy, gamma, theta, max_iterations)
        policy = policy_improvement(env, value, gamma)
        current_iteration += 1

    value = policy_evaluation(env, policy, gamma, theta, max_iterations)
    return policy, value


def value_iteration(env, gamma, theta, max_iterations, value=None):
    if value is None:
        value = np.zeros(env.n_states)
    else:
        value = np.array(value, dtype=np.float)

    curr_iteration = 0
    stop = False
    # get models of the enviroment
    p = env.probs
    r = env.rewards

    while curr_iteration < max_iterations and not stop:
        delta = 0
        for s in range(env.n_states):
            current_value = value[s]
            value[s] = np.max(np.sum(p[:,s,:] * (r[:,s,:] + (gamma * value.reshape(-1, 1))), axis=0))
            delta = max(delta, abs(current_value - value[s]))

        curr_iteration += 1
        stop = delta < theta

    policy = policy_improvement(env, value, gamma)

    return policy, value