import numpy as np
import timeit
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
    
    prev_policy = np.zeros(env.n_states, dtype=int) # keep track of previous policy, to know when there is no improvement and hence terminate the policy

    current_iteration = 0

    start = timeit.default_timer() # to time the algorithms

    while current_iteration < max_iterations:
        value = policy_evaluation(env, policy, gamma, theta, max_iterations)
        policy = policy_improvement(env, value, gamma)
        current_iteration += 1

        if np.all(np.equal(policy, prev_policy)): # if the previous policy is equal to the new improved policy, stop the algorithm.
            break
        else:
            prev_policy = policy

    end = timeit.default_timer()

    value = policy_evaluation(env, policy, gamma, theta, max_iterations)
    print("**********************policy iteration took {} iterations to find the optimal policy. The algorithm ran in {} ms********************************".format(current_iteration,end-start))
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

    start = timeit.default_timer() # to time the algorithm

    while curr_iteration < max_iterations and not stop:
        curr_iteration += 1
        delta = 0
        for s in range(env.n_states):
            current_value = value[s]
            value[s] = np.max(np.sum(p[:,s,:] * (r[:,s,:] + (gamma * value.reshape(-1, 1))), axis=0))
            delta = max(delta, abs(current_value - value[s]))

        stop = delta < theta
    end = timeit.default_timer()
    policy = policy_improvement(env, value, gamma)
    print("**********************value iteration took {} iterations to find the optimal policy. The algorithms ran in {} ms********************************".format(curr_iteration,end-start))
    return policy, value