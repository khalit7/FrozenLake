from frozen_lake_enviroment import *

################ Model-based algorithms ################

def policy_evaluation(env, policy, gamma, theta, max_iterations):
    value = np.zeros(env.n_states, dtype=np.float)

    # TODO:
    stop = False
    iteration_counter = 0
    while not stop:
        for iteration in range(max_iterations) :
            delta = 0

            for s in range(env.n_states):
                current_value = value[s]
                # policy_action_prob = identity[policy[s]]
                action_value = []
                for next_s in range(env.n_states):
                    action_value.append(env.p(next_s, s, policy[s]) * (env.r(next_s, s, policy[s]) + gamma * value[next_s]))
                
                value[s] = sum(action_value)

                delta = max(delta, abs(current_value - value[s]))
            if delta < theta:
                stop = True
                break
    return value
    
def policy_improvement(env, value, gamma):
    policy = np.zeros(env.n_states, dtype=int)
    
    # TODO:
    for s in range(env.n_states):
        actions_list = []
        for a in range(env.n_actions):
            action_value = []
            for next_s in range(env.n_states):
                action_value.append(env.p(next_s, s, a) * (env.r(next_s, s, a) + gamma * value[next_s]))
        
            sum_action_value = sum(action_value)
            actions_list.append(sum_action_value)
        policy[s] = np.argmax(actions_list)

    return policy
    
def policy_iteration(env, gamma, theta, max_iterations, policy=None):
    if policy is None:
        policy = np.zeros(env.n_states, dtype=int)
    else:
        policy = np.array(policy, dtype=int)
    
    # TODO:
    for i in range(max_iterations):
        values = policy_evaluation(env, policy, gamma, theta,max_iterations)
        policy = policy_improvement(env, values, gamma)

    return policy, value
    
def value_iteration(env, gamma, theta, max_iterations, value=None):
    if value is None:
        value = np.zeros(env.n_states)
    else:
        value = np.array(value, dtype=np.float)
    
    # TODO:
    stop = False
    iteration_counter = 0
    while not stop:
        for iteration in range(max_iterations) :
            delta = 0

            for s in range(env.n_states):
                current_value = value[s]

                actions_list = []
                for a in range(env.n_actions):
                    action_value = []
                    for next_s in range(env.n_states):
                        action_value.append(env.p(next_s, s, a) * (env.r(next_s, s, a) + gamma * value[next_s]))
                
                    sum_action_value = sum(action_value)
                    actions_list.append(sum_action_value)

                value[s] = max(actions_list)
                delta = max(delta, abs(current_value - value[s]))
            if delta < theta:
                stop = True
                break

        policy = policy_improvement(env, value, gamma)

    return policy, value
