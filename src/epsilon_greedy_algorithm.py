import numpy as np
import math

class epsilon_greedy:
    """
    This class implements the epsilon greedy algorithm. the next action for the algorithm will depend on the value of epsilon
    """

    def __init__(self, epsilon, random_state=None):
        """
        This is the constructor of the class
        Parameters:
            (float): epsilon value that the algorithm will use to select the next node to explore
            (np.RandomState): Random state of the current game
        """
        if random_state is None:
            self.random_state = np.random.RandomState()
        else:
            self.random_state = random_state

        self.epsilon = epsilon
        

    def selection(self, actions):
        """
        this function is used for the algorithm to select an action 

        """
        # generate a random probablity using uniform distribution
        if self.random_state.uniform(0, 1) < self.epsilon: # if less than epsioln
            # choose a random action
            return self.random_state.randint(0, len(actions))
        else:
            #otherwise, pick the maximum the action that will give the maximum result
            max_action = np.max(actions)
            #get the indexes of the max action
            max_index = np.flatnonzero(max_action == actions)

            return self.random_state.choice(max_index)