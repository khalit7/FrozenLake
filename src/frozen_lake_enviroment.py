from enviroment_setup import *
import contextlib

# Configures numpy print options
@contextlib.contextmanager
def _printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally: 
        np.set_printoptions(**original)
################ Environment ################

class FrozenLake(Environment):
    def __init__(self, lake, slip, max_steps, seed=None):
        """
        lake: A matrix that represents the lake. For example:
        lake =  [['&', '.', '.', '.'],
                ['.', '#', '.', '#'],
                ['.', '.', '.', '#'],
                ['#', '.', '.', '$']]
        slip: The probability that the agent will slip
        max_steps: The maximum number of time steps in an episode
        seed: A seed to control the random number generator (optional)
        """
        # start (&), frozen (.), hole (#), goal ($)
        self.lake = np.array(lake)
        self.rows, self.columns = self.lake.shape
        self.lake_flat = self.lake.reshape(-1)
        
        self.slip = slip
        
        n_states = self.lake.size + 1
        n_actions = 4
        
        pi = np.zeros(n_states, dtype=float)
        pi[np.where(self.lake_flat == '&')[0]] = 1.0
        
        self.absorbing_state = n_states - 1
        
        # TODO:
        # call super constructor
        super(FrozenLake, self).__init__(n_states, n_actions, max_steps, pi, seed)

        # calcualte models of the enviroment
        self.probs = np.zeros((self.n_states, self.n_states, self.n_actions), dtype=float)
        self.calculate_probs()
        '''target = np.load("p.npy")
        if target == self._p:
            print("correctly calculated propabilities")
        else:
            print("propabilities calculation implemented wrong !")'''
        self.rewards = np.zeros((self.n_states, self.n_states, self.n_actions), dtype=float)
        self.calculate_rewards()

    def calculate_probs(self):
        for state in range(self.n_states):
            if state == self.absorbing_state or self.lake_flat[state] in ("#","$"):
                self.probs[self.absorbing_state,state,:] = 1
                continue
            for action in range(self.n_actions):
                for slip_action in range (self.n_actions):
                    next_state = self._make_action(slip_action,state)
                    self.probs[next_state,state,action] += self.slip/self.n_actions
                    if action == slip_action:
                        self.probs[next_state,state,action]+=1-self.slip

    def calculate_rewards(self):
        for state in range(self.n_states):
            if state != self.absorbing_state:
                if self.lake_flat[state] == "$":
                    self.rewards[self.absorbing_state,state,:] = 1
                    

    def _make_action(self,action,state):
        '''
        performs given action and returns the new state if the action is valid. Return the current  state 
        if the action is not valid
        '''
        if (state+1)%self.columns == 0: # cant go right
            if action== 3:
                return state
        if state % self.columns == 0: # cant go left
            if action == 1:
                return state
        if state - self.columns < 0: # cant go up
            if action ==0:
                return state
        if state + self.columns >= self.n_states-1: # cant go down
            if action == 2:
                return state
        if action == 0: # action up
            new_state = state - self.columns
        if action == 1: #action left
            new_state = state - 1
        if action == 2: # action down
            new_state = state + self.columns
        if action == 3: # actoin right
            new_state = state + 1
        return new_state
        
    def step(self, action):
        state, reward, done = Environment.step(self, action)
        
        done = (state == self.absorbing_state) or done
        
        return state, reward, done


    def p(self, next_state, state, action):

        return self.probs[next_state,state,action]

        


    def r(self, next_state, state, action):

        return self.rewards[next_state,state,action]
   
    def render(self, policy=None, value=None):
        if policy is None:
            lake = np.array(self.lake_flat)
            
            if self.state < self.absorbing_state:
                lake[self.state] = '@'
                
            print(lake.reshape(self.lake.shape))
        else:
            # UTF-8 arrows look nicer, but cannot be used in LaTeX
            # https://www.w3schools.com/charsets/ref_utf_arrows.asp
            actions = ['↑', '←', '↓', '→']
            
            print('Lake:')
            print(self.lake)
        
            print('Policy:')
            policy = np.array([actions[a] for a in policy[:-1]])
            print(policy.reshape(self.lake.shape))
            
            print('Value:')
            with _printoptions(precision=3, suppress=True):
                print(value[:-1].reshape(self.lake.shape))