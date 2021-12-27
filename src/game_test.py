from enviroment.frozen_lake_enviroment import *

def play(env):
    actions = ['w', 'a', 's', 'd']
    
    state = env.reset()
    env.render()
    
    done = False
    while not done:
        c = input('\nMove: ')
        if c not in actions:
            raise Exception('Invalid action')
            
        state, r, done = env.step(actions.index(c))
        #print(state)
        env.render()
        print('Reward: {0}.'.format(r))

lake =  [['&', '.', '.', '.'],
        ['.', '#', '.', '#'],
        ['.', '.', '.', '#'],
        ['#', '.', '.', '$']]
slip=0.1
max_steps=20
play(FrozenLake(lake,slip,max_steps))