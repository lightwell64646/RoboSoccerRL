import gym
from tf_agents.environments import suite_gym
from tf_agents.environments.batched_py_environment import BatchedPyEnvironment
from tf_agents.environments.tf_py_environment import TFPyEnvironment

import numpy as np

class randomActor:
    def __init__(self, game):
        self.action_space = game.action_space
    def __call__(self, observation):
        return np.array([self.action_space.sample()])
        #return np.array([[self.action_space.sample()] for i in range(8)])
    def reset(self):
        pass

def gatherTrace(game, actor, num_steps = 1024, observers = [], render = False):
    actor.reset()
    step = game.reset()
    total_return = 0
    for _ in range(num_steps):
        action = actor(step.observation)
        step = game.step(action)
        if render:
            game.render()

        for obs in observers:
            obs(step, action)
        
        total_return += step.reward
    return total_return

if __name__ == "__main__":
    game = suite_gym.load('LunarLander-v2')
    env = BatchedPyEnvironment([suite_gym.load('LunarLander-v2') for _ in range (8)])
    actor = randomActor(game)

    gatherTrace(game, actor, observers = [lambda x,a : print(x,a)], render=True)