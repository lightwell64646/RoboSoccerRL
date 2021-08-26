import tensorflow as tf
import numpy as np

from tf_agents.environments import suite_gym
from tf_agents.environments.batched_py_environment import BatchedPyEnvironment
from tf_agents.environments.tf_py_environment import TFPyEnvironment

def gather_traces(game, agent, steps = 1024, observers = [], render = False, batches = 8):
    for _ in range(batches):
        gather_trace(game, agent, steps, observers, render)

def gather_trace(game, agent, steps = 1024, observers = [], render = False):
    #agent.net.reset()
    step = game.reset()
    total_return = 0
    trace = [[] for _ in range(4)]
    for _ in range(steps):
        action = agent(step.observation)
        step = game.step(action)
        if render:
            game.render()

        info = (step.observation, action, step.reward, step.discount)
        for i in range(4):
            trace[i].append(info[i])
        
        total_return += step.reward
    
    formed_batch = [tf.stack(t, axis = 1) for t in trace]
    for obs in observers:
        obs(formed_batch)
    return total_return

class randomActor:
    def __init__(self, game):
        self.action_space = game.action_space
    def __call__(self, observation):
        return np.array([self.action_space.sample()])
    def reset(self):
        pass

if __name__ == "__main__":
    game = suite_gym.load('LunarLander-v2')
    env = BatchedPyEnvironment([suite_gym.load('LunarLander-v2') for _ in range (8)])
    actor = randomActor(game)

    gather_trace(game, actor, observers = [lambda x,a : print(x,a)], render=True)