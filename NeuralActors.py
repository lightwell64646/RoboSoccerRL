import sys
sys.path.insert(0,'..')

from gatherGameRecords import gatherTrace


import gym
from tf_agents.environments import suite_gym
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.environments.batched_py_environment import BatchedPyEnvironment
from tf_agents.utils.common import scale_to_spec
from tf_agents.drivers import dynamic_step_driver
from tf_agents.agents.tf_agent import TFAgent

from absl import flags
from absl import app

import tensorflow as tf
import numpy as np

def nameSpec(spec, name):
    return tf.TensorSpec(spec.shape, spec.dtype, name)

def get_action_layer(game):
    if game.action_spec().dtype == np.int64:
        num_actions = game.action_spec().maximum - game.action_spec().minimum + 1
        return tf.keras.layers.Dense(num_actions, name = "actions", 
                activation = tf.nn.softmax, kernel_regularizer=None)
    
    return tf.keras.layers.Dense(game.action_spec().shape[0], name = "actions", 
            activation = tf.nn.tanh, kernel_regularizer=None)

class AgentNet (tf.keras.Model):
    def __init__(self, game, flags):
        super(AgentNet, self).__init__()
        self.act = get_action_layer(game)

    def reverseDiscretes(self, acts):
        return tf.one_hot(acts, self.act.units)

    def reset(self):
        pass

    def call(self, x):
        y = self.act(x)
        return y

class CriticNet (tf.keras.Model):
    def __init__(self, game, flags):
        super(CriticNet, self).__init__()
        self.l1 = tf.keras.layers.Dense(50, name = "critic_l1", activation = tf.nn.relu)
        self.l2 = tf.keras.layers.Dense(1, name = "critic_l2")

    def call(self, x):
        y = self.l1(x)
        y = self.l2(y)
        return y

class ExplorePolicy:
    def __init__(self, net, are_discrete, sticky_epsilon = 0.2):
        self.net = net
        self.state = None
        self.last_action = None
        self.sticky_epsilon = sticky_epsilon
        self.are_discrete = are_discrete
        
    def reset(self):
        self.net.reset()

    def make_choice(self, res):
        if self.are_discrete is not list:
            return tf.argmax(res, axis = -1).numpy() if self.are_discrete else res
        return [tf.argmax(r).numpy() if d else r
            for d, r in zip(self.are_discrete, res)]

    def __call__(self, observation):
        res = self.make_choice(self.net(observation))
        if self.last_action is not None:
            res_stick = []
            stick_rand = tf.random.uniform(res.shape)
            for i in range(res.shape[0]):
                if stick_rand[i] < self.sticky_epsilon:
                    res_stick.append(self.last_action[i])
                else:
                    res_stick.append(res[i])
            res = np.stack(res_stick, axis = 0)
        self.last_action = res
        return res
    

class MyAgent:
    def __init__(self, game, flags, agent_learning_rate = 1E-4, critic_learning_rate = 1E-4, discount_rate = 0.99):
        self.collect_spec = (
            nameSpec(game.observation_spec(), 'observation'),
            nameSpec(game.action_spec(), 'action'),
            tf.TensorSpec([1], tf.float32, 'reward'),
            tf.TensorSpec([1], tf.float32, 'step_type')
        )
        self.game = game
        self.state = None
        self.discount_rate = discount_rate

        # agent
        self.AgentNet = AgentNet(game, flags)
        self.AgentOptimizer = tf.keras.optimizers.RMSprop(learning_rate=agent_learning_rate)

        # agent
        self.CriticNet = CriticNet(game, flags)
        self.CriticOptimizer = tf.keras.optimizers.RMSprop(learning_rate=critic_learning_rate)

    def getCollectionPolicy(self):
        return ExplorePolicy(self.AgentNet, True)

    def format_step(self, step, action):
        return (step.observation, action, step.step_type, step.reward)

    def train(self, dataset, iterations = 16):
        '''
        b - batch
        t - time
        '''
        for b in dataset:
            traj, _ = b
            obs, act, rew, typ = traj
            with tf.GradientTape() as tape:
                naturalized_act = self.AgentNet.reverseDiscretes(act)
                state_and_action = tf.concat([obs, naturalized_act], axis = -1) # (b,t,?)
                estimates = self.CriticNet(state_and_action) # (b,t,1)
                estimates = tf.squeeze(estimates, axis = -1)
                estimates_no_grad = tf.stop_gradient(estimates) #accept future returns as fact while updating
                expected_reward = rew[:,:-1] + (1 - typ[:,:-1]) * self.discount_rate * estimates_no_grad[:,:-1] # (b,t)
                critic_loss = tf.keras.losses.MSE(expected_reward, estimates[:,:-1])
            grads = tape.gradient(critic_loss, self.CriticNet.trainable_weights)
            self.CriticOptimizer.apply_gradients(list(zip(grads, self.CriticNet.trainable_weights)))

            with tf.GradientTape() as tape:
                hindsight = self.AgentNet(obs) # ensure this can't look at the future (I would be curios how it fails though)
                hypothetical = tf.concat([obs, hindsight], axis = -1)
                critique = -self.CriticNet(hypothetical)
            grads = tape.gradient(critique, self.AgentNet.trainable_weights)
            self.AgentOptimizer.apply_gradients(list(zip(grads, self.AgentNet.trainable_weights)))
            
            '''
            print(len(b))
            for bb in b[0]:
                print('0 - ', bb.shape, bb[0])
            for bb in b[1]:
                print('1 - ', bb.shape, bb[0])
            '''
            iterations -= 1
            if iterations == 0:
                return

    def reset(self):
        self.AgentNet.reset()



def trainAgent(env, agent, flags, epochs = 1000, render = False):
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_spec,
        batch_size=flags.parallel_environments,
        max_length=flags.replay_buffer_capacity)
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=tf.data.experimental.AUTOTUNE, 
        sample_batch_size=flags.train_batch_size, 
        num_steps=2).prefetch(tf.data.experimental.AUTOTUNE)

    # populate initial actions
    gatherTrace(
        env, 
        agent.getCollectionPolicy(), 
        observers=[lambda step, act : replay_buffer.add_batch(agent.format_step(step, act))],
        num_steps=flags.initial_collect_steps)

    for epoch in range(epochs):
        total_reward = gatherTrace(
            env, 
            agent.getCollectionPolicy(), 
            observers=[lambda step, act : replay_buffer.add_batch(agent.format_step(step, act))],
            num_steps=flags.collect_steps_per_iteration)
        if epoch % 50 == 0:
            print("current progress", tf.reduce_mean(total_reward))

        agent.train(dataset)




        

flags.DEFINE_integer('train_batch_size', 64, "number of elements in a batch")
flags.DEFINE_integer('replay_buffer_capacity', 1024, "maximum number of elements to store in replay buffer")
flags.DEFINE_integer('initial_collect_steps', 128, "maximum number of elements to store in replay buffer")
flags.DEFINE_integer('collect_steps_per_iteration', 32, "experiences added per cycle")
flags.DEFINE_integer('parallel_environments', 8, "number of parallel games to play")
FLAGS = flags.FLAGS

def main(argv):
    game = suite_gym.load('LunarLander-v2')
    env = BatchedPyEnvironment([suite_gym.load('LunarLander-v2') for _ in range (FLAGS.parallel_environments)])
    
    actor = MyAgent(env, FLAGS)
    trainAgent(env, actor, FLAGS)

if __name__ == "__main__":
    app.run(main)

