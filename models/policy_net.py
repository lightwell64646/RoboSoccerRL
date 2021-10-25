import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from .wikitext_Quac_npestes.commons.train_utils import learn_net

class dualNet(Model):
    def __init__(self, critic, duplicates=2, *args, **kwargs):
        super().__init__()
        self.critics = [critic(*args, **kwargs) for _ in range(duplicates)]
    def call(self, *args, **kwargs):
        return tf.math.minimum(*[
            critic(*args, **kwargs) for critic in self.critics
        ])

class Agent(Model):
    def __init__(self, action_net, critic_net, discount_rate = 0.99):
        super().__init__()
        self.discount_rate = discount_rate

        self.action = action_net
        self.critic = critic_net

        # Prioritized replay methods
        self.sample_idxs = None

        self.actor_preped_loss = None

    def format_step(self, step, action):
        return (step.observation, action, step.step_type, step.reward)

    def call(self, x):
        return self.action(x)

    def call_train_actor(self, x):
        obs, discount = x
        hindsight_actions = self.action(obs, discount) # (b,t,actions)
        critique = self.critic(obs, hindsight_actions) # (b,t,1)
        self.actor_preped_loss = -tf.reduce_mean(critique, axis=0)

    def train_step(self, trace, idx, sample_weight, **kwargs):
        obs, act, rew, discount = trace
        with tf.GradientTape() as agent_tape:
            hindsight_actions = self.action(obs, discount) # (b,t,actions)
            critique = self.critic(obs, hindsight_actions) # (b,t,1)
            actor_loss = -tf.reduce_mean(critique)
        learn_net(self.action, actor_loss, agent_tape, **kwargs)

        with tf.GradientTape() as critic_tape:
            continuous_act = self.action.reverseDiscretes(act) # (b,t,actions)
            estimates = self.critic(obs, continuous_act) # (b,t,1)
            estimates = tf.squeeze(estimates, axis = -1) # (b,t)

            # note we use critique to calculate expected reward under "optimal" future actions
            expected_reward = rew[:,:-1] + discount[:,:-1] * self.discount_rate * critique[:,1:] # (b,t)
            expected_reward = tf.stop_gradient(expected_reward)
            critic_loss = tf.keras.losses.MSE(expected_reward, estimates[:,:-1])
        learn_net(self.critic, critic_loss, critic_tape, **kwargs)

        return critic_loss


class GreedyExplorePolicy(Model):
    def __init__(self, model, sticky_epsilon = 0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.sticky_epsilon = sticky_epsilon
        self.last_action = None

    def make_greedy_choice(self, res):
        if self.model.action.actions_discrete is not list:
            return tf.argmax(res, -1).numpy() if self.model.action.actions_discrete else res
        return [tf.argmax(r, -1).numpy() if d else r
            for d, r in zip(self.model.action.actions_discrete, res)]

    def sticky_decision(self, action):
        if self.last_action is None:
            self.last_action = action
        else:
            stick_rand = np.random.uniform(size=action.shape) > self.sticky_epsilon
            self.last_action = stick_rand * action + (1 - stick_rand) * self.last_action
        return self.last_action

    def call(self, obs):
        res = self.make_greedy_choice(self.model.action(obs))
        return ([
            self.sticky_decision(r)
            for r in res
        ] if isinstance(res, list) else self.sticky_decision(res))

def nameSpec(spec, name):
    return tf.TensorSpec(spec.shape, spec.dtype, name)

def get_action_layer(game):
    if game.action_spec().dtype == np.int64:
        num_actions = game.action_spec().maximum - game.action_spec().minimum + 1
        return tf.keras.layers.Dense(num_actions, name = "actions", 
                activation = tf.nn.softmax, kernel_regularizer=None), [True]
    
    return tf.keras.layers.Dense(game.action_spec().shape[0], name = "actions", 
            activation = tf.nn.tanh, kernel_regularizer=None), [False]