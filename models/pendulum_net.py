import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from .policy_net import Agent, dualNet, get_action_layer

class PendulumAgentNet (Model):
    def __init__(self, game, **kwargs):
        super().__init__()
        self.l = Dense(64, activation = 'relu')
        self.act, self.actions_discrete = get_action_layer(game)

    def reverseDiscretes(self, acts):
        return tf.one_hot(acts, self.act.units)

    def call(self, x, *args, **kwargs):
        x = self.l(x)
        y = self.act(x)
        return y

class PendulumCriticNet (Model):
    def __init__(self, **kwargs):
        super().__init__()
        self.l1 = Dense(50, name="critic_l1", activation=tf.nn.relu)
        self.l2 = Dense(1, name="critic_l2")

    def call(self, state, action):
        y = self.l1(tf.concat([state, action], -1))
        y = self.l2(y)
        return y

def get_pendulum_net(game, discount_rate, **kwargs):
    return Agent(
        PendulumAgentNet(game, **kwargs), 
        dualNet(PendulumCriticNet, **kwargs), 
        discount_rate)





