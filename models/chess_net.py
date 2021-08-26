import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from .dream_net import DreamShuffleAgent
from .policy_net import get_action_layer

class ChessActorNet (Model):
    def __init__(self, flags, game):
        super().__init__()
        self.l = Dense(64, activation = 'relu')
        self.act, self.actions_discrete = get_action_layer(game)

    def reverseDiscretes(self, acts):
        return tf.one_hot(acts, self.act.units)

    def call(self, x, *args, **kwargs):
        x = self.l(x)
        y = self.act(x)
        return y

class ChessCriticNet (Model):
    def __init__(self, flags):
        super().__init__()
        self.l1 = Dense(50, activation=tf.nn.relu)
        self.l2 = Dense(1)

    def call(self, state, action):
        y = self.l1(tf.concat([state, action], -1))
        y = self.l2(y)
        return y

class ChessCoreNet (Model):
    def __init__(self, flags):
        super().__init__()
        self.l1 = Dense(50, activation=tf.nn.relu)
        self.l2 = Dense(width)

    def call(self, state):
        y = self.l1(state)
        y = self.l2(y)
        return y

class ChessOrderNet (Model):
    def __init__(self, num_offsets_to_predict, flags):
        super().__init__()
        self.l1 = Dense(50, activation=tf.nn.relu)
        self.l2 = Dense(num_offsets_to_predict)

    def call(self, x):
        y = self.l1(x)
        y = self.l2(y)
        return y

class ChessGenNet (Model):
    def __init__(self, width, flags):
        super().__init__()
        self.l1 = Dense(50, activation=tf.nn.relu)
        self.l2 = Dense(width)

    def call(self, seed):
        y = self.l1(seed)
        y = self.l2(y)
        return y

def get_chess_net(connections=[1,2,3,7], *args, **kwargs):
    return DreamShuffleAgent(
        ChessActorNet(*args, **kwargs), 
        ChessCriticNet(*args, **kwargs), 
        ChessCoreNet(*args, **kwargs), 
        ChessOrderNet(len(connections) + 1, *args, **kwargs), 
        ChessGenNet(*args, **kwargs)
    )




