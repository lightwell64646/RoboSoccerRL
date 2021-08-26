import tensorflow as tf
from tensorflow.keras import Model
from wikitext_Quac_npestes.commons.utils import learn_net, learn_nets
from wikitext_Quac_npestes.reorder_utils.shuffle_utils import shuffleNet
from .policy_net import Agent, GreedyExplorePolicy

class DreamNet(Model):
    def __init__(self, gen_net, act_net, policy, cycles=2):
        super().__init__()
        self.gen = gen_net
        self.act = act_net
        self.policy = policy
        self.cycles = cycles
    def call(self, x):
        for _ in range(self.cycles):
            trace = self.gen(x)
            acts = self.policy(self.act(trace))
            x = tf.concat([trace, acts], axis=-1)
        return x

class DreamShuffleAgent(Model):
    def __init__(self, act_net, critic_net, core_net, order_net, gen_net, 
                 prep_net=lambda x:x, policy=None, cycles=2):
        super().__init__()
        policy = policy or GreedyExplorePolicy(self.call_act)
        self.act = act_net
        self.critic = critic_net
        self.prep = prep_net
        self.core = core_net
        self.order = order_net
        self.gen = gen_net
        self.policy = policy
        self.agent = Agent(self.call_act, self.call_critic)
        self.language = shuffleNet(
            prep_net, core_net, order_net, 
            DreamNet(
                gen_net, self.call_act_train, policy, cycles=cycles))
    def call_act_train(self, x):
        return self.agent.call_train_actor(
                    tf.stop_gradient(
                        self.core(self.prep(x))
                    ))
    def call_act(self, x):
        return self.agent(
                    tf.stop_gradient(
                        self.core(self.prep(x))
                    ))
    def call_critic(self, x):
        return self.critic(
                    tf.stop_gradient(
                        self.core(self.prep(x))
                    ))
    def train_step(self, x, *args):
        self.agent.train_step(x, *args)
        with tf.GradientTape(persistent=True) as tape:
            self.language(x, training=True)
            language_loss = self.prep.losses + self.core.losses + self.order.losses
            agent_loss = self.agent.actor_preped_loss * self.language.batch_generator_loss / self.cycles
        learn_nets([self.core, self.prep, self.order], language_loss, tape)
        learn_net(self.gen, self.gen.losses, tape)
        learn_net(self.act, agent_loss, tape)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, x):
        self(x, training=False)
        return {m.name: m.result() for m in self.metrics}



