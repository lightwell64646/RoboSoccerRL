from models.trainAgent import trainAgent
from models.pendulum_net import get_pendulum_net
from commons.flags import get_replay_flags
from tf_agents.environments import suite_gym, make_batched_environment

from absl import app

FLAGS = get_replay_flags()
       

def main(argv):
    env = make_batched_environment(suite_gym.load, FLAGS.parallel_environments, 'LunarLander-v2')
    
    actor = get_pendulum_net(FLAGS, env)
    trainAgent(env, actor, FLAGS)

if __name__ == "__main__":
    app.run(main)

