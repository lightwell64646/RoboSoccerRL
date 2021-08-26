from models.trainAgent import trainAgent
from models.pendulum_net import get_pendulum_net
from commons.flags import get_replay_flags
from tf_agents.environments import suite_gym, make_batched_environment
from gym_chess.alphazero import BoardEncoding

from absl import app

FLAGS = get_replay_flags()
       

def main(argv):
    encoded_chess = lambda : BoardEncoding(suite_gym.load('Chess-v0'))
    env = make_batched_environment(encoded_chess, FLAGS.parallel_environments)
    
    actor = get_pendulum_net(FLAGS, env)
    trainAgent(env, actor, FLAGS)

if __name__ == "__main__":
    app.run(main)

