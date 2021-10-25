from absl import app

from tf_agents.environments import suite_gym, make_batched_environment
from gym_chess.alphazero import BoardEncoding

from models.trainAgent import trainAgent
from models.pendulum_net import get_pendulum_net
from commons.flags import get_replay_flags


def main(argv):
    do_main(FLAGS)

def do_main(flags):
    encoded_chess = lambda : BoardEncoding(suite_gym.load('Chess-v0'))
    env = make_batched_environment(encoded_chess, flags.parallel_environments)
    
    actor = get_pendulum_net(env, **flags.flag_values_dict())
    trainAgent(env, actor, **flags.flag_values_dict())

if __name__ == "__main__":
    FLAGS = get_replay_flags()
    app.run(main)

