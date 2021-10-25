from absl import app

from tf_agents.environments import suite_gym, make_batched_environment

from models.trainAgent import trainAgent
from models.pendulum_net import get_pendulum_net
from commons.flags import get_replay_flags
       

def main(argv):
    do_main(FLAGS)

def do_main(flags):
    env = make_batched_environment(suite_gym.load, flags.parallel_environments, 'LunarLander-v2')
    
    actor = get_pendulum_net(env, **flags.flag_values_dict())
    trainAgent(env, actor, **flags.flag_values_dict())

if __name__ == "__main__":
    FLAGS = get_replay_flags()
    app.run(main)

