from absl import flags
from wikitext_Quac_npestes.commons.common_flags import get_basic_flags

def get_replay_flags():
    get_basic_flags()
    flags.DEFINE_integer('capacity', 128, "maximum number of elements to store in replay buffer")

def get_reenforcement_flags():
    get_replay_flags()
    flags.DEFINE_integer('batch_size', 64, "experiences added per cycle")
    flags.DEFINE_integer('parallel_environments', 8, "number of parallel games to play")
    flags.DEFINE_integer('train_cycles', 8, "number of steps to train on in each batch")
    flags.DEFINE_integer('explore_cycles', 16, "number of steps to train on in each batch")
    flags.DEFINE_integer('steps', 32, "number of steps to train on in each batch")
    flags.DEFINE_float('sticky_epsilon', 0.4, "chance to repeat an action durring exploration")
    flags.DEFINE_float('discount_rate', 0.99, "value of future rewards")
    return flags.FLAGS