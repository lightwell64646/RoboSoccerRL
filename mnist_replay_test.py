import tensorflow as tf
import matplotlib.pyplot as plt
from absl import app
from absl import flags

from replay.prioritized_replay import compile_with_memory, Memory

from commons.flags import get_replay_flags
from wikitext_Quac_npestes.commons.MnistDataLoader import get_mnist_datset
from wikitext_Quac_npestes.commons.basic_fit import do_basic_fit

from wikitext_Quac_npestes.models.mnist_test_models import conv_model, test_mnist


def main(argv):
    model = conv_model()
    test_mnist(model, 1000)

    model = conv_model()
    test_mnist(model, None)
    
if __name__ == "__main__":
    get_replay_flags()
    app.run(main)