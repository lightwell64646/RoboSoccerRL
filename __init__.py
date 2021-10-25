from RoboSoccerRL import mnist_replay_test
from RoboSoccerRL import train_chess
from RoboSoccerRL import train_pendulum
from RoboSoccerRL import wikitext_Quac_npestes

from RoboSoccerRL.mnist_replay_test import (FLAGS, main, test_mnist,)
from RoboSoccerRL.train_chess import (main,)
from RoboSoccerRL.train_pendulum import (main,)
from RoboSoccerRL.wikitext_Quac_npestes import (M, P, PerfectFixedOrderNet,
                                                PerfectMarcovOrderNet,
                                                SimpleExpandNet, SimplePassNet,
                                                cifar_gan_test,
                                                cifar_shuffle_gan_test,
                                                covar_mnist_test,
                                                covariance_test, f, g,
                                                generate_vocab_fixed,
                                                generate_vocab_markov,
                                                getPerfectModelFixed,
                                                getPerfectModelMarkov,
                                                grammar_test, h, i,
                                                inspect_test, j, k, l,
                                                learn_net, main,
                                                make_noise_dataset,
                                                make_synthetic_dataset,
                                                multi_loss_test,
                                                persistent_gradient,
                                                persistent_gradient_test,
                                                pretrain_wikitext,
                                                tape_reentry_test,
                                                tensorflow_compiler_test, test,
                                                test_mnist, test_models,
                                                test_routing_entropy_on_mnist,
                                                timeing,)

__all__ = ['FLAGS', 'M', 'P', 'PerfectFixedOrderNet', 'PerfectMarcovOrderNet',
           'SimpleExpandNet', 'SimplePassNet', 'cifar_gan_test',
           'cifar_shuffle_gan_test', 'covar_mnist_test', 'covariance_test',
           'f', 'g', 'generate_vocab_fixed', 'generate_vocab_markov',
           'getPerfectModelFixed', 'getPerfectModelMarkov', 'grammar_test',
           'h', 'i', 'inspect_test', 'j', 'k', 'l', 'learn_net', 'main',
           'make_noise_dataset', 'make_synthetic_dataset', 'mnist_replay_test',
           'multi_loss_test', 'persistent_gradient',
           'persistent_gradient_test', 'pretrain_wikitext',
           'tape_reentry_test', 'tensorflow_compiler_test', 'test',
           'test_mnist', 'test_models', 'test_routing_entropy_on_mnist',
           'timeing', 'train_chess', 'train_pendulum', 'wikitext_Quac_npestes']
