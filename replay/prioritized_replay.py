
"""
The DQN improvement: Prioritized Experience Replay (based on https://arxiv.org/abs/1511.05952)
View more on 莫烦Python: https://morvanzhou.github.io/tutorials/
Using:
Tensorflow: 1.0
"""

import numpy as np
import tensorflow as tf

from tensorflow.keras.callbacks import Callback
from tensorflow.keras import Model

np.random.seed(1)
tf.random.set_seed(1)


class SumTree(object):
    """
    This SumTree code is modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py
    Story the data with it priority in tree and data frameworks.
    """
    data_pointer = 0
    full = False

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add_new_priority(self, p, data):
        leaf_idx = self.data_pointer + self.capacity - 1

        self.data[self.data_pointer] = data  # update data_frame
        self.update(leaf_idx, p)  # update tree_frame
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0
            self.full = True

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        self._propagate_change(tree_idx, change)

    def _propagate_change(self, tree_idx, change):
        """change the sum of priority value in all parent nodes"""
        parent_idx = (tree_idx - 1) // 2
        self.tree[parent_idx] += change
        if parent_idx != 0:
            self._propagate_change(parent_idx, change)

    def get_leaf(self, lower_bound):
        leaf_idx = self._retrieve(lower_bound)  # search the max leaf priority based on the lower_bound
        data_idx = leaf_idx - self.capacity + 1
        return [leaf_idx, self.tree[leaf_idx], self.data[data_idx]]

    def _retrieve(self, lower_bound, parent_idx=0):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        left_child_idx = 2 * parent_idx + 1
        right_child_idx = left_child_idx + 1

        if left_child_idx >= len(self.tree):  # end search when no more child
            return parent_idx

        if self.tree[left_child_idx] == self.tree[right_child_idx]:
            return self._retrieve(lower_bound, np.random.choice([left_child_idx, right_child_idx]))
        if lower_bound <= self.tree[left_child_idx]:  # downward search, always search for a higher priority node
            return self._retrieve(lower_bound, left_child_idx)
        else:
            return self._retrieve(lower_bound - self.tree[left_child_idx], right_child_idx)

    @property
    def root_priority(self):
        return self.tree[0]  # the root


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This SumTree code is modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    epsilon = 0.001  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 1e-4  # annealing the bias
    abs_err_upper = 1   # for stability refer to paper

    data_groups = 1
    data_groups_initialized = False

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, error, transition):
        p = self._get_priority(error)
        self.tree.add_new_priority(p, transition)
        if self.data_groups_initialized == False:
            self.data_groups = len(transition)
            self.data_groups_initialized = True

    def sample(self, n):
        batch_idx, ISWeights = [], []
        data_packed = [[] for _ in range(self.data_groups)]
        segment = self.tree.root_priority / n
        self.beta = np.min([1, self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.root_priority
        maxiwi = np.power(self.tree.capacity * min_prob, -self.beta)  # for later normalizing ISWeights
        for i in range(n):
            a = segment * i
            b = segment * (i + 1)
            lower_bound = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(lower_bound)
            prob = p / self.tree.root_priority
            ISWeights.append(self.tree.capacity * prob)
            batch_idx.append(idx)
            for j in range(self.data_groups):
                data_packed[j].append(data[j])

        data_packed = [tf.stack(d, axis=0) for d in data_packed]

        ISWeights = np.vstack(ISWeights)
        ISWeights = np.power(ISWeights, -self.beta) / maxiwi  # normalize
        return data_packed, np.array(batch_idx), ISWeights

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)

    def _get_priority(self, error):
        error += self.epsilon   # avoid 0
        clipped_error = np.clip(error, 0, self.abs_err_upper)
        return np.power(clipped_error, self.alpha)

def get_replay_dataset(memory, batch):
    data, batch_idx, ISWeights = memory.sample(batch)
    obs, act, rew, discount = data
    def gen():
        while 1:
            yield memory.sample(batch)
    return tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            (tf.TensorSpec(shape=obs.shape, dtype=obs.dtype),
            tf.TensorSpec(shape=act.shape, dtype=act.dtype),
            tf.TensorSpec(shape=rew.shape, dtype=rew.dtype),
            tf.TensorSpec(shape=discount.shape, dtype=discount.dtype)),
            tf.TensorSpec(shape=batch_idx.shape, dtype=batch_idx.dtype),
            tf.TensorSpec(shape=ISWeights.shape, dtype=ISWeights.dtype),
        )
    )

def compile_with_memory(dataset, memory, model, loss, update=False, samples=None, *args, **kwargs):
    batch_size = None
    for data in dataset:
        if type(data) is tuple:
            samples = samples or data[0].shape[0]
            batch_size = data[0].shape[0]
        else:
            samples = samples or data.shape[0]
            batch_size = data.shape[0]
        break
    for data in dataset:
        for i in range(batch_size):
            memory.store(1, [d[i] for d in data])
        if memory.tree.full:
            break

    callback = prioritized_update_callback(memory) if update else prioritized_replace_callback(memory, lambda x: x > 0.5)
    def append_memory(*args):
        data_packed, batch_idx, ISWeights = memory.sample(samples)
        if update:
            callback.last_sample_idxs = batch_idx
        res = []
        for i, m in enumerate(data_packed):
            res.append(tf.concat([args[i], m], axis=0))
        return res
    dataset = dataset.map(append_memory).map(get_post_batch(callback))
    model.compile(loss=wraped_loss(loss, get_loss_reporter(callback)), *args, **kwargs)
    return dataset, callback


class prioritized_update_callback(Callback):
    def __init__(self, mem):
        self.mem = mem
        self.last_sample_idxs = None
        self.last_loss = None
    def on_train_batch_end(self, batch, logs=None):
        idxs = self.last_sample_idxs
        for j in range(len(idxs)):
            self.mem.update(idxs[j], self.last_loss[j])

class prioritized_replace_callback(Callback):
    def __init__(self, mem, filter):
        self.mem = mem
        self.filter = filter
        self.last_batch = None
        self.last_loss = None
    def on_train_batch_end(self, batch, logs=None):
        print(logs)
        print(self.last_loss)
        self.last_loss = np.array(self.last_loss)
        for i, sample in enumerate(self.last_batch):
            if self.filter(self.last_loss[i]):
                self.mem.store(self.last_loss[i], sample)

def get_loss_reporter(callback):
    def report(loss):
        callback.last_loss = loss
    return report

def wraped_loss(real_loss, util):
    def wl(true, pred):
        loss = real_loss(true, pred)
        util(loss)
        return loss
    return wl

def get_post_batch(report):
    def post(*args):
        report.last_batch = args
        return args
    return post