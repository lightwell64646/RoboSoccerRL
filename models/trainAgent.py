from tensorflow.keras.callbacks import Callback

from models.policy_net import GreedyExplorePolicy
from replay.prioritized_replay import Memory, prioritized_update_callback, get_memory_dataset
from replay.gather_game_records import gather_traces
from tf_agents.environments.batched_py_environment import BatchedPyEnvironment


def trainAgent(env, agent, capacity, steps, sticky_epsilon, parallel_environments, batch_size, train_cycles, epochs = 1000, render = False):
    memory = Memory(capacity)
    def store_new_batches(step):
        confusion = 9999 # agent.get_loss(*step)
        for i in range(step[0].shape[0]):
            memory.store(confusion, [s[i] for s in step])
    class gather_traces_callback(Callback):
        def __init__(self):
            pass
        def on_epoch_end(self, epoch, logs=None):
            gather_traces(
                        env, 
                        explore_policy, 
                        steps, 
                        observers = [store_new_batches], 
                        batches = explore_cycles)


    explore_policy = GreedyExplorePolicy(agent, sticky_epsilon)
    batches_till_full = int(capacity / parallel_environments)
    gather_traces(env, explore_policy, steps, observers = [store_new_batches], batches = batches_till_full)
    
    replay_dataset = get_replay_dataset(memory, batch_size)
    for args in replay_dataset:
        for arg, name in zip(args, ["batch_idx", "obs", "act", "rew", "discount", "ISWeights"]):
            print(f"{name}: {arg}")
        agent.train_step(*args)
    agent.compile("adam")
    agent.fit(
            replay_dataset, 
            epochs = epochs, 
            steps_per_epoch = train_cycles, 
            callbacks = [
                prioritized_update_callback(memory, agent),
                gather_traces_callback()])

def make_batched_environment(env_fn, parallel, *args, **kwargs):
    return BatchedPyEnvironment([env_fn(*args, **kwargs) for _ in range(parallel)])