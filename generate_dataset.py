"""
Generates a dataset of random-action CoinRun episodes.
WARNING: Default (10,000 episode) dataset is very large at >100GB.
"""

from dataclasses import dataclass

from gym3 import types_np
import numpy as np
from procgen import ProcgenGym3Env
import tyro


@dataclass
class Args:
    num_episodes: int = 10000


args = tyro.cli(Args)
data = []
for i in range(args.num_episodes):
    seed = np.random.randint(0, 10000)
    env = ProcgenGym3Env(num=1, env_name="coinrun", start_level=seed)
    step = 0
    dataseq = []
    for j in range(1000):
        env.act(types_np.sample(env.ac_space, bshape=(env.num,)))
        rew, obs, first = env.observe()
        step += 1
        dataseq.append(obs["rgb"])
    dataseq = np.concatenate(dataseq, axis=0)
    data.append(dataseq)
data_to_file = np.array(data)
np.save("data/coinrun.npy", data_to_file)
