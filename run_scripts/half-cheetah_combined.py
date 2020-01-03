import sys
sys.path.append("./")
from meta_policy_search.utils.PrompExperiment import PrompExperiment
experiment = PrompExperiment(api_key="KWwx7zh6I2uw6oQMkpEo3smu0",
                            project_name="ml4l3", workspace="glenb")

import sys
sys.path.append("./")
from meta_policy_search.baselines.linear_baseline import LinearFeatureBaseline
from meta_policy_search.envs.mujoco_envs.half_cheetah_rand_vel import HalfCheetahRandVelEnv
from meta_policy_search.envs.normalized_env import normalize
from meta_policy_search.meta_algos.pro_mp import ProMP
from meta_policy_search.meta_trainer import Trainer
from meta_policy_search.rl_trainer import RLTrainer
from meta_policy_search.samplers.meta_sampler import MetaSampler
from meta_policy_search.samplers.meta_sample_processor import MetaSampleProcessor
from meta_policy_search.policies.meta_gaussian_mlp_policy import MetaGaussianMLPPolicy
from meta_policy_search.utils import logger
from meta_policy_search.utils.utils import set_seed, ClassEncoder
import run_scripts.half_cheetah_rl as hcrl
import run_scripts.promp_run_mujoco_vel as promphc
import numpy as np
import tensorflow as tf
import os
import json
import argparse
import time
import pickle

meta_policy_search_path = '/'.join(os.path.realpath(os.path.dirname(__file__)).split('/')[:-1])

def main(config):
    TASKS1=[0, 0.2, 0.4]
    TASKS2=[0, 0.2, 0.4, 0.6]
    # step one
    config['n_itr'] = 1
    promphc.TASKS = TASKS1
    promphc.main(config)



if __name__=="__main__":
    config = {
        'seed': 1,

        'baseline': 'LinearFeatureBaseline',

        'env': 'HalfCheetahRandVelEnv',

        # sampler config
        'rollouts_per_meta_task': 20,
        'max_path_length': 100,
        'parallel': True,

        # sample processor config
        'discount': 0.99,
        'gae_lambda': 1,
        'normalize_adv': True,

        # policy config
        'hidden_sizes': (64, 64),
        'learn_std': True,  # whether to learn the standard deviation of the gaussian policy

        # ProMP config
        'inner_lr': 0.1,  # adaptation step size
        'learning_rate': 1e-3,  # meta-policy gradient step size
        'num_promp_steps': 5,  # number of ProMp steps without re-sampling
        'clip_eps': 0.3,  # clipping range
        'target_inner_step': 0.01,
        'init_inner_kl_penalty': 5e-4,
        'adaptive_inner_kl_penalty': False,  # whether to use an adaptive or fixed KL-penalty coefficient
        'n_itr': 0,  # number of overall training iterations
        'meta_batch_size': 40,  # number of sampled meta-tasks per iterations
        'num_inner_grad_steps': 1,  # number of inner / adaptation gradient steps

    }
    main()
