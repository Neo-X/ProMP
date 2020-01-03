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

def main():
    TASKS1=[0, 0.2, 0.4]
    TASKS2=[0, 0.2, 0.4, 0.6]
    # step one
    promphc.TASKS = TASKS1
    promphc.main()



if __name__=="__main__":
    main()
