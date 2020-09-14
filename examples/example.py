# Based on https://keras.io/examples/rl/ddpg_pendulum/

## Get rid of some very verbose logging of TF
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import gym
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import random
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
from scipy.stats import sem
import datetime
import ray
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining
import json
from pprint import pprint
from SpringBoxMixing.trainable import DDPG_Trainable
import sys


if __name__ == "__main__":
    IS_ON_CLUSTER = ('dominiks' in os.environ['HOME']) # Bad proxy for this property, but up to now didn't find anything better
    if IS_ON_CLUSTER:
        ray.init(num_cpus=int(os.environ.get('SLURM_NTASKS', '1')), include_dashboard=False, temp_dir = '/tmp/ray-dominiks')
    else:
        ray.init(num_cpus=2)

    # Hyper-Hyper parameters
    epochs_per_generation = 25
    population_size = 32
    num_generations = 10

    hyperparam_mutations = dict()
    hyperparam_mutations["actor_lr"] = np.geomspace(1e-5, 1e-1, 9).tolist()
    hyperparam_mutations["critic_lr"] = np.geomspace(1e-5, 1e-1, 9).tolist()
    hyperparam_mutations["THRESH"] = np.linspace(.01,.99, 15).tolist()
    hyperparam_mutations["noise_std_dev"] = np.linspace(.01, .91, 4).tolist()

    hyperparam_mutations["actor_n_layers"        ] = [1,2,3              ]
    hyperparam_mutations["actor_channels_0"      ] = [8,16,32,64,128     ]
    hyperparam_mutations["actor_channels_1"      ] = [8,16,32,64,128     ]
    hyperparam_mutations["actor_channels_2"      ] = [8,16,32,64,128     ]
    hyperparam_mutations["actor_kernel_0"        ] = [2,3,4              ]
    hyperparam_mutations["actor_kernel_1"        ] = [2,3,4              ]
    hyperparam_mutations["actor_kernel_2"        ] = [2,3,4              ]
    hyperparam_mutations["actor_kernel_final"    ] = [2,3,4              ]
    hyperparam_mutations["critic_n_conv_layers"  ] = [1,2,3              ]
    hyperparam_mutations["critic_channels_0"     ] = [8,16,32,64,128     ]
    hyperparam_mutations["critic_channels_1"     ] = [8,16,32,64,128     ]
    hyperparam_mutations["critic_channels_2"     ] = [8,16,32,64,128     ]
    hyperparam_mutations["critic_kernel_0"       ] = [2,3,4              ]
    hyperparam_mutations["critic_kernel_1"       ] = [2,3,4              ]
    hyperparam_mutations["critic_kernel_2"       ] = [2,3,4              ]
    hyperparam_mutations["critic_n_dense_layers" ] = [1,2,3              ]
    hyperparam_mutations["critic_dense_0"        ] = [8,16,32,64,128,256 ]
    hyperparam_mutations["critic_dense_1"        ] = [8,16,32,64,128,256 ]
    hyperparam_mutations["critic_dense_2"        ] = [8,16,32,64,128,256 ]

    schedule = PopulationBasedTraining(
            time_attr='epoch',
            metric='episode_reward',
            mode='max',
            perturbation_interval=epochs_per_generation,
            hyperparam_mutations=hyperparam_mutations)

    ## If this code throws an error bytes has no readonly flag, comment out a line in cloudpickle_fast (see this discussion: https://github.com/ray-project/ray/issues/8262)

    BASE_DIR = './raytune' ## TODO make the data part of this basedir...
    for resume in [True, False]:
        try:
            tune.run(DDPG_Trainable,
                     verbose=0,
                     local_dir=BASE_DIR,
                     config = dict(
                             total_episodes = epochs_per_generation,
                             n_epochs = epochs_per_generation,
                             grid_size = 16,
                             probability_to_make_video = 0,
                             do_video = False,
                             light_density_punishment = .01,
                             THRESH = tune.sample_from(lambda _: random.choice(hyperparam_mutations['THRESH'])),
                             noise_std_dev = tune.sample_from(lambda _: random.choice(hyperparam_mutations['noise_std_dev'])),
                             actor_n_layers       = tune.sample_from(lambda _: random.choice(hyperparam_mutations["actor_n_layers"        ])),
                             actor_channels_0     = tune.sample_from(lambda _: random.choice(hyperparam_mutations["actor_channels_0"      ])),
                             actor_channels_1     = tune.sample_from(lambda _: random.choice(hyperparam_mutations["actor_channels_1"      ])),
                             actor_channels_2     = tune.sample_from(lambda _: random.choice(hyperparam_mutations["actor_channels_2"      ])),
                             actor_kernel_0       = tune.sample_from(lambda _: random.choice(hyperparam_mutations["actor_kernel_0"        ])),
                             actor_kernel_1       = tune.sample_from(lambda _: random.choice(hyperparam_mutations["actor_kernel_1"        ])),
                             actor_kernel_2       = tune.sample_from(lambda _: random.choice(hyperparam_mutations["actor_kernel_2"        ])),
                             actor_kernel_final   = tune.sample_from(lambda _: random.choice(hyperparam_mutations["actor_kernel_final"    ])),
                             critic_n_conv_layers = tune.sample_from(lambda _: random.choice(hyperparam_mutations["critic_n_conv_layers"  ])),
                             critic_channels_0    = tune.sample_from(lambda _: random.choice(hyperparam_mutations["critic_channels_0"     ])),
                             critic_channels_1    = tune.sample_from(lambda _: random.choice(hyperparam_mutations["critic_channels_1"     ])),
                             critic_channels_2    = tune.sample_from(lambda _: random.choice(hyperparam_mutations["critic_channels_2"     ])),
                             critic_kernel_0      = tune.sample_from(lambda _: random.choice(hyperparam_mutations["critic_kernel_0"       ])),
                             critic_kernel_1      = tune.sample_from(lambda _: random.choice(hyperparam_mutations["critic_kernel_1"       ])),
                             critic_kernel_2      = tune.sample_from(lambda _: random.choice(hyperparam_mutations["critic_kernel_2"       ])),
                             critic_n_dense_layers= tune.sample_from(lambda _: random.choice(hyperparam_mutations["critic_n_dense_layers" ])),
                             critic_dense_0       = tune.sample_from(lambda _: random.choice(hyperparam_mutations["critic_dense_0"        ])),
                             critic_dense_1       = tune.sample_from(lambda _: random.choice(hyperparam_mutations["critic_dense_1"        ])),
                             critic_dense_2       = tune.sample_from(lambda _: random.choice(hyperparam_mutations["critic_dense_2"        ])),
                             buffer_capacity = 50000,
                             batch_size=32,
                             num_generations = num_generations,
                             actor_lr =  tune.sample_from(lambda _: random.choice(hyperparam_mutations['actor_lr'])),
                             critic_lr = tune.sample_from(lambda _: random.choice(hyperparam_mutations['critic_lr'])),
                         ),
                     scheduler = schedule,
                     stop = {'training_iteration': num_generations*epochs_per_generation},
                     resources_per_trial={'cpu': 1},
                     num_samples=population_size,
                     resume=resume,
                     global_checkpoint_period=60,
                     #fail_fast=True,
                    )
            break
        except ValueError:
            print('No checkpoint data found! Starting a new one!')
