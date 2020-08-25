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
from env import SpringBoxEnv
import sys
from tqdm import tqdm
from scipy.stats import sem
import datetime
import ray
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining
import json
from pprint import pprint

from OUnoise import OUActionNoise

BASE_DIR = './raytune'

class DDPG_Trainable(tune.Trainable):
    def setup(self, config):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        import tensorflow as tf
        self.config = config
        self.N = config['total_episodes']
        self.std_dev = config['noise_std_dev']

        self.THRESH = config['THRESH']

        # Environment variables
        self.env = SpringBoxEnv(grid_size=self.config['grid_size'], THRESH=self.THRESH)
        self.env.reset()
        self.num_states = self.env.observation_space.shape
        self.num_actions = self.env.action_space.shape
        self.upper_bound = self.env.action_space.high[0][0]
        self.lower_bound = self.env.action_space.low[0][0]

        ## To store reward history of each episode
        self.ep_reward_list = []
        ## To store average reward history of last few episodes
        self.avg_reward_list = []
        self.n=0

        ## Hyperparams 

        # Discount factor for future rewards
        self.gamma = 0.99
        # Used to update target networks
        self.tau = 0.005

        ## Training hyperparameters
        # Learning rate for actor-critic models
        self.actor_lr = config['actor_lr']
        self.critic_lr = config['critic_lr']

        ## Setup buffer
        self.buffer_capacity = config['buffer_capacity']
        self.batch_size = config['batch_size']
        # Its tells us num of times record() was called.
        self.buffer_counter = 0
        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer      = np.zeros((self.buffer_capacity, *self.num_states ))
        self.action_buffer     = np.zeros((self.buffer_capacity, *self.num_actions))
        self.reward_buffer     = np.zeros((self.buffer_capacity, 1                ))
        self.next_state_buffer = np.zeros((self.buffer_capacity, *self.num_states ))


        def get_actor(): # TODO make them a bit more flexible (and probably smaller)
            inputs = layers.Input(shape=self.num_states)
            out = layers.Conv2D( 64, 3, activation='relu', padding='same', data_format="channels_first")(inputs)
            out = layers.BatchNormalization()(out)
            out = layers.Conv2D( 128, 3, activation='relu', padding='same', data_format="channels_first")(out)
            out = layers.BatchNormalization()(out)
            out = layers.Conv2D( 16, 3, activation='relu', padding='same', data_format="channels_first")(out)
            out = layers.BatchNormalization()(out)
            out = layers.Conv2D( 1, 3, activation='sigmoid', padding='same', data_format="channels_first")(out)
            # Our upper bound is 2.0 for Pendulum.
            out = out * self.upper_bound
            model = tf.keras.Model(inputs, out)
            return model


        def get_critic(): # TODO Same as actor
            # Inputs
            state_input = layers.Input(shape=self.num_states)
            action_input = layers.Input(shape=(1,*self.num_actions)) ## [BATCH_SIZE (None), Channels (1), grid_size_x, grid_size_y] where BATCH_SIZE is inferred at runtime

            # Concatenate
            out = tf.keras.layers.Concatenate(axis=1)([state_input, action_input])

            # Apply convolutions
            out = layers.Conv2D( 64, 3, activation='relu', padding='same', data_format="channels_first")(out)
            out = layers.BatchNormalization()(out)
            out = layers.Conv2D( 64, 3, activation='relu', padding='same', data_format="channels_first")(out)
            out = layers.BatchNormalization()(out)

            out = layers.Flatten()(out)

            out = layers.Dense(512, activation='relu')(out)
            out = layers.Dense(1, activation=None)(out) ## Linear layer

            # Outputs single value for give state-action
            model = tf.keras.Model([state_input, action_input], out)

            return model
        def get_critic_old(): ## TODO Treat state_input and action_input the same by concatenating them along the first dimension
            # tf.keras.layers.Concatenate(axis=1)([state_input, action_input])
            # State as input
            state_input = layers.Input(shape=self.num_states)
            state_out = layers.Conv2D( 64, 3, activation='relu', padding='same', data_format="channels_first")(state_input)
            state_out = layers.BatchNormalization()(state_out)
            state_out = layers.Flatten()(state_out)

            # # Action as input
            action_input = layers.Input(shape=(1,*self.num_actions)) ## [BATCH_SIZE (None), Channels (1), grid_size_x, grid_size_y] where BATCH_SIZE is inferred at runtime
            action_out = layers.Conv2D( 64, 3, activation='relu', padding='same', data_format="channels_first")(action_input)
            action_out = layers.BatchNormalization()(action_out)
            action_out = layers.Flatten()(action_out)

            # Both are passed through seperate layer before concatenating
            out = layers.Concatenate()([state_out, action_out])
            out = layers.Dense(512, activation='relu')(out)
            out = layers.Dense(1, activation=None)(out) ## Linear layer

            # Outputs single value for give state-action
            model = tf.keras.Model([state_input, action_input], out)

            return model



        ## Initialize models
        #self.learning_buffer = LearningBuffer(self.gamma)
        self.actor = get_actor()
        self.critic = get_critic()
        self.target_actor = get_actor()
        self.target_critic = get_critic()
        # Making the weights equal initially
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())
        self.noise = OUActionNoise(mean=np.zeros(self.num_actions), std_deviation=float(self.std_dev) * np.ones(self.num_actions)) 

        ## Initialize optimizer
        self.critic_optimizer = tf.keras.optimizers.Adam(self.critic_lr, clipnorm=1.0)
        self.actor_optimizer = tf.keras.optimizers.Adam(self.actor_lr, clipnorm=1.0)
        # Compile to make them saveable
        self.actor.compile(self.actor_optimizer)
        self.critic.compile(self.critic_optimizer)
        self.target_actor.compile(self.actor_optimizer)
        self.target_critic.compile(self.critic_optimizer)

    # This update target parameters slowly
    # Based on rate `tau`, which is much less than one.
    def update_target(self):
        new_weights = []
        target_variables = self.target_critic.weights
        for i, variable in enumerate(self.critic.weights):
            new_weights.append(variable * self.tau + target_variables[i] * (1 - self.tau))

        self.target_critic.set_weights(new_weights)

        new_weights = []
        target_variables = self.target_actor.weights
        for i, variable in enumerate(self.actor.weights):
            new_weights.append(variable * self.tau + target_variables[i] * (1 - self.tau))

        self.target_actor.set_weights(new_weights)
        
    def step(self): # Play one game ## Should only do one step of a game
        prev_state = self.env.reset()
        episodic_reward = 0
        ep_frame = 0
        while True: # Play one game
            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

            action = self.policy(tf_prev_state)[0]

            # Recieve state and reward from environment.
            state, reward, done, info = self.env.step(action)

            self.record((prev_state, action, reward, state))

            episodic_reward += reward

            self.learn()
            self.update_target()

            if done: # or ep_frame >= 20-1: # TODO get this number out of a proper config file
                break

            prev_state = state
            ep_frame += 1

        self.ep_reward_list.append(episodic_reward)
        self.n += 1

        return {
                "epoch": self.n,
                "total_reward": self.ep_reward_list,
                "avg_reward": self.ep_reward_list[-1],
            }

    def policy(self, state):
        sampled_actions = self.actor(state)
        sampled_actions = tf.squeeze(sampled_actions).numpy()
        noise = self.noise()
        sampled_actions = sampled_actions + noise

        # We make sure action is within bounds
        legal_action = np.clip(sampled_actions, self.lower_bound, self.upper_bound)
        return [np.squeeze(legal_action)]
    
    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity
        self.state_buffer[index]      = obs_tuple[0]
        self.action_buffer[index]     = obs_tuple[1]
        self.reward_buffer[index]     = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]
        self.buffer_counter += 1


    def learn(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch      = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch     = tf.convert_to_tensor(self.action_buffer[batch_indices])
        action_batch     = tf.expand_dims(action_batch, axis=1)
        reward_batch     = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch     = tf.cast(reward_batch, dtype= tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_state_batch)
            y = reward_batch + self.gamma * self.target_critic([next_state_batch, target_actions])
            critic_value = self.critic([state_batch, action_batch])
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(critic_grad, self.critic.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = self.actor(state_batch)
            critic_value = self.critic([state_batch, actions])
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor.trainable_variables)
        )

    def save_checkpoint(self, checkpoint_dir):
        self.actor.save(f'{checkpoint_dir}/actor.model')
        self.critic.save(f'{checkpoint_dir}/critic.model')
        self.target_actor.save(f'{checkpoint_dir}/target_actor.model')
        self.target_critic.save(f'{checkpoint_dir}/target_critic.model')
        save_dict = {
                'config':self.config,
                'ep_reward_list':self.ep_reward_list,
                'actor_lr': self.actor_lr,
                'critic_lr': self.critic_lr,
                'n':self.n,
                'THRESH': self.THRESH, 
                }
        with open(f'{checkpoint_dir}/other_data.json', 'w') as f:
            json.dump(save_dict, f)
        return checkpoint_dir

    def load_checkpoint(self, checkpoint_dir):
        self.actor = tf.keras.models.load_model(f'{checkpoint_dir}/actor.model')
        self.critic = tf.keras.models.load_model(f'{checkpoint_dir}/critic.model')
        self.target_actor = tf.keras.models.load_model(f'{checkpoint_dir}/target_actor.model')
        self.target_critic = tf.keras.models.load_model(f'{checkpoint_dir}/target_critic.model')
        with open(f'{checkpoint_dir}/other_data.json', 'r') as f:
            save_dict = json.load(f)
        self.config = save_dict['config']
        self.ep_reward_list = save_dict['ep_reward_list']
        self.n = save_dict['n']
        self.actor_lr = save_dict['actor_lr']
        self.critic_lr = save_dict['critic_lr']
        self.THRESH = save_dict['THRESH']
        self.env = SpringBoxEnv(grid_size=self.config['grid_size'], THRESH=self.THRESH) # Required since self.THRESH is possibly not initialized
        self.env.reset()
        self.critic_optimizer = tf.keras.optimizers.Adam(self.critic_lr, clipnorm=1.0)
        self.actor_optimizer = tf.keras.optimizers.Adam(self.actor_lr, clipnorm=1.0)
        # Compile to make them saveable
        self.actor.compile(self.actor_optimizer)
        self.critic.compile(self.critic_optimizer)
        self.target_actor.compile(self.actor_optimizer)
        self.target_critic.compile(self.critic_optimizer)

if __name__ == "__main__":
    ray.init(num_cpus=1, num_gpus=1)

    # Hyper-Hyper parameters
    epochs_per_generation = 10
    population_size = 10
    num_generations = 4

    hyperparam_mutations = dict()
    hyperparam_mutations["actor_lr"] = np.geomspace(1e-5, 1e-1, 9).tolist()
    hyperparam_mutations["critic_lr"] = np.geomspace(1e-5, 1e-1, 9).tolist()
    hyperparam_mutations["THRESH"] = np.linspace(.01,.99, 15).tolist()

    schedule = PopulationBasedTraining(
            time_attr='epoch',
            metric='avg_reward',
            mode='max',
            perturbation_interval=epochs_per_generation,
            hyperparam_mutations=hyperparam_mutations)

    ## If this code throws an error bytes has no readonly flag, comment out a line in cloudpickle_fast (see this discussion: https://github.com/ray-project/ray/issues/8262)
    for resume in [True, False]:
        try: 
            tune.run(DDPG_Trainable,
                     verbose=1,
                     local_dir=BASE_DIR,
                     config = dict(
                             total_episodes = epochs_per_generation,
                             n_epochs = epochs_per_generation,
                             grid_size = 16,
                             THRESH = tune.sample_from(lambda _: random.choice(hyperparam_mutations['THRESH'])),
                             noise_std_dev = .2,
                             buffer_capacity = 50000,
                             batch_size=32,
                             num_generations = num_generations,
                             actor_lr =  tune.sample_from(lambda _: random.choice(hyperparam_mutations['actor_lr'])),
                             critic_lr = tune.sample_from(lambda _: random.choice(hyperparam_mutations['critic_lr'])),
                         ),
                     scheduler = schedule,
                     stop = {'training_iteration': num_generations*epochs_per_generation},
                     resources_per_trial={'gpu': 1, 'cpu': 1},
                     num_samples=population_size,
                     resume=resume,
                     global_checkpoint_period=60,
                    )
            break
        except ValueError:
            print('No checkpoint data found! Starting a new one!')
