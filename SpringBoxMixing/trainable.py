# Based on https://keras.io/examples/rl/ddpg_pendulum/

## Get rid of some very verbose logging of TF
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import gym
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import random
from ray import tune
import json
from .env import SpringBoxEnv
from .OUnoise import OUActionNoise

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
        self.env = SpringBoxEnv(grid_size=self.config['grid_size'], THRESH=self.THRESH, PROB_VIDEO=config['probability_to_make_video'], light_density_punishment = config['light_density_punishment'] )
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


        def get_actor(config):
            inputs = layers.Input(shape=self.num_states)
            for i in range(config['actor_n_layers']):
                out = layers.Conv2D( config[f'actor_channels_{i}'], config[f'actor_kernel_{i}'], activation='relu', padding='same')(inputs)
                out = layers.BatchNormalization()(out)
            out = layers.Conv2D( 1, config[f'actor_kernel_final'], activation='sigmoid', padding='same')(out)
            model = tf.keras.Model(inputs, out)
            return model


        def get_critic(config):
            # Inputs
            state_input = layers.Input(shape=self.num_states)
            action_input = layers.Input(shape=(*self.num_actions, 1)) ## [BATCH_SIZE (None), grid_size_x, grid_size_y, Channels (1)] where BATCH_SIZE is inferred at runtime

            # Concatenate
            out = tf.keras.layers.Concatenate(axis=-1)([state_input, action_input])

            # Apply convolutions
            for i in range(config['critic_n_conv_layers']):
                out = layers.Conv2D( config[f'critic_channels_{i}'], config[f'critic_kernel_{i}'], activation='relu', padding='same')(out)
                out = layers.BatchNormalization()(out)

            # Flatten
            out = layers.Flatten()(out)

            # Apply dense layers
            for i in range(config['critic_n_dense_layers']):
                out = layers.Dense(config[f'critic_dense_{i}'], activation='relu')(out)
            out = layers.Dense(1, activation=None)(out) ## Linear layer

            # Outputs single value for give state-action
            model = tf.keras.Model([state_input, action_input], out)

            return model

        ## Initialize models
        self.actor = get_actor(self.config)
        self.critic = get_critic(self.config)
        self.target_actor = get_actor(self.config)
        self.target_critic = get_critic(self.config)
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
        rewards = []
        mixing_reward = []
        light_sparsity_reward = []
        ep_frame = 0
        while True: # Play one game
            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

            action = self.policy(tf_prev_state)[0]

            # Recieve state and reward from environment.
            state, reward, done, info = self.env.step(action)

            self.record((prev_state, action, reward, state))

            rewards.append(reward)
            mixing_reward.append(info['mixing_reward'])
            light_sparsity_reward.append(info['light_sparsity_reward'])

            self.learn()
            self.update_target()

            if done:
                break

            prev_state = state
            ep_frame += 1

        print(f"Buffer filled: {self.buffer_counter}")

        self.ep_reward_list.append(rewards[-1])
        self.n += 1

        return {
                "epoch": self.n,
                "total_reward": self.ep_reward_list,
                "episode_reward": rewards[-1],
                "mixing_reward": np.mean(mixing_reward),
                "light_sparsity_reward": np.mean(light_sparsity_reward),
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
        for _ in range(int(record_range/self.batch_size)): # Do more than one batch (i.e. use as much of the data as possible)
            # Randomly sample indices
            batch_indices = np.random.choice(record_range, self.batch_size)

            # Convert to tensors
            state_batch      = tf.convert_to_tensor(self.state_buffer[batch_indices])
            action_batch     = tf.convert_to_tensor(self.action_buffer[batch_indices])
            action_batch     = tf.expand_dims(action_batch, axis=-1)
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
        ## in most situations the networks will change... Let's first find a proper network architecture, fix these values, then do PopBased training for the successful architecture
        #self.actor = tf.keras.models.load_model(f'{checkpoint_dir}/actor.model')
        #self.critic = tf.keras.models.load_model(f'{checkpoint_dir}/critic.model')
        #self.target_actor = tf.keras.models.load_model(f'{checkpoint_dir}/target_actor.model')
        #self.target_critic = tf.keras.models.load_model(f'{checkpoint_dir}/target_critic.model')
        with open(f'{checkpoint_dir}/other_data.json', 'r') as f:
            save_dict = json.load(f)
        self.config = save_dict['config']
        self.ep_reward_list = save_dict['ep_reward_list']
        self.n = save_dict['n']
        self.actor_lr = save_dict['actor_lr']
        self.critic_lr = save_dict['critic_lr']
        self.THRESH = save_dict['THRESH']
        self.env = SpringBoxEnv(grid_size=self.config['grid_size'], THRESH=self.THRESH, PROB_VIDEO=self.config['probability_to_make_video'], light_density_punishment = self.config['light_density_punishment']) # Required since self.THRESH is possibly not initialized
        self.env.reset()
        self.critic_optimizer = tf.keras.optimizers.Adam(self.critic_lr, clipnorm=1.0)
        self.actor_optimizer = tf.keras.optimizers.Adam(self.actor_lr, clipnorm=1.0)
        # Compile to make them saveable
        self.actor.compile(self.actor_optimizer)
        self.critic.compile(self.critic_optimizer)
        self.target_actor.compile(self.actor_optimizer)
        self.target_critic.compile(self.critic_optimizer)

