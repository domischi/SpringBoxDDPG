
# Based on https://keras.io/examples/rl/ddpg_pendulum/

import gym
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from env import SpringBoxEnv
import sys
from tqdm import tqdm
from scipy.stats import sem

grid_size = 10

THRESH = 0.5

env = SpringBoxEnv(grid_size=grid_size, THRESH=THRESH)

num_states = env.observation_space.shape

print("Size of State Space ->  {}".format(num_states))
num_actions = env.action_space.shape[0]
print("Size of Action Space ->  {}".format(num_actions))


upper_bound = env.action_space.high[0][0]
lower_bound = env.action_space.low[0][0]


print("Max Value of Action ->  {}".format(upper_bound))
print("Min Value of Action ->  {}".format(lower_bound))



class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


class Buffer:
    def __init__(self, buffer_capacity=100000, batch_size=64):
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element

        self.state_buffer      = np.zeros((self.buffer_capacity, *num_states ))
        self.action_buffer     = np.zeros((self.buffer_capacity, *num_actions))
        self.reward_buffer     = np.zeros((self.buffer_capacity, 1           ))
        self.next_state_buffer = np.zeros((self.buffer_capacity, *num_states ))

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

    # We compute the loss and update parameters
    def learn(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch      = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch     = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch     = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch     = tf.cast(reward_batch, dtype= tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = target_actor(next_state_batch)
            y = reward_batch + gamma * target_critic([next_state_batch, target_actions])
            critic_value = critic_model([state_batch, action_batch])
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(critic_grad, critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:  # actor_model.summary(line_length=None, positions=None, print_fn=None)
            # critic_model.summary(line_length=None, positions=None, print_fn=None)

            actions = actor_model(state_batch)
            critic_value = critic_model([state_batch, actions])
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(actor_grad, actor_model.trainable_variables)
        )


# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
def update_target(tau):
    new_weights = []
    target_variables = target_critic.weights
    for i, variable in enumerate(critic_model.weights):
        new_weights.append(variable * tau + target_variables[i] * (1 - tau))

    target_critic.set_weights(new_weights)

    new_weights = []
    target_variables = target_actor.weights
    for i, variable in enumerate(actor_model.weights):
        new_weights.append(variable * tau + target_variables[i] * (1 - tau))

    target_actor.set_weights(new_weights)


"""
Here we define the Actor and Critic networks. These are basic Dense models
with `ReLU` activation. `BatchNormalization` is used to normalize dimensions across
samples in a mini-batch, as activations can vary a lot due to fluctuating values of input
state and action.
Note: We need the initialization for last layer of the Actor to be between
`-0.003` and `0.003` as this prevents us from getting `1` or `-1` output values in
the initial stages, which would squash our gradients to zero,
as we use the `tanh` activation.
"""


def get_actor():
    # Initialize weights between -3e-3 and 3-e3

    # last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
    last_init = tf.keras.initializers.GlorotNormal(seed=None)
    #
    # inputs = layers.Input(shape=(num_states,))
    # out = layers.Dense(512, activation="relu")(inputs)
    # out = layers.BatchNormalization()(out)
    # out = layers.Dense(512, activation="relu")(out)
    # out = layers.BatchNormalization()(out)
    # outputs = layers.Dense(num_actions[0]*num_actions[1], activation="tanh", kernel_initializer=last_init)(out)
    # outputs = layers.Reshape((num_actions[0],num_actions[1]))(outputs)
    #
    inputs = layers.Input(shape=num_states)
    # inputs1 = layers.Reshape((grid_size*10,grid_size*10,3))(inputs)
    # out = layers.BatchNormalization()(inputs)
    out = layers.Conv2D(
        64,
        3,
        strides=1,
        #activation="relu",
        #kernel_initializer=last_init,
        data_format="channels_first",
    )(inputs)
    out = layers.BatchNormalization()(out)
    out = layers.Conv2D(
        128, 4, strides=1, activation="relu", kernel_initializer=last_init
    )(out)
    out = layers.BatchNormalization()(out)
    outputs = layers.Conv2D(
        256, 5, strides=1, activation="relu", kernel_initializer=last_init
    )(out)
    out = layers.Flatten()(out)
    out = layers.BatchNormalization()(out)
    outputs = layers.Dense(
        num_actions[0] * num_actions[1], activation="relu", kernel_initializer=last_init
    )(out)
    outputs = layers.Reshape((num_actions[0], num_actions[1]))(outputs)
    # Our upper bound is 2.0 for Pendulum.
    outputs = outputs * upper_bound
    model = tf.keras.Model(inputs, outputs)
    return model


def get_critic():
    # State as input
    # state_input = layers.Input(shape=(num_states))
    # state_out = layers.Dense(16, activation="relu")(state_input)
    # state_out = layers.BatchNormalization()(state_out)
    # state_out = layers.Dense(32, activation="relu")(state_out)
    # state_out = layers.BatchNormalization()(state_out)
    state_input = layers.Input(shape=(num_states))
    state_out = layers.Reshape((grid_size*10,grid_size*10,1))(state_input)
    state_out = layers.Conv2D(128, 5, strides = 2, activation = "relu")(state_out)
    state_out = layers.Flatten()(state_out)
    state_out = layers.BatchNormalization()(state_out)

    state_out = layers.Dense(128, activation="relu")(state_input)
    state_out = layers.BatchNormalization()(state_out)
    state_out = layers.Dense(32, activation="relu")(state_out)
    state_out = layers.BatchNormalization()(state_out)

    #
    # state_input = layers.Input(shape = num_states)
    # #out = layers.Reshape((grid_size*10,grid_size*10,1))(state_input)
    # out = layers.Conv2D(64, 3 ,strides = 2, activation="relu")(state_input)
    # out = layers.BatchNormalization()(out)
    # out = layers.Conv2D(128, 3 ,strides = 2, activation="relu")(out)
    # out = layers.BatchNormalization()(out)
    # out = layers.Flatten()(out)
    # out = layers.Dense(32, activation="relu")(out)
    # state_out = layers.BatchNormalization()(out)
    #
    # # Action as input
    action_input = layers.Input(shape=num_actions)
    action_input1 = layers.Flatten()(action_input)
    out = layers.Dense(128, activation="relu")(action_input1)
    out = layers.BatchNormalization()(out)
    action_out = layers.Dense(32, activation="relu")(out)
    action_out = layers.BatchNormalization()(action_out)

    #
    # action_input = layers.Input(shape = num_actions)
    # out = layers.Reshape((grid_size,grid_size,1))(action_input)
    # out = layers.Conv2D(64, 3 ,strides = 2, activation="relu")(out)
    # out = layers.BatchNormalization()(out)
    # out = layers.Conv2D(128, 3 ,strides = 2, activation="relu")(out)
    # out = layers.BatchNormalization()(out)
    # out = layers.Flatten()(out)
    # out = layers.Dense(32, activation="relu")(out)
    # action_out = layers.BatchNormalization()(out)


    # Both are passed through seperate layer before concatenating
    concat = layers.Concatenate()([state_out, action_out])

    out = layers.Dense(512, activation="relu")(concat)
    out = layers.BatchNormalization()(out)
    out = layers.Dense(512, activation="relu")(out)
    out = layers.BatchNormalization()(out)
    outputs = layers.Dense(1)(out)

    # Outputs single value for give state-action
    model = tf.keras.Model([state_input, action_input], outputs)

    return model


"""
`policy()` returns an action sampled from our Actor network plus some noise for
exploration.
"""


def policy(state, noise_object):
    sampled_actions = actor_model(state)
    sampled_actions = tf.squeeze(sampled_actions)

    noise = noise_object()
    sampled_actions = sampled_actions.numpy() + noise

    # We make sure action is within bounds
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)
    return [np.squeeze(legal_action)]


"""
## Training hyperparameters
"""

std_dev = 0.2
ou_noise = OUActionNoise(
    mean=np.zeros(num_actions), std_deviation=float(std_dev) * np.ones(num_actions)
)


actor_model = get_actor()
critic_model = get_critic()

target_actor = get_actor()
target_critic = get_critic()

# Making the weights equal initially
target_actor.set_weights(actor_model.get_weights())
target_critic.set_weights(critic_model.get_weights())


reload =False
if reload:
    actor_model.load_weights("SpringBox_actor.h5")
    critic_model.load_weights("SpringBox_critic.h5")

    target_actor.load_weights("SpringBox_target_actor.h5")
    target_critic.load_weights("SpringBox_target_critic.h5")

# Learning rate for actor-critic models
critic_lr = 0.0002
actor_lr = 0.0001

critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

total_episodes = 100
#total_episodes = 5

# Discount factor for future rewards
gamma = 0.99
# Used to update target networks
tau = 0.005

buffer = Buffer(50000, 64)

"""
Now we implement our main training loop, and iterate over episodes.
We sample actions using `policy()` and train with `learn()` at each time step,
along with updating the Target networks at a rate `tau`.
"""

# To store reward history of each episode
ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []
std_reward_list = []

reload = False

if reload:
    actor_model.load_weights("SpringBox_actor.h5")
    critic_model.load_weights("SpringBox_critic.h5")

    target_actor.load_weights("SpringBox_target_actor.h5")
    target_critic.load_weights("SpringBox_target_critic.h5")

# Takes about 20 min to train
render = False
for ep in tqdm(range(total_episodes)):

    prev_state = env.reset()
    episodic_reward = 0
    ep_frame = 0


    while True:
        # Uncomment this to see the Actor in action
        # But not in a python notebook.
        # env.render()
        if render:
            env.render()

        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
        action = policy(tf_prev_state, ou_noise)[0]

        # Recieve state and reward from environment.
        state, reward, done, info = env.step(action)

        buffer.record((prev_state, action, reward, state))
        episodic_reward += reward

        buffer.learn()
        update_target(tau)

        if done or ep_frame > 18
            break

        prev_state = state
        ep_frame += 1

    ep_reward_list.append(episodic_reward)


    render = False
    #if ep % 5 == 0:
    #    render = True
    avg_reward = np.mean(ep_reward_list[-40:])
    std_reward = sem(ep_reward_list[-40:])
    #if ep % 1 == 0:
    #    #print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
    avg_reward_list.append(avg_reward)
    std_reward_list.append(std_reward)

    if ep % 10 == 0:
        actor_model.save_weights("SpringBox_actor.h5")
        critic_model.save_weights("SpringBox_critic.h5")

        target_actor.save_weights("SpringBox_target_actor.h5")
        target_critic.save_weights("SpringBox_target_critic.h5")


# Plotting graph
# Episodes versus Avg. Rewards
plt.errorbar(range(total_episodes),avg_reward_list, std_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Epsiodic Reward")
plt.tight_layout()
plt.savefig('avg_reward.png')
