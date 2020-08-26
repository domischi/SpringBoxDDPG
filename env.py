import gym
from gym import spaces

from sacred import Experiment, SETTINGS
from sacred.dependencies import PackageDependency
from sacred.observers import FileStorageObserver, MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds
from functools import partial
from copy import deepcopy
from tqdm import tqdm as std_tqdm

tqdm = partial(std_tqdm, ncols=100)
import uuid
import os
import shutil
import numba
from numba.core.errors import NumbaWarning
import warnings
warnings.simplefilter("ignore", category=NumbaWarning)

import SpringBox
from SpringBox.integrator import integrate_one_timestep
from SpringBox.illustration import get_mixing_hists
from SpringBox.activation import *
from SpringBox.post_run_hooks import post_run_hooks
from SpringBox.measurements import (
    do_measurements,
    do_one_timestep_correlation_measurement,
    get_mixing_score,
)

import matplotlib
# matplotlib.use('tkagg')
import matplotlib.pyplot as plt

ex = Experiment("SpringBox")
# ex.observers.append(MongoObserver.create())
ex.observers.append(FileStorageObserver.create(f"data/"))
ex.dependencies.add(PackageDependency("SpringBox", SpringBox.__version__))


@ex.config
def cfg():
    ## Simulation parameters
    sweep_experiment = False
    mixing_experiment = True
    run_id = 0
    savefreq_fig = int(1e6) 
    savefreq_data_dump = 100000
    # Speeds up the computation somewhat, but incurs an error due to oversmoothing of fluids (which could however be somewhat physical)
    use_interpolated_fluid_velocities = True
    dt = 0.05
    T = 1.
    particle_density = 6.
    MAKE_VIDEO = False
    SAVEFIG = False
    const_particle_density = False
    measure_one_timestep_correlator = False
    periodic_boundary = True

    ## Geometry parameters / Activation Fn
    # activation_fn_type = 'const-rectangle' # For the possible choices, see the activation.py file
    activation_fn_type = "activation_matrix"
    #AR = 0.75
    L = 2
    n_part = int(particle_density * ((2 * L) ** 2))
    if mixing_experiment:
        assert n_part % 2 == 0

    ## Interaction parameters
    # Particle properties
    m_init = 1.0
    activation_decay_rate = 10.0  # Ex. at dt=0.01 this leads to an average deactivation of 10% of the particles
    # Spring properties
    spring_cutoff = 1.5
    spring_lower_cutoff = spring_cutoff / 25
    spring_k = 3.0
    spring_r0 = 0.2
    # LJ properties
    LJ_eps = 0.0
    LJ_r0 = 0.05
    LJ_cutoff = 2.5 / 1.122 * LJ_r0  # canonical choice
    # Brownian properties
    brownian_motion_delta = 0.0

    ## Fluid parameters
    mu = 10.0
    Rdrag = 0.0
    drag_factor = 1


def get_sim_info(old_sim_info, _config, i):
    sim_info = old_sim_info
    dt = _config["dt"]
    L = _config["L"]
    T = _config["T"]
    savefreq_fig = _config["savefreq_fig"]
    savefreq_dd = _config["savefreq_data_dump"]
    sim_info["t"] = i * dt
    sim_info["time_step_index"] = i
    sim_info["x_min"] = -L
    sim_info["y_min"] = -L
    sim_info["x_max"] = L
    sim_info["y_max"] = L
    sim_info["plotting_this_iteration"] = savefreq_fig != None and i % savefreq_fig == 0
    sim_info["data_dump_this_iteration"] = savefreq_dd != None and (
        i % savefreq_dd == 0 or i == int(T / dt) - 1
    )
    sim_info["get_fluid_velocity_this_iteration"] = (
        sim_info["plotting_this_iteration"] or sim_info["data_dump_this_iteration"]
    )
    sim_info["measure_one_timestep_correlator"] = (
        "measure_one_timestep_correlator" in _config.keys()
        and _config["measure_one_timestep_correlator"]
    )
    return sim_info


class SpringBoxEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    metadata = {"render.modes": ["human"]}

    def __init__(self, grid_size, THRESH, CAP=4):
        super(SpringBoxEnv, self).__init__()

        self.THRESH = THRESH
        self.CAP = CAP
        self.grid_size = grid_size
        self._config = cfg()

        run_id = self._config["run_id"]
        unique_id = str(uuid.uuid4())
        data_dir = f"/tmp/boxspring-{run_id}-{unique_id}"
        os.makedirs(data_dir)
        self.sim_info = {"data_dir": data_dir}
        self.sim_info = get_sim_info(self.sim_info, self._config, 0)

        ## Initialize particlesself.pXs> -0.2
        self.pXs = (
            (np.random.rand(self._config["n_part"], 2) - 0.5) * 2 * self._config["L"]
        )
        self.pXs[: self._config["n_part"] // 2, 0] = (
            -np.random.rand(self._config["n_part"] // 2) * self._config["L"]
        )
        self.pXs[self._config["n_part"] // 2 :, 0] = (
            +np.random.rand(self._config["n_part"] // 2) * self._config["L"]
        )
        self.pVs = np.zeros_like(self.pXs)
        self.acc = np.zeros(len(self.pXs))
        self.ms = self._config["m_init"] * np.ones(len(self.pXs))

        L = self._config["L"]
        self.X = (((np.indices((self.grid_size + 1,))[0]) / self.grid_size) * L * 2) - L
        self.Y = (((np.indices((self.grid_size + 1,))[0]) / self.grid_size) * L * 2) - L

        self.N_steps = int(self._config["T"] / self._config["dt"])
        self.current_step = 0

        # self.action_space = spaces.Box(low = 0, high = 1, shape = (self.grid_size * self.grid_size,))
        ## Example for using image as input:
        # self.observation_space = spaces.Box(low=0, high=1, shape= (self.grid_size*10 * self.grid_size*10,))
        self.action_space = spaces.Box(
            low=0, high=1, shape=(self.grid_size, self.grid_size,)
        )
        ## Example for using image as input:
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(2, self.grid_size, self.grid_size)
        )
        self.obs = np.zeros_like(self.observation_space.sample())

        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.sc = self.ax.scatter([], [])
        self.sc2 = self.ax.scatter([], [])
        plt.draw()

        self.lights = np.zeros(shape=(self.grid_size, self.grid_size))
        self.previous_score = None

    def calculate_obs(self):
        _, _, H1, H2 = get_mixing_hists(
            self.pXs, self.grid_size, self.sim_info, cap=self.CAP
        )
        return np.array([H1, H2])

    def sample_action(self):
        return self.action_space.sample()

    def sample_observation(self):
        return self.observation_space.sample()

    def step(self, action):
        done = False
        self.sim_info = get_sim_info(self.sim_info, self._config, self.current_step)

        A = (action.reshape((self.grid_size, self.grid_size)) > self.THRESH).astype(int)

        self.lights = np.copy(A)

        reward = 0

        if self.previous_score == None:
            obs = self.calculate_obs()
            self.previous_score = get_mixing_score(self.pXs, self._config)

        activation_fn = activation_fn_dispatcher(
            self._config, self.sim_info["t"], lx=self.X, ly=self.Y, lh=np.transpose(A)
        )

        # run every action pattern for 5 iterations
        for i in range(5):
            (
                self.pXs,
                self.pVs,
                self.acc,
                self.ms,
                self.fXs,
                self.fVs,
            ) = integrate_one_timestep(
                pXs=self.pXs,
                pVs=self.pVs,
                acc=self.acc,
                ms=self.ms,
                activation_fn=activation_fn,
                sim_info=self.sim_info,
                _config=self._config,
                get_fluid_velocity=self.sim_info["get_fluid_velocity_this_iteration"],
                use_interpolated_fluid_velocities=self._config[
                    "use_interpolated_fluid_velocities"
                ],
            )

        score = get_mixing_score(self.pXs, self._config)

        if self.current_step > self.N_steps:
            done = True
        obs = self.calculate_obs()
        self.current_step += 1

        reward = score - self.previous_score
        self.previous_score = score

        if done:
            self.clean_up()

        return obs, reward, done, {}
    
    def clean_up(self):
        shutil.rmtree(data_dir)

    def reset(self):
        self.pXs = (
            (np.random.rand(self._config["n_part"], 2) - 0.5) * 2 * self._config["L"]
        )
        self.pXs[: self._config["n_part"] // 2, 0] = (
            -np.random.rand(self._config["n_part"] // 2) * self._config["L"]
        )
        self.pXs[self._config["n_part"] // 2 :, 0] = (
            +np.random.rand(self._config["n_part"] // 2) * self._config["L"]
        )
        self.pVs = np.zeros_like(self.pXs)
        self.acc = np.zeros(len(self.pXs))
        self.ms = self._config["m_init"] * np.ones(len(self.pXs))
        self.obs = np.zeros_like(self.observation_space.sample())
        self._config = cfg()
        self.previous_score = None
        data_dir = f"/tmp/boxspring-{run_id}-{unique_id}"
        os.makedirs(data_dir)
        self.sim_info = {"data_dir": data_dir}
        self.sim_info = get_sim_info(self.sim_info, self._config, 0)
        self.current_step = 0

        obs = self.calculate_obs()

        return obs

    def render(self, mode="human", close=False, first=False):
        L = self._config["L"]
        self.ax.set_xlim(self.sim_info["x_min"], self.sim_info["x_max"])
        self.ax.set_ylim(self.sim_info["y_min"], self.sim_info["y_max"])

        split = len(self.pXs) // 2
        x = self.pXs[split:, 0]
        y = -self.pXs[split:, 1]

        x2 = self.pXs[:split, 0]
        y2 = -self.pXs[:split, 1]

        self.sc.set_offsets(np.c_[x, y])
        self.sc2.set_offsets(np.c_[x2, y2])

        self.ax.imshow(self.lights, extent=[-L, L, -L, L])

        self.fig.canvas.draw_idle()
        plt.pause(0.01)
