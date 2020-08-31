import gym
from gym import spaces

import uuid
import os
import json
import random
import shutil
import numba
from numba.core.errors import NumbaWarning
import warnings
warnings.simplefilter("ignore", category=NumbaWarning)

import SpringBox
from SpringBox.integrator import integrate_one_timestep
from SpringBox.illustration import get_mixing_hists, plot_mixing_on_axis, plot_light_pattern, generate_video_from_png
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
import time

def default_cfg():
    config=dict(
        ## Simulation parameters
        sweep_experiment = False,
        mixing_experiment = True,
        run_id = 0,
        savefreq_fig = int(1e6) ,
        savefreq_data_dump = 100000,
        # Speeds up the computation somewhat, but incurs an error due to oversmoothing of fluids (which could however be somewhat physical)
        use_interpolated_fluid_velocities = True,
        dt = 0.05,
        T = 1.,
        particle_density = 6.,
        MAKE_VIDEO = False,
        SAVEFIG = False,
        const_particle_density = False,
        measure_one_timestep_correlator = False,
        periodic_boundary = True,

        ## Geometry parameters / Activation Fn
        # activation_fn_type = 'const-rectangle' # For the possible choices, see the activation.py file
        activation_fn_type = "activation_matrix",
        L = 2,
        ## Interaction parameters
        # Particle properties
        m_init = 1.0,
        activation_decay_rate = 10.0,  # Ex. at dt=0.01 this leads to an average deactivation of 10% of the particles
        # Spring properties
        spring_cutoff = 1.5,
        spring_k = 3.0,
        spring_r0 = 0.2,
        # LJ properties
        LJ_eps = 0.0,
        #LJ_r0 = 0.05
        #LJ_cutoff = 2.5 / 1.122 * LJ_r0  # canonical choice
        # Brownian properties
        brownian_motion_delta = 0.0,

        ## Fluid parameters
        mu = 10.0,
        Rdrag = 0.0,
        drag_factor = 1,
    )

    config['spring_lower_cutoff'] = config['spring_cutoff'] / 25,
    config['n_part'] = int(config['particle_density'] * ((2 * config['L']) ** 2))
    if config['mixing_experiment']:
        assert config['n_part'] % 2 == 0
    return config

def cfg(do_video=False):
    config_file = '../environment_config.json'
    if os.path.isfile(config_file):
        with open(config_file, 'r') as f:
            conf_dict = json.load(f)
    else:
        print("Did not find configuration file, generate a default one!")
        conf_dict = default_cfg()

        with open(config_file, 'w') as f:
            json.dump(conf_dict, f, indent=4)
    if do_video:
        conf_dict['savefreq_fig'] = 1
        conf_dict['MAKE_VIDEO'] = True
    return conf_dict


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

    def __init__(self, grid_size, THRESH, CAP=4, PROB_VIDEO=.1):
        super(SpringBoxEnv, self).__init__()

        self.THRESH = THRESH
        self.CAP = CAP
        self.grid_size = grid_size
        self.do_video = random.random() < PROB_VIDEO
        self._config = cfg(self.do_video)

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
            low=0, high=1, shape=(self.grid_size, self.grid_size, 2)
        )
        self.obs = np.zeros_like(self.observation_space.sample())

        self.lights = np.zeros(shape=(self.grid_size, self.grid_size))
        self.previous_score = None

    def calculate_obs(self):
        _, _, H1, H2 = get_mixing_hists(
            self.pXs, self.grid_size, self.sim_info, cap=self.CAP
        )
        return np.stack([H1, H2], axis=-1) # Channels last

    def sample_action(self):
        return self.action_space.sample()

    def sample_observation(self):
        return self.observation_space.sample()

    def plot_frame(self):
        fname=f"{self.sim_info['data_dir']}/frame_{self.current_step:03}.png"
        title=f"Step: {self.current_step:03}, Score: {self.previous_score:.4f}"
        fig = plt.figure(figsize=(5,5))
        plot_mixing_on_axis(plt.gca(), self.pXs, self.sim_info, title, fix_frame=True, SAVEFIG=False, ex=None, nbins=self.grid_size, cap=self.CAP, alpha=.85)
        plot_light_pattern(plt.gca(), self.lights, self.sim_info, alpha=.3)
        plt.gca().get_xaxis().set_visible(False)
        plt.gca().get_yaxis().set_visible(False)
        plt.tight_layout()

        plt.savefig(fname)
        plt.close(fig)


    def collect_video(self):
        fname = generate_video_from_png(self.sim_info["data_dir"])
        if fname is None:
            raise RuntimeError("Wanted to collect video, but apparently this failed. Check the folder self.sim_info['data_dir']")
        return fname

    def step(self, action):
        done = False
        self.sim_info = get_sim_info(self.sim_info, self._config, self.current_step)

        A = (action > self.THRESH).astype(int)

        self.lights = np.copy(A)

        if self.previous_score == None:
            obs = self.calculate_obs()
            self.previous_score = get_mixing_score(self.pXs, self._config)

        if self.do_video:
            self.plot_frame()


        activation_fn = activation_fn_dispatcher(
            self._config, self.sim_info["t"], lx=self.X, ly=self.Y, lh=np.transpose(A)
        )

        self.pXs, self.pVs, self.acc, self.ms, self.fXs, self.fVs, = integrate_one_timestep(
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
        if self.do_video:
            self.plot_frame()
            old_location = self.collect_video()
            new_location = f'../video_{int(time.time())}.avi'
            shutil.move(old_location, new_location)
            print(f'Collected example video: {new_location}')
        shutil.rmtree(self.sim_info['data_dir'])

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
        unique_id = str(uuid.uuid4())
        data_dir = f"/tmp/boxspring-{self._config['run_id']}-{unique_id}"
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
