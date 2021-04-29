import numpy as np
import gym
from gym import spaces

import uuid
import os
import json
import random
import shutil

import warnings
try:
    from numba.core.errors import NumbaWarning
except ImportError:
    from numba.errors import NumbaWarning
warnings.simplefilter("ignore", category=NumbaWarning)

import SpringBox
from SpringBox.integrator import integrate_one_timestep
from SpringBox.illustration import get_mixing_hists, plot_mixing_on_axis, plot_light_pattern, generate_video_from_png, generate_gif_from_png
from SpringBox.activation import *
from SpringBox.post_run_hooks import post_run_hooks
from SpringBox.measurements import (
    do_measurements,
    do_one_timestep_correlation_measurement,
    get_mixing_score,
    store_dict_to_h5_by_filename
)

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
        spring_k_rep = 3.0,
        spring_r0 = 0,
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

    config['spring_lower_cutoff'] = config['spring_cutoff'] / 100,
    config['n_part'] = int(config['particle_density'] * ((2 * config['L']) ** 2))
    if config['mixing_experiment']:
        assert config['n_part'] % 2 == 0
    return config

def cfg(env_config):
    env_configs_dir = 'environment_configs' 
    uid = env_config['uid'] ## TODO replace with a get
    os.makedirs(env_configs_dir, exist_ok=True)
    config_file = f'{env_configs_dir}/{str(uid)}.json'
    if os.path.isfile(config_file):
        with open(config_file, 'r') as f:
            conf_dict = json.load(f)
    else:
        print("Did not find configuration file, generate a default one!")
        conf_dict = default_cfg()
        for k in env_config:
            if 'sim_config_' in k:
                k_sim = k.replace('sim_config_', '')
                conf_dict[k_sim]=env_config[k]
        ## make sure all interactions are loaded, but not more than necessary
        if env_config['do_attractive']:
            assert(conf_dict['spring_k']>0)
        else:
            conf_dict['spring_k']=0
        if env_config['do_repulsive']:
            assert(conf_dict['spring_k_rep']>0)
        else:
            conf_dict['spring_k_rep']=0
        with open(config_file, 'w') as f:
            json.dump(conf_dict, f, indent=4)
    if env_config.get('do_video', False):
        conf_dict['savefreq_fig'] = 1
        conf_dict['MAKE_VIDEO'] = True
    if env_config.get('do_data_dump', False):
        conf_dict['compute_update_matrix'] = True
        conf_dict['savefreq_data_dump'] = 1
        conf_dict['hdf_file'] = 'dd_tmp.h5'

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
    sim_info['compute_update_matrix'] = _config.get('do_')
    return sim_info


class SpringBoxEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    metadata = {"render.modes": ["human"]}

    def __init__(self, env_config):
        FLATTENED_SPACES = env_config.get("FLATTENED_SPACES", True)
        self.FLATTENED_OBSERVATION_SPACE = env_config.get("FLATTENED_OBSERVATION_SPACE", FLATTENED_SPACES)
        self.FLATTENED_ACTION_SPACE = env_config.get("FLATTENED_ACTION_SPACE", FLATTENED_SPACES)
        self.grid_size = env_config.get("grid_size",16)
        self.reward_scaling_factor = env_config.get("reward_scaling_factor",1)
        self.mixing_score_type = env_config.get("mixing_score_type") ## hist_quad, hist_lin, delaunay

        self.do_attractive = env_config.get("do_attractive", True)
        self.do_repulsive = env_config.get("do_repulsive", False)

        assert(self.mixing_score_type in ["hist_quad", "hist_lin", "delaunay"])
        super(SpringBoxEnv, self).__init__()

        self.do_data_dump = env_config.get('do_data_dump', False)
        if self.do_data_dump:
            self.update_matrix = None
        self.do_video = env_config['do_video']
        self.do_avi = env_config.get('do_avi', False)
        self._config = cfg(env_config)

        run_id = self._config["run_id"]
        unique_id = str(uuid.uuid4())
        data_dir = f"/tmp/boxspring-{run_id}-{unique_id}"
        os.makedirs(data_dir)
        self.sim_info = {"data_dir": data_dir}
        self.sim_info = get_sim_info(self.sim_info, self._config, 0)

        #self.CAP = env_config.get("CAP",int(self._config["n_part"]/(self.grid_size**2)*4))
        self.CAP = env_config.get("CAP",None)

        self.shift_rewards_0_to_1 = env_config.get("shift_rewards_0_to_1",False)

        ## Initialize particles
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

        high_val = self.CAP if (not self.CAP is None) else self._config["n_part"]//2
        self.min_action_value = -1 if self.do_repulsive  else 0
        self.max_action_value =  1 if self.do_attractive else 0
        observation_space_shape = (self.grid_size*self.grid_size*2,) if self.FLATTENED_OBSERVATION_SPACE else (self.grid_size, self.grid_size, 2)
        self.observation_space = spaces.Box( low=0, high=high_val, shape=observation_space_shape)
        if self.FLATTENED_ACTION_SPACE:
            self.action_space = spaces.Box(low=self.min_action_value, high=self.max_action_value, shape=(self.grid_size*self.grid_size,))
        else:
            self.action_space = spaces.Box(low=self.min_action_value, high=self.max_action_value, shape=(self.grid_size, self.grid_size,))
        self.obs = np.zeros_like((self.grid_size, self.grid_size, 2))
        self.lights = np.zeros_like(self.action_space.sample())

        ## Setup rewards
        # Set mix of rewards
        self.homogeneity_multiplier = env_config.get("homogeneity_multiplier")
        self.mixing_multiplier = env_config.get("mixing_multiplier")
        self.light_multiplier = env_config.get("light_multiplier")
        self.total_multipliers = self.homogeneity_multiplier + self.mixing_multiplier + self.light_multiplier 

        # Set stuff required for proper normalization of rewards
        self.avg_cell_cnt = self._config["n_part"]/(self.grid_size**2)
        self.max_inhomogeneity_score = (1-1/(self.grid_size**2))*self._config["n_part"]**2
        if self.mixing_score_type == "hist_quad":
            self.max_unmixing_score = (self._config["n_part"]/self.grid_size)**2
        if self.mixing_score_type == "hist_lin":
            self.max_unmixing_score = self._config["n_part"]

        self.set_hist_mixing_score_cap(env_config.get("hist_mixing_score_cap_factor", None))

        # Set the variables for scores and so on
        self.homogeneity_score = None
        self.mixing_score = None
        self.light_score = None
        self.homogeneity_reward = None
        self.mixing_reward = None
        self.light_reward = None
        self.total_reward = None

    def set_hist_mixing_score_cap(self, hsmcf=None):
        if hsmcf is None:
            self.hist_mixing_score_factor = self.grid_size**2
            self.hist_mixing_score_cap = None
        else:
            self.hist_mixing_score_factor = hsmcf
            self.hist_mixing_score_cap = int(np.ceil(self._config["n_part"]/self.grid_size**2 * hsmcf))

    def calculate_obs(self):
        _, _, H1, H2 = get_mixing_hists(
            self.pXs, self.grid_size, self.sim_info, cap=self.CAP
        )
        self.obs = np.stack([H1, H2], axis=-1) # Channels last
        return self.obs

    def sample_action(self):
        return self.action_space.sample()

    def sample_observation(self):
        return self.observation_space.sample()

    def plot_frame(self):
        fname=f"{self.sim_info['data_dir']}/frame_{self.current_step:03}.png"
        title=f"Step: {self.current_step:03}, Score: {self.total_reward:.4f}"
        fig = plt.figure(figsize=(5,5))
        plot_mixing_on_axis(plt.gca(), self.pXs, self.sim_info, title, fix_frame=True, SAVEFIG=False, ex=None, plot_density_map=False, nbins=self.grid_size, cap=self.CAP if (not self.CAP is None) else self.avg_cell_cnt*4, alpha=.85)
        plot_light_pattern(plt.gca(), self.lights, self.sim_info, alpha=.3) ## TODO also commit in SpringBox
        plt.gca().get_xaxis().set_visible(False)
        plt.gca().get_yaxis().set_visible(False)
        plt.tight_layout()

        plt.savefig(fname)
        plt.close(fig)

    def get_data_dump_location(self):
        return f"{self.sim_info['data_dir']}/dd.h5"

    def store_data_dump(self):
        fname = self.get_data_dump_location()
        iteration_group_name = f"iteration_{self.current_step:03}"
        d = {
                'pXs': self.pXs,
                'pVs': self.pVs,
                'acc': self.acc,
                'm'  : self.ms,
                'current_step' : self.current_step,
                'obs': self.obs,
                'activation': self.lights,
                'homogeneity_score'  :self.homogeneity_score   ,
                'mixing_score'       :self.mixing_score        ,
                'light_score'        :self.light_score         ,
                'homogeneity_reward' :self.homogeneity_reward  ,
                'mixing_reward'      :self.mixing_reward       ,
                'light_reward'       :self.light_reward        ,
                'total_reward'       :self.total_reward        ,
                }
        store_dict_to_h5_by_filename(d, fname, iteration_group_name)
        return fname


    def activate_do_video(self):
        self.do_video=True
    def deactivate_do_video(self):
        self.do_video=False
    def activate_data_dump(self):
        self.do_data_dump=True
        self.update_matrix = None
    def deactivate_data_dump(self):
        self.do_data_dump=False
        del self.update_matrix

    def set_multipliers(self, d):
        self.homogeneity_multiplier = d.get("homogeneity_multiplier", self.homogeneity_multiplier)
        self.mixing_multiplier      = d.get("mixing_multiplier"     , self.mixing_multiplier     )
        self.light_multiplier       = d.get("light_multiplier"      , self.light_multiplier      )
        self.total_multipliers =  self.homogeneity_multiplier + self.mixing_multiplier + self.light_multiplier 

    def collect_video(self):
        if self.do_avi:
            fname = generate_video_from_png(self.sim_info["data_dir"], destroyAllWindows=False)
        else: 
            fname = generate_gif_from_png(self.sim_info["data_dir"])
        if fname is None:
            raise RuntimeError("Wanted to collect video, but apparently this failed. Check the folder self.sim_info['data_dir']")
        return fname

    def step(self, action):
        done = False
        self.sim_info = get_sim_info(self.sim_info, self._config, self.current_step)

        if np.isnan(action).any():
            action = np.random.rand(self.grid_size, self.grid_size) # doing purely random should be sufficiently bad that it is not optimized for
            print('Warning: Caught an nan output in env.step...')
            print(f'Debug Info: {np.min(self.obs)} {np.max(self.obs)} {np.isnan(self.obs).any()}')
        self.lights = np.round(np.clip(action.reshape(self.grid_size, self.grid_size),
                           a_min = self.min_action_value,
                           a_max = self.max_action_value)).astype(int)
        assert(np.min(self.lights)>=-1 and np.max(self.lights)<=1 )

        if (self.homogeneity_score is None) or (self.mixing_score is None) or (self.light_score is None) or (self.total_reward is None):
            self.compute_rewards()

        if self.do_video:
            self.plot_frame()

        
        activation_fn = activation_fn_dispatcher(
            self._config, self.sim_info["t"], lx=self.X, ly=self.Y, lh=np.transpose(self.lights)
        )

        self.pXs, self.pVs, self.acc, self.ms, self.fXs, self.fVs, self.update_matrix = integrate_one_timestep(
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
                inverted_update=True,
                
            )

        self.current_step += 1
        if self.current_step >= self.N_steps:
            done = True

        self.calculate_obs() ## Has to be executed before compute_rewards to be able to determine homogeneity
        self.compute_rewards()

        if self.do_data_dump:
            self.store_data_dump()
        if done:
            self.clean_up()

        info_dir = {"mixing_score"           : self.mixing_score,
                    "mixing_reward"          : self.mixing_reward,
                    "mixing_multiplier"      : self.mixing_multiplier,
                    "light_score"            : self.light_score,
                    "light_reward"           : self.light_reward,
                    "light_multiplier"       : self.light_multiplier,
                    "homogeneity_score"      : self.homogeneity_score,
                    "homogeneity_reward"     : self.homogeneity_reward,
                    "homogeneity_multiplier" : self.homogeneity_multiplier,
                    "reward_multiplier"      : self.reward_scaling_factor,
                    "fraction_attractive_lights_activated": (self.lights>0).sum()/self.lights.size,
                    "fraction_repulsive_lights_activated": (self.lights<0).sum()/self.lights.size,
                    "avg_light": self.lights.sum()/self.lights.size,
                    "hist_mixing_score_factor": self.hist_mixing_score_factor,
                    "hist_mixing_score_cap"   : self.hist_mixing_score_cap ,
                    "total_reward_unscaled"  : self.total_reward/self.reward_scaling_factor,
                    }

        if self.FLATTENED_OBSERVATION_SPACE:
            return self.obs.flatten(), self.total_reward, done, info_dir
        else:
            return self.obs, self.total_reward, done, info_dir
    
    def compute_rewards(self):
        if self.mixing_score_type == "hist_quad" or self.mixing_score_type == "hist_lin":
            abs_diff_obs = abs(self.obs[:,:, 0]-self.obs[:,:,1])
            if not self.hist_mixing_score_cap is None:
                abs_diff_obs = np.clip(abs_diff_obs, a_min=0, a_max = self.hist_mixing_score_cap)
            if self.mixing_score_type == "hist_quad":
                abs_diff_obs = abs_diff_obs**2
            self.mixing_score = -np.sum(abs_diff_obs)/self.max_unmixing_score
            if self.shift_rewards_0_to_1:
                self.mixing_score+=1
        elif self.mixing_score_type == "delaunay":
            self.mixing_score      = get_mixing_score(self.pXs, self._config)
        else:
            raise RuntimeError(f"Unrecognized mixing_score_type: {self.mixing_score_type}")
        self.homogeneity_score = -np.sum((np.sum(self.obs, axis = -1)-self.avg_cell_cnt)**2)/self.max_inhomogeneity_score
        self.light_score       = -abs(self.lights).sum()/self.lights.size
        if self.shift_rewards_0_to_1:
            self.homogeneity_score+=1
            self.light_score      +=1
        self.mixing_score      /= self.N_steps
        self.homogeneity_score /= self.N_steps
        self.light_score       /= self.N_steps
        self.mixing_reward      = self.mixing_multiplier * self.mixing_score
        self.homogeneity_reward = self.homogeneity_multiplier * self.homogeneity_score
        self.light_reward       = self.light_multiplier * self.light_score
        self.total_reward = self.reward_scaling_factor*(self.mixing_reward + self.light_reward + self.homogeneity_reward) / self.total_multipliers


    def clean_up(self):
        if self.do_video:
            self.plot_frame()
            old_location = self.collect_video()
            if self.do_avi:
                new_location = f'video_{int(time.time())}.avi'
            else:
                new_location = f'video_{int(time.time())}.gif'
            shutil.move(old_location, new_location)
            print(f'Collected example video: {new_location}')
        if self.do_data_dump:
            old_location = self.get_data_dump_location()
            assert(not old_location is None)
            new_location = f'dd_tmp_{int(time.time())}.h5'
            shutil.move(old_location, new_location)
            print(f'Collected data_dump: {new_location}')
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
        self.homogeneity_score = None
        self.mixing_score = None
        self.light_score = None
        self.homogeneity_reward = None
        self.mixing_reward = None
        self.light_reward = None
        self.total_reward = None
        unique_id = str(uuid.uuid4())
        data_dir = f"/tmp/boxspring-{self._config['run_id']}-{unique_id}"
        os.makedirs(data_dir)
        self.sim_info = {"data_dir": data_dir}
        self.sim_info = get_sim_info(self.sim_info, self._config, 0)
        self.current_step = 0

        self.calculate_obs()

        if self.FLATTENED_OBSERVATION_SPACE:
            return self.obs.flatten()
        else:
            return self.obs
