import numpy as np
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.utils import ENV_REGISTRY, deep_merge_dicts
from easydict import EasyDict
import gymnasium as gym

@ENV_REGISTRY.register('health_metrics_lightzero')
class HealthMetricsEnvLightZero(BaseEnv):
    config = dict(
        env_id='health_metrics_optimization',
        max_step=50,  # Define how many steps in an episode
        health_metric_ranges={
            "bp_systolic": (90, 120),
            "bp_diastolic": (60, 80),
            "pulse": (60, 100),
            "temperature": (36.5, 37.5),
        },
        action_effects={
            0: {"bp_systolic": -10, "bp_diastolic": -5},
            1: {"pulse": +10},
            2: {"temperature": -0.5},
            3: {},  # No action
        },
        initial_health_state={
            "bp_systolic": 120,
            "bp_diastolic": 80,
            "pulse": 70,
            "temperature": 37.0,
        },
        reward_for_health_optimization=10,
    )

    def __init__(self, cfg: dict) -> None:
        self._cfg = deep_merge_dicts(self.default_config(), cfg)
        self._init_flag = False
        self.health_metrics = self._cfg.initial_health_state.copy()
        self._current_step = 0
        self._total_reward = 0
        self.reset()

    @classmethod
    def default_config(cls: type) -> EasyDict:
        return EasyDict(cls.config)

    def reset(self):
        self.health_metrics = self._cfg.initial_health_state.copy()
        self._current_step = 0
        self._total_reward = 0
        return self._get_obs(), {}

    def step(self, action: int):
        self._apply_action_effects(action)
        reward = self._calculate_reward()
        self._total_reward += reward
        self._current_step += 1
        done = self._current_step >= self._cfg.max_step
        return BaseEnvTimestep(self._get_obs(), reward, done, {})

    def _apply_action_effects(self, action):
        effects = self._cfg.action_effects[action]
        for metric, effect in effects.items():
            self.health_metrics[metric] += effect
            # Ensure health metrics stay within possible physiological ranges
            min_val, max_val = self._cfg.health_metric_ranges[metric]
            self.health_metrics[metric] = np.clip(self.health_metrics[metric], min_val, max_val)

    def _calculate_reward(self):
        # Simple reward: +1 for each metric in optimal range, -1 otherwise
        reward = 0
        for metric, (low, high) in self._cfg.health_metric_ranges.items():
            if low <= self.health_metrics[metric] <= high:
                reward += 1
            else:
                reward -= 1
        return reward

    def _get_obs(self):
        # Normalize metrics to [0, 1] for the neural network
        obs = np.array([self.health_metrics[metric] for metric in self._cfg.health_metric_ranges.keys()])
        obs_norm = (obs - np.array(list(self._cfg.health_metric_ranges.values()))[:, 0]) / \
                   (np.array(list(self._cfg.health_metric_ranges.values()))[:, 1] -
                    np.array(list(self._cfg.health_metric_ranges.values()))[:, 0])
        return obs_norm

    @property
    def observation_space(self):
        # Observation space is the normalized health metrics
        return gym.spaces.Box(low=0, high=1, shape=(len(self.health_metrics),), dtype=np.float32)

    @property
    def action_space(self):
        # Action space: actions that can influence the health metrics
        return gym.spaces.Discrete(len(self._cfg.action_effects))

    def seed(self, seed=None):
        np.random.seed(seed)

    def close(self):
        pass

    def __repr__(self):
        return "Health Metrics Optimization Environment for LightZero"
