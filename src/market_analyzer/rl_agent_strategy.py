"""
rl_agent_strategy.py
Uses stable-baselines or a custom RL approach to produce [-1,0,1].
"""

import pandas as pd
import numpy as np
from typing import Optional
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import DummyVecEnv

from .strategy import TradingStrategy

class DummyTradingEnv:
    """
    A minimal example environment for demonstration.
    Real code would define:
     - observation_space
     - action_space
     - step/reward logic
    """
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.current_step = 0
        # placeholder logic

    def reset(self):
        self.current_step = 0
        return np.zeros(5, dtype=np.float32)  # e.g. 5-d observation

    def step(self, action):
        # placeholder => no real reward
        self.current_step += 1
        done = (self.current_step >= len(self.data))
        obs = np.zeros(5, dtype=np.float32)
        reward = 0.0
        return obs, reward, done, {}

    def render(self, mode="human"):
        pass

class RLAgentStrategy(TradingStrategy):
    """
    A 'real' RL-based approach using stable-baselines as an example.
    We'll assume a discrete action space in {0,1,2} => Sell/Hold/Buy
    """

    def __init__(self, name="rl_agent"):
        super().__init__(name)
        self.model = None

    def train(self, data: pd.DataFrame, timesteps=10000):
        """
        Build a gym environment around the data, train a PPO agent on it.
        """
        # Placeholder environment
        env = DummyTradingEnv(data)
        # stable-baselines requires a gym-like environment => we skip details
        def make_env():
            return env
        vec_env = DummyVecEnv([make_env])
        # Create PPO with discrete action => real code would require custom env
        self.model = PPO("MlpPolicy", vec_env, verbose=0)
        self.model.learn(total_timesteps=timesteps)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0.0, index=data.index)
        if self.model is None:
            print("[RLAgent] Model not trained, returning zeros.")
            return signals

        # Step through data => pick action
        obs = np.zeros((1,5), dtype=np.float32)
        for i, idx in enumerate(data.index):
            # In real code => build an observation from data row i
            action, _ = self.model.predict(obs)
            # action in {0,1,2} => map 0->-1,1->0,2->1
            mapping = {0:-1,1:0,2:1}
            signals.loc[idx] = mapping[action]
            # next observation => placeholder
        return signals
