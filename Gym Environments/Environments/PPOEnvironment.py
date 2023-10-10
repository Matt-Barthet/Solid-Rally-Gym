from abc import ABC
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from Utils.Tensorboard_Callbacks import TensorboardCallback
from BaseEnvironment import BaseEnvironment
import numpy as np
import pickle
from Utils.Scalers import VectorScaler


class PPO_Environment(BaseEnvironment, ABC):

    def calculate_reward(self, state):
        rotation_component = (180 - state[-1]) / 180
        speed_component = np.linalg.norm([state[0], state[1], state[2]]) / 60
        distance_component = (1 + state[-2])
        self.reward = rotation_component * speed_component / distance_component

    def reset_condition(self):
        self.episode_length += 1
        if self.episode_length > 600:
            self.episode_length = 0
            self.reset()

    def update_stats(self):
        self.max_score = np.max([self.score, self.max_score])
        self.max_reward = np.max([self.max_reward, self.reward])

    def step(self, action):
        # Move the env forward 1 tick and receive messages through side-channel.
        state, env_score, d, info = self.env.step(np.asarray([tuple([action[0] - 1, action[1] - 1])]))
        state = state[0]
        self.score = env_score
        self.calculate_reward(state)
        self.update_stats()
        self.reset_condition()
        return state, self.reward, d, info


if __name__ == "__main__":

    load_scaler = False

    if load_scaler:
        with open('../Models_Pkls/MinMaxScaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
    else:
        scaler = VectorScaler(49)

    env = DummyVecEnv([lambda:PPO_Environment(counter, True, scaler) for counter in [4]])
    model = PPO("MlpPolicy", env=env, tensorboard_log="../Tensorboard")
    model.learn(total_timesteps=1500000, progress_bar=True, callback=TensorboardCallback(), tb_log_name="PPO")
    model.save("ppo_solid_test")

    with open("../Models_Pkls/MinMaxScaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
