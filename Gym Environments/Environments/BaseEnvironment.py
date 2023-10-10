from abc import ABC

import numpy as np
import gym
from mlagents_envs.side_channel import OutgoingMessage
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

from Utils.SideChannels import MySideChannel
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.exception import UnityEnvironmentException


class BaseEnvironment(gym.Env, ABC):

    def __init__(self, id_number, graphics, scaler):
        super(BaseEnvironment, self).__init__()
        self.engineConfigChannel = EngineConfigurationChannel()
        self.engineConfigChannel.set_configuration_parameters(capture_frame_rate=10)
        self.customSideChannel = MySideChannel()
        self.env = self.load_environment(id_number, graphics)
        self.env = UnityToGymWrapper(self.env, uint8_visual=False, allow_multiple_obs=True)

        self.action_space = self.env.action_space
        self.action_size = self.env.action_size
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(49,))
        self.create_and_send_message("[Generate Arousal]:{}".format(True))

        self.reward = 0
        self.max_reward = 0
        self.mean_reward = 0
        self.score = 0
        self.max_score = 0
        self.episode_length = 0
        self.steps = 0
        self.scaler = scaler

    def step(self, action):
        # Move the env forward 1 tick and receive messages through side-channel.
        state, env_score, d, info = self.env.step(np.asarray([tuple([action[0] - 1, action[1] - 1])]))
        self.score = env_score
        self.max_score = np.max([self.score, self.max_score])
        return state, env_score, d, info

    @staticmethod
    def tuple_to_vector(s):
        obs = []
        for i in range(len(s[0])):
            obs.append(s[0][i])
        return obs

    def reset(self):
        self.steps = 0
        return self.tuple_to_vector(self.env.reset())

    def create_and_send_message(self, contents):
        message = OutgoingMessage()
        message.write_string(contents)
        self.customSideChannel.queue_message_to_send(message)

    def load_environment(self, identifier, graphics):
        try:
            env = UnityEnvironment("../Builds/Mac Build/solidrally",
                                   side_channels=[self.engineConfigChannel, self.customSideChannel],
                                   worker_id=identifier,
                                   no_graphics=not graphics)
        except UnityEnvironmentException:
            env = UnityEnvironment("../Builds/Windows Build/Racing.exe",
                                   side_channels=[self.engineConfigChannel, self.customSideChannel],
                                   worker_id=identifier,
                                   no_graphics=not graphics)
        return env
