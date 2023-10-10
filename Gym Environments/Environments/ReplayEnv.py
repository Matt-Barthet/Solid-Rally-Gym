from abc import ABC

import pandas as pd
import numpy as np
import gym
from mlagents_envs.side_channel import OutgoingMessage
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from stable_baselines3.common.vec_env import DummyVecEnv

from SideChannels import MySideChannel
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.exception import UnityEnvironmentException


class ReplayEnv(gym.Env, ABC):

    def __init__(self, id_number, graphics):
        super(ReplayEnv, self).__init__()
        self.engineConfigChannel = EngineConfigurationChannel()
        self.engineConfigChannel.set_configuration_parameters(capture_frame_rate=10)
        self.customSideChannel = MySideChannel()
        self.env = self.load_environment(id_number, graphics)
        self.env = UnityToGymWrapper(self.env, uint8_visual=False, allow_multiple_obs=True)

        self.action_space = self.env.action_space
        self.action_size = self.env.action_size
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(19,))
        self.create_and_send_message("[Generate Arousal]:{}".format(True))

    def step(self, action):
        # Move the env forward 1 tick and receive messages through side-channel.
        state, env_score, d, info = self.env.step(action)
        return state[0], float(env_score), d, info

    def reset(self):
        pass

    def create_and_send_message(self, contents):
        message = OutgoingMessage()
        message.write_string(contents)
        self.customSideChannel.queue_message_to_send(message)

    def load_environment(self, identifier, graphics):
        try:
            env = UnityEnvironment("./Mac Build/solidrally",
                                   side_channels=[self.engineConfigChannel, self.customSideChannel],
                                   worker_id=identifier,
                                   no_graphics=not graphics)
        except UnityEnvironmentException:
            env = UnityEnvironment("../Windows Build/Racing.exe",
                                   side_channels=[self.engineConfigChannel, self.customSideChannel],
                                   worker_id=identifier,
                                   no_graphics=not graphics)
        return env


def replayActions():
    env = DummyVecEnv([lambda: ReplayEnv(counter, True) for counter in [1]])
    sideChannel = env.envs[0].customSideChannel
    target_trace = pd.read_csv("Data/Target/Human_Trace.csv").drop(['Score'], axis=1)

    positions_x = target_trace['Position_X'].values
    positions_y = target_trace['Position_Y'].values
    positions_z = target_trace['Position_Z'].values

    rotations_x = target_trace['Rotation_X'].values
    rotations_y = target_trace['Rotation_Y'].values
    rotations_z = target_trace['Rotation_Z'].values

    velocities_x = target_trace['Velocity_X'].values
    velocities_y = target_trace['Velocity_Y'].values
    velocities_z = target_trace['Velocity_Z'].values

    pedals = target_trace['Pedal'].values
    steering = target_trace['Steering'].values

    env.reset()
    for i in range(len(positions_x)):
        env.step(np.asarray([tuple([steering[i], pedals[i]])]))
        position_message = OutgoingMessage()
        position_message.write_string("[Set Position]:{},{},{}/{},{},{}/{},{},{}".format(positions_x[i],
                                                                                         positions_y[i],
                                                                                         positions_z[i],
                                                                                         rotations_x[i],
                                                                                         rotations_y[i],
                                                                                         rotations_z[i],
                                                                                         velocities_x[i],
                                                                                         velocities_y[i],
                                                                                         velocities_z[i]))
        sideChannel.queue_message_to_send(position_message)

    save_message = OutgoingMessage()
    save_message.write_string("[Save Dict]")
    sideChannel.queue_message_to_send(save_message)
    env.step(np.asarray([tuple([0, 0])]))
    env.close()