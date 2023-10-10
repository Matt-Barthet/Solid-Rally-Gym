from abc import ABC

import numpy as np
import gym
import vg
from mlagents_envs.side_channel import OutgoingMessage
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from SideChannels import MySideChannel
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.exception import UnityEnvironmentException


class BackwardAlgorithmEnv(gym.Env, ABC):

    def __init__(self, id_number, graphics, target_trace, target_states, scaler, arousal):
        super(BackwardAlgorithmEnv, self).__init__()

        self.engineConfigChannel = EngineConfigurationChannel()
        self.engineConfigChannel.set_configuration_parameters(capture_frame_rate=5)
        self.customSideChannel = MySideChannel()

        self.env = self.load_environment(id_number, graphics)
        self.env = UnityToGymWrapper(self.env, uint8_visual=False, allow_multiple_obs=True)
        self.action_space = self.env.action_space
        self.action_size = self.env.action_size
        self.observation_space = gym.spaces.Box(low=-1, high=2, shape=(19,))

        self.target_trace = target_trace
        self.target_states = target_states
        self.scaler = scaler

        self.reward = 0
        self.max_reward = 0
        self.mean_reward = 0

        self.step_count = 0
        self.total_count = 0

        self.arousal = arousal
        self.score = 8

        self.create_and_send_message("[Save Cells]:false")
        self.create_and_send_message("[Generate Arousal]:{}".format(arousal))

        self.startingID = len(target_trace) - 1
        self.target_id = 0
        self.message_and_reset()
        self.best_distance = 0
        np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

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

    @staticmethod
    def tuple_to_vector(s):
        obs = []
        for i in range(len(s[0])):
            obs.append(s[0][i])
        return obs

    def create_and_send_message(self, contents):
        message = OutgoingMessage()
        message.write_string(contents)
        self.customSideChannel.queue_message_to_send(message)

    def message_and_reset(self):
        self.create_and_send_message("[Cell Name]:Target/Cells/Time-step-{}".format(self.startingID * 10))
        return self.reset()

    def reset(self):
        self.reward = 0
        self.step_count = 0
        self.target_id = int(self.startingID)
        self.set_target_point()
        save_message = OutgoingMessage()
        save_message.write_string("[Save Dict]")
        self.customSideChannel.queue_message_to_send(save_message)
        return self.tuple_to_vector(self.env.reset())

    def set_target_point(self):
        point_a = np.asarray(self.target_states[self.target_id - 1])
        point_b = np.asarray(self.target_states[self.target_id])
        self.best_distance = np.round(np.linalg.norm(point_a-point_b), 1)
        self.create_and_send_message("[Target]:{},{},{}".format(self.target_trace[self.target_id][0],
                                                                self.target_trace[self.target_id][1],
                                                                self.target_trace[self.target_id][2]))

    @staticmethod
    def angle_to_target(direct, pos, target_pos):
        relative_position = target_pos - pos
        angle_to_target = vg.angle(direct, relative_position)
        return angle_to_target

    def step(self, action):

        # Move the env forward 1 tick and receive messages through side-channel.
        state, env_score, d, info = self.env.step(action)
        direction_vector = np.asarray(self.customSideChannel.direction)
        position = np.asarray([int(state[0][0]), int(state[0][1]), int(state[0][2])])

        target_position = np.asarray(self.target_trace[self.target_id])

        angle_to_target = self.angle_to_target(direction_vector, position, target_position)

        self.step_count += 1
        self.total_count += 1
        positional_distance = np.round(np.linalg.norm(position - target_position), 1)
        reward = 0.0

        # If the target position is behind us, see where the next one is and shift to that if it is infront of us
        if angle_to_target > 90:
            try:
                next_position = np.asarray(self.target_trace[self.target_id + 1])
                angle_to_next_target = self.angle_to_target(direction_vector, position, next_position)
                if angle_to_next_target < 90 and positional_distance < 15:
                    # if the angle is acute, it's infront of us and we can shift to it
                    self.target_id += 1
                    self.set_target_point()
                elif angle_to_next_target > 90:
                    next_distance = np.round(np.linalg.norm(position - next_position), 1)
                    if next_distance < positional_distance:
                        self.target_id += 1
                        self.set_target_point()
            except IndexError:
                pass

            """try:
                previous_position = np.asarray(self.target_trace[self.target_id - 1])
                angle_to_previous = self.angle_to_target(direction_vector, position, previous_position)
                trace_direction = target_position - previous_position
                angle_to_trace_direction = vg.angle(trace_direction, direction_vector)

                if angle_to_previous > 90 and angle_to_trace_direction > 90:
                    self.target_id -= 1
                    self.set_target_point()

            except IndexError:
                pass
            """

        if np.linalg.norm(np.asarray([state[0][3], state[0][4], state[0][5]])) < 1:
            reward = -0

        state_norm = self.scaler.transform(state)[0]
        target_state = np.asarray(self.target_states[self.target_id])
        state_distance = np.round(np.linalg.norm(target_state - state_norm))

        # if state_distance < self.best_distance:
            # reward += 1 / (1 + state_distance)

        if positional_distance <= 5:
            reward += 1

        if positional_distance <= 5:
            self.target_id += 1
            if self.target_id == len(self.target_trace):
                self.startingID -= 1
                self.message_and_reset()
            else:
                self.set_target_point()

        self.score = np.min([float(env_score), self.score])
        self.reward += float(reward)
        self.mean_reward = (self.mean_reward * (self.total_count - 1) / self.total_count) + (self.reward / self.total_count)
        self.max_reward = np.max([self.reward, self.max_reward])

        if self.step_count == 1500 or env_score == 8:
            self.message_and_reset()

        return self.tuple_to_vector(state), float(reward), d, info
