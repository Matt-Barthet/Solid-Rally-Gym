import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv
import pickle
from stable_baselines3 import PPO
from BaseEnvironment import BaseEnvironment
from ImitationLearning import state_dim, action_dim_x, action_dim_y, ImitationNetwork, states, actions
from PPOEnvironment import PPO_Environment
import pandas as pd

with open("../Models_Pkls/MinMaxScaler.pkl", 'rb') as f:
    scaler = pickle.load(f)


def optimizer_demonstrations():
    model = PPO.load("ppo_solid_optimizer.zip")
    # model.learn(total_timesteps=2500000, progress_bar=True, callback=TensorboardCallback(), tb_log_name="PPO")
    # model.save("ppo_solid_optimizer")

    # This list will hold the demonstrations
    demonstrations = []

    for _ in range(50):

        new_demonstration = []
        observation = env.reset()

        for _ in range(600):
            # Use the model to choose an action
            action, _ = model.predict(observation)

            # Take a step in the environment with the chosen action
            observation, reward, done, info = env.step(action)

            # Save the state, action, and reward
            new_demonstration.append((observation, action, reward))

        if new_demonstration[-1][-1] > 12:
            demonstrations += new_demonstration

    # Save the demonstrations to a file
    with open('demonstrations.pkl', 'wb') as f:
        pickle.dump(demonstrations, f)


if __name__ == '__main__':

    env = DummyVecEnv([lambda:PPO_Environment(counter, True, scaler) for counter in [0]])

    # Define the model architecture (make sure it's the same as when it was trained!)
    model = ImitationNetwork(state_dim, action_dim_x, action_dim_y)

    # Load the state dict into the model
    model.load_state_dict(torch.load('../Models_Pkls/model_min_maxed_large.pth'))

    for i in range(len(states)):
        state = states[i]
        action = actions[i]
        action_logits_x, action_logits_y = model(state)
        action_x = torch.argmax(action_logits_x).item()
        action_y = torch.argmax(action_logits_y).item()
        pred = [action_x, action_y]

        if pred != [torch.argmax(action[0]), torch.argmax(action[1])]:
            print(pred, [torch.argmax(action[0]).item(), torch.argmax(action[1]).item()])

    for _ in range(50):
        observation = env.reset()
        env.step(np.asarray([env.action_space.sample()]))
        for _ in range(600):

            observation_tensor = torch.from_numpy(scaler.transform([np.asarray(observation[0])])).float()
            action_logits_x, action_logits_y = model(observation_tensor[0])
            action_x = torch.argmax(action_logits_x).item()
            action_y = torch.argmax(action_logits_y).item()
            action = [action_x, action_y]

            # Take a step in the environment with the chosen action
            observation, reward, done, info = env.step(np.asarray([action]))
