import torch
import numpy as np
from torch import nn
from torch.distributions import Categorical


class BF(nn.Module):
    def __init__(self, state_space, action_space, hidden_size, seed):
        super(BF, self).__init__()
        torch.manual_seed(seed)
        self.actions = np.arange(action_space)
        self.action_space = action_space
        self.fc1 = nn.Linear(state_space, hidden_size)
        self.commands = nn.Linear(2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, action_space)
        self.sigmoid = nn.Sigmoid()

    def forward(self, state, command):
        out = self.sigmoid(self.fc1(state))
        command_out = self.sigmoid(self.commands(command))
        out = out * command_out
        out = torch.relu(self.fc2(out))
        out = torch.relu(self.fc3(out))
        out = torch.relu(self.fc4(out))
        out = self.fc5(out)

        return out

    def action(self, state, desire, horizon, return_scale, horizon_scale):
        """
        Samples the action based on their probability
        """
        command = torch.cat((desire * return_scale, horizon * horizon_scale), dim=-1)
        action_prob = self.forward(state, command)
        probs = torch.softmax(action_prob, dim=-1)
        m = Categorical(probs)
        action = m.sample()
        return action

    def greedy_action(self, state, desire, horizon, return_scale, horizon_scale):
        """
        Returns the greedy action
        """
        command = torch.cat((desire * return_scale, horizon * horizon_scale), dim=-1)
        action_prob = self.forward(state, command)
        probs = torch.softmax(action_prob, dim=-1)
        action = torch.argmax(probs).item()
        return action


class ReplayBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = []

    def add_sample(self, states, actions, rewards):
        episode = {"states": states, "actions": actions, "rewards": rewards, "summed_rewards": sum(rewards)}
        self.buffer.append(episode)

    def sort(self):
        self.buffer = sorted(self.buffer, key=lambda i: i["summed_rewards"], reverse=True)
        self.buffer = self.buffer[:self.max_size]

    def get_random_samples(self, batch_size):
        self.sort()
        idxs = np.random.randint(0, len(self.buffer), batch_size)
        batch = [self.buffer[idx] for idx in idxs]
        return batch

    def get_nbest(self, n):
        self.sort()
        return self.buffer[:n]

    def __len__(self):
        return len(self.buffer)