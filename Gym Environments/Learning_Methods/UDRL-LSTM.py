import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import gym
import random
from collections import deque

from torch.autograd.grad_mode import F

# Create the Unity environment with gym wrapper
env = gym.make('UnityRallyEnv-v0')

# Set seeds for reproducibility
seed = 42
env.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Initialize agent and optimizer
input_dim = 18
output_dim = 2
agent = LSTMAgent(input_dim, output_dim)
optimizer = optim.Adam(agent.parameters())

# Initialize replay buffer
replay_buffer = deque(maxlen=100000)

# Training parameters
num_episodes = 1000
max_timesteps = 200  # 2 minutes * 100 timesteps per minute
batch_size = 64
desired_horizon = 10

# Initialize the arousal target generator
# arousal_target_generator = ArousalTargetGenerator()


class LSTMAgent(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMAgent, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden):
        lstm_out, hidden = self.lstm(x, hidden)
        output = self.fc(lstm_out[:, -1, :])
        return output, hidden


def preprocess_observation(obs):
    return torch.tensor(obs, dtype=torch.float32)


def continuous_to_discrete(action):
    discrete_action = np.zeros(action.shape, dtype=int)
    discrete_action[action < -0.5] = -1
    discrete_action[action > 0.5] = 1
    return discrete_action


def train(agent, memory, batch_size, desired_horizon):
    if len(memory) < batch_size:
        return

    # Sample a batch of transitions from the replay buffer
    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states = zip(*batch)

    # Convert the batch to tensors
    states = torch.stack(states).view(batch_size, -1, agent.input_dim)
    actions = torch.tensor(actions, dtype=torch.float32).view(batch_size, -1, 1)
    rewards = torch.tensor(rewards, dtype=torch.float32).view(batch_size, -1, 1)
    next_states = torch.stack(next_states).view(batch_size, -1, agent.input_dim)

    # Initialize the LSTM hidden states
    hidden = (torch.zeros(1, batch_size, agent.hidden_dim), torch.zeros(1, batch_size, agent.hidden_dim))

    # Compute the Q-values for the current states and actions
    q_values, _ = agent(states, hidden)
    q_values = torch.sum(q_values * actions, dim=-1)

    # Compute the target Q-values
    with torch.no_grad():
        target_q_values, _ = agent(next_states, hidden)
        target_q_values, _ = torch.max(target_q_values, dim=-1)
        target_q_values = rewards + desired_horizon * target_q_values

    # Compute the loss and update the agent's parameters
    loss = F.mse_loss(q_values, target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def calculate_reward(agent_arousal_trace, agent_score_trace, target_arousal_trace, target_score_trace):
    arousal_distance = np.linalg.norm(np.array(agent_arousal_trace) - np.array(target_arousal_trace))
    score_distance = np.linalg.norm(np.array(agent_score_trace) - np.array(target_score_trace))
    reward = - (arousal_distance + score_distance)
    return reward


def preprocess_expert_demonstrations(expert_demonstrations):
    processed_demonstrations = []
    for demonstration in expert_demonstrations:
        state, action, arousal, next_state = demonstration
        state = preprocess_observation(state)
        next_state = preprocess_observation(next_state)
        processed_demonstrations.append((state, action, arousal, next_state))
    return processed_demonstrations


# expert_demonstrations = load_expert_demonstrations()  # Load the expert demonstrations from your data source
# processed_demonstrations = preprocess_expert_demonstrations(expert_demonstrations)

# Add expert demonstrations to the replay buffer
# replay_buffer.extend(processed_demonstrations)


# pretrain_iterations = 1000
# for _ in range(pretrain_iterations):
    # train(agent, processed_demonstrations, batch_size, desired_horizon)

history_length = 5


for episode in range(num_episodes):
    obs = env.reset()
    hidden = (torch.zeros(1, 1, agent.hidden_dim), torch.zeros(1, 1, agent.hidden_dim))
    state_history = deque(maxlen=history_length)
    action_history = deque(maxlen=history_length)
    arousal_history = deque(maxlen=history_length)
    score_history = deque(maxlen=history_length)
    total_reward = 0

    for t in range(max_timesteps):
        state_history.append(state)
        if len(state_history) < history_length:
            continue

        state_batch = torch.stack(list(state_history)).view(1, -1, agent.input_dim)
        action, hidden = agent(state_batch, hidden)
        action_discrete = continuous_to_discrete(action.detach().numpy())

        next_obs, reward, done, info = env.step(action_discrete)

        next_state = preprocess_observation(next_obs)
        action_history.append(action)
        arousal_history.append(info["arousal"])
        if len(action_history) < history_length or len(arousal_history) < history_length:
            state = next_state
            continue

        # Calculate the reward based on the Euclidean distance between the agent's arousal trace and the target cluster
        arousal_distance = np.linalg.norm(np.array(arousal_history) - target_arousal_trace[:len(arousal_history)])
        score_distance = np.linalg.norm(np.array(score_history) - target_score_trace[:len(score_history)])
        reward = -arousal_distance - score_distance

        total_reward += reward
        replay_buffer.append((state, action_discrete, reward, next_state))
        state = next_state

        train(agent, replay_buffer, batch_size, desired_horizon)

        if done:
            break

    print(f"Episode {episode}: Total reward: {total_reward}")
