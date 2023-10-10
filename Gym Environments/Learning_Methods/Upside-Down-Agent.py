import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import gym
import random
from collections import deque

# Create the Unity environment with gym wrapper
env = gym.make('UnityRallyEnv-v0')

# Set seeds for reproducibility
seed = 42
env.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


# Define agent's neural network
class UpsideDownAgent(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(UpsideDownAgent, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Initialize agent and optimizer
input_dim = 18
output_dim = 2
agent = UpsideDownAgent(input_dim, output_dim)
optimizer = optim.Adam(agent.parameters())

# Initialize replay buffer
replay_buffer = deque(maxlen=100000)


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
    states = torch.stack(states)
    actions = torch.tensor(actions, dtype=torch.float32)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.stack(next_states)

    # Compute the Q-values for the current states and actions
    q_values = agent(states)
    q_values = torch.sum(q_values * actions, dim=-1)

    # Compute the target Q-values
    with torch.no_grad():
        target_q_values = agent(next_states)
        pred_discrete_actions = continuous_to_discrete(target_q_values.numpy())
        pred_discrete_actions = torch.tensor(pred_discrete_actions, dtype=torch.float32)
        target_q_values = torch.sum(target_q_values * pred_discrete_actions, dim=-1)
        target_q_values = rewards + desired_horizon * target_q_values

    # Compute the loss and update the agent's parameters
    loss = nn.MSELoss(q_values, target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def calculate_reward(agent_arousal_trace, agent_score_trace, target_arousal_trace, target_score_trace):
    arousal_distance = np.linalg.norm(np.array(agent_arousal_trace) - np.array(target_arousal_trace))
    score_distance = np.linalg.norm(np.array(agent_score_trace) - np.array(target_score_trace))
    reward = - (arousal_distance + score_distance)
    return reward


# Training parameters
num_episodes = 1000
max_timesteps = 200  # 2 minutes * 100 timesteps per minute
batch_size = 64
desired_horizon = 10

# Initialize the arousal target generator
# arousal_target_generator = ArousalTargetGenerator()


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


# Main training loop
for episode in range(num_episodes):
    obs = env.reset()
    state = preprocess_observation(obs)
    score_trace = []
    arousal_trace = []

    for t in range(max_timesteps):
        # Select an action using the agent's neural network
        action = agent(state)
        action_discrete = continuous_to_discrete(action.detach().numpy())

        # Step the environment and observe the result
        next_obs, _, done, info = env.step(action_discrete)
        next_state = preprocess_observation(next_obs)

        # Update the arousal trace
        arousal_trace.append(0)

        # Retrieve the target arousal and score trace for the current time step
        target_arousal_trace, target_score_trace = (0, 0)
        # target_arousal_trace, target_score_trace = arousal_target_generator.get_target_traces(t)

        # Calculate the reward based on how well the arousal and score traces align with the target traces
        reward = calculate_reward(arousal_trace, score_trace, target_arousal_trace, target_score_trace)

        # Update the score trace
        score_trace.append(reward)

        # Store the transition in the replay buffer
        replay_buffer.append((state, action_discrete, reward, next_state))

        # Train the agent using UDRL
        train(agent, replay_buffer, batch_size, desired_horizon)

        # Update the current state
        state = next_state

        # Check if the episode is done
        if done:
            break

    # Calculate the total score and average arousal for the episode
    total_score = np.sum(score_trace)
    avg_arousal = np.mean(arousal_trace)

    # Print the episode results
    print(f"Episode {episode + 1}/{num_episodes}: Total Score = {total_score}, Average Arousal = {avg_arousal}")
