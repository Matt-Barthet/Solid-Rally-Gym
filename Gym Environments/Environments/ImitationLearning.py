import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle

import pandas as pd
import matplotlib.pyplot as plt
from torchmetrics import Accuracy
from sklearn.preprocessing import OneHotEncoder


with open('../Models_Pkls/demonstrations.pkl', 'rb') as f:
    demonstrations = pickle.load(f)

with open("../Models_Pkls/MinMaxScaler.pkl", 'rb') as f:
    scaler = pickle.load(f)

states = np.asarray([demo[0][0] for demo in demonstrations])
actions_orig = np.asarray([demo[1][0] for demo in demonstrations])
rewards = np.asarray([demo[2] for demo in demonstrations])

# states = scaler.transform(states)

actions_x = pd.Series(actions_orig[:, 0])
actions_x = pd.get_dummies(actions_x).astype(int).values

actions_y = pd.Series(actions_orig[:, 1])
actions_y = pd.get_dummies(actions_y).astype(int).values

actions = np.asarray([(actions_x[i], actions_y[i]) for i in range(len(actions_orig))])

# Convert numpy arrays to PyTorch tensors
states = torch.from_numpy(states).float()
actions = torch.from_numpy(actions).float()

# Create the model
state_dim = states.shape[1]

# The +1 is there assuming actions are 0-indexed
action_dim_x = 3
action_dim_y = 3


class ImitationNetwork(nn.Module):
    def __init__(self, state_dim, action_dim_x, action_dim_y):
        super(ImitationNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64 * 2)
        self.fc2 = nn.Linear(64 * 2, 64 * 2)
        self.fc3_x = nn.Linear(64 * 2, action_dim_x)
        self.fc3_y = nn.Linear(64 * 2, action_dim_y)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x_out = self.fc3_x(x)
        y_out = self.fc3_y(x)
        return x_out, y_out


def train(model, data_loader, criterion_x, criterion_y, optimizer, num_epochs):
    accuracy_x = Accuracy(task="multiclass", num_classes=3)
    accuracy_y = Accuracy(task="multiclass", num_classes=3)

    loss_list, steering_accuracy, pedal_accuracy = [], [], []

    for epoch in range(num_epochs):
        for states, actions in data_loader:

            # Move data to the appropriate device
            states = states.to(device)

            actions_x, actions_y = actions[:, 0], actions[:, 1]
            actions_x = actions_x.to(device)
            actions_y = actions_y.to(device)

            # Forward pass
            action_logits_x, action_logits_y = model(states)

            # Compute loss
            loss_x = criterion_x(action_logits_x, actions_x)
            loss_y = criterion_y(action_logits_y, actions_y)
            loss = loss_x + loss_y

            # Compute accuracy
            acc_x = accuracy_x(action_logits_x.softmax(dim=1), actions_x)
            acc_y = accuracy_y(action_logits_y.softmax(dim=1), actions_y)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        steering_accuracy.append(acc_x.item())
        pedal_accuracy.append(acc_y.item())
        loss_list.append(loss.item())

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}, Acc Steering: {acc_x.item()}, Acc Pedal: {acc_y.item()}")

    plt.figure()
    plt.title("Predicting Steering + Pedals")
    plt.ylabel("Metric")
    plt.xlabel("Epoch")
    plt.plot(loss_list, label="Loss")
    plt.plot(steering_accuracy, label="Steering Acc")
    plt.plot(pedal_accuracy, label="Pedals Acc")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Calculate class frequencies
    class_counts = np.bincount(actions_orig[:, 0])
    majority_class = np.argmax(class_counts)
    majority_percentage = (class_counts[majority_class] / len(actions_orig[:, 0])) * 100
    formatted_percentage = "{:.2f}".format(majority_percentage)

    # Plot histogram for steering actions
    plt.hist(actions_orig[:, 0], bins=3, alpha=0.5, color='blue', label='Steering')
    plt.xlabel('Action')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of Steering Actions (Majority Class %: {formatted_percentage}')
    plt.xticks([0, 1, 2], ['Left', 'Straight', 'Right'])
    plt.show()

    # Calculate class frequencies
    class_counts = np.bincount(actions_orig[:, 1])
    majority_class = np.argmax(class_counts)
    majority_percentage = (class_counts[majority_class] / len(actions_orig[:, 1])) * 100
    formatted_percentage = "{:.2f}".format(majority_percentage)

    # Plot histogram for pedal actions
    plt.hist(actions_orig[:, 1], bins=3, alpha=0.5, color='green', label='Pedals')
    plt.xlabel('Action')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of Pedal Actions (Majority Class %: {formatted_percentage}')
    plt.xticks([0, 1, 2], ['Backward', 'Neutral', 'Forward'])
    plt.show()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Wrap the tensors in a dataset and create a dataloader for mini-batch processing
    dataset = TensorDataset(states, actions)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    # Create the model with two action dimensions
    model = ImitationNetwork(state_dim, action_dim_x, action_dim_y)

    # You will now need two loss functions, one for each action head
    criterion_x = nn.CrossEntropyLoss()
    criterion_y = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train(model, dataloader, criterion_x, criterion_y, optimizer, 5000)

    print('Finished Training')
    torch.save(model.state_dict(), 'model_min_maxed_large.pth')
