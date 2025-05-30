import numpy as np

data = np.loadtxt("obsmat.txt")

ped_ids = np.unique(data[:, 1])
"""
def get_trajectories(data, obs_len=8, pred_len=12):
    trajs = []
    for pid in np.unique(data[:, 1]):
        ped_data = data[data[:, 1] == pid]
        ped_data = ped_data[np.argsort(ped_data[:, 0])]
        
        if len(ped_data) < obs_len + pred_len:
            continue
        
        for i in range(len(ped_data) - obs_len - pred_len + 1):
            sub_traj = ped_data[i:i+obs_len+pred_len, 2:4]  # (x, y)
            trajs.append(sub_traj)
    
    return np.array(trajs)  # Shape: [N, obs_len+pred_len, 2]


trajs = get_trajectories(data)
obs = trajs[:, :8, :]
pred = trajs[:, 8:, :]



import torch
import torch.nn as nn

num_epochs = 50

class TrajectoryLSTM(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, num_layers=1, pred_len=12):
        super(TrajectoryLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.pred_len = pred_len
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.Linear(hidden_dim, 2)

    def forward(self, obs_traj):
        # obs_traj: [batch_size, obs_len, 2]
        _, (hidden, cell) = self.lstm(obs_traj)

        # Initialize prediction sequence with last observed position
        last_pos = obs_traj[:, -1, :]  # [batch_size, 2]
        preds = []

        for _ in range(self.pred_len):
            # Expand input for LSTM: [batch_size, 1, 2]
            lstm_input = last_pos.unsqueeze(1)
            output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
            delta = self.decoder(output.squeeze(1))  # Predict offset
            last_pos = last_pos + delta  # Predict next position
            preds.append(last_pos.unsqueeze(1))

        return torch.cat(preds, dim=1)  # [batch_size, pred_len, 2]


model = TrajectoryLSTM()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for epoch in range(num_epochs):
    model.train()
    for obs_batch, pred_batch in dataloader:  # Shape: [batch_size, seq_len, 2]
        obs_batch = obs_batch.float()
        pred_batch = pred_batch.float()

        pred_output = model(obs_batch)
        loss = loss_fn(pred_output, pred_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class TrajectoryDataset(Dataset):
    def __init__(self, file_path, obs_len=8, pred_len=12):
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.data = np.loadtxt(file_path)
        self.trajectories = self._build_trajectories()

    def _build_trajectories(self):
        trajectories = []
        for pid in np.unique(self.data[:, 1]):
            ped_data = self.data[self.data[:, 1] == pid]
            ped_data = ped_data[np.argsort(ped_data[:, 0])]
            if len(ped_data) < self.obs_len + self.pred_len:
                continue
            for i in range(len(ped_data) - self.obs_len - self.pred_len + 1):
                traj = ped_data[i:i+self.obs_len+self.pred_len, 2:4]  # (x, y)
                traj -= traj[0]  # Normalize relative to first position
                trajectories.append(traj)
        return np.array(trajectories)

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        obs = traj[:self.obs_len]
        pred = traj[self.obs_len:]
        return torch.tensor(obs, dtype=torch.float32), torch.tensor(pred, dtype=torch.float32)

import torch.nn as nn

class TrajectoryLSTM(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, num_layers=1, pred_len=12):
        super(TrajectoryLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.pred_len = pred_len
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.Linear(hidden_dim, 2)

    def forward(self, obs_traj):
        _, (hidden, cell) = self.lstm(obs_traj)
        last_pos = obs_traj[:, -1, :]  # [batch, 2]
        preds = []

        for _ in range(self.pred_len):
            lstm_input = last_pos.unsqueeze(1)
            output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
            delta = self.decoder(output.squeeze(1))
            last_pos = last_pos + delta
            preds.append(last_pos.unsqueeze(1))

        return torch.cat(preds, dim=1)  # [batch, pred_len, 2]

# Load your dataset
dataset = TrajectoryDataset("obsmat.txt", obs_len=8, pred_len=12)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Initialize model
model = TrajectoryLSTM(pred_len=12)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# Training loop
for epoch in range(50):
    model.train()
    epoch_loss = 0
    for obs, pred in dataloader:
        pred_output = model(obs)
        loss = loss_fn(pred_output, pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")
