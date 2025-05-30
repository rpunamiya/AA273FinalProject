import os, math, torch, pandas as pd
from torch.utils.data import IterableDataset, DataLoader
import torch.nn as nn, torch.nn.functional as F
from torch.utils.data import ConcatDataset
from torch.utils.data import Dataset

# ---------- Configuration ----------
VIDEO_DIR   = r"C:\Users\rpuna\OneDrive - Stanford\Spring 2025\AA 273\AA273FinalProject\video0"
CHECKPOINT  = "noise_model_sdd.pt"
EPOCHS      = 100

# ---------- Helper: Cholesky Projection ----------
def triangular_param(vec, d, eps=1e-3):
    L = torch.zeros(vec.size(0), d, d, device=vec.device)
    idx = torch.tril_indices(d, d, 0)
    L[:, idx[0], idx[1]] = vec
    diag = torch.arange(d, device=vec.device)
    L[:, diag, diag] = F.softplus(L[:, diag, diag]) + eps
    return L @ L.transpose(-1, -2)

# ---------- 1. LSTM-based Model ----------
class LSTMEncoder(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

    def forward(self, x_seq):  # (B, T, 2)
        _, (h, _) = self.lstm(x_seq)
        return h.squeeze(0)    # (B, hidden_dim)

class DynamicsNet(nn.Module):
    def __init__(self, hidden_dim, out_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, h):
        return self.fc(h)

class NoiseNet(nn.Module):
    def __init__(self, hidden_dim, out_dim):
        super().__init__()
        m = out_dim * (out_dim + 1) // 2
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, m)
        )
        self.out_dim = out_dim

    def forward(self, h):
        return triangular_param(self.fc(h), self.out_dim)

class NoiseModel(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=2):
        super().__init__()
        self.encoder = LSTMEncoder(input_dim, hidden_dim)
        self.dyn     = DynamicsNet(hidden_dim, output_dim)
        self.noise   = NoiseNet(hidden_dim, output_dim)

    def forward(self, x_seq):
        h = self.encoder(x_seq)
        x_pred = x_seq[:, -1] + self.dyn(h)
        Q = self.noise(h)
        return x_pred, Q

class NoiseNLL(nn.Module):
    def forward(self, x_pred, x_next, Q):
        resid = (x_next - x_pred).unsqueeze(-1)
        maha = torch.matmul(resid.transpose(-1,-2), torch.linalg.solve(Q, resid)).squeeze()
        return 0.5 * (torch.logdet(Q) + maha).mean()

# ---------- 2. Dataset with History ----------
class SDDPairs(Dataset):
    COLS = ["track_id","xmin","ymin","xmax","ymax",
            "frame","lost","occ","gen","label"]

    def __init__(self, video_dir, hist_len=4):
        self.samples = []
        self.hist_len = hist_len

        path = os.path.join(video_dir, "annotations.txt")
        df = pd.read_csv(path, sep=r'\s+', header=None, names=self.COLS, quotechar='"', engine="python")
        df = df[(df["label"] == '"Pedestrian"') & (df["lost"] == 0)]
        df["cx"] = (df["xmin"] + df["xmax"]) / 2.0
        df["cy"] = (df["ymin"] + df["ymax"]) / 2.0

        for _, group in df.groupby("track_id"):
            group = group.sort_values("frame")
            traj = group[["cx", "cy"]].to_numpy(dtype="float32")
            if len(traj) > hist_len:
                for t in range(hist_len, len(traj) - 1):
                    hist = traj[t - hist_len:t]
                    future = traj[t + 1]
                    self.samples.append((hist, future))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        hist, future = self.samples[idx]
        return torch.from_numpy(hist), torch.from_numpy(future)

    def __iter__(self):
        for traj in self.data.values():
            for t in range(self.hist_len, len(traj)-1):
                hist = traj[t-self.hist_len:t]  # shape (hist_len, 2)
                next = traj[t+1]                # shape (2,)
                yield torch.from_numpy(hist), torch.from_numpy(next)

# Walk recursively to find every folder with annotations.txt
def find_video_dirs(root="data"):
    """
    Recursively finds all video folders under each scene subfolder
    that contain an `annotations.txt` file.
    """
    video_dirs = []
    for scene in os.listdir(root):
        scene_path = os.path.join(root, scene)
        if not os.path.isdir(scene_path):
            continue
        for video in os.listdir(scene_path):
            video_path = os.path.join(scene_path, video)
            if os.path.isdir(video_path) and "annotations.txt" in os.listdir(video_path):
                video_dirs.append(video_path)
    return video_dirs

# ---------- 3. Training Loop ----------
def train():
    video_dirs = find_video_dirs("data")

    # Create a dataset for each video and combine them
    datasets = [SDDPairs(v, hist_len=10) for v in video_dirs]
    ds = ConcatDataset(datasets)
    loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = NoiseModel().to(device)
    loss_fn = NoiseNLL()
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)

    for ep in range(EPOCHS):
        total_loss, n_batches = 0.0, 0
        for hist_seq, x_next in loader:
            hist_seq, x_next = hist_seq.to(device), x_next.to(device)
            x_pred, Q = model(hist_seq)
            loss = loss_fn(x_pred, x_next, Q)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item(); n_batches += 1

        if n_batches:
            print(f"Epoch {ep:03d} | NLL {total_loss / n_batches:.4f}")
        else:
            print(f"Epoch {ep:03d} | ⚠ No data")
            break

    torch.save(model.state_dict(), CHECKPOINT)
    print(f"✔ Finished. Weights saved to {CHECKPOINT}")

if __name__ == "__main__":
    train()
