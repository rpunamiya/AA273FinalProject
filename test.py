"""
sdd_noise_train.py
------------------
Train a simple process-noise predictor on **one** Stanford Drone Dataset video.

Assumptions
-----------
* Video directory layout:
    <video_dir>/
        annotations.txt   (# 10-column MOT-style file)
        reference.jpg     (top-down reference frame, not used here)
* We use CENTRE OF BOUNDING BOX  in pixel coords; no homography to metres.
  (Fine for a noise-estimation demo; add homography later if needed.)
"""

# ---------- user paths -------------------------------------------------
VIDEO_DIR   = r"C:\Users\rpuna\OneDrive - Stanford\Spring 2025\AA 273\AA273FinalProject\video0"   # <-- change me
CHECKPOINT  = "noise_model_sdd.pt"               # saved weights
EPOCHS      = 1000
# -----------------------------------------------------------------------

import os, math, torch, pandas as pd
from torch.utils.data import IterableDataset, DataLoader
import torch.nn as nn, torch.nn.functional as F

# ---------- 1. tiny helper: noise network from earlier -----------------
def triangular_param(vec, d, eps=1e-3):
    L = torch.zeros(vec.size(0), d, d, device=vec.device)
    idx = torch.tril_indices(d, d, 0)
    L[:, idx[0], idx[1]] = vec
    diag = torch.arange(d, device=vec.device)
    L[:, diag, diag] = F.softplus(L[:, diag, diag]) + eps
    return L @ L.transpose(-1, -2)

class DynamicsNet(nn.Module):
    def __init__(self, n, hid=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n, hid), nn.ReLU(),
                                 nn.Linear(hid, n))
    def forward(self, x):        # residual velocity model
        return x + self.net(x)

class NoiseNet(nn.Module):
    def __init__(self, n, hid=64):
        super().__init__()
        self.n = n
        m = n * (n + 1) // 2
        self.enc = nn.Sequential(nn.Linear(n, hid), nn.Tanh(),
                                 nn.Linear(hid, hid), nn.Tanh())
        self.head = nn.Linear(hid, m)
    def forward(self, x):
        Q = triangular_param(self.head(self.enc(x)), self.n)
        return Q

class NoiseModel(nn.Module):
    def __init__(self, n=2):
        super().__init__()
        self.dyn   = DynamicsNet(n)
        self.noise = NoiseNet(n)
    def forward(self, x):
        x_pred = self.dyn(x)
        Q_t    = self.noise(x)
        return x_pred, Q_t

class NoiseNLL(nn.Module):
    def forward(self, x_pred, x_next, Q):
        resid  = (x_next - x_pred).unsqueeze(-1)          # (B,n,1)
        maha   = torch.matmul(resid.transpose(-1,-2),
                              torch.linalg.solve(Q, resid)).squeeze()
        return 0.5 * (torch.logdet(Q) + maha).mean()

# ---------- 2. dataset for one SDD video -------------------------------
class SDDPairs(IterableDataset):
    """
    Streams (x_t, x_{t+1}) centre-pixel pairs for *pedestrians* in one video.
    """
    COLS = ["track_id","xmin","ymin","xmax","ymax",
            "frame","lost","occ","gen","label"]

    def __init__(self, video_dir: str, hist=1):
        super().__init__()
        self.path = os.path.join(video_dir, "annotations.txt")
        self.hist = hist

        # read once into memory (a single SDD video ≈ few MB)
        df = pd.read_csv(
            self.path,
            sep=r'\s+',
            header=None,
            names=self.COLS,
            quotechar='"',
            engine="python"
        )

        df = df[(df["label"] == '"Pedestrian"') & (df["lost"] == 0)]
        df["cx"] = (df["xmin"] + df["xmax"]) / 2.0
        df["cy"] = (df["ymin"] + df["ymax"]) / 2.0

        self.data = {}
        for tid, grp in df.groupby("track_id"):
            g = grp.sort_values("frame")
            xy = g[["cx","cy"]].to_numpy()
            if len(xy) > 1:
                self.data[tid] = xy.astype("float32")

    def __iter__(self):
        for traj in self.data.values():
            for t in range(len(traj)-1):
                yield torch.from_numpy(traj[t]), torch.from_numpy(traj[t+1])

# ---------- 3. training loop ------------------------------------------
def train():
    ds      = SDDPairs(VIDEO_DIR)
    loader  = DataLoader(ds, batch_size=512,
                         num_workers=0, shuffle=False)
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    model   = NoiseModel().to(device)
    crit    = NoiseNLL()
    opt     = torch.optim.Adam(model.parameters(), lr=3e-4)

    for ep in range(EPOCHS):
        epoch_loss, n_batches = 0.0, 0
        for x_t, x_n in loader:
            x_t, x_n = x_t.to(device), x_n.to(device)
            x_pred, Q = model(x_t)
            loss = crit(x_pred, x_n, Q)
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item()
            n_batches += 1

        if n_batches:
            print(f"Epoch {ep:02d} | NLL {epoch_loss}")
        else:
            print(f"Epoch {ep:02d} | ⚠ No batches (video < batch_size?)")
            break

    torch.save(model.state_dict(), CHECKPOINT)
    print(f"✔ finished, weights saved to {CHECKPOINT}")

if __name__ == "__main__":
    train()
