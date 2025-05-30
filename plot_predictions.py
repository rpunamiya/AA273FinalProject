import os, torch, pandas as pd
import matplotlib.pyplot as plt
from test2 import NoiseModel, triangular_param
import numpy as np
from matplotlib.patches import Ellipse


# ---------- Configuration ----------
VIDEO_DIR  = r"C:\Users\rpuna\OneDrive - Stanford\Spring 2025\AA 273\AA273FinalProject\data\quad\video0"  # <-- match your train video
CHECKPOINT = "noise_model_sdd.pt"
N_STEPS    = 500   # How many steps ahead to roll out
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
tid = 1 # <-- change this to select a different track_id
# ------------------------------------

COLS = ["track_id","xmin","ymin","xmax","ymax",
        "frame","lost","occ","gen","label"]

def plot_cov_ellipse(Q, center, ax, n_std=2.0, **kwargs):
    """
    Plots an n-std confidence ellipse for a 2x2 covariance matrix Q.
    Args:
        Q: 2x2 positive semi-definite covariance matrix (numpy)
        center: 2D mean location [x, y]
        ax: matplotlib axis
        n_std: number of standard deviations (2.0 = 95% confidence)
        kwargs: passed to Ellipse()
    """
    if Q.shape != (2,2):
        return
    vals, vecs = np.linalg.eigh(Q)
    angle = np.degrees(np.arctan2(*vecs[:,1][::-1]))
    width, height = 2 * n_std * np.sqrt(vals)
    ell = Ellipse(xy=center, width=width, height=height, angle=angle, **kwargs)
    ax.add_patch(ell)

# ---------- Load GT trajectory ----------
df = pd.read_csv(os.path.join(VIDEO_DIR, "annotations.txt"),
                 sep=r"\s+", header=None, names=COLS, quotechar='"', engine="python")

df = df[(df["label"] == '"Pedestrian"') & (df["lost"] == 0)]
df["cx"] = (df["xmin"] + df["xmax"]) / 2.0
df["cy"] = (df["ymin"] + df["ymax"]) / 2.0

# choose a random pedestrian
# tid = 1 # <-- change this to select a different track_id
traj = df[df["track_id"] == tid].sort_values("frame")[["cx", "cy"]].to_numpy()
print(f"Track ID: {tid}, Trajectory Length: {len(traj)}")

# ---------- Load model ----------
model = NoiseModel().to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT))
model.eval()

# ---------- Predict N_STEPS ahead ----------
HIST_LEN = 4
x_seq = torch.tensor(traj[:HIST_LEN], dtype=torch.float32).unsqueeze(0).to(DEVICE)  # shape: (1, 4, 2)
gt = [traj[HIST_LEN - 1]]
pred = [traj[HIST_LEN - 1]]

for i in range(HIST_LEN, min(HIST_LEN + N_STEPS, len(traj) - 1)):
    with torch.no_grad():
        x_pred, Q = model(x_seq)
    pred.append(x_pred.squeeze(0).cpu().numpy())
    gt.append(traj[i + 1])
    # Update sequence: drop oldest, append new prediction
    new_input = x_pred.unsqueeze(1)
    x_seq = torch.cat([x_seq[:, 1:], new_input], dim=1)

gt = np.array(gt)
pred = np.array(pred)

# Print out the loss/error between ground truth and prediction
error = np.linalg.norm(gt - pred, axis=1)
print(f"Prediction Error (Euclidean distance): {error[-1]:.2f} pixels")

# ---------- Plot ----------
fig, ax = plt.subplots(figsize=(6,6))
ax.plot(pred[:,0], pred[:,1], 's--', label="Prediction", color="blue", alpha=0.7)
ax.plot(gt[:,0], gt[:,1], 'o-', label="Ground Truth", color="green")

# Plot uncertainty ellipses
# x = torch.tensor(traj[0], dtype=torch.float32).to(DEVICE).unsqueeze(0)
# for i in range(len(pred)):
#     with torch.no_grad():
#         _, Q = model(x)
#         Q_np = Q.squeeze(0).cpu().numpy()
#         center = pred[i]
#         plot_cov_ellipse(Q_np, center, ax, n_std=2.0, alpha=0.3, color="blue")

#         x, _ = model(x)  # roll forward


plt.title(f"Trajectory Prediction (Track ID: {tid})")
plt.xlabel("x [pixels]"); plt.ylabel("y [pixels]")
plt.legend()
plt.grid(True)
plt.axis("equal")
plt.tight_layout()
plt.show()
