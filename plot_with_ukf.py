# ------------------------------------------------------------
# plot_with_ukf.py  –  UKF-fused roll-out & visualisation
# ------------------------------------------------------------
import os, torch, pandas as pd, numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.animation as animation
from create_meas import *
from matplotlib.animation import FuncAnimation

# ---------- Configuration ----------
VIDEO_DIR  = r"C:\Users\rpuna\OneDrive - Stanford\Spring 2025\AA 273\AA273FinalProject\data\quad\video0"
CHECKPOINT = "noise_model_sdd.pt"
TRACK_ID   = 0     # choose a pedestrian id present in this video
N_STEPS    = 500    # max rollout horizon
HIST_LEN   = 10      # must match training
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
# ------------------------------------

# ---------- Utils ------------------------------------------------------
def triangular_param(vec, d, eps=1e-3):
    L = torch.zeros(vec.size(0), d, d, device=vec.device)
    idx = torch.tril_indices(d, d, 0)
    L[:, idx[0], idx[1]] = vec
    diag = torch.arange(d, device=vec.device)
    L[:, diag, diag] = F.softplus(L[:, diag, diag]) + eps
    return L @ L.transpose(-1, -2)

def plot_cov_ellipse(Q, center, ax, n_std=2.0, **kw):
    if Q.shape != (2,2): return
    vals, vecs = np.linalg.eigh(Q)
    angle = np.degrees(np.arctan2(*vecs[:,1][::-1]))
    w, h = 2*n_std*np.sqrt(vals)
    ax.add_patch(Ellipse(xy=center, width=w, height=h, angle=angle, **kw))

# ---------- NoiseModel (same as training) ------------------------------
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


# ---------- Load trajectory -------------------------------------------
COLS = ["track_id","xmin","ymin","xmax","ymax",
        "frame","lost","occ","gen","label"]

df = pd.read_csv(os.path.join(VIDEO_DIR,"annotations.txt"),
                 sep=r"\s+", header=None, names=COLS,
                 quotechar='"', engine="python")
df = df[df["label"].isin(['"Pedestrian"', '"Biker"']) & (df["lost"] == 0)]
df["cx"] = (df["xmin"]+df["xmax"])/2
df["cy"] = (df["ymin"]+df["ymax"])/2

traj = (df[df["track_id"]==TRACK_ID]
        .sort_values("frame")[["cx","cy"]].to_numpy(dtype="float32"))
if len(traj) < HIST_LEN+1:
    raise ValueError(f"Track {TRACK_ID} too short ({len(traj)} frames).")

# ---------- Load model -------------------------------------------------
model = NoiseModel().to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.eval()

# ---------- UKF helper -------------------------------------------------
def fx(x, dt, delta):
    return x + delta                    # constant-velocity residual

def hx(x):
    return x                            # we “observe” true position exactly

def make_ukf(init_x):
    pts = MerweScaledSigmaPoints(n=2, alpha=1e-3, beta=2, kappa=0)
    ukf = UnscentedKalmanFilter(2, 2, dt=1.0, hx=hx,
                                fx=lambda x,dt: fx(x,dt,np.zeros(2)),
                                points=pts)
    ukf.x = init_x.astype(float)
    ukf.P = np.eye(2)*1e-2              # small initial cov
    ukf.R = np.diag((10,10))             # zero measurement noise
    return ukf

# ---------- Roll-out with UKF -----------------------------------------
y = create_measurements(df)
y = y.astype(np.float32)  # ensure float32 for consistency
seq = torch.tensor(y[:HIST_LEN], device=DEVICE).unsqueeze(0)  # (1,T,2)
gt, ukf_mu, ukf_P = [traj[HIST_LEN-1]], [], []
ukf = make_ukf(traj[HIST_LEN-1])

for step in range(HIST_LEN, min(HIST_LEN+N_STEPS, len(traj)-1)):
    # start_time = time.time()

    with torch.no_grad():
        mu_torch, Q_torch = model(seq)
    mu = mu_torch.squeeze(0).cpu().numpy()
    Q  = Q_torch.squeeze(0).cpu().numpy()
    print(f"Step {step-HIST_LEN+1}: mu={mu}, Q={Q}")

    ukf.Q = Q
    ukf.fx = lambda x,dt,d=mu-ukf.x: fx(x,dt,d)
    ukf.predict()

    z = y[step+1]                    # perfect measurement
    ukf.update(z)

    ukf_mu.append(ukf.x.copy())
    ukf_P.append(ukf.P.copy())
    gt.append(traj[step+1])

    new = torch.tensor(ukf.x, dtype=torch.float32, device=DEVICE).unsqueeze(0).unsqueeze(0)
    seq = torch.cat([seq[:,1:], new], dim=1)

    # end_time = time.time()
    # print(f"Step {step-HIST_LEN+1}: UKF update took {end_time - start_time:.4f} seconds")

gt      = np.array(gt)
ukf_mu  = np.array(ukf_mu)

# ---------- Compute and Print Error -----------------------------------
errors = np.linalg.norm(ukf_mu - gt[1:], axis=1)  # Skip first gt which was initial
rmse = np.sqrt(np.mean(errors**2))
final_error = errors[-1]

print(f"RMSE over rollout: {rmse:.2f} pixels")
# print(f"Final frame error: {final_error:.2f} pixels")
# # Print difference between UKF and ground truth for all frames
# diff = np.linalg.norm(ukf_mu - gt[1:], axis=1)
# print("Frame-wise differences (UKF vs GT):")
# for i, d in enumerate(diff):
#     print(f"Frame {i+1}: {d:.2f} pixels")
# # Print final ground truth and UKF position
# print(f"Final GT position: {gt[-1]}")
# print(f"Final UKF position: {ukf_mu[-1]}")

# ---------- Plot -------------------------------------------------------
# fig, ax = plt.subplots(figsize=(6,6))
# ax.plot(ukf_mu[:,0], ukf_mu[:,1], 's--', label="UKF fused", color="blue")
# ax.plot(gt[:,0],     gt[:,1], 'o-', label="Ground Truth",  color="green")

# # for m,P in zip(ukf_mu, ukf_P):
# #     plot_cov_ellipse(P, m, ax, n_std=2.0, alpha=0.2, color="blue")

# ax.set_title(f"UKF fusion – Track {TRACK_ID}")
# ax.set_xlabel("x [px]"); ax.set_ylabel("y [px]")
# ax.axis("equal"); ax.grid(True); ax.legend(); plt.tight_layout(); plt.show()

# PLOT SINGLE AGENT ROLL-OUT WITH ANIMATION

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(0, 1968)
ax.set_ylim(0, 1080)
ax.set_title(f"UKF Fusion – Track {TRACK_ID}")
ax.set_xlabel("x [px]")
ax.set_ylabel("y [px]")
ax.grid(True)
ax.set_aspect("equal")

gt_line, = ax.plot([], [], 'o-', color='green', label="Ground Truth", markersize=1)
ukf_line, = ax.plot([], [], 's--', color='blue', label="UKF Fused", markersize=1, alpha=0.5)
gt_scatter = ax.scatter([], [], color='green', s=2)
ukf_scatter = ax.scatter([], [], color='blue', s=2)

ax.legend()

ellipse_patches = []

def init():
    gt_line.set_data([], [])
    ukf_line.set_data([], [])
    gt_scatter.set_offsets(np.empty((0, 2)))
    ukf_scatter.set_offsets(np.empty((0, 2)))
    for e in ellipse_patches:
        e.remove()
    ellipse_patches.clear()
    return gt_line, ukf_line, gt_scatter, ukf_scatter

ellipse_interval = 10   # Keep every 10th ellipse

def update(frame):
    gt_line.set_data(gt[:frame+1, 0], gt[:frame+1, 1])
    ukf_line.set_data(ukf_mu[:frame+1, 0], ukf_mu[:frame+1, 1])
    gt_scatter.set_offsets(gt[frame])
    ukf_scatter.set_offsets(ukf_mu[frame])

    # Add a persistent ellipse only every `ellipse_interval` frames
    if frame < len(ukf_P) and frame % ellipse_interval == 0:
        P = ukf_P[frame]
        center = ukf_mu[frame]
        plot_cov_ellipse(P, center, ax, n_std=2.0, alpha=0.2, color='blue')
        ellipse_patches.append(ax.patches[-1])

    return gt_line, ukf_line, gt_scatter, ukf_scatter, *ellipse_patches

ani = animation.FuncAnimation(fig, update, frames=len(ukf_mu),
                              init_func=init, blit=False, interval=1, repeat=False)

plt.tight_layout()
plt.show()

ani.save("pedestrian_ukf_rollout.gif", writer='pillow', fps=10)

# ADD THIS AT THE BOTTOM OF YOUR SCRIPT AFTER MODEL AND UTILS ARE LOADED

# Filter valid tracks (with enough length)
# Filter first 10 valid track_ids with enough history
# frame_stride = 5  # Show every 5th frame (adjust as needed)

# valid_ids = []
# id_to_df = {}

# for tid in df["track_id"].unique():
#     traj_df = df[df["track_id"] == tid].sort_values("frame")
#     if len(traj_df) >= HIST_LEN + 1:
#         valid_ids.append(tid)
#         id_to_df[tid] = traj_df
#     if len(valid_ids) == 10:
#         break

# ukf_results = {}
# max_global_frame = 0

# for tid in valid_ids:
#     traj_df = id_to_df[tid]
#     traj = traj_df[["cx", "cy"]].to_numpy(dtype="float32")
#     start_frame = int(traj_df["frame"].iloc[HIST_LEN])  # UKF starts after history
#     y = traj.copy()

#     seq = torch.tensor(y[:HIST_LEN], device=DEVICE).unsqueeze(0)
#     gt = [traj[HIST_LEN - 1]]
#     ukf_mu = []
#     ukf = make_ukf(traj[HIST_LEN - 1])

#     for step in range(HIST_LEN, len(traj) - 1):
#         with torch.no_grad():
#             mu_torch, Q_torch = model(seq)
#         mu = mu_torch.squeeze(0).cpu().numpy()
#         Q = Q_torch.squeeze(0).cpu().numpy()

#         ukf.Q = Q
#         ukf.fx = lambda x, dt, d=mu - ukf.x: fx(x, dt, d)
#         ukf.predict()
#         z = y[step + 1]
#         ukf.update(z)

#         ukf_mu.append(ukf.x.copy())
#         gt.append(traj[step + 1])
#         new = torch.tensor(ukf.x, dtype=torch.float32, device=DEVICE).unsqueeze(0).unsqueeze(0)
#         seq = torch.cat([seq[:, 1:], new], dim=1)

#     ukf_results[tid] = {
#         "start_frame": start_frame,
#         "gt": np.array(gt),
#         "ukf": np.array(ukf_mu)
#     }

#     max_global_frame = max(max_global_frame, start_frame + len(ukf_mu))

# # Initialize figure
# fig, ax = plt.subplots(figsize=(8, 8))
# ax.set_title("UKF Real-Time Rollouts – First 10 Agents")
# ax.set_xlabel("x [px]")
# ax.set_ylabel("y [px]")
# ax.grid(True)
# ax.set_aspect("equal")

# colors = plt.cm.get_cmap("tab10", len(valid_ids))
# lines_gt, lines_ukf = {}, {}

# # Axis limits
# all_gt = np.concatenate([v["gt"] for v in ukf_results.values()])
# ax.set_xlim(all_gt[:, 0].min() - 10, all_gt[:, 0].max() + 10)
# ax.set_ylim(all_gt[:, 1].min() - 10, all_gt[:, 1].max() + 10)

# for i, tid in enumerate(valid_ids):
#     color = colors(i % 10)
#     lines_gt[tid], = ax.plot([], [], '-', color=color, label=f"GT {tid}", alpha=0.5)
#     lines_ukf[tid], = ax.plot([], [], '--', color=color, label=f"UKF {tid}")

# ax.legend(fontsize="x-small", ncol=2, loc='upper right')

# def init():
#     for line in lines_gt.values():
#         line.set_data([], [])
#     for line in lines_ukf.values():
#         line.set_data([], [])
#     return list(lines_gt.values()) + list(lines_ukf.values())

# def update(global_frame):
#     for tid in valid_ids:
#         res = ukf_results[tid]
#         start = res["start_frame"]
#         rel_frame = global_frame - start
#         if rel_frame >= 0 and rel_frame < len(res["ukf"]):
#             lines_gt[tid].set_data(res["gt"][:rel_frame+1, 0], res["gt"][:rel_frame+1, 1])
#             lines_ukf[tid].set_data(res["ukf"][:rel_frame+1, 0], res["ukf"][:rel_frame+1, 1])
#     return list(lines_gt.values()) + list(lines_ukf.values())

# ani = FuncAnimation(fig, update,
#                     frames=range(0, max_global_frame, frame_stride),
#                     init_func=init, blit=False, interval=50, repeat=False)


# plt.tight_layout()
# plt.show()

# # ani.save("ukf_rollouts.gif", writer='pillow', fps=10)

# from matplotlib.animation import FFMpegWriter

# writer = FFMpegWriter(fps=20, metadata=dict(artist='UKF Tracker'), bitrate=1800)
# ani.save("ukf_rollouts.mp4", writer=writer)

