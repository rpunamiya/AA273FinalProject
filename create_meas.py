from test2 import SDDPairs
import pdb 
import pandas as pd 
import os
import numpy as np 

# ---------- Configuration ----------
VIDEO_DIR  = "data/quad/video0"  # <-- match your train video
CHECKPOINT = "noise_model_sdd.pt"

COLS = ["track_id","xmin","ymin","xmax","ymax",
        "frame","lost","occ","gen","label"]

df = pd.read_csv(os.path.join(VIDEO_DIR, "annotations.txt"),
                 sep=r"\s+", header=None, names=COLS, quotechar='"', engine="python")



df = df[(df["label"] == '"Pedestrian"') & (df["lost"] == 0)]
df["cx"] = (df["xmin"] + df["xmax"]) / 2.0
df["cy"] = (df["ymin"] + df["ymax"]) / 2.0

# Create a dataset for each video and combine them

def create_measurements(df):
    Q = np.eye(2)
    cx, cy = np.array(df["cx"]), np.array(df["cy"]) 
    y = np.vstack([cx, cy])
    y_noisy = y.T + np.random.multivariate_normal(np.zeros(2), Q, size=y.shape[1])
    # pdb.set_trace()
    return y_noisy

create_measurements(df)
