import cv2
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

"""

1. IMAGE OVERLAY WITH TRACK NODES 

2. EXAMPLE OF CUMULATIVE TRACKS 

"""

""" 1. IMAGE OVERLAY WITH TRACK NODES """

video_path = '/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/group-housed/2025-03-07_14-59-00_td11.mp4'

df = pd.read_feather('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/group-housed/2025-03-07_14-59-00_td11.tracks.feather')
df = df.sort_values('frame')
df = df[df['frame'] < 300]


output = '/Users/cochral/repos/behavioural-analysis/plots/lrs_paper/tracks'

# =========================
# 1) Frame 20 overlay -> PDF
# =========================
FRAME_TO_EXPORT = 22
frame_df = df[df['frame'] == FRAME_TO_EXPORT].copy()

# cap = cv2.VideoCapture(video_path)
# cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_TO_EXPORT)
# ok, frame_img = cap.read()
# cap.release()

cap = cv2.VideoCapture(video_path)
for _ in range(FRAME_TO_EXPORT + 1):
    ok, frame_img = cap.read()
cap.release()


# BGR -> RGB for matplotlib
frame_rgb = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)

black_thresh = 60  # 0–255

mask = np.all(frame_rgb < black_thresh, axis=2)
frame_rgb[mask] = [255, 255, 255]
# plt.figure(figsize=(8, 8)) 
# plt.imshow(frame_rgb)
plt.figure(figsize=(6, 6), dpi=600)          # 14*100 = 1400 px
plt.imshow(frame_rgb, interpolation="nearest") # stops smoothing blur


plt.axis("off")

# overlay head/body/tail points (and labels) per track
for tid, sub in frame_df.groupby("track_id"):
    # head
    xh, yh = sub["x_head"].iloc[0], sub["y_head"].iloc[0]
    xb, yb = sub["x_body"].iloc[0], sub["y_body"].iloc[0]
    xt, yt = sub["x_tail"].iloc[0], sub["y_tail"].iloc[0]

    # skip if missing
    if np.isnan([xh, yh, xb, yb, xt, yt]).any():
        continue

    plt.plot(
        [xh, xb, xt],
        [yh, yb, yt],
        color="navy",
        linewidth=0.6,
        alpha=0.4,
        zorder=1
    )


    # points
    plt.scatter([xh], [yh], s=1, color='lightskyblue')
    plt.scatter([xb], [yb], s=1, color='steelblue')
    plt.scatter([xt], [yt], s=1, color='cornflowerblue')


    # # tiny labels
    # plt.text(xh + 4, yh + 4, "H", fontsize=6,  color='darkred', fontweight='bold')
    # plt.text(xb + 4, yb + 4, "B", fontsize=6,  color='indianred', fontweight='bold')
    # plt.text(xt + 4, yt + 4, "T", fontsize=6,  color='lightcoral', fontweight='bold')

out_pdf_frame = f"{output}/overlay{FRAME_TO_EXPORT}.pdf"
plt.savefig(out_pdf_frame, format="pdf", bbox_inches="tight", pad_inches=0)
plt.close()




# ==========================================
# 2) Cumulative body paths (0–99) -> PDF
# ==========================================
# Use video size so coordinates align naturally

df = pd.read_feather('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/group-housed/2025-03-03_11-40-39_td1.tracks.feather')
df = df.sort_values('frame')
df = df[df['frame'] < 200]

cap = cv2.VideoCapture(video_path)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) 
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

canvas = np.full((H, W, 3), 255, dtype=np.uint8)  # white background

frames = sorted(df["frame"].unique())
n_frames = len(frames)

track_ids = sorted(df["track_id"].dropna().unique())
cmap = cm.get_cmap("Blues_r")  # red palette

def rgba_to_bgr(rgba):
    r, g, b, a = rgba
    return (int(b*255), int(g*255), int(r*255))  # OpenCV wants BGR

last_xy = {}

vals = np.linspace(0, 1, len(track_ids))

for i, frame in enumerate(frames):
    fdf = df[df["frame"] == frame]

    t = i / (n_frames - 1)
    max_val = 0.8
    color = rgba_to_bgr(cmap(max_val * t))

    for tid, sub in fdf.groupby("track_id"):
        xb, yb = sub["x_body"].iloc[0], sub["y_body"].iloc[0]
        if np.isnan([xb, yb]).any():
            continue

        xb_i, yb_i = int(xb), int(yb)

        if tid in last_xy:
            x_prev, y_prev = last_xy[tid]
            cv2.line(canvas, (x_prev, y_prev), (xb_i, yb_i), color, 6)
        last_xy[tid] = (xb_i, yb_i)


out_pdf_path = f"{output}/tracks.pdf"
fig, ax = plt.subplots(figsize=(2, 2), dpi=600)  # 14*100 = 1400 px
ax.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), interpolation="nearest")
ax.axis("off")
fig.savefig(out_pdf_path, format="pdf", bbox_inches="tight", pad_inches=0)
plt.close(fig)








