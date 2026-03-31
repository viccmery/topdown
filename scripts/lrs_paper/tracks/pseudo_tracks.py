# %%
import cv2
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ===========================================================
# PLOT REAL GROUP IN DARK BLUE WHICH I HAVE BEEN PLOTTING IN
# ===========================================================

df = pd.read_feather('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/group-housed/2025-03-03_11-40-39_td1.tracks.feather')
df = df.sort_values('frame')
df = df[df['frame'] < 300]
output = '/Users/cochral/repos/behavioural-analysis/plots/lrs_paper/tracks_pseudo'

color = '#4B769A'

fig, ax = plt.subplots(figsize=(2, 2), dpi=600)

for track_id, sub in df.groupby("track_id"):
    sub = sub.sort_values("frame")

    ax.plot(
        sub["x_body"],
        sub["y_body"],
        color=color,
        linewidth=0.5,
        solid_capstyle="round"
    )

ax.set_aspect("equal")
ax.set_xlim(100, 1300)
ax.set_ylim(100, 1300)
ax.axis("off")

fig.savefig(
    f"{output}/real_group.pdf",
    bbox_inches="tight",
    pad_inches=0
)
plt.close(fig)



# %%

import cv2
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# =================
# PLOT PSEUDO GROUP
# =================

df = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/pseudo-n10/group-housed/pseudo_population_9.csv')
df = df.sort_values('frame')
df = df[df['frame'] < 300]
output = '/Users/cochral/repos/behavioural-analysis/plots/lrs_paper/tracks_pseudo'


track_colors = {
    0: "#a50026",  # deep red
    1: "#d73027",
    2: "#f46d43",
    3: "#fdae61",
    4: "#fee08b",  # yellow-ish midpoint
    5: "#d9ef8b",
    6: "#a6d96a",
    7: "#66bd63",
    8: "#1a9850",
    9: "#006837",  # deep green
}

for track_id, sub in df.groupby("track_id"):
    sub = sub.sort_values("frame")

    fig, ax = plt.subplots(figsize=(2, 2), dpi=600)

    ax.plot(
        sub["x_body"],
        sub["y_body"],
        color=track_colors[track_id],
        linewidth=1.5,
        solid_capstyle="round"
    )

    ax.set_aspect("equal")
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.axis("off")

    fig.savefig(
        f"{output}/track_{track_id}.pdf",
        bbox_inches="tight",
        pad_inches=0
    )
    plt.close(fig)




# %%
import cv2
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ===============
# PLOT FAKE GROUP 
# ===============

df = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/pseudo-n10/group-housed/pseudo_population_9.csv')
df = df.sort_values('frame')
df = df[df['frame'] < 300]
output = '/Users/cochral/repos/behavioural-analysis/plots/lrs_paper/tracks_pseudo'

track_colors = {
    0: "#893344",
    1: "#bb242c",
    2: "#de8934",
    3: "#6aa629",
    4: "#307832",
    5: "#73b09e",
    6: "#2e767d",
    7: "#484570",
    8: "#aa76b8",
    9: "#eb647b",
}

track_colors = {
    0: "#61c562",
    1: "#90be74",
    2: "#7ab460",
    3: "#6aa629",
    4: "#307832",
    5: "#73b09e",
    6: "#229b91",
    7: "#2e7c36",
    8: "#279951",
    9: "#7c9753",
}

track_colors = {
    0: "#db3554",
    1: "#c86e64",
    2: "#7a1f21",
    3: "#6aa629",
    4: "#307832",
    5: "#73b09e",
    6: "#229b91",
    7: "#2e7c36",
    8: "#279951",
    9: "#aa76b8",
}

track_colors = {
    0: "#a50026",  # deep red
    1: "#d73027",
    2: "#f46d43",
    3: "#fdae61",
    4: "#fee08b",  # yellow-ish midpoint
    5: "#d9ef8b",
    6: "#a6d96a",
    7: "#66bd63",
    8: "#1a9850",
    9: "#006837",  # deep green
}

fig, ax = plt.subplots(figsize=(2, 2), dpi=600)

for track_id, sub in df.groupby("track_id"):
    sub = sub.sort_values("frame")

    ax.plot(
        sub["x_body"],
        sub["y_body"],
        color=track_colors[track_id],
        linewidth=0.5,
        solid_capstyle="round"
    )

ax.set_aspect("equal")
ax.set_xlim(-50, 50)
ax.set_ylim(-50, 50)
ax.axis("off")

fig.savefig(
    f"{output}/fake_group.pdf",
    bbox_inches="tight",
    pad_inches=0
)
plt.close(fig)

# %%
