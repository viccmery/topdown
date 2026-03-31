
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import matplotlib.patches as mpatches
import matplotlib as mpl
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

""" 
ROSEPLOT APPROACH ANGLE X SPEED DIFFERENCE FOR GH VS PSEUDO POPULATION 

Load GH and PSEUDO nearest-neighbour data and merge them.

Filter interactions to 1–5 mm distance.

Bin approach angles into 30° bins (0–180°).

For each file × condition × angle × distance bin, compute mean speed.

Compare GH vs PSEUDO per angle bin using Mann–Whitney U tests.

Apply FDR correction across angle bins.

Compute the absolute mean speed difference (|PSEUDO − GH|). | average speed (PSEUDO) − average speed (GH) |,

Plot those differences as a polar (rose) plot vs approach angle.

"""

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']

PALETTE = {
    "Socially Isolated": 'darkorange',
     "Pseudo Control": "#F7D455", }

HUE_ORDER = ["Socially Isolated", 'Pseudo Control']

df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/socially-isolated/nearest_neighbour.csv')
df1['condition'] = 'Socially Isolated'
df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/pseudo-n10/socially-isolated/nearest_neighbour.csv')
df2['condition'] = 'Pseudo Control'

df = pd.concat([df1, df2], ignore_index=True)
df["condition"] = df["condition"].astype(str)


angle_edges = np.arange(0, 181, 30)  # 0,30,60,90,120,150,180
df['angle_bin'] = pd.cut(df['approach_angle'], angle_edges, include_lowest=True, right=False)
df['angle_bin'] = df['angle_bin'].astype(str)


df = df[(df["closest_node_distance"] > 1) & (df["closest_node_distance"] <= 5)].copy()

bins = np.arange(1, 5.1, 0.5)
df["bin"] = pd.cut(df["closest_node_distance"], bins, include_lowest=True)
df["bin_center"] = df["bin"].apply(lambda x: x.mid)


replicates = (
    df
    .groupby(
        ["filename", "condition", "angle_bin", "bin_center"],
        observed=True
    )["speed"]
    .mean()
    .reset_index(name="mean_speed")
)
replicates = replicates[replicates["angle_bin"] != "nan"]


results = []

for angle_bin, sub in replicates.groupby("angle_bin"):
    si = sub.loc[sub["condition"] == "Socially Isolated", "mean_speed"].dropna()
    pseudo = sub.loc[sub["condition"] == "Pseudo Control", "mean_speed"].dropna()

    if gh.size == 0 or pseudo.size == 0:
        continue

    u, p = mannwhitneyu(si, pseudo, alternative="two-sided")

    results.append({
        "angle_bin": angle_bin,
        "n_SI": si.size,
        "n_PSEUDO": pseudo.size,
        "u_stat": u,
        "p_raw": p,
        "mean_SI": si.mean(),
        "mean_PSEUDO": pseudo.mean(),
        "delta_mean": pseudo.mean() - si.mean()
    })

stats_df = pd.DataFrame(results)

passed, p_corr, _, _ = multipletests(
    stats_df["p_raw"],
    alpha=0.05,
    method="fdr_bh"
)

stats_df["p_corrected"] = p_corr
stats_df["passes_multiple_test_correction"] = passed

print(stats_df)



# ---- prepare data ----
plot_df = stats_df.copy()

# absolute difference (magnitude only)
plot_df["abs_delta"] = plot_df["delta_mean"].abs()

# extract angle-bin centers from strings like "[0, 30)"
def angle_center(label):
    left = float(label.split(",")[0].strip("[("))
    right = float(label.split(",")[1].strip(" )]"))
    return (left + right) / 2

plot_df["angle_center_deg"] = plot_df["angle_bin"].apply(angle_center)
plot_df = plot_df.sort_values("angle_center_deg")

theta = np.deg2rad(plot_df["angle_center_deg"].to_numpy())
r = plot_df["abs_delta"].to_numpy()

# width = 30° bins
width = np.deg2rad(30)

# ---- plot ----
fig = plt.figure(figsize=(7, 7))
ax = plt.subplot(111, projection="polar")

bars = ax.bar(
    theta,
    r,
    width=width,
    align="center",
    alpha=0.8, color="steelblue",  edgecolor="0.2", linewidth=1.5
)

# optional: mark significant bins
sig = plot_df["passes_multiple_test_correction"].to_numpy()

ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.set_yticklabels([])


ax.set_thetamin(0)
ax.set_thetamax(180)
ax.yaxis.grid(False)


ax.set_yticklabels([])          # no 0.01/0.02 labels
ax.yaxis.grid(False)            # remove circular (radial) grid lines
ax.xaxis.grid(True, linewidth=1, alpha=0.8, color='0.3')  # faint angle spokes


# --- make the outer border less heavy ---
ax.spines["polar"].set_linewidth(1)

# # --- make bars cleaner ---
# for b in bars:
#     b.set_linewidth(0)          # no bar outlines
#     b.set_alpha(0.9)            # slightly softer fill

# --- fix the radius so wedges don’t slam into the border ---
rmax = plot_df["abs_delta"].max() * 1.15      # headroom
ax.set_rlim(0, rmax)

ax.set_frame_on(False)


# --- nicer angle tick labels (optional but usually better) ---
ax.set_xticks(np.deg2rad([0, 30, 60, 90, 120, 150, 180]))
ax.set_xticklabels(["0°","30°","60°","90°","120°","150°","180°"])


ax.set_title(
    "Magnitude of speed difference (|PSEUDO − GH|)\n1–5 mm, per approach angle",
    pad=20
)

plt.tight_layout()
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/lrs_paper/GHxSI/head-approach-angle_speed-difference-roseplot.pdf',format='pdf', bbox_inches='tight')
plt.show()
