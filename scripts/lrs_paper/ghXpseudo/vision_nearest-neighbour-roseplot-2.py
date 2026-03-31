
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
"""

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']

PALETTE = {
    "GH": 'steelblue',     
    "PSEUDO": 'skyblue'}

HUE_ORDER = ["GH", "PSEUDO"]


df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/group-housed/nearest_neighbour.csv')
df1['condition'] = 'GH'
df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/pseudo-n10/group-housed/nearest_neighbour.csv')
df2['condition'] = 'PSEUDO'

df = pd.concat([df1, df2], ignore_index=True)
df["condition"] = df["condition"].astype(str)


angle_edges = np.arange(0, 181, 30)  # 0,30,60,90,120,150,180
df['angle_bin'] = pd.cut(df['approach_angle'], angle_edges, include_lowest=True, right=False)
df['angle_bin'] = df['angle_bin'].astype(str)

print('BETWEEN 1-5')
df = df[(df["closest_node_distance"] > 1) & (df["closest_node_distance"] <= 5)].copy()


# ONE value per video × condition × angle_bin
per_video = (
    df.groupby(["filename", "condition", "angle_bin"], observed=True)["speed"]
          .mean()
          .reset_index(name="video_mean_speed")
)
per_video = per_video[per_video["angle_bin"] != "nan"].copy()

# stats per angle bin (10 vs 10)
results = []
for angle_bin, sub in per_video.groupby("angle_bin"):
    gh = sub.loc[sub["condition"] == "GH", "video_mean_speed"].dropna()
    pseudo = sub.loc[sub["condition"] == "PSEUDO", "video_mean_speed"].dropna()
    if gh.size == 0 or pseudo.size == 0:
        continue
    u, p = mannwhitneyu(gh, pseudo, alternative="two-sided")
    results.append({
        "angle_bin": angle_bin,
        "n_GH": gh.size,
        "n_PSEUDO": pseudo.size,
        "u_stat": u,
        "p_raw": p,
        "mean_GH": gh.mean(),
        "mean_PSEUDO": pseudo.mean(),
        "delta_mean": pseudo.mean() - gh.mean()
    })

stats = pd.DataFrame(results)
passed, p_corr, _, _ = multipletests(stats["p_raw"], alpha=0.05, method="fdr_bh")
stats["p_corrected"] = p_corr
stats["passes_multiple_test_correction"] = passed
print(stats)





bins = np.arange(0, 5.1, 0.5)
df["bin"] = pd.cut(df["closest_node_distance"], bins, include_lowest=True)
df["bin_center"] = df["bin"].apply(lambda x: x.mid)

# 1) mean speed per video x condition x angle-bin x distance-bin
replicates = (
    df.groupby(["filename", "condition", "angle_bin", "bin_center"], observed=True)["speed"]
      .mean()
      .reset_index(name="mean_speed")
)
replicates = replicates[replicates["angle_bin"] != "nan"].copy()

# 2) AUC (area under speed vs distance) per video x condition x angle-bin
def auc_speed_vs_dist(g):
    g = g.sort_values("bin_center")
    x = g["bin_center"].to_numpy()
    y = g["mean_speed"].to_numpy()
    if len(x) < 2:
        return np.nan
    return np.trapz(y, x)   # integrates speed across 1–5 mm

auc_df = (
    replicates.groupby(["filename", "condition", "angle_bin"], observed=True)
              .apply(auc_speed_vs_dist)
              .reset_index(name="auc_speed_1to5mm")
              .dropna()
)

# sanity check: should be ~10 GH + ~10 PSEUDO per angle bin
print(
    auc_df.groupby(["angle_bin", "condition"])["auc_speed_1to5mm"]
          .count()
          .unstack(fill_value=0)
)

# OPTIONAL: per-angle-bin GH vs PSEUDO stats on these per-video AUC values
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

results = []
for angle_bin, sub in auc_df.groupby("angle_bin"):
    gh = sub.loc[sub["condition"] == "GH", "auc_speed_1to5mm"].dropna()
    pseudo = sub.loc[sub["condition"] == "PSEUDO", "auc_speed_1to5mm"].dropna()
    if gh.size == 0 or pseudo.size == 0:
        continue
    u, p = mannwhitneyu(gh, pseudo, alternative="two-sided")
    results.append({
        "angle_bin": angle_bin,
        "n_GH": gh.size,
        "n_PSEUDO": pseudo.size,
        "u_stat": u,
        "p_raw": p,
        "mean_GH_auc": gh.mean(),
        "mean_PSEUDO_auc": pseudo.mean(),
        "delta_mean_auc": pseudo.mean() - gh.mean()
    })

stats_auc = pd.DataFrame(results)
passed, p_corr, _, _ = multipletests(stats_auc["p_raw"], alpha=0.05, method="fdr_bh")
stats_auc["p_corrected"] = p_corr
stats_auc["passes_multiple_test_correction"] = passed

print(stats_auc)

# auc_df is the thing you wanted: 10 values per condition per angle bin (per video)
# columns: filename, condition, angle_bin, auc_speed_1to5mm


# replicates = (
#     df
#     .groupby(
#         ["filename", "condition", "angle_bin", "bin_center"],
#         observed=True
#     )["speed"]
#     .mean()
#     .reset_index(name="mean_speed")
# )
# replicates = replicates[replicates["angle_bin"] != "nan"]


# results = []

# for angle_bin, sub in replicates.groupby("angle_bin"):
#     gh = sub.loc[sub["condition"] == "GH", "mean_speed"].dropna()
#     pseudo = sub.loc[sub["condition"] == "PSEUDO", "mean_speed"].dropna()

#     if gh.size == 0 or pseudo.size == 0:
#         continue

#     u, p = mannwhitneyu(gh, pseudo, alternative="two-sided")

#     results.append({
#         "angle_bin": angle_bin,
#         "n_GH": gh.size,
#         "n_PSEUDO": pseudo.size,
#         "u_stat": u,
#         "p_raw": p,
#         "mean_GH": gh.mean(),
#         "mean_PSEUDO": pseudo.mean(),
#         "delta_mean": pseudo.mean() - gh.mean()
#     })

# stats_df = pd.DataFrame(results)

# passed, p_corr, _, _ = multipletests(
#     stats_df["p_raw"],
#     alpha=0.05,
#     method="fdr_bh"
# )

# stats_df["p_corrected"] = p_corr
# stats_df["passes_multiple_test_correction"] = passed

# print(stats_df)



# # ---- prepare data ----
# plot_df = stats_df.copy()

# # absolute difference (magnitude only)
# plot_df["abs_delta"] = plot_df["delta_mean"].abs()

# # extract angle-bin centers from strings like "[0, 30)"
# def angle_center(label):
#     left = float(label.split(",")[0].strip("[("))
#     right = float(label.split(",")[1].strip(" )]"))
#     return (left + right) / 2

# plot_df["angle_center_deg"] = plot_df["angle_bin"].apply(angle_center)
# plot_df = plot_df.sort_values("angle_center_deg")

# theta = np.deg2rad(plot_df["angle_center_deg"].to_numpy())
# r = plot_df["abs_delta"].to_numpy()

# # width = 30° bins
# width = np.deg2rad(30)

# # ---- plot ----
# fig = plt.figure(figsize=(7, 7))
# ax = plt.subplot(111, projection="polar")

# bars = ax.bar(
#     theta,
#     r,
#     width=width,
#     align="center",
#     alpha=0.8, color="steelblue",  edgecolor="0.2", linewidth=1.5
# )

# # optional: mark significant bins
# sig = plot_df["passes_multiple_test_correction"].to_numpy()

# ax.set_theta_zero_location("N")
# ax.set_theta_direction(-1)
# ax.set_yticklabels([])


# ax.set_thetamin(0)
# ax.set_thetamax(180)
# ax.yaxis.grid(False)


# ax.set_yticklabels([])          # no 0.01/0.02 labels
# ax.yaxis.grid(False)            # remove circular (radial) grid lines
# ax.xaxis.grid(True, linewidth=1, alpha=0.8, color='0.3')  # faint angle spokes


# # --- make the outer border less heavy ---
# ax.spines["polar"].set_linewidth(1)

# # # --- make bars cleaner ---
# # for b in bars:
# #     b.set_linewidth(0)          # no bar outlines
# #     b.set_alpha(0.9)            # slightly softer fill

# # --- fix the radius so wedges don’t slam into the border ---
# rmax = plot_df["abs_delta"].max() * 1.15      # headroom
# ax.set_rlim(0, rmax)

# ax.set_frame_on(False)


# # --- nicer angle tick labels (optional but usually better) ---
# ax.set_xticks(np.deg2rad([0, 30, 60, 90, 120, 150, 180]))
# ax.set_xticklabels(["0°","30°","60°","90°","120°","150°","180°"])


# ax.set_title(
#     "Magnitude of speed difference (|PSEUDO − GH|)\n1–5 mm, per approach angle",
#     pad=20
# )

# plt.tight_layout()
# plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/lrs_paper/ghXpseudo/head-approach-angle_speed-difference-roseplot.pdf',format='pdf', bbox_inches='tight')
# plt.show()
