
# %%

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
ROSEPLOT APPROACH ANGLE X SPEED DIFFERENCE FOR SI VS PSEUDO POPULATION 
"""

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']

PALETTE = {
    "PSEUDO": "#F7D455",     
    "SI": 'darkorange',}

HUE_ORDER = [ "SI", "PSEUDO"]


df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/socially-isolated/nearest_neighbour.csv')
df1['condition'] = 'SI'
df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/pseudo-n10/socially-isolated/nearest_neighbour.csv')
df2['condition'] = 'PSEUDO'

df = pd.concat([df1, df2], ignore_index=True)
df["condition"] = df["condition"].astype(str)

df["approach_angle"] = pd.to_numeric(df["approach_angle"], errors="coerce")
angle_edges = np.arange(0, 181, 30)  # 0,30,60,90,120,150,180
df['angle_bin'] = pd.cut(df['approach_angle'], angle_edges, include_lowest=True, right=False)
df = df.dropna(subset=["angle_bin"])
df['angle_bin'] = df['angle_bin'].astype(str)


# --- per-file baseline mean speed at 10–20 mm ---
baseline = (
    df.loc[df["closest_node_distance"].between(10, 20, inclusive="both")]
      .groupby(["condition", "filename"])["speed"]
      .mean()
      .rename("mean_speed_10_20")
      .reset_index()
)

# merge baseline back onto all rows
df = df.merge(baseline, on=["condition", "filename"], how="left")
# drop files that don't have any baseline data in 10–20 mm
df = df.dropna(subset=["mean_speed_10_20"]).copy()
# normalize
df["speed_norm"] = df["speed"] / df["mean_speed_10_20"]





df = df[(df["closest_node_distance"] > 1) & (df["closest_node_distance"] <= 5)].copy()

bins = np.arange(1, 5.1, 0.5)
df["bin"] = pd.cut(df["closest_node_distance"], bins, include_lowest=True, right=True)
df["bin_center"] = df["bin"].apply(lambda x: x.mid)

## GROUP BY FILE FIRST 
distance_means = df.groupby(['condition', 'filename', 'angle_bin', 'bin_center'])['speed_norm'].mean().reset_index()


## GROUP BY CONDITION
condition_mean_curves = distance_means.groupby(['condition', 'angle_bin', 'bin_center'])['speed_norm'].mean().reset_index().rename(columns={'speed_norm': 'mean_speed'})


wide = condition_mean_curves.pivot(
    index=["angle_bin", "bin_center"],
    columns="condition",
    values="mean_speed"
).reset_index()

wide = wide.dropna(subset=["SI", "PSEUDO"]).sort_values(["angle_bin", "bin_center"])

# difference between curves at each distance bin
wide["speed_diff"] = wide["PSEUDO"] - wide["SI"]
wide = wide.sort_values(["angle_bin", "bin_center"])

# integrate per angle bin
delta_area_real = (
    wide
    .groupby("angle_bin")
    .apply(
        lambda g: np.trapz(
            g["speed_diff"].to_numpy(),
            g["bin_center"].to_numpy()
        )))

print(delta_area_real)



## FUNCTION TO COMPUTE AREA DIFFERENCE
def compute_area_difference(dataframe):
    distance_means = dataframe.groupby(['condition', 'filename', 'angle_bin', 'bin_center'])['speed_norm'].mean().reset_index()
    condition_mean_curves = distance_means.groupby(['condition', 'angle_bin', 'bin_center'])['speed_norm'].mean().reset_index().rename(columns={'speed_norm': 'mean_speed'})
    wide = condition_mean_curves.pivot(index=["angle_bin", "bin_center"], columns="condition", values="mean_speed").reset_index()
    wide = wide.dropna(subset=["SI", "PSEUDO"]).sort_values(["angle_bin", "bin_center"])
    wide["speed_diff"] = wide["PSEUDO"] - wide["SI"]
    return wide.groupby("angle_bin").apply(lambda g: np.trapz(g["speed_diff"].to_numpy(), g["bin_center"].to_numpy()))


## BOOTSTRAP
si_files = df[df["condition"]=="SI"]["filename"].unique()
ps_files = df[df["condition"]=="PSEUDO"]["filename"].unique()

resamples = 1000
boot = []

for b in range(resamples):
    si_sample = np.random.choice(si_files, size=10, replace=True)
    ps_sample = np.random.choice(ps_files, size=10, replace=True)

    si_boot = pd.concat([df[(df["condition"]=="SI") & (df["filename"]==f)] for f in si_sample],ignore_index=True)
    ps_boot = pd.concat([df[(df["condition"]=="PSEUDO") & (df["filename"]==f)] for f in ps_sample],ignore_index=True)

    df_boot = pd.concat([si_boot, ps_boot], ignore_index=True)

    boot.append(compute_area_difference(df_boot).rename(b))

boot_df = pd.concat(boot, axis=1)

print(boot_df)


## RESULTS TABLE
## MEAN AREA DIFFERENCE
results = delta_area_real.reset_index()
results.columns = ["angle_bin", "relative_area"]

# BOOTSTRAP CONFIDENCE INTERVALS
results["ci_low"]  = boot_df.quantile(0.025, axis=1).values
results["ci_high"] = boot_df.quantile(0.975, axis=1).values

results["p_value"] = (
    2 * np.minimum(
        (boot_df <= 0).mean(axis=1),
        (boot_df >= 0).mean(axis=1)
    ).values
)

print(results)

results.to_csv('/Users/cochral/repos/behavioural-analysis/plots/lrs_paper/GHxSI/bootstrap_results.csv', index=False)




# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

results = pd.read_csv('/Users/cochral/repos/behavioural-analysis/plots/lrs_paper/GHxSI/bootstrap_results.csv')

# --- angle centres ---
def angle_center(label):
    left = float(label.split(",")[0].strip("[("))
    right = float(label.split(",")[1].strip(" )]"))
    return (left + right) / 2

results["angle_center_deg"] = results["angle_bin"].apply(angle_center)
results = results.sort_values("angle_center_deg")

theta = np.deg2rad(results["angle_center_deg"].to_numpy())
width = np.deg2rad(30)

# --- SIGNED effect (PSEUDO − GH) ---
delta = results["relative_area"].to_numpy()
ci_low = results["ci_low"].to_numpy()
ci_high = results["ci_high"].to_numpy()

r = np.abs(delta)

# convert signed CI into magnitude CI
mag_lo = np.where(delta >= 0, ci_low, -ci_high)
mag_hi = np.where(delta >= 0, ci_high, -ci_low)

crosses_zero = (ci_low <= 0) & (ci_high >= 0)
mag_lo = np.where(crosses_zero, 0.0, mag_lo)

ci_low_plot = np.clip(mag_lo, 0, None)
ci_high_plot = np.clip(mag_hi, 0, None)


# --- plot ---
fig = plt.figure(figsize=(7, 7))
ax = plt.subplot(111, projection="polar")

ax.bar(
    theta,
    r,
    width=width,
    align="center",
    color="darkorange",
    edgecolor="0.2",
    linewidth=1.5,
)

# CI lines (correct for the signed effect)
for th, lo, hi in zip(theta, ci_low_plot, ci_high_plot):
    ax.plot([th, th], [lo, hi], color="black", linewidth=2, solid_capstyle="round")


# formatting
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.set_thetamin(0)
ax.set_thetamax(180)

ax.set_yticklabels([])
ax.yaxis.grid(False)
ax.xaxis.grid(True, linewidth=1, alpha=0.8, color="0.3")
ax.spines["polar"].set_linewidth(1)

rmax = max(abs(ci_high_plot).max(), abs(r).max()) * 1.2
ax.set_rlim(0, rmax)



ax.text(np.deg2rad(15), r[0] + (rmax * 0.12), "*",
        ha="center", va="bottom", fontsize=18, fontweight="bold")

# [30,60) → ***
ax.text(np.deg2rad(45), r[1] + (rmax * 0.12), "***",
        ha="center", va="bottom", fontsize=18, fontweight="bold")



ax.set_frame_on(False)

ax.set_xticks(np.deg2rad([0, 30, 60, 90, 120, 150, 180]))
ax.set_xticklabels(["0°","30°","60°","90°","120°","150°","180°"])

ax.set_title(
    "Signed speed difference",
    pad=20
)

plt.tight_layout()
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/lrs_paper/GHxSI/roseplot.pdf',
            format='pdf', bbox_inches='tight')
plt.show()


# %%
