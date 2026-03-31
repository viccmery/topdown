import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# ------------------ plotting style ------------------
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']

# ------------------ load data ------------------
df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/group-housed/nearest_neighbour.csv')
df1['condition'] = 'Group Housed'

df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/socially-isolated/nearest_neighbour.csv')
df2['condition'] = 'Socially Isolated'

df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/pseudo-n10/group-housed/nearest_neighbour.csv')
df2['condition'] = 'Socially Isolated'

df = pd.concat([df1, df2], ignore_index=True)

# ------------------ bins ------------------
bins = np.linspace(0, 90, 90)
df['bin'] = pd.cut(df['head_distance'], bins, include_lowest=True)
df = df.dropna(subset=['bin'])  # drop any rows outside bin range

# Use categorical bins directly (this avoids float-midpoint weirdness)
bin_order = df['bin'].cat.categories

# ------------------ counts per file per bin (WITH ZERO FILL) ------------------
# table: rows = (filename, condition), cols = bins, values = counts
count_table = (
    df.groupby(['filename', 'condition', 'bin'])
      .size()
      .unstack(fill_value=0)          # <<< critical: add missing bins as zeros
      .reindex(columns=bin_order, fill_value=0)
)

# convert to fractions within each file (row sums to 1)
frac_table = count_table.div(count_table.sum(axis=1), axis=0)

# long format for bootstrapping per bin
frac_long = (
    frac_table
    .stack()
    .reset_index()
    .rename(columns={0: 'fraction'})
)

# ------------------ bootstrap difference per bin ------------------
def bootstrap_diff_for_bin(df_bin, n_boot=2000):
    gh = df_bin.loc[df_bin['condition'] == 'Group Housed', 'fraction'].to_numpy()
    si = df_bin.loc[df_bin['condition'] == 'Socially Isolated', 'fraction'].to_numpy()

    # if either condition truly missing, skip
    if gh.size == 0 or si.size == 0:
        return np.nan, np.nan, np.nan

    diffs = np.empty(n_boot)
    for i in range(n_boot):
        diffs[i] = np.random.choice(gh, gh.size, replace=True).mean() - np.random.choice(si, si.size, replace=True).mean()

    return diffs.mean(), np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)

diff_df = (
    frac_long
    .groupby('bin')
    .apply(lambda x: pd.Series(
        bootstrap_diff_for_bin(x),
        index=['diff', 'ci_low', 'ci_high']
    ))
    .dropna()
    .reset_index()
)

# ------------------ x values (bin centers) ------------------
diff_df['bin_center'] = diff_df['bin'].apply(lambda b: b.mid)

# ------------------ plot ------------------
plt.figure(figsize=(8, 8))

plt.plot(diff_df['bin_center'], diff_df['diff'], color='black', lw=2)
plt.fill_between(diff_df['bin_center'], diff_df['ci_low'], diff_df['ci_high'], alpha=0.3)

plt.axhline(0, color='grey', linestyle='--')

plt.xlabel('Nearest Neighbour Distance (mm)', fontsize=16, fontweight='bold')
plt.ylabel('Δ Fraction of Frames\n(Group Housed − Socially Isolated)', fontsize=16, fontweight='bold')

plt.xlim(0, 60)
sns.despine()
plt.tight_layout()
plt.show()



import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

# ------------------ style ------------------
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']

PALETTE = {"Group Housed": "steelblue", "Pseudo Control": "skyblue"}
ORDER = ["Group Housed", "Pseudo Control"]

# ------------------ load ------------------
df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/group-housed/nearest_neighbour.csv')
df1["condition"] = "Group Housed"
df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/pseudo-n10/group-housed/nearest_neighbour.csv')
df2["condition"] = "Pseudo Control"
df = pd.concat([df1, df2], ignore_index=True)

# ------------------ OPTION 1: mean NN distance per track ------------------
opt1 = (
    df.groupby(["filename", "condition", 'track_id'], as_index=False)["head_distance"]
      .mean()
      .rename(columns={"head_distance": "mean_nn_distance"})
)

plt.figure(figsize=(6, 6))
ax = sns.barplot(
    data=opt1, x="condition", y="mean_nn_distance",
    order=ORDER, palette=PALETTE, errorbar=("ci", 95), capsize=0.15
)
sns.stripplot(
    data=opt1, x="condition", y="mean_nn_distance",
    order=ORDER, color="black", size=4, alpha=0.6, jitter=0.15
)
ax.set_xlabel("")
ax.set_ylabel("Mean nearest-neighbour distance (mm)", fontsize=14, fontweight="bold")
sns.despine()
plt.tight_layout()
plt.show()



# ------------------ OPTION 2: fraction of frames < threshold per file ------------------
THRESH_MM = 10
opt2 = (
    df.assign(close=(df["head_distance"] < THRESH_MM))
      .groupby(["filename", "condition"], as_index=False)["close"]
      .mean()
      .rename(columns={"close": "frac_close"})
)

plt.figure(figsize=(6, 6))
ax = sns.barplot(
    data=opt2, x="condition", y="frac_close",
    order=ORDER, palette=PALETTE, errorbar=("ci", 95), capsize=0.15
)
sns.stripplot(
    data=opt2, x="condition", y="frac_close",
    order=ORDER, color="black", size=4, alpha=0.6, jitter=0.15
)
ax.set_xlabel("")
ax.set_ylabel(f"Fraction of frames with NN < {THRESH_MM} mm", fontsize=14, fontweight="bold")
sns.despine()
plt.tight_layout()
plt.show()


# ================== PER-LARVA CLOSE-PROXIMITY METRIC ==================
# “For each larva we computed the fraction of observations in which its nearest neighbour was within X mm.”

import seaborn as sns
import matplotlib.pyplot as plt

PALETTE = {"Group Housed": "steelblue", "Pseudo Control": "skyblue"}
ORDER = ["Group Housed", "Pseudo Control"]

X_MM = 8  # <<< choose your "close" threshold here (e.g. 5, 8, 10)

# df at this point should be your GH vs Pseudo dataframe (the one used for opt1)
# and must contain: ['filename', 'condition', 'track_id', 'head_distance']

per_larva_close = (
    df.assign(is_close=df["head_distance"] < X_MM)
      .groupby(["filename", "condition", "track_id"], as_index=False)["is_close"]
      .mean()
      .rename(columns={"is_close": "p_close"})
)

plt.figure(figsize=(6, 6))
ax = sns.barplot(
    data=per_larva_close,
    x="condition",
    y="p_close",
    order=ORDER,
    palette=PALETTE,
    errorbar=("ci", 95),
    capsize=0.15
)
sns.stripplot(
    data=per_larva_close,
    x="condition",
    y="p_close",
    order=ORDER,
    color="black",
    size=4,
    alpha=0.6,
    jitter=0.15
)

ax.set_xlabel("")
ax.set_ylabel(f"P(NN distance < {X_MM} mm)\n(per larva)", fontsize=14, fontweight="bold")
sns.despine()
plt.tight_layout()
plt.show()
