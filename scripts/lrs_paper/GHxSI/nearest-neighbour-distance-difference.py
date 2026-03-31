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



