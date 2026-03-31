
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import matplotlib.patches as mpatches
import matplotlib as mpl
from scipy.stats import mannwhitneyu

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']

PALETTE = {
    "Group Housed": 'steelblue',     
    "Socially Isolated": 'darkorange',
     "Pseudo Control": "#F7D455", }

HUE_ORDER = ["Group Housed", "Socially Isolated", "Pseudo Control"]

df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/socially-isolated/nearest_neighbour.csv')
df1['condition'] = 'Socially Isolated'

df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/group-housed/nearest_neighbour.csv')
df2['condition'] = 'Group Housed'

df3 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/pseudo-n10/socially-isolated/nearest_neighbour.csv')
df3['condition'] = 'Pseudo Control'


df = pd.concat([df1, df2, df3], ignore_index=True)


# 1) Mean speed per track per file (within 5–10 mm)
track_means = (
    df.groupby(['filename', 'condition', 'track_id'])['speed']
    .mean()
    .reset_index()
)

# 2) Collapse tracks → single value per file
file_means = (
    track_means
    .groupby(['filename', 'condition'])['speed']
    .mean()
    .reset_index()
)

gh = file_means.loc[file_means['condition'] == 'Group Housed', 'speed']
si = file_means.loc[file_means['condition'] == 'Socially Isolated', 'speed']

u_gh_si, p_gh_si = mannwhitneyu(gh, si, alternative='two-sided')
print(f"GH vs SI: U = {u_gh_si:.3f}, p = {p_gh_si:.4e}")

# SI vs Pseudo
pc = file_means.loc[file_means['condition'] == 'Pseudo Control', 'speed']

u_si_pc, p_si_pc = mannwhitneyu(si, pc, alternative='two-sided')
print(f"SI vs Pseudo: U = {u_si_pc:.3f}, p = {p_si_pc:.4e}")


plt.figure(figsize=(6, 8))

ax = sns.barplot(
    data=file_means,
    x='condition',
    y='speed',
    errorbar='sd', palette=PALETTE, order=HUE_ORDER)


# plt.xlabel('Nearest Neighbour Distance From Head (mm)', fontsize=14)
plt.ylabel('Speed (mm/s)', fontsize=14)
plt.subplots_adjust(wspace=0.3, hspace=0.4)
# plt.suptitle('Speed vs Nearest Neighour Distance From Head', fontsize=16, fontweight='bold')

ax.legend(
    title=None,
    frameon=False,
    fontsize=11,
    loc='upper right'
)

y = file_means['speed'].max() * 1.08

# SI vs Pseudo (0 ↔ 1) — significant
ax.plot([0, 0, 1, 1], [y*0.98, y, y, y*0.98], lw=1.5, c='k')
ax.text(0.5, y, '*', ha='center', va='bottom', fontsize=16)

# GH vs SI (1 ↔ 2) — not significant
ax.plot([1, 1, 2, 2], [y*1.06, y*1.08, y*1.08, y*1.06], lw=1.5, c='k')
ax.text(1.5, y*1.08, 'ns', ha='center', va='bottom', fontsize=12)


sns.despine()
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/lrs_paper/GHxSI/barplot-speed.pdf', format='pdf', bbox_inches='tight')

plt.close()