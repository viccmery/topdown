
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

df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/socially-isolated/distance_from_centre.csv')
df1['condition'] = 'Socially Isolated'

df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/group-housed/distance_from_centre.csv')
df2['condition'] = 'Group Housed'

df3 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/pseudo-n10/socially-isolated/distance_from_centre.csv')
df3['condition'] = 'Pseudo Control'


df = pd.concat([df1, df2, df3], ignore_index=True)


track_means = (
    df.groupby(['file', 'condition', 'track'])['distance_from_centre']
    .mean()
    .reset_index()
)

file_means = (
    track_means
    .groupby(['file', 'condition'])['distance_from_centre']
    .mean()
    .reset_index()
)

gh = file_means.loc[file_means['condition'] == 'Group Housed', 'distance_from_centre']
si = file_means.loc[file_means['condition'] == 'Socially Isolated', 'distance_from_centre']

u_gh_si, p_gh_si = mannwhitneyu(gh, si, alternative='two-sided')
print(f"GH vs SI: U = {u_gh_si:.3f}, p = {p_gh_si:.4e}")

# SI vs Pseudo
pc = file_means.loc[file_means['condition'] == 'Pseudo Control', 'distance_from_centre']

u_si_pc, p_si_pc = mannwhitneyu(si, pc, alternative='two-sided')
print(f"SI vs Pseudo: U = {u_si_pc:.3f}, p = {p_si_pc:.4e}")


plt.figure(figsize=(6, 8))

ax = sns.barplot(
    data=file_means,
    x='condition',
    y='distance_from_centre',
    errorbar='sd', palette=PALETTE, order=HUE_ORDER)


plt.ylabel('Distance From Centre (mm)', fontsize=14)
plt.subplots_adjust(wspace=0.3, hspace=0.4)

ax.legend(
    title=None,
    frameon=False,
    fontsize=11,
    loc='upper right'
)

y = file_means['distance_from_centre'].max() * 1.10
h = file_means['distance_from_centre'].max() * 0.03

# SI (1) vs Pseudo (2)
ax.plot([1, 2], [y, y], lw=1.5, c='k')
ax.text(1.5, y + h, '*', ha='center', va='bottom', fontsize=16)


sns.despine()
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/lrs_paper/GHxSI/barplot-distance-from-centre.pdf', format='pdf', bbox_inches='tight')

plt.close()