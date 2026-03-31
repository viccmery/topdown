
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import matplotlib.patches as mpatches
import matplotlib as mpl

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']

PALETTE = {
    "Pseudo Control": "#F7D455",     
    "Socially Isolated": 'darkorange',}

HUE_ORDER = [ "Socially Isolated", "Pseudo Control"]


df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/socially-isolated/nearest_neighbour.csv')
df1['condition'] = 'Socially Isolated'

df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/pseudo-n10/socially-isolated/nearest_neighbour.csv')
df2['condition'] = 'Pseudo Control'


df = pd.concat([df1, df2], ignore_index=True)


from scipy.stats import mannwhitneyu

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

si = file_means.loc[file_means['condition'] == 'Socially Isolated', 'speed']
pc = file_means.loc[file_means['condition'] == 'Pseudo Control', 'speed']

u, p = mannwhitneyu(si, pc, alternative='two-sided')

print(f"Mann–Whitney U = {u:.3f}, p = {p:.4e}")
print(f"N files — SI: {len(si)}, PSEUDO: {len(pc)}")

# per_vid = maybe my plots dh be i like this idk/

plt.figure(figsize=(4, 8))

ax = sns.barplot(
    data=file_means,
    x='condition',
    y='speed',
    errorbar='sd', order=HUE_ORDER, palette=PALETTE)


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
plt.ylim(0,1.2)

y = file_means['speed'].max() * 1.08
ax.text(0.5, y, '*', ha='center', va='bottom', fontsize=20)

sns.despine()
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/lrs_paper/GHxSI/barplot-speed-pseudo.pdf', format='pdf', bbox_inches='tight')

plt.close()