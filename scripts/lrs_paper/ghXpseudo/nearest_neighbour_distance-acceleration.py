
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
    "Group Housed": 'steelblue',     
    "Pseudo Control": 'skyblue'
}

HUE_ORDER = ["Group Housed", "Pseudo Control"]

df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/group-housed/nearest_neighbour.csv')
df1['condition'] = 'Group Housed'

df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/pseudo-n10/group-housed/nearest_neighbour.csv')
df2['condition'] = 'Pseudo Control'


df = pd.concat([df1, df2], ignore_index=True)

bins = list(range(0, 90, 1))  # [0, 10, 20, ..., 100]
df['bin'] = pd.cut(df['head_distance'], bins, include_lowest=True) #body-body ; closest_node_distance ; head_distance
df['bin_center'] = df['bin'].apply(lambda x: x.mid)


plt.figure(figsize=(8, 8))

ax = sns.lineplot(
    data=df,
    x='bin_center',
    y='acceleration',
    hue='condition', errorbar=('ci', 95), palette=PALETTE, hue_order=HUE_ORDER)

# plt.ylim(0,1.5)
plt.ylim(-0.1, 0.1)
plt.xlim(0,10)
plt.xlabel('Nearest Neighbour Distance (mm)', fontsize=16, fontweight='bold')
plt.ylabel('Acceleration (mm/s²)', fontsize=16, fontweight='bold')
plt.subplots_adjust(wspace=0.3, hspace=0.4)
# plt.suptitle('Speed vs Nearest Neighour Distance From Head', fontsize=16, fontweight='bold')

ax.legend(
    title=None,
    frameon=False,
    fontsize=11,
    loc='upper right'
)

for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_color('black')
    label.set_fontweight('bold')

sns.despine()
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/lrs_paper/ghXpseudo/head_min_dist-acceleration.pdf', dpi=300, bbox_inches='tight')

plt.close()