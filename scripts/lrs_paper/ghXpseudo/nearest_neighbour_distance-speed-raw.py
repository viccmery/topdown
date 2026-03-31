
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

plt.figure(figsize=(8, 8))

sns.scatterplot(
    data=df,
    x='head_distance',
    y='speed',
    hue='condition', palette=PALETTE, hue_order=HUE_ORDER)

# plt.ylim(0,1.1)
# plt.ylim(0, 1.1)
# plt.xlim(0,10)
plt.xlabel('Nearest Neighbour Distance (mm)', fontsize=16, fontweight='bold')
plt.ylabel('Speed (mm/s)', fontsize=16, fontweight='bold')
plt.subplots_adjust(wspace=0.3, hspace=0.4)
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/lrs_paper/ghXpseudo/head_min_dist-speed-raw.png', dpi=300, bbox_inches='tight')

plt.show()