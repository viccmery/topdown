
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

bins = list(range(0, 90, 1))  # [0, 10, 20, ..., 100]
df['bin'] = pd.cut(df['head_distance'], bins, include_lowest=True) #body-body ; closest_node_distance ; head_distance
df['bin_center'] = df['bin'].apply(lambda x: x.right)

max_mean_speed = (
    df[df['bin_center'] == 10]
    .groupby('condition')['speed']
    .mean()
)


plt.figure(figsize=(4, 2))

ax = sns.lineplot(
    data=df,
    x='bin_center',
    y='speed',
    hue='condition',
    errorbar=('ci', 95), palette=PALETTE, hue_order=HUE_ORDER)

plt.ylim(0.6,1.1)
# plt.xlim(0,30)
plt.xlabel('Nearest Neighbour Distance (mm)', fontsize=14)
plt.ylabel('Speed (mm/s)', fontsize=14)
plt.subplots_adjust(wspace=0.3, hspace=0.4)
# plt.suptitle('Speed vs Nearest Neighour Distance From Head', fontsize=16, fontweight='bold')

ax.legend(
    title=None,
    frameon=False,
    fontsize=11,
    loc='upper right'
)

sns.despine()
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/lrs_paper/GHxSI/SI_PSEUDO_speed-not-normalised.pdf', format='pdf', bbox_inches='tight')

plt.close()