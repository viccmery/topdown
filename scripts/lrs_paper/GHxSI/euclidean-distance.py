
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

# ---- Adobe / Illustrator friendly PDFs ----
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']

PALETTE = {
    "GH": 'steelblue',     
    "SI": 'darkorange',}

HUE_ORDER = ["GH", "SI"]

df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/group-housed/euclidean_distances.csv')
df1['condition'] = 'GH'

df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/socially-isolated/euclidean_distances.csv')
df2['condition'] = 'SI'


plt.figure(figsize=(2,2), dpi=600)

df = pd.concat([df1, df2], ignore_index=True)
ax = sns.lineplot(data=df, x='time', y='average_distance',  errorbar=('ci', 95), hue='condition', palette=PALETTE, hue_order=HUE_ORDER)

plt.ylim(0, 60)
plt.xlim(0,3600)
ax.set_xticks([0, 900, 1800, 2700, 3600])
ax.set_xticklabels(['0', '900', '1800', '2700', '3600'])
plt.xlabel('Time (S)', fontsize=12, fontweight='bold')
plt.ylabel('Average Distance (mm)', fontsize=12, fontweight='bold')

sns.despine()
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/lrs_paper/GHxSI/euclidean-distance.pdf', format='pdf', bbox_inches='tight')
plt.show()
