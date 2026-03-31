
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
    "PSEUDO": 'skyblue'}

HUE_ORDER = ["GH", "PSEUDO"]

df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/group-housed/euclidean_distances.csv')
df1['condition'] = 'GH'

df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/pseudo-n10/group-housed/euclidean_distances.csv')
df2['condition'] = 'PSEUDO'


plt.figure(figsize=(8,6))

df = pd.concat([df1, df2], ignore_index=True)

# df = df[df['time'] <= 300]

ax = sns.lineplot(data=df, x='time', y='average_distance',  errorbar=('ci', 95), hue='condition', palette=PALETTE, hue_order=HUE_ORDER)


plt.xlabel('Time (S)', fontsize=12, fontweight='bold')
plt.ylabel('Average Distance (mm)', fontsize=12, fontweight='bold')

sns.despine()
ax.legend(frameon=False, title=None, fontsize=11, loc="upper right")


# plt.title('Euclidean Distances', fontsize=16, fontweight='bold')


plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/lrs_paper/ghXpseudo/euclidean-distance.pdf', format='pdf', bbox_inches='tight')
plt.show()
