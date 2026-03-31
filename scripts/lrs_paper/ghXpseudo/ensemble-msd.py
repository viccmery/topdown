
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']

PALETTE = {
    "GH": 'steelblue',     
    "PSEUDO": 'skyblue'}

HUE_ORDER = ["GH", "PSEUDO"]


df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/group-housed/ensemble_msd.csv')
df1['condition'] = 'GH'

df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/pseudo-n10/group-housed/ensemble_msd.csv')
df2['condition'] = 'PSEUDO'

df = pd.concat([df1, df2], ignore_index=True)


plt.figure(figsize=(8,8))

ax = sns.lineplot(data=df, x='time', y='squared_distance',  errorbar=('ci', 95), hue='condition', palette=PALETTE, hue_order=HUE_ORDER)  

plt.xlabel('Time (S)', fontsize=12)
plt.ylabel('Ensemble Mean Squared Distance (mm)', fontsize=12)

# plt.ylim(0, 90)
# plt.xlim(0,600)
# plt.yscale('log')
# plt.xscale('log')

sns.despine()
ax.legend(frameon=False, title=None, fontsize=11, loc="upper right")

# plt.title('Ensemble Mean Squared Distance', fontsize=16, fontweight='bold')


plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/lrs_paper/ghXpseudo/ensemble-msd.pdf', format='pdf', bbox_inches='tight')

plt.show()
