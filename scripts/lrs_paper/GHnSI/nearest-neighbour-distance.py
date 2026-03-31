
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

df = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/grouped+isolated/nearest_neighbour.csv')
df["social_experience"] = np.where(df["track_id"] <= 4, "SI", "GH")

bins = np.linspace(0, 90, 90)  # 0 to 2.5 in 0.1 increments
df['bin'] = pd.cut(df['head_distance'], bins, include_lowest=True)
df['bin_center'] = df['bin'].apply(lambda x: x.mid)


counts = (
    df.groupby(['filename', 'social_experience', 'bin_center'])
    .size()
    .groupby(['filename', 'social_experience'], group_keys=False)
    .apply(lambda x: x / x.sum())
    .reset_index(name='density')
)


plt.figure(figsize=(2,3))

ax = sns.lineplot(data=counts, x='bin_center', y='density', hue='social_experience', errorbar=('ci', 95)) 
plt.xlabel('Nearest Neighbour Distance (mm)', fontsize=16, fontweight='bold')
plt.ylabel('Fraction of Animals', fontsize=16, fontweight='bold')
sns.despine()
ax.legend(frameon=False, title=None, fontsize=11, loc="upper right")

# plt.title('Nearest Neighour Distance Distriubtion', fontsize=16, fontweight='bold')

plt.tight_layout(rect=[1, 1, 1, 1])

plt.ylim(0, 0.07)

plt.xlim(0, 60)
ax.set_xticks(np.arange(0, 61, 5))

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/lrs_paper/GHnSI/nearest-neighbour-distance.pdf', format='pdf', bbox_inches='tight')
plt.close()
