
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

# df = df[df['frame'] < 600].copy()


speed = (
    df.groupby(['filename', 'social_experience'])['speed'].mean().reset_index(name='mean_speed')

)
plt.figure(figsize=(2,3))

ax = sns.barplot(data=speed, x='social_experience', y='mean_speed') 
plt.xlabel('Social Experience', fontsize=16, fontweight='bold')
plt.ylabel('Mean Speed (mm/s)', fontsize=16, fontweight='bold')
sns.despine()
ax.legend(frameon=False, title=None, fontsize=11, loc="upper right")

# plt.title('Nearest Neighour Distance Distriubtion', fontsize=16, fontweight='bold')

plt.tight_layout(rect=[1, 1, 1, 1])


plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/lrs_paper/GHnSI/speed.pdf', format='pdf', bbox_inches='tight')
plt.close()
