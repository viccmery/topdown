
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import matplotlib.patches as mpatches
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


df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/group-housed/speed_over_time.csv')
df1['condition'] = 'GH'

df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/pseudo-n10/group-housed/speed_over_time.csv')
df2['condition'] = 'PSEUDO'

plt.figure(figsize=(1.5,3))

df = pd.concat([df1, df2], ignore_index=True)

ax = sns.barplot(data=df, x='condition', y='speed', errorbar='sd', palette=PALETTE, order=HUE_ORDER, linewidth=2, edgecolor='black')


plt.ylabel('Probability', fontsize=12, fontweight='bold', labelpad=15)
plt.xlabel('Speed (mm/s)', fontsize=12, fontweight='bold', labelpad=15)

sns.despine()
ax.legend(frameon=False, title=None, fontsize=11, loc="upper right")

for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight('bold')

plt.ylim(0, None)

# plt.title('Speed', fontsize=16, fontweight='bold')

plt.tight_layout(rect=[1, 1, 1, 1])
plt.xticks(fontweight='bold')
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/lrs_paper/ghXpseudo/speed_bar.pdf', format='pdf', bbox_inches='tight')

# Show the plot
plt.show()
