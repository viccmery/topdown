
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
    "GH": 'steelblue',     
    "SI": 'darkorange',}

HUE_ORDER = ["GH", "SI"]


df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/group-housed/interaction_type_bout.csv')
df1['condition'] = 'GH'

df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/socially-isolated/interaction_type_bout.csv')
df2['condition'] = 'SI'


df = pd.concat([df1, df2], ignore_index=True)


###### RAW NUMBER OF BOUTS
plt.figure(figsize=(2,4))
grouped = (
    df.groupby(['condition', 'file'])['bout_id']
      .size()
      .reset_index(name='num_bouts')
)

ax = sns.barplot(data=grouped, x='condition', y='num_bouts', hue='condition', edgecolor='black', linewidth=2, errorbar='sd', palette=PALETTE,order=HUE_ORDER)

plt.xlabel('', fontsize=12, fontweight='bold')
plt.ylabel('Frequency', fontsize=12, fontweight='bold')
sns.despine()
ax.legend(frameon=False, title=None, fontsize=11, loc="upper right")
# plt.title('Total Interaction Bouts', fontsize=16, fontweight='bold')
plt.tight_layout(rect=[1, 1, 1, 1])
plt.ylim(0, None)
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/lrs_paper/GHxSI/number_bouts.pdf', format='pdf', bbox_inches='tight')
plt.show()


###### BOUT LENGTHS

length_bouts = df.groupby(['condition', 'file'])['duration'].mean().reset_index(name='length_bout')

plt.figure(figsize=(2,4))
ax = sns.barplot(data=length_bouts, x='condition', y='length_bout',  edgecolor='black', linewidth=2, errorbar='sd', palette=PALETTE,  order=HUE_ORDER)
plt.xlabel('', fontsize=12, fontweight='bold')
plt.ylabel('Average Bout Length (S)', fontsize=12, fontweight='bold')
# plt.title('Average Contact Bout Length', fontsize=16, fontweight='bold')
sns.despine()
ax.legend(frameon=False, title=None, fontsize=11, loc="upper right")
plt.tight_layout(rect=[1, 1, 1, 1])
plt.ylim(0, None)
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/lrs_paper/GHxSI/average_bout_length.pdf', format='pdf', bbox_inches='tight')
plt.show()










