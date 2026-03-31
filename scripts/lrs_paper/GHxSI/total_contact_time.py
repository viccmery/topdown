import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import matplotlib.patches as mpatches
import matplotlib as mpl
from scipy.stats import mannwhitneyu

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']

PALETTE = {
    "Group Housed": 'steelblue',     
    "Socially Isolated": 'darkorange',
     "Pseudo Control": "#F7D455", }

HUE_ORDER = ["Group Housed", "Socially Isolated", 'Pseudo Control']


df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/group-housed/closest_contacts_1mm.csv')
df1['condition'] = 'Group Housed'

df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/socially-isolated/closest_contacts_1mm.csv')
df2['condition'] = 'Socially Isolated'

df3 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/pseudo-n10/socially-isolated/closest_contacts_1mm.csv')
df3['condition'] = 'Pseudo Control'


df = pd.concat([df1, df2, df3], ignore_index=True)

pairs_long = (
    df[['condition', 'file', 'frame', 'Interaction Pair']]
    .assign(
        track=lambda d: (
            d['Interaction Pair']
            .astype(str)
            .str.replace(r'[\(\)\[\]\s]', '', regex=True)   # remove (), [], spaces
            .str.split(',')                                 # -> ['0','1']
        )
    )
    .explode('track')
)

pairs_long['track'] = pairs_long['track'].astype(int)


grouped_track = (
    pairs_long.groupby(['file', 'condition', 'track'])
    .size()
    .reset_index(name='count')
)

# collapse tracks -> ONE value per video (pick one)
grouped = (
    grouped_track.groupby(['file', 'condition'])['count']
    # .sum()          # TOTAL contact frames in the video (summed over larvae)
    .mean()       # average contact frames per larva in that video
    .reset_index()
)

gh = grouped.loc[grouped['condition'] == 'Group Housed', 'count']
si = grouped.loc[grouped['condition'] == 'Socially Isolated', 'count']
pc = grouped.loc[grouped['condition'] == 'Pseudo Control', 'count']

u, p = mannwhitneyu(gh, si, alternative='two-sided')
print('Group Housed vs Socially Isolated:')
print(f"Mann–Whitney U = {u:.3f}")
print(f"p-value = {p:.4e}")

u, p = mannwhitneyu(pc, si, alternative='two-sided')
print('Pseudo Control vs Socially Isolated:')
print(f"Mann–Whitney U = {u:.3f}")
print(f"p-value = {p:.4e}")


plt.figure(figsize=(4,8))
ax = sns.barplot(data=grouped, x='condition', y='count', linewidth=2, errorbar='sd', palette=PALETTE, order=HUE_ORDER, edgecolor='black',alpha=0.8)

plt.xlabel('', fontsize=12, fontweight='bold')
plt.ylabel('Total Contact Time (s)', fontsize=12, fontweight='bold')

# plt.title('Total Contacts', fontsize=16, fontweight='bold')

sns.despine()
ax.legend(frameon=False, title=None, fontsize=11, loc="upper right")


y = 460
h = 10

# GH vs SI
ax.plot([0, 1], [y, y], lw=1.5, c='black')
ax.text(0.5, y + h, '**', ha='center', va='bottom',
        fontsize=14, fontweight='bold')

# SI vs Pseudo
y2 = y + 30
ax.plot([1, 2], [y2, y2], lw=1.5, c='black')
ax.text(1.5, y2 + h, '**', ha='center', va='bottom',
        fontsize=14, fontweight='bold')



plt.tight_layout(rect=[1, 1, 1, 1])

plt.ylim(0, None)



plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/lrs_paper/GHxSI/total_contact_frames.pdf', 
            format='pdf', bbox_inches='tight')

plt.close()

