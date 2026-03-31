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
    "Pseudo Control": "#F7D455",     
    "Socially Isolated": 'darkorange',}

HUE_ORDER = [ "Socially Isolated", "Pseudo Control"]


df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/socially-isolated/closest_contacts_1mm.csv')
df1['condition'] = 'Socially Isolated'

df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/pseudo-n10/socially-isolated/closest_contacts_1mm.csv')
df2['condition'] = 'Pseudo Control'




df = pd.concat([df1, df2], ignore_index=True)

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


grouped = (
    pairs_long.groupby(['file', 'condition', 'track'])
    .size()
    .reset_index(name='count')
)

gh = grouped.loc[grouped['condition'] == 'Socially Isolated', 'count']
si = grouped.loc[grouped['condition'] == 'Pseudo Control', 'count']

u, p = mannwhitneyu(gh, si, alternative='two-sided')

print(f"Mann–Whitney U = {u:.3f}")
print(f"p-value = {p:.4e}")



plt.figure(figsize=(4,8))
ax = sns.barplot(data=grouped, x='condition', y='count', linewidth=2, errorbar='sd', palette=PALETTE, order=HUE_ORDER, edgecolor='black',alpha=0.8)

plt.xlabel('', fontsize=12, fontweight='bold')
plt.ylabel('Total Contact Time (s)', fontsize=12, fontweight='bold')

# plt.title('Total Contacts', fontsize=16, fontweight='bold')

sns.despine()
ax.legend(frameon=False, title=None, fontsize=11, loc="upper right")


y_max = 500
h = y_max * 0.04          # height of bracket
y = y_max + h * 1.05       # vertical position

# x positions of bars (0 = GH, 1 = SI)
x1, x2 = 0, 1

# draw bracket
# ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c='black')
ax.plot([x1, x2], [y+h, y+h], lw=1.5, c='black')

# add stars
ax.text((x1 + x2) / 2, y + h * 1.1, '***',
        ha='center', va='bottom', fontsize=16, fontweight='bold')



plt.tight_layout(rect=[1, 1, 1, 1])

plt.ylim(0, None)

for label in ax.get_xticklabels():
    label.set_color('black')
    label.set_fontweight('bold')
    label.set_fontsize(12)



plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/lrs_paper/GHxSI/total_contact_frames_pseudo.pdf', 
            format='pdf', bbox_inches='tight')

plt.close()

