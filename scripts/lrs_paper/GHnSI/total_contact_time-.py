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



df = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/grouped+isolated/closest_contacts_1mm.csv')


pairs_long = (
    df[['file', 'frame', 'Interaction Pair']]
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

pairs_long["social_experience"] = np.where(pairs_long["track"] <= 4, "SI", "GH")

grouped_track = (
    pairs_long.groupby(['file', 'social_experience', 'track'])
    .size()
    .reset_index(name='count')
)

grouped = (
    grouped_track.groupby(['file', 'social_experience'])['count']
    # .sum()          # TOTAL contact frames in the video (summed over larvae)
    .mean()       # average contact frames per larva in that video
    .reset_index(name='mean_contact_frames_per_larva')
)

plt.figure(figsize=(4,8))
ax = sns.barplot(data=grouped, x='social_experience', y='mean_contact_frames_per_larva', linewidth=2, errorbar='sd', edgecolor='black',alpha=0.8)

plt.xlabel('', fontsize=12, fontweight='bold')
plt.ylabel('Total Contact Time (s)', fontsize=12, fontweight='bold')


sns.despine()
ax.legend(frameon=False, title=None, fontsize=11, loc="upper right")

plt.tight_layout(rect=[1, 1, 1, 1])

plt.ylim(0, None)

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/lrs_paper/GHnSI/total_contact_frames.pdf', 
            format='pdf', bbox_inches='tight')

plt.close()

