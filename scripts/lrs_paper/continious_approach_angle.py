from math import comb
from operator import index
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import matplotlib.patches as mpatches
import os
import matplotlib as mpl

gh = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/group-housed/individual_approach_responses_consistent_angle_10.csv')  
gh['condition'] = 'group-housed'
si = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/socially-isolated/individual_approach_responses_consistent_angle_10.csv')  
si['condition'] = 'socially-isolated'

gh = gh[gh['rel_time'] < 10].copy()
si = si[si['rel_time'] < 10].copy()


# ---- event length summary (counts + %) ----
def length_table(df):
    lengths = df.groupby('event_id').size()              # frames per event
    dist = lengths.value_counts().sort_index()           # how many events of each length
    out = pd.DataFrame({
        'n_events': dist,
        'percent': (dist / dist.sum() * 100).round(2)
    })
    return out

print("\n=== group-housed (thr=10) event length distribution ===")
print(length_table(gh))

print("\n=== socially-isolated (thr=10) event length distribution ===")
print(length_table(si))




combined = pd.concat([gh, si], ignore_index=True)

max_d = int(np.ceil(combined['distance'].max()))
bins = np.arange(0, max_d + 1, 1)

combined['distance_binned'] = pd.cut(
    combined['distance'],
    bins=bins,
    right=False  # bin_right = False
)

combined['distance_binned'] = combined['distance_binned'].apply(lambda x: x.left).astype(int)


import matplotlib.pyplot as plt
import seaborn as sns

# make sure we're using the right column names
# angle bin column is 'angle_bin' (not 'approach_angle')
# y is 'stim_speed'

angle_bins = sorted(combined['angle_bin'].dropna().unique())

fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharey=True)
axes = axes.flatten()

for i, angle_bin in enumerate(angle_bins[:6]):  # 6 bins -> 2x3
    ax = axes[i]
    subset = combined[combined['angle_bin'] == angle_bin]

    sns.lineplot(
        data=subset,
        x='distance_binned',
        y='stim_speed',
        hue='condition',
        ax=ax
    )

    ax.set_title(f'Angle bin: {angle_bin}')
    ax.set_xlabel('Distance (binned)')
    ax.set_ylabel('Stim speed' if i % 3 == 0 else '')
    ax.tick_params(axis='x', rotation=45)


plt.tight_layout()
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/lrs_paper/GHxSI/continious-approach-angle_speed-distance.pdf', format='pdf', bbox_inches='tight')
plt.show()

