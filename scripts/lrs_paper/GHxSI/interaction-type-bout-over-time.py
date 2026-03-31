
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

unified_types = [
    'head_head', 'tail_tail', 'body_body',
    'body_head', 'body_tail', 'head_tail'
]


#### BOUT DURATION OVER TIME

bin_size = 600
df['time_bin'] = (df['start_frame'] // bin_size + 1) * bin_size
length_bouts = df.groupby(['condition', 'file', 'time_bin'])['duration'].mean().reset_index(name='length_bout')

bins = sorted(length_bouts['time_bin'].unique())

plt.figure(figsize=(3,2))
ax = sns.lineplot(data=length_bouts, x='time_bin', y='length_bout', hue='condition',  errorbar=('ci', 95), palette=PALETTE)
plt.xlabel('Time Bin (S)', fontsize=12, fontweight='bold')
plt.ylabel('Mean Bout Duration (S)', fontsize=12, fontweight='bold')
# plt.title("Mean duration Over Time", fontsize=14)
plt.ylim(0,8)
plt.xlim(600,3600)
plt.xticks(np.arange(600, 3601, 600))
sns.despine()
ax.legend(frameon=False, title=None, fontsize=11, loc="upper right")
plt.tight_layout()
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/lrs_paper/GHxSI/bout-duration-over-time.pdf', format='pdf', bbox_inches='tight')
plt.show()


#### BOUT DURATION FREQUENCY TIME
plt.figure(figsize=(3,2))

bin_size = 600
df['time_bin'] = (df['start_frame'] // bin_size +1) * bin_size
freq_bouts = df.groupby(['condition', 'file', 'time_bin'])['bout_id'].size().reset_index(name='num_bouts')
sns.lineplot(data=freq_bouts, x='time_bin', y='num_bouts', hue='condition',  palette=PALETTE, errorbar=('ci', 95))
plt.xlabel('Time Bin (S)', fontsize=12, fontweight='bold')
plt.ylabel('Count', fontsize=12, fontweight='bold')
plt.title('Total Interaction Bouts', fontsize=16, fontweight='bold')
plt.xlim(600,3600)
plt.xticks(np.arange(600, 3601, 600))
sns.despine()
plt.tight_layout(rect=[1, 1, 1, 1])
plt.ylim(0, None)
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/lrs_paper/GHxSI/bout-frequency-over-time.pdf', format='pdf', bbox_inches='tight')
plt.show()

