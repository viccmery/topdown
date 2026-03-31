import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import matplotlib.patches as mpatches
import os
import matplotlib as mpl

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']

PALETTE = {
    "Group Housed": 'steelblue',     
    "Socially Isolated": 'darkorange',}

HUE_ORDER = ["Group Housed", "Socially Isolated"]

##### GROUPHOUSE 
df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/group-housed/individual_approach_responses_2.csv')
df2['condition'] = 'Group Housed'
df2['response_distance'] = '2'

df3 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/group-housed/individual_approach_responses_3.csv')
df3['condition'] = 'Group Housed'
df3['response_distance'] = '3'

df4 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/group-housed/individual_approach_responses_4.csv')
df4['condition'] = 'Group Housed'
df4['response_distance'] = '4'

df5 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/group-housed/individual_approach_responses_5.csv')
df5['condition'] = 'Group Housed'
df5['response_distance'] = '5'

df6 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/group-housed/individual_approach_responses_6.csv')
df6['condition'] = 'Group Housed'
df6['response_distance'] = '6'

df7 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/group-housed/individual_approach_responses_7.csv')
df7['condition'] = 'Group Housed'
df7['response_distance'] = '7'

df8 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/group-housed/individual_approach_responses_8.csv')
df8['condition'] = 'Group Housed'
df8['response_distance'] = '8'

df9 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/group-housed/individual_approach_responses_9.csv')
df9['condition'] = 'Group Housed'
df9['response_distance'] = '9'

df10 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/group-housed/individual_approach_responses_10.csv')
df10['condition'] = 'Group Housed'
df10['response_distance'] = '10'

##### SOCIALLY ISOLATED  
si2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/socially-isolated/individual_approach_responses_2.csv')
si2['condition'] = 'Socially Isolated'
si2['response_distance'] = '2'

si3 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/socially-isolated/individual_approach_responses_3.csv')
si3['condition'] = 'Socially Isolated'
si3['response_distance'] = '3'

si4 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/socially-isolated/individual_approach_responses_4.csv')
si4['condition'] = 'Socially Isolated'
si4['response_distance'] = '4'

si5 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/socially-isolated/individual_approach_responses_5.csv')
si5['condition'] = 'Socially Isolated'
si5['response_distance'] = '5'

si6 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/socially-isolated/individual_approach_responses_6.csv')
si6['condition'] = 'Socially Isolated'
si6['response_distance'] = '6'

si7 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/socially-isolated/individual_approach_responses_7.csv')
si7['condition'] = 'Socially Isolated'
si7['response_distance'] = '7'

si8 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/socially-isolated/individual_approach_responses_8.csv')
si8['condition'] = 'Socially Isolated'
si8['response_distance'] = '8'

si9 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/socially-isolated/individual_approach_responses_9.csv')
si9['condition'] = 'Socially Isolated'
si9['response_distance'] = '9'

si10 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/socially-isolated/individual_approach_responses_10.csv')
si10['condition'] = 'Socially Isolated'
si10['response_distance'] = '10'

df_all = pd.concat([df2, df3, df4, df5, df6, df7, df8, df9, df10, si2, si3, si4, si5, si6, si7, si8, si9, si10], ignore_index=True)
df_all['response_distance'] = pd.to_numeric(df_all['response_distance'])

df_all = df_all[df_all['dist_from_centre'] < 10]

per_video = (
    df_all
    .groupby(['condition', 'filename', 'response_distance']) #, 'filename'])
    .agg(
        n_encounters=('touch', 'size'),   # total encounters
        n_touch=('touch', 'sum'),          # encounters with touch
        touch_rate=('touch', 'mean') ,      # fraction that touched
        no_touch_rate=('touch', lambda s: 1 - s.mean()),
    )
    .reset_index()
)


plt.figure(figsize=(6,6))

ax = sns.lineplot(data=per_video, x='response_distance', y='touch_rate', hue='condition', errorbar=('ci', 95), palette=PALETTE, hue_order=HUE_ORDER)

sns.despine()
ax.legend(frameon=False, title=None, fontsize=11, loc="upper right")

# plt.title('Potential Interactions\n', fontsize=14, fontweight='bold')
plt.ylabel('P(touch)', fontsize=12, fontweight='bold')
plt.xlabel('Distance (mm)', fontsize=12, fontweight='bold')
plt.ylim(0, 1)
plt.tight_layout()
# plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/lrs_paper/GHxSI//potential_interactions_individual_combined.pdf', format='pdf', bbox_inches='tight')
plt.show()



