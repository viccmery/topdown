import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import matplotlib.patches as mpatches
import matplotlib as mpl

# ---- Adobe-friendly fonts (must be set BEFORE plotting) ----
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/fed-fed/head_head_contacts_kinematics_over_time.csv')
df1['condition'] = 'fed-fed'
df1['role'] = 'fed'

df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/starved-fed/head_head_contacts_kinematics_over_time.csv')
df2['condition'] = 'fed-starved'
df2['role'] = df2['track_id'].map({0: 'fed', 1: 'starved'})

df3 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/starved-starved/head_head_contacts_kinematics_over_time.csv')
df3['condition'] = 'starved-starved'
df3['role'] = 'starved'


df = pd.concat([df1, df2, df3], ignore_index=True)

df['heading_angle_change'] = (
    df
    .sort_values(['file', 'interaction_number', 'track_id', 'frame'])
    .groupby(['file', 'condition', 'interaction_number', 'track_id'])['heading_angle']
    .diff()
    .abs() )


df['interaction_group'] = np.where(df['interaction_number'] == 1, 'first', 'other')


df2['interaction_group'] = np.where(df2['interaction_number'] == 1, 'first', 'other')

df_rel10 = df[df['rel_frame'] == 10].copy()

plt.figure(figsize = (8, 8))

sns.barplot(
    data=df_rel10,
    x='condition',
    y='dist_from_start',
    hue='interaction_group',
    ci='sd'
)
plt.xlabel('Condition', fontsize=12, fontweight='bold')
plt.ylabel('Distance from start position', fontsize=12, fontweight='bold')
plt.title('Distance from Initial Position at Rel Frame 10', fontsize=16, fontweight='bold') 

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/head-head/dist/dist_10.png', dpi=300, bbox_inches='tight')
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/head-head/dist/dist_10.pdf',
             format='pdf', bbox_inches='tight')
plt.close()




df_rel10 = df2[df2['rel_frame'] == 10].copy()

plt.figure(figsize = (8, 8))

sns.barplot(
    data=df_rel10,
    x='role',
    y='dist_from_start',
    hue='interaction_group',
    ci='sd'
)
plt.xlabel('Role', fontsize=12, fontweight='bold')
plt.ylabel('Distance from start position', fontsize=12, fontweight='bold')
plt.title('Distance from Initial Position at Rel Frame 10', fontsize=16, fontweight='bold') 

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/head-head/dist/dist_10.png', dpi=300, bbox_inches='tight')
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/head-head/dist/dist_10.pdf',
             format='pdf', bbox_inches='tight')
plt.close()


plt.figure(figsize = (8, 8))

sns.barplot(
    data=df2,
    x='role',
    y='dist_from_start',
    hue='interaction_group',
    ci='sd'
)
plt.xlabel('Role', fontsize=12, fontweight='bold')
plt.ylabel('Distance from start position', fontsize=12, fontweight='bold')
plt.title('Distance from Initial Position at Rel Frame 10', fontsize=16, fontweight='bold') 

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/head-head/dist/dist_10.png', dpi=300, bbox_inches='tight')
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/head-head/dist/dist_10.pdf',
             format='pdf', bbox_inches='tight')
plt.close()