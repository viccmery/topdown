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


df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/2/group-housed/fed-fed/closest_contacts_1mm.csv')
df1['condition'] = 'GH-fed-fed'

df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/2/group-housed/fed-starved/closest_contacts_1mm.csv')
df2['condition'] = 'GH-fed-starved'

df3 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/2/group-housed/starved-starved/closest_contacts_1mm.csv')
df3['condition'] = 'GH-starved-starved'

df4 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/2/socially-isolated/fed-fed/closest_contacts_1mm.csv')
df4['condition'] = 'SI-fed-fed'

df5 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/2/socially-isolated/fed-starved/closest_contacts_1mm.csv')
df5['condition'] = 'SI-fed-starved'

df6 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/2/socially-isolated/starved-starved/closest_contacts_1mm.csv')
df6['condition'] = 'SI-starved-starved'


plt.figure(figsize=(8,8))


df = pd.concat([df1, df2, df3, df4, df5, df6], ignore_index=True)

# Sum across all frame bins per file + interaction type
grouped = (
    df.groupby(['file', 'condition', 'Closest Interaction Type'])
    .size()
    .reset_index(name='count')
)

sns.barplot(data=grouped, x='Closest Interaction Type', y='count', hue='condition', edgecolor='black', linewidth=2, errorbar='sd', alpha=0.8)

plt.xlabel('Closest Interaction Type', fontsize=12, fontweight='bold')
plt.ylabel('Frame Count', fontsize=12, fontweight='bold')

# Add an overall title to the entire figure
plt.title('Closest Node per Frame <1mm', fontsize=16, fontweight='bold')

# Adjust layout to prevent overlap, considering the overall title
# plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.tight_layout(rect=[1, 1, 1, 1])

plt.ylim(0, None)

plt.xticks(rotation=45)

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/head-head/interaction_type_closest_nodes.png', dpi=300, bbox_inches='tight')

plt.savefig(
    '/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/head-head/interaction_type_closest_nodes.pdf',
    format='pdf',
    bbox_inches='tight'
)

plt.close()