import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


##### 2 LARVAE DIGGING 
# df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/2/group-housed/fed-fed/digging_distances_pair.csv')
# df1['condition'] = 'GH-fed-fed'

df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/2/group-housed/fed-starved/digging_distances_pair.csv')
df2['condition'] = 'GH-fed-starved'

df3 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/2/group-housed/starved-starved/digging_distances_pair.csv')
df3['condition'] = 'GH-starved-starved'

# df4 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/2/socially-isolated/fed-fed/digging_distances_pair.csv')
# df4['condition'] = 'SI-fed-fed'

# df5 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/2/socially-isolated/fed-starved/digging_distances_pair.csv')
# df5['condition'] = 'SI-fed-starved'

df6 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/2/socially-isolated/starved-starved/digging_distances_pair.csv')
df6['condition'] = 'SI-starved-starved'


##### 1 LARVAE DIGGING 

df7 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/2/group-housed/fed-fed/digging_distances_single.csv')
df7['condition'] = 'GH-fed-fed'

df8 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/2/group-housed/fed-starved/digging_distances_single.csv')
df8['condition'] = 'GH-fed-starved'

df9 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/2/group-housed/starved-starved/digging_distances_single.csv')
df9['condition'] = 'GH-starved-starved'

df10 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/2/socially-isolated/fed-fed/digging_distances_single.csv')
df10['condition'] = 'SI-fed-fed'

df11 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/2/socially-isolated/fed-starved/digging_distances_single.csv')
df11['condition'] = 'SI-fed-starved'

df12 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/2/socially-isolated/starved-starved/digging_distances_single.csv')
df12['condition'] = 'SI-starved-starved'



plt.figure(figsize=(8,8))

df = pd.concat([df7, df8, df9, df10, df11, df12], ignore_index=True)
df = pd.concat([df3, df6], ignore_index=True)


bins = np.linspace(0, 90, 10)  
df['bin'] = pd.cut(df['distance'], bins, include_lowest=True)
df['bin_center'] = df['bin'].apply(lambda x: x.mid)


# Count per file-condition-bin
counts = (
    df.groupby(['file', 'condition', 'bin_center'])
    .size()
    .groupby(['file', 'condition'], group_keys=False)
    .apply(lambda x: x / x.sum())
    .reset_index(name='density')
)


sns.lineplot(data=counts, x='bin_center', y='density', hue='condition', errorbar='sd')



# sns.histplot(data=df, x='distance', hue='condition', stat='density', common_norm=False, alpha=0.5)


plt.xlabel('Distance', fontsize=12)
plt.ylabel('Density', fontsize=12)

plt.ylim(0, None)

# Add an overall title to the entire figure
plt.title('Distance Between Larval Digging', fontsize=16, fontweight='bold')

# Adjust layout to prevent overlap, considering the overall title
plt.tight_layout(rect=[0, 0, 1, 0.95])

# plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/digging/n2-one-digging-pseudo-si.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()

