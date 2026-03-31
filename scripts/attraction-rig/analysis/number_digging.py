import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/2/group-housed/fed-fed/number_digging.csv')
df1['condition'] = 'GH-fed-fed'

df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/2/group-housed/fed-starved/number_digging.csv')
df2['condition'] = 'GH-fed-starved'

df3 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/2/group-housed/starved-starved/number_digging.csv')
df3['condition'] = 'GH-starved-starved'

df4 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/2/socially-isolated/fed-fed/number_digging.csv')
df4['condition'] = 'SI-fed-fed'

df5 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/2/socially-isolated/fed-starved/number_digging.csv')
df5['condition'] = 'SI-fed-starved'

df6 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/2/socially-isolated/starved-starved/number_digging.csv')
df6['condition'] = 'SI-starved-starved'


plt.figure(figsize=(8,8))

df = pd.concat([df4, df5, df6], ignore_index=True)


sns.lineplot(data=df, x='frame', y='normalised_digging', hue='condition', errorbar='sd') ## normalised_digging / number_digging


plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('% Digging', fontsize=12)
plt.ylim(0,100)


# Add an overall title to the entire figure
plt.title('Number Digging', fontsize=16, fontweight='bold')

# Adjust layout to prevent overlap, considering the overall title
plt.tight_layout(rect=[0, 0, 1, 0.95])

# plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/digging/gh-n2-normalised.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()

