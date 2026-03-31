
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys



# df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n1/group-housed/angle_over_time.csv')
# df1['condition'] = 'GH_N1'

# df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n1/socially-isolated/angle_over_time.csv')
# df2['condition'] = 'SI_N1'

# df3 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n2/group-housed/angle_over_time.csv')
# df3['condition'] = 'GH_N2'

# df4 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n2/socially-isolated/angle_over_time.csv')
# df4['condition'] = 'SI_N2'

df5 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/group-housed/angle_over_time.csv')
df5['condition'] = 'GH_N10'

df6 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/socially-isolated/angle_over_time.csv')
df6['condition'] = 'SI_N10'

df5 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/group-housed/angle_over_time.csv')
df5['condition'] = 'GH_N10'

df7 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/pseudo-n10/socially-isolated/angle_over_time.csv')
df7['condition'] = 'PSEUDO-SI_N10'

df8 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/pseudo-n10/group-housed/angle_over_time.csv')
df8['condition'] = 'PSEUDO-GH_N10'

df9 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/pseudo-n2/socially-isolated/angle_over_time.csv')
df9['condition'] = 'PSEUDO-SI_N2'

df10 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/pseudo-n2/group-housed/angle_over_time.csv')
df10['condition'] = 'PSEUDO-GH_N2'

## ALL DF
# df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9, df10], ignore_index=True)

## N1
# df = pd.concat([df1, df2], ignore_index=True)

## N2
# df = pd.concat([df3, df4], ignore_index=True)

## N10
# df = pd.concat([df5, df6], ignore_index=True)

## PEUDO N10 - GROUP
# df = pd.concat([df5, df8,], ignore_index=True)

## PEUDO N10 - ISO
# df = pd.concat([df6, df7], ignore_index=True)

## PEUDO N2- GROUP
# df = pd.concat([df3, df10], ignore_index=True)

## PEUDO N2 - ISO
# df = pd.concat([ df4, df9,], ignore_index=True)


## GH
# df = pd.concat([df1, df3, df5], ignore_index=True)

## SI
df = pd.concat([df5, df6], ignore_index=True)


plt.figure(figsize=(8,6))


bins = np.linspace(0, 180, 18)  # 0 to 2.5 in 0.1 increments
df['bin'] = pd.cut(df['angle'], bins, include_lowest=True)
df['bin_center'] = df['bin'].apply(lambda x: x.mid)


# Count per file-condition-bin
counts = (
    df.groupby(['file', 'condition', 'bin_center'])
    .size()
    .groupby(['file', 'condition'], group_keys=False)
    .apply(lambda x: x / x.sum())
    .reset_index(name='density')
)


sns.lineplot(data=counts, x='bin_center', y='density', hue='condition', errorbar=('ci', 95))

# sns.histplot(data=df, x='angle', hue='condition', stat='density', common_norm=False, alpha=0.5)


plt.xlabel('Angle', fontsize=12)
plt.ylabel('Probability', fontsize=12)

# plt.ylim(0, 0.06)


# plt.yscale('log')


# Add an overall title to the entire figure
plt.title('Trajectory Angle Probability Distribution', fontsize=16, fontweight='bold')

# Adjust layout to prevent overlap, considering the overall title
plt.tight_layout(rect=[0, 0, 1, 0.95])

# plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/angle/si.png', dpi=300, bbox_inches='tight')


# Show the pl
plt.show()
