
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import matplotlib.patches as mpatches
import matplotlib as mpl

# ---- Adobe / Illustrator friendly PDFs ----
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']

PALETTE = {
    "GH": 'steelblue',     
    "SI": 'darkorange',}

HUE_ORDER = ["GH", "SI"]


df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/group-housed/speed_over_time.csv')
df1['condition'] = 'GH'

df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/socially-isolated/speed_over_time.csv')
df2['condition'] = 'SI'


df = pd.concat([df1, df2], ignore_index=True)

bins = np.linspace(0, 2.5, 26)  # 0 to 2.5 in 0.1 increments
df['speed_bin'] = pd.cut(df['speed'], bins, include_lowest=True)
df['bin_center'] = df['speed_bin'].apply(lambda x: x.mid)

counts = (
    df.groupby(['file', 'condition', 'bin_center'])
    .size()
    .groupby(['file', 'condition'], group_keys=False)
    .apply(lambda x: x / x.sum())
    .reset_index(name='density')
)


plt.figure(figsize=(1.5,1.5), dpi=600)

ax = sns.lineplot(data=counts, x='bin_center', y='density', hue='condition', errorbar=('ci', 95), palette=PALETTE, hue_order=HUE_ORDER)


plt.ylabel('Probability', fontsize=12, fontweight='bold', labelpad=15)
plt.xlabel('Speed (mm/s)', fontsize=12, fontweight='bold', labelpad=15)

sns.despine()
plt.ylim(0, 0.4)
plt.xlim(0, 0.5)
plt.tight_layout(rect=[1, 1, 1, 1])


plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/lrs_paper/GHxSI/speed.pdf', format='pdf', bbox_inches='tight')

# Show the plot
plt.show()
