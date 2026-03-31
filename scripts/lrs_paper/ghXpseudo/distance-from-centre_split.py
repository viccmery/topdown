
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys  
import matplotlib as mpl


mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']

PALETTE = {
    "GH": 'steelblue',     
    "PSEUDO": 'skyblue'}

HUE_ORDER = ["GH", "PSEUDO"]


df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/group-housed/distance_from_centre.csv')
df1['condition'] = 'GH'

df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/pseudo-n10/group-housed/distance_over_time.csv')
df2['condition'] = 'PSEUDO'

df = pd.concat([df1, df2], ignore_index=True)


bins = np.linspace(0, 50, 25)  # 0 to 2.5 in 0.1 increments
df['distance_bin'] = pd.cut(df['distance_from_centre'], bins, include_lowest=True)
df['bin_center'] = df['distance_bin'].apply(lambda x: x.mid)


counts = (
    df.groupby(['file', 'condition', 'bin_center'])
    .size()
    .groupby(['file', 'condition'], group_keys=False)
    .apply(lambda x: x / x.sum())
    .reset_index(name='density')
)

for cond in counts['condition'].unique():
    cond_data = counts[counts['condition'] == cond]

    plt.figure(figsize=(1.5,1.5))
    ax = sns.lineplot(data=cond_data, x='bin_center', y='density', color=PALETTE[cond], errorbar=('ci', 95))



    plt.xlabel('Distance From Centre (mm) ', fontsize=12)
    plt.ylabel('Probability', fontsize=12)

    sns.despine()
    ax.legend(frameon=False, title=None, fontsize=11, loc="upper right")
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')
    plt.ylim(0, 0.4)
    plt.xlim(0, 50)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.savefig(f'/Users/cochral/repos/behavioural-analysis/plots/lrs_paper/ghXpseudo/distance-from-centre_{cond}.pdf', format='pdf', bbox_inches='tight')
    plt.show()
