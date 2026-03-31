
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import matplotlib.patches as mpatches
import matplotlib as mpl
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests


'''
Kruskal–Wallis test (non-parametric ANOVA equivalent).

Then if significant, do pairwise Mann–Whitney tests with correction.'''


mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']

df = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/grouped+isolated/closest_contacts_1mm.csv')

# df = df[df['frame'] < 2000].copy()

grouped = (
    df.groupby(['file', 'social_experience', 'Closest Interaction Type'])
    .size()
    .reset_index(name='count')
)

plt.figure(figsize=(12,8))
ax = sns.barplot(data=grouped, x='Closest Interaction Type', y='count', hue='social_experience',  edgecolor='black', linewidth=2, errorbar='sd', alpha=0.8)
plt.xlabel('Interaction Type', fontsize=12, fontweight='bold')
plt.ylabel('Total Contact Time (s)', fontsize=12, fontweight='bold')
sns.despine()
ax.legend(frameon=False, title=None, loc="upper right")
plt.tight_layout()
plt.ylim(0, 300)
plt.xticks(rotation=45)

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/lrs_paper/GHnSI/interaction_type-closestnode_n10.pdf', 
            format='pdf', bbox_inches='tight')
plt.close()



sub = df[(df["social_experience"] == "GH-SI") &
         (df["Closest Interaction Type"] == "head_tail")].copy()

# map track id -> group
def grp(t): 
    return "SI" if int(t) <= 4 else "GH"

# who is the HEAD in this head_tail contact?
sub["head_group"] = np.where(
    sub["track_0_node"] == "head",
    sub["track_0"].apply(grp),
    sub["track_1"].apply(grp)
)

# counts: head is SI vs head is GH
plot_df = sub["head_group"].value_counts().rename_axis("group").reset_index(name="count")

sns.barplot(data=plot_df, x="group", y="count", edgecolor="black", linewidth=1.5, errorbar=None)
plt.ylabel("SI-GH head_tail: who is head?")
plt.xlabel("")
sns.despine()
plt.tight_layout()
plt.show()