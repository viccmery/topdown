
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
    "Group Housed": 'steelblue',     
    "Pseudo Control": 'skyblue'}

HUE_ORDER = ["Group Housed", "Pseudo Control"]


df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/group-housed/nearest_neighbour.csv')
df1['condition'] = 'Group Housed'

df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/pseudo-n10/group-housed/nearest_neighbour.csv')
df2['condition'] = 'Pseudo Control'

df = pd.concat([df1, df2], ignore_index=True)


from scipy.stats import mannwhitneyu

# 1) Mean per track per file
track_means = (
    df.groupby(['filename', 'condition', 'track_id'])['head_distance']
    .mean()
    .reset_index()
)

# 2) Collapse tracks → single value per file
file_means = (
    track_means
    .groupby(['filename', 'condition'])['head_distance']
    .mean()
    .reset_index()
)

# 3) Split conditions
gh = file_means.query("condition == 'Group Housed'")['head_distance']
pc = file_means.query("condition == 'Pseudo Control'")['head_distance']

# 4) Mann–Whitney U test
u, p = mannwhitneyu(gh, pc, alternative='two-sided')

print(f"Mann–Whitney U = {u:.3f}, p = {p:.4e}")
print(f"N files — Group Housed: {len(gh)}, Pseudo Control: {len(pc)}")

def cliffs_delta(x, y):
    nx = len(x)
    ny = len(y)
    greater = sum(xi > yj for xi in x for yj in y)
    less = sum(xi < yj for xi in x for yj in y)
    return (greater - less) / (nx * ny)

delta = cliffs_delta(gh, pc)
print(f"Cliff's delta = {delta:.3f}")


plt.figure(figsize=(4,8))
sns.barplot(data=file_means, x='condition', y='head_distance', palette=PALETTE, order=HUE_ORDER
)
plt.ylim(0,16)
sns.despine()
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/lrs_paper/ghXpseudo/nearest-neighour-distance-barplot.pdf', format='pdf', bbox_inches='tight')
plt.close()








