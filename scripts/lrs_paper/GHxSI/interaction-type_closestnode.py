
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
    "PSEUDO": "#F7D455",     
    "SI": 'darkorange',}

HUE_ORDER = [ "SI", "PSEUDO"]



df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/group-housed/closest_contacts_1mm.csv')
df1['condition'] = 'PSEUDO'

df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/socially-isolated/closest_contacts_1mm.csv')
df2['condition'] = 'SI'


df = pd.concat([df1, df2], ignore_index=True)

# Sum across all frame bins per file + interaction type
grouped = (
    df.groupby(['file', 'condition', 'Closest Interaction Type'])
    .size()
    .reset_index(name='count')
)

from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

results = []

for interaction, sub in grouped.groupby("Closest Interaction Type"):
    gh = sub.loc[sub["condition"] == "GH", "count"]
    si = sub.loc[sub["condition"] == "SI", "count"]

    # skip if one group is missing (safety)
    if gh.empty or si.empty:
        continue

    u, p = mannwhitneyu(gh, si, alternative="two-sided")

    results.append({
        "interaction_type": interaction,
        "n_GH": len(gh),
        "n_SI": len(si),
        "u_stat": u,
        "p_raw": p,
        "median_GH": gh.median(),
        "median_SI": si.median(),
        "delta_median": si.median() - gh.median()
    })

stats_df = pd.DataFrame(results)

# --- multiple comparisons correction (FDR) ---
passed, p_corr, _, _ = multipletests(
    stats_df["p_raw"],
    alpha=0.05,
    method="fdr_bh"
)

stats_df["p_corrected"] = p_corr
stats_df["passes_multiple_test_correction"] = passed

print(stats_df)

# --- ADD THIS BLOCK RIGHT HERE ---
def p_to_stars(p):
    if p <= 1e-4:
        return "****"
    if p <= 1e-3:
        return "***"
    if p <= 1e-2:
        return "**"
    if p <= 5e-2:
        return "*"
    return ""

# IMPORTANT: use the SAME label source as your plot x-axis
star_map = dict(
    zip(stats_df["interaction_type"], stats_df["p_corrected"].apply(p_to_stars))
)



plt.figure(figsize=(12,8))
ax = sns.barplot(data=grouped, x='Closest Interaction Type', y='count', hue='condition', hue_order = HUE_ORDER, edgecolor='black', linewidth=2, errorbar='sd', palette=PALETTE, alpha=0.8)

plt.xlabel('Interaction Type', fontsize=12, fontweight='bold')
plt.ylabel('Total Contact Time (s)', fontsize=12, fontweight='bold')

sns.despine()
ax.legend(frameon=False, title=None, loc="upper right")

# plt.title('Interaction Type (Closest Node)', fontsize=16, fontweight='bold')

plt.tight_layout()

plt.ylim(0, 1000)

plt.xticks(rotation=45)

# --- ADD STARS HERE (TRULY ROBUST) ---
# make a dict: category_label -> x_position (tick center)
tick_x = {t.get_text(): x for t, x in zip(ax.get_xticklabels(), ax.get_xticks())}

# collect bar centers + heights
bar_info = []
for p in ax.patches:
    x_center = p.get_x() + p.get_width() / 2
    bar_info.append((x_center, p.get_height()))

# for each category, find the two bars closest to its tick center, take max height, place star at tick center
for label, x in tick_x.items():
    stars = star_map.get(label, "")
    if not stars:
        continue

    # distance from each bar center to this category tick
    dists = [(abs(bx - x), h) for bx, h in bar_info]

    # take the TWO closest bars (GH + PSEUDO)
    dists.sort(key=lambda t: t[0])
    closest_two = dists[:2]

    # safety: if bars missing, skip
    if len(closest_two) == 0:
        continue

    max_height = max(h for _, h in closest_two)

    ax.text(
        x,                    # ALWAYS centered on the category tick
        max_height * 1.06,    # slightly above tallest bar in that category
        stars,
        ha="center",
        va="bottom",
        fontsize=14,
        fontweight="bold",
        zorder=10
    )
# --- END STARS ---







plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/lrs_paper/GHxSI/interaction_type-closestnode_n10.pdf', 
            format='pdf', bbox_inches='tight')
plt.show()


