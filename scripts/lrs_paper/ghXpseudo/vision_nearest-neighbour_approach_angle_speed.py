
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
    "GH": 'steelblue',     
    "PSEUDO": 'skyblue'
}

HUE_ORDER = ["GH", "PSEUDO"]


df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/group-housed/nearest_neighbour.csv')
df1['condition'] = 'GH'
df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/pseudo-n10/group-housed/nearest_neighbour.csv')
df2['condition'] = 'PSEUDO'


df = pd.concat([df1, df2], ignore_index=True)


######### MIN DISTANCE X SPEED ############
bins = list(range(0, 90, 1))  # [0, 10, 20, ..., 100]
df['bin'] = pd.cut(df['closest_node_distance'], bins, include_lowest=True)
df['bin_right'] = df['bin'].apply(lambda x: x.right).astype(float)


angle_edges = np.arange(0, 181, 30)  # 0,30,60,90,120,150,180
df['angle_bin'] = pd.cut(df['approach_angle'], angle_edges, include_lowest=True, right=False)


baseline = (
    df.loc[df["closest_node_distance"].between(10, 20, inclusive="both")]
      .groupby(["condition", "filename"])["speed"]
      .mean()
      .rename("mean_speed_10_20")
      .reset_index()
)

# merge baseline back onto all rows
df = df.merge(baseline, on=["condition", "filename"], how="left")
# drop files that don't have any baseline data in 10–20 mm
df = df.dropna(subset=["mean_speed_10_20"]).copy()
# normalize
df["speed_norm"] = df["speed"] / df["mean_speed_10_20"]

df = df[df['bin_right'] <= 15].copy()

for binned_angle in df['angle_bin'].dropna().unique():
    df_sub = df[df['angle_bin'] == binned_angle]

    df_plot = (
    df_sub.groupby(['condition', 'filename', 'bin_right'], as_index=False)['speed_norm']
          .mean()
)

    plt.figure(figsize=(3,3))

    sns.lineplot(
        data=df_plot,
        x='bin_right',
        y='speed_norm',
        hue='condition',
        errorbar=('ci', 95),
        palette=PALETTE, hue_order=HUE_ORDER
    )

    plt.xlim(0, 5)
    plt.ylim(0.5, 1.1)
    # plt.xlabel("Nearest neighbour distance")
    # plt.ylabel("Speed")
    plt.legend().remove()
    sns.despine()
    # plt.axvline(x=1, color='gray', linestyle='--', linewidth=1)

    plt.title(f'Approach Angle {binned_angle}')

    plt.tight_layout(rect=[0, 0, 0.95, 0.93])
    plt.savefig(f'/Users/cochral/repos/behavioural-analysis/plots/lrs_paper/ghXpseudo/approach_angle-{binned_angle.left:.0f}–{binned_angle.right:.0f}.pdf', format='pdf', bbox_inches='tight')
    plt.close()




