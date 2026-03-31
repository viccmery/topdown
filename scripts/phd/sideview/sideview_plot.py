import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import matplotlib.patches as mpatches
import os
import matplotlib as mpl

################################################################
####### --- PLOTTING SIDEVIEW ANALYSIS FOR NDD-GENES --- #######
################################################################

output = '/Users/cochral/repos/behavioural-analysis/plots/phd/ndd-sideview'

################################################################
######## --------- PLOT PERCENTAGE IN CLUSTERS -------- ########
################################################################

df = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/PhD/NDD/EXPERIMENTS/SIDEVIEW/ANALYSIS/percent_in_clusters.csv')
print('Percentage in clusters data loaded:')

bin_size = 300 # 5fps *  60s
df["frame_bin"] = (df["frame"] // bin_size) * bin_size

average_df = (
    df.groupby(["condition", "file_id", "frame_bin"], as_index=False)["percent_in_clusters"]
      .mean()
      .rename(columns={"frame_bin": "frame"}))


control = "T21-A08"
conditions = sorted([c for c in average_df["condition"].unique() if c != control])

for cond in conditions:
    print(f"Plotting {cond}...")
    plot_df = average_df[average_df["condition"].isin([control, cond])].copy()
    plot_df["condition"] = pd.Categorical(plot_df["condition"], categories=[control, cond], ordered=True)

    plt.figure(figsize=(7, 4))
    sns.lineplot(
        data=plot_df,
        x="frame", y="percent_in_clusters",
        hue="condition",
        errorbar=("ci", 95)   # if seaborn old, replace with ci=95
    )

    plt.xlabel("Frame")
    plt.ylabel("Percent in clusters")
    plt.title(f"Percentage in Clusters")
    plt.legend(title="Condition")
    sns.despine()

    outpath = os.path.join(output, f"{cond}__percent_in_clusters.pdf")
    plt.savefig(outpath, format="pdf", bbox_inches="tight")
    plt.close()



################################################################
###### --------- PLOT NUMBER OF CLUSTERS / TIME -------- #######
################################################################

df = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/PhD/NDD/EXPERIMENTS/SIDEVIEW/ANALYSIS/n_clusters_over_time.csv')
print('Number of clusters data loaded:')

bin_size = 300 # 5fps *  60s
df["frame_bin"] = (df["frame"] // bin_size) * bin_size

number_of_clusters = (
    df.groupby(["condition", "file_id", "frame_bin"], as_index=False)["n_clusters"]
      .mean()
      .rename(columns={"frame_bin": "frame"}))


control = "T21-A08"
conditions = sorted([c for c in number_of_clusters["condition"].unique() if c != control])

for cond in conditions:
    print(f"Plotting {cond}...")
    plot_df = number_of_clusters[number_of_clusters["condition"].isin([control, cond])].copy()
    plot_df["condition"] = pd.Categorical(plot_df["condition"], categories=[control, cond], ordered=True)

    plt.figure(figsize=(7, 4))
    sns.lineplot(
        data=plot_df,
        x="frame", y="n_clusters",
        hue="condition",
        errorbar=("ci", 95)   # if seaborn old, replace with ci=95
    )

    plt.xlabel("Frame")
    plt.ylabel("Number of Clusters")
    plt.title(f"Number of Clusters")
    plt.legend(title="Condition")
    sns.despine()

    outpath = os.path.join(output, f"{cond}__number_of_clusters.pdf")
    plt.savefig(outpath, format="pdf", bbox_inches="tight")
    plt.close()


################################################################
####### --------- PLOT SIZE OF CLUSTERS / TIME -------- ########
################################################################

df = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/PhD/NDD/EXPERIMENTS/SIDEVIEW/ANALYSIS/avg_cluster_size_over_time.csv')
print('Average cluster size data loaded:')

bin_size = 300 # 5fps *  60s
df["frame_bin"] = (df["frame"] // bin_size) * bin_size

cluster_size = (
    df.groupby(["condition", "file_id", "frame_bin"], as_index=False)["avg_cluster_size"]
      .mean()
      .rename(columns={"frame_bin": "frame"}))


control = "T21-A08"
conditions = sorted([c for c in cluster_size["condition"].unique() if c != control])

for cond in conditions:
    print(f"Plotting {cond}...")
    plot_df = cluster_size[cluster_size["condition"].isin([control, cond])].copy()
    plot_df["condition"] = pd.Categorical(plot_df["condition"], categories=[control, cond], ordered=True)

    plt.figure(figsize=(7, 4))
    sns.lineplot(
        data=plot_df,
        x="frame", y="avg_cluster_size",
        hue="condition",
        errorbar=("ci", 95)   # if seaborn old, replace with ci=95
    )

    plt.xlabel("Frame")
    plt.ylabel("Average Cluster Size")
    plt.title(f"Average Cluster Size")
    plt.legend(title="Condition")
    sns.despine()

    outpath = os.path.join(output, f"{cond}__average_cluster_size.pdf")
    plt.savefig(outpath, format="pdf", bbox_inches="tight")
    plt.close()


################################################################
####### --------- PLOT MAX CLUSTER SIZE / TIME -------- ########
################################################################
df = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/PhD/NDD/EXPERIMENTS/SIDEVIEW/ANALYSIS/max_cluster_size_over_time.csv')
print('Max cluster size data loaded:')

bin_size = 300 # 5fps *  60s
df["frame_bin"] = (df["frame"] // bin_size) * bin_size

cluster_size = (
    df.groupby(["condition", "file_id", "frame_bin"], as_index=False)["max_cluster_size"]
      .mean()
      .rename(columns={"frame_bin": "frame"}))


control = "T21-A08"
conditions = sorted([c for c in cluster_size["condition"].unique() if c != control])

for cond in conditions:
    print(f"Plotting {cond}...")
    plot_df = cluster_size[cluster_size["condition"].isin([control, cond])].copy()
    plot_df["condition"] = pd.Categorical(plot_df["condition"], categories=[control, cond], ordered=True)

    plt.figure(figsize=(7, 4))
    sns.lineplot(
        data=plot_df,
        x="frame", y="max_cluster_size",
        hue="condition",
        errorbar=("ci", 95)   # if seaborn old, replace with ci=95
    )

    plt.xlabel("Frame")
    plt.ylabel("Max Cluster Size")
    plt.title(f"Max Cluster Size")
    plt.legend(title="Condition")
    sns.despine()

    outpath = os.path.join(output, f"{cond}__max_cluster_size.pdf")
    plt.savefig(outpath, format="pdf", bbox_inches="tight")
    plt.close()

################################################################
######### --------- DEPTH DIFFERENCE IN FOOD -------- ##########
################################################################
df = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/PhD/NDD/EXPERIMENTS/SIDEVIEW/ANALYSIS/depth_difference_food.csv')
print('Depth difference in food data loaded:')

plt.figure(figsize=(7, 4))
sns.stripplot(
    data=df,
    x="condition",
    y="depth_difference_food",
    color="grey",
    alpha=0.6,
    jitter=True
)

sns.pointplot(
    data=df,
    x="condition",
    y="depth_difference_food",
    errorbar="sd",
    color="green",
    linestyle="none"
)

plt.xlabel("Condition")
plt.ylabel("Food Depth Difference")
plt.title("Depth Difference in Food")
sns.despine()

outpath = os.path.join(output, f"depth_difference_food.pdf")
plt.savefig(outpath, format="pdf", bbox_inches="tight")
plt.close()


################################################################
######### --------- AVERAGE NEAREST NEIGHBOR -------- ##########
################################################################
df = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/PhD/NDD/EXPERIMENTS/SIDEVIEW/ANALYSIS/nearest_neighbour.csv')
print('Nearest neighbour data loaded:')

bin_size = 300 # 5fps *  60s
df["frame_bin"] = (df["frame"] // bin_size) * bin_size

nearest_neighbour = (
    df.groupby(["condition", "file_id", "frame_bin"], as_index=False)["mean_nearest_neighbour"]
      .mean()
      .rename(columns={"frame_bin": "frame"}))


control = "T21-A08"
conditions = sorted([c for c in nearest_neighbour["condition"].unique() if c != control])

for cond in conditions:
    print(f"Plotting {cond}...")
    plot_df = nearest_neighbour[nearest_neighbour["condition"].isin([control, cond])].copy()
    plot_df["condition"] = pd.Categorical(plot_df["condition"], categories=[control, cond], ordered=True)

    plt.figure(figsize=(7, 4))
    sns.lineplot(
        data=plot_df,
        x="frame", y="mean_nearest_neighbour",
        hue="condition",
        errorbar=("ci", 95)   # if seaborn old, replace with ci=95
    )

    plt.xlabel("Frame")
    plt.ylabel("Mean Nearest Neighbour")
    plt.title(f"Mean Nearest Neighbour")
    plt.legend(title="Condition")
    sns.despine()

    outpath = os.path.join(output, f"{cond}__mean_nearest_neighbour.pdf")
    plt.savefig(outpath, format="pdf", bbox_inches="tight")
    plt.close()
