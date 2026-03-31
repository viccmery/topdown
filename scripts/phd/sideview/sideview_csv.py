from operator import index
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import matplotlib.patches as mpatches
import os
from scipy.spatial.distance import cdist
import matplotlib as mpl

##### --- SIDEVIEW CLUSTER PERCENTAGE --- #####
def cluster_percentage(parent_folder):

    all_results = []

    for condition in os.listdir(parent_folder):
        print(f"Processing {condition}")

        cluster_folder = os.path.join(parent_folder, condition, 'clustering')

        if not os.path.isdir(cluster_folder):
            print(f"Skipping {condition} as it does not contain a 'clustering' folder.")
            continue

        cluster_files = [f for f in os.listdir(cluster_folder) if f.endswith('.feather')]

        dfs = []

        for cluster_file in cluster_files:
            df = pd.read_feather(os.path.join(cluster_folder, cluster_file))
            file_id = cluster_file.split(".predictions")[0]
            df['file_id'] = file_id
            dfs.append(df)

        if not dfs:
            print(f"No feather files found in {condition}")
            continue

        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df['frame'] = combined_df['frame'].astype(int)

        result = (
            combined_df
            .groupby(['file_id', 'frame'])
            .apply(lambda x: (x['cluster'] != -1).sum() / len(x) * 100)
            .reset_index(name='percent_in_clusters'))
        
        result.insert(0, 'condition', condition)

        all_results.append(result)

        result.to_csv(
            os.path.join(cluster_folder, 'percent_in_clusters.csv'),
            index=False
        )

    analysis_folder = os.path.join(parent_folder, "ANALYSIS")
    os.makedirs(analysis_folder, exist_ok=True)

    final_df = pd.concat(all_results, ignore_index=True)
    final_df.to_csv(os.path.join(analysis_folder, 'percent_in_clusters.csv'), index=False)



##### --- SIDEVIEW NUMBER OF CLUSTERS OVER TIME --- #####
def number_of_clusters(parent_folder):

    all_results = []

    for condition in os.listdir(parent_folder):
        print(f"Processing {condition}")

        cluster_folder = os.path.join(parent_folder, condition, 'clustering')

        if not os.path.isdir(cluster_folder):
            print(f"Skipping {condition} as it does not contain a 'clustering' folder.")
            continue

        cluster_files = [f for f in os.listdir(cluster_folder) if f.endswith('.feather')]

        dfs = []
        for cluster_file in cluster_files:
            df = pd.read_feather(os.path.join(cluster_folder, cluster_file))
            file_id = cluster_file.split(".predictions")[0]
            df["file_id"] = file_id
            dfs.append(df)

        if not dfs:
            print(f"No feather files found in {condition}")
            continue

        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df["frame"] = combined_df["frame"].astype(int)

        # count unique clusters per frame (excluding -1)
        result = (
            combined_df[combined_df["cluster"] != -1]
            .groupby(["file_id", "frame"])["cluster"]
            .nunique()
            .reset_index(name="n_clusters")
        )

        # if a frame had only -1, it won't appear above; optionally fill those with 0:
        all_frames = combined_df.groupby(["file_id", "frame"]).size().reset_index()[["file_id", "frame"]]
        result = all_frames.merge(result, on=["file_id", "frame"], how="left").fillna({"n_clusters": 0})

        result.insert(0, "condition", condition)
        all_results.append(result)

        result.to_csv(os.path.join(cluster_folder, "n_clusters_over_time.csv"), index=False)
    

    if all_results:
        analysis_folder = os.path.join(parent_folder, "ANALYSIS")
        os.makedirs(analysis_folder, exist_ok=True)
        final_df = pd.concat(all_results, ignore_index=True)
        final_df.to_csv(os.path.join(analysis_folder, "n_clusters_over_time.csv"), index=False)


##### --- SIDEVIEW AVERAGE CLUSTER SIZE OVER TIME --- #####
def average_cluster_size(parent_folder):

    all_results = []

    for condition in os.listdir(parent_folder):
        print(f"Processing {condition}")

        cluster_folder = os.path.join(parent_folder, condition, 'clustering')

        if not os.path.isdir(cluster_folder):
            print(f"Skipping {condition} as it does not contain a 'clustering' folder.")
            continue

        cluster_files = [f for f in os.listdir(cluster_folder) if f.endswith('.feather')]

        dfs = []
        for cluster_file in cluster_files:
            df = pd.read_feather(os.path.join(cluster_folder, cluster_file))
            file_id = cluster_file.split(".predictions")[0]
            df["file_id"] = file_id
            dfs.append(df)

        if not dfs:
            print(f"No feather files found in {condition}")
            continue

        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df["frame"] = combined_df["frame"].astype(int)

        # compute cluster sizes per frame (excluding -1), then average those sizes per frame
        clustered = combined_df[combined_df["cluster"] != -1].copy()

        cluster_sizes = (
            clustered
            .groupby(["file_id", "frame", "cluster"])
            .size()
            .reset_index(name="cluster_size")
        )

        result = (
            cluster_sizes
            .groupby(["file_id", "frame"])["cluster_size"]
            .mean()
            .reset_index(name="avg_cluster_size")
        )

        # frames with no clusters present -> avg_cluster_size = 0 (or NaN if you prefer)
        all_frames = combined_df.groupby(["file_id", "frame"]).size().reset_index()[["file_id", "frame"]]
        result = all_frames.merge(result, on=["file_id", "frame"], how="left").fillna({"avg_cluster_size": 0})

        result.insert(0, "condition", condition)
        all_results.append(result)

        result.to_csv(os.path.join(cluster_folder, "avg_cluster_size_over_time.csv"), index=False)

    if all_results:
        analysis_folder = os.path.join(parent_folder, "ANALYSIS")
        os.makedirs(analysis_folder, exist_ok=True)
        final_df = pd.concat(all_results, ignore_index=True)
        final_df.to_csv(os.path.join(analysis_folder, "avg_cluster_size_over_time.csv"), index=False)


##### --- SIDEVIEW AVERAGE CLUSTER SIZE OVER TIME --- #####
def max_cluster_size(parent_folder):

    all_results = []

    for condition in os.listdir(parent_folder):
        print(f"Processing {condition}")

        cluster_folder = os.path.join(parent_folder, condition, 'clustering')

        if not os.path.isdir(cluster_folder):
            print(f"Skipping {condition} as it does not contain a 'clustering' folder.")
            continue

        cluster_files = [f for f in os.listdir(cluster_folder) if f.endswith('.feather')]

        dfs = []
        for cluster_file in cluster_files:
            df = pd.read_feather(os.path.join(cluster_folder, cluster_file))
            file_id = cluster_file.split(".predictions")[0]
            df["file_id"] = file_id
            dfs.append(df)

        if not dfs:
            print(f"No feather files found in {condition}")
            continue

        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df["frame"] = combined_df["frame"].astype(int)

        # compute cluster sizes per frame (excluding -1), then average those sizes per frame
        clustered = combined_df[combined_df["cluster"] != -1].copy()

        cluster_sizes = (
            clustered
            .groupby(["file_id", "frame", "cluster"])
            .size()
            .reset_index(name="cluster_size")
        )

        result = (
            cluster_sizes
            .groupby(["file_id", "frame"])["cluster_size"]
            .max()
            .reset_index(name="max_cluster_size")
        )

        # frames with no clusters present -> max_cluster_size = 0 (or NaN if you prefer)
        all_frames = combined_df.groupby(["file_id", "frame"]).size().reset_index()[["file_id", "frame"]]
        result = all_frames.merge(result, on=["file_id", "frame"], how="left").fillna({"max_cluster_size": 0})

        result.insert(0, "condition", condition)
        all_results.append(result)

        result.to_csv(os.path.join(cluster_folder, "max_cluster_size_over_time.csv"), index=False)

    if all_results:
        analysis_folder = os.path.join(parent_folder, "ANALYSIS")
        os.makedirs(analysis_folder, exist_ok=True)
        final_df = pd.concat(all_results, ignore_index=True)
        final_df.to_csv(os.path.join(analysis_folder, "max_cluster_size_over_time.csv"), index=False)



def depth_difference_food(parent_folder):

    all_results = []

    for condition in os.listdir(parent_folder):
        print(f"Processing {condition}")

        cluster_folder = os.path.join(parent_folder, condition, "clustering")
        if not os.path.isdir(cluster_folder):
            print(f"Skipping {condition} as it does not contain a 'clustering' folder.")
            continue

        cluster_files = [f for f in os.listdir(cluster_folder) if f.endswith(".feather")]

        for cluster_file in cluster_files:

            df = pd.read_feather(os.path.join(cluster_folder, cluster_file))
            file_id = cluster_file.split(".predictions")[0]

            # first 500 frames
            first = df[df["frame"] < 500]["y_tail"].mean()

            # last 500 frames
            last_frames = df["frame"].max()
            # last = df[df["frame"] > (last_frames - 500)]["y_tail"].mean()

            last_df = df[df["frame"] > (last_frames - 500)]
            # keep only larvae that are not above the initial depth
            last_df = last_df[last_df["y_tail"] >= first]
            last = last_df["y_tail"].mean()

            
            diff = last - first

            result = pd.DataFrame({
                "condition":[condition],
                "file_id":[file_id],
                "depth_difference_food":[diff]
            })

            all_results.append(result)

    if all_results:
        analysis_folder = os.path.join(parent_folder, "ANALYSIS")
        os.makedirs(analysis_folder, exist_ok=True)

        final_df = pd.concat(all_results, ignore_index=True)
        final_df.to_csv(os.path.join(analysis_folder,"depth_difference_food.csv"),index=False)



def nearest_neighbour(parent_folder):

    all_results = []

    for condition in os.listdir(parent_folder):
        print(f"Processing {condition}")

        cluster_folder = os.path.join(parent_folder, condition, "clustering")
        if not os.path.isdir(cluster_folder):
            print(f"Skipping {condition} as it does not contain a 'clustering' folder.")
            continue

        cluster_files = [f for f in os.listdir(cluster_folder) if f.endswith(".feather")]

        for cluster_file in cluster_files:

            df = pd.read_feather(os.path.join(cluster_folder, cluster_file))
            file_id = cluster_file.split(".predictions")[0]

            frames = []

            for frame, frame_df in df.groupby("frame"):

                coords = frame_df[["x_head","y_head"]].values

                if len(coords) < 2:
                    continue

                dist_matrix = cdist(coords, coords)

                np.fill_diagonal(dist_matrix, np.inf)

                nearest = dist_matrix.min(axis=1)

                frames.append({
                    "condition":condition,
                    "file_id":file_id,
                    "frame":frame,
                    "mean_nearest_neighbour":nearest.mean()
                })

            all_results.append(pd.DataFrame(frames))

    if all_results:

        analysis_folder = os.path.join(parent_folder, "ANALYSIS")
        os.makedirs(analysis_folder, exist_ok=True)

        final_df = pd.concat(all_results, ignore_index=True)
        final_df.to_csv(os.path.join(analysis_folder,"nearest_neighbour.csv"),index=False)




##### --- RUNNING ANALYSIS FOR SIDEVIEW VIDEOS --- #####

parent_folder = '/Volumes/lab-windingm/home/users/cochral/PhD/NDD/EXPERIMENTS/SIDEVIEW'
# cluster_percentage(parent_folder)
# number_of_clusters(parent_folder)
# average_cluster_size(parent_folder)
# max_cluster_size(parent_folder)
# depth_difference_food(parent_folder)
nearest_neighbour(parent_folder)    

