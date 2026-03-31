import sys
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pyarrow.feather as feather
import matplotlib.patches as mpatches
import cv2
from joblib import Parallel, delayed
from matplotlib.colors import ListedColormap
import re 
import math
import ast 
from sklearn.decomposition import PCA



syllable_groups = {
    "orange1": [33, 43], ##orange
    "orange": [44, 45], ##orange
    "orange2": [2, 19], ##orange
    "orange3": [28, 37], ##orange
    'orange4': [6, 11], ##orange
    'blue': [1], ##blue
    'green': [16], ##green
    'green2': [39, 27, 40, 26, 25, 22, 30, 9, 23, 10, 24, 12, 17, 35, 29, 42], ##green
    'green3': [5, 36, 13, 15, 4 ,21, ], ##green
    'green4': [3, 32], ##green
    'green5': [0, 14, 38, 18, 20], ##green
    'green6': [34, 8, 41, 7, 31], ##green
}

group_colors = {
    "orange4": 'chocolate',
    "orange": 'yellow',
    "orange1": 'darkorange',
    "orange2": 'lightsalmon',
    "orange3": 'orange',
    "blue": 'dodgerblue',
    "green2": 'mediumseagreen',
    "green": 'forestgreen',
    "green3": 'limegreen',
    "green4": 'palegreen',
    "green5": 'lightgreen',
    "green6": 'olivedrab',
}


# map syllable id -> group name
syll_to_group = {
    s: g for g, syls in syllable_groups.items() for s in syls
}


""" ANALYSIS PIPELINE FOR MOSEQ DATA IN ATTRACTION RIG """
# --------------------------------------------------------
# BASIC_STATS: duration, frequency from stats_df
# --------------------------------------------------------
def basic_stats(df, output):
    
    plt.figure(figsize=(8,6))
    sns.barplot(data=df, x='syllable', y='duration',  ci='sd')
    plt.title('Syllable Duration by Condition')
    plt.ylim(0, None)
    plt.xticks(rotation=90)
    plt.savefig(os.path.join(output, 'syllable_duration.png'), dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8,6))
    sns.barplot(data=df, x='syllable', y='frequency',  ci='sd')
    plt.title('Syllable Frequency by Condition')
    plt.ylim(0, None)
    plt.xticks(rotation=90)
    plt.savefig(os.path.join(output, 'syllable_frequency.png'), dpi=300, bbox_inches='tight')
    plt.close()
# --------------------------------------------------------
# DURATIONS: test durations of syllables
# --------------------------------------------------------
def durations(df, output):

    def durations_by_onset(df):

        out = []
        for name, g in df.groupby("name", sort=False):
            g = g.reset_index(drop=False)  # keep original index as 'index'
            onset_rows = g[g["onset"] == True]

            start_idxs = onset_rows["index"].tolist()               # Frames: 0  1  2  3  4  5  6
            sylls = onset_rows["syllable"].tolist()                 # Onset:  T  F  F  T  F  T  F
            # add sentinel “end” at one past the last row index     # Sylls:  37 37 37 40 40 41 41
            end_sentinel = g["index"].iloc[-1] + 1                  # → Start indices: [0, 3, 5]
            next_starts = start_idxs[1:] + [end_sentinel]           # → Next starts:  [3, 5, 7]
                                                                    # → Durations:    [3, 2, 2]
            for s, n, sy in zip(start_idxs, next_starts, sylls):
                out.append({
                    "name": name,
                    "syllable": sy,
                    "start_idx": s,
                    "end_idx_exclusive": n,
                    "duration_frames": n - s
                })

        return pd.DataFrame(out)
    
    df = df.sort_values(["name", "frame_index"]).reset_index(drop=True)
    durations = durations_by_onset(df)
    print(durations)

    output = os.path.join(output, 'syllable_duration_distributions')
    if not os.path.exists(output):
        os.makedirs(output)

    # number of bouts per syllable
    bout_counts = (
        durations.groupby("syllable")
                .size()
                .reset_index(name="num_bouts"))

    plt.figure(figsize=(8,6))
    sns.barplot(data=bout_counts, x="syllable", y="num_bouts", color="steelblue")
    plt.title("Number of Bouts per Syllable")
    plt.xlabel("Syllable")
    plt.ylabel("Number of Bouts")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(output, "bouts_per_syllable.png"), dpi=300)
    plt.close()


    # total frames per syllable

    total_frames = (durations.groupby(["name", "syllable"])["duration_frames"].sum().reset_index(name="total_frames"))
    plt.figure(figsize=(8,6))
    sns.barplot(data=total_frames, x='syllable', y='total_frames', ci='sd')
    plt.title('Total Frames per Syllable by Condition')
    plt.ylim(0, None)
    plt.xticks(rotation=90)
    plt.savefig(os.path.join(output, 'total_frames_per_syllable.png'), dpi=300, bbox_inches='tight')
    plt.close()

    for syllable in durations['syllable'].unique():

        sub = durations[durations['syllable'] == syllable]
        plt.figure(figsize=(8,6))
        sns.histplot(sub['duration_frames'], bins=20, kde=False)
        plt.title(f'Syllable {syllable} Duration Distribution')
        plt.xlabel('Duration (frames)')
        plt.ylabel('Count')
        plt.xticks(rotation=90)
        plt.savefig(os.path.join(output, f'syllable_{syllable}_duration_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    

    grouped_1_frame = durations[durations['duration_frames'] < 2].groupby(['syllable', 'name']).size().reset_index(name='count_1_frame')

    plt.figure(figsize=(8,6))
    sns.barplot(data=grouped_1_frame, x='syllable', y='count_1_frame', ci='sd')
    plt.title('Frequency of 1 frame Syllable Occurrences')
    plt.ylim(0, None)
    plt.xticks(rotation=90)
    plt.savefig(os.path.join(output, 'frequency_1_frame_syllable_occurrences.png'), dpi=300, bbox_inches='tight')
    plt.close()

    total_counts = (durations.groupby(["syllable", 'name']).size().reset_index(name="total_bouts"))
    freqs = total_counts.merge(grouped_1_frame, on=["name", "syllable"], how="left").fillna(0)
    freqs["fraction_1_frame"] = freqs["count_1_frame"] / freqs["total_bouts"]

    plt.figure(figsize=(8,6))
    sns.barplot(data=freqs, x='syllable', y='fraction_1_frame', ci='sd')
    plt.title('Fraction of 1 frame Syllable Occurrences')
    plt.ylim(0, None)
    plt.xticks(rotation=90)
    plt.savefig(os.path.join(output, 'fraction_1_frame_syllable_occurrences.png'), dpi=300, bbox_inches='tight')
    plt.close()
# --------------------------------------------------------
# SYLLABLE_OVERLAY: overlay syllables on video 
# --------------------------------------------------------
def syllable_overlay(df, output, video_track_name, video_path, output_name):

    f = df[df['name'] == video_track_name]
    print(f)
    f = f.sort_values('frame_index')
    coord_columns = ['centroid_x', 'centroid_y']  # replace with your actual centroid column names

    image_size = 1400
    original_video = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_output = cv2.VideoWriter(os.path.join(output, output_name), fourcc, 25.0, (image_size, image_size))

    frame_number = 0
    while original_video.isOpened():
        ret, frame = original_video.read()
        if not ret:
            break

        frame_df = f[f['frame_index'] == frame_number]

        for _, row in frame_df.iterrows():
            x = row['centroid_x']
            y = row['centroid_y']
            if np.isnan([x, y]).any():
                continue
            x, y = int(x), int(y)
            color = (0, 0, 255)
            cv2.circle(frame, (x, y), radius=8, color=color, thickness=-1)
            syllable = str(row['syllable'])  # make sure it's a string
            cv2.putText(frame, syllable, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        1.2, (0, 0, 255), thickness=1, lineType=cv2.LINE_AA)

        video_output.write(frame)
        frame_number += 1
    original_video.release()
    video_output.release()


df = pd.read_csv('/Users/cochral/Desktop/MOSEQ/KEYPOINT-KAPPA1500/moseq_df.csv')
output = '/Users/cochral/Desktop/MOSEQ/KEYPOINT-KAPPA1500/plots/testing_empty_syllables'
video_track_name = "N10-SI-_2025-02-18_13-53-15_td12_track4"
video_path = "/Users/cochral/Desktop/MOSEQ/videos/SI-N10_2025-02-18_13-53-15_td12.mp4"
output_name = "SI-N10_2025-02-18_13-53-15_td12_track4.mp4"

syllable_overlay(df, output, video_track_name, video_path, output_name)



# --------------------------------------------------------
# SYLLABLE_FEATURES: quantify syllable features 
# --------------------------------------------------------
def syllable_features(df, output):

    output = os.path.join(output, 'syllable_quantifications')
    if not os.path.exists(output):
        os.makedirs(output)

    df = df.sort_values(["name", "frame_index"]).reset_index(drop=True)
    df["bout_id"] = df.groupby("name")["onset"].cumsum()
    df["frame_per_bout"] = df.groupby(["name", "bout_id"]).cumcount()

    df.to_csv(os.path.join(output, 'moseq_df_normalized_frame.csv'), index=False)

    # Fixed Y axis range 
    hmin, hmax   = -1, 1    # heading range (radians)
    avmin, avmax = -8, 8          # angular velocity
    vmin, vmax   = 0, 450          # speed (px/s)

    for syllable in sorted(df['syllable'].unique()):
        sub = df[df["syllable"] == syllable]  

        fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)

        durations = ( sub.groupby(["name", "bout_id"]).size())  # length of each syllable bout
    

        median_dur = durations.median()
        low95, high95 = durations.quantile([0.025, 0.975]).values

        ax1 = axes[0]
        sns.lineplot(data=sub, x='frame_per_bout', y='heading', ax=ax1, legend=False, ci=95)
        ax2 = axes[1]
        sns.lineplot(data=sub, x='frame_per_bout', y='angular_velocity', ax=ax2, legend=False, ci=95)
        ax3 = axes[2]
        sns.lineplot(data=sub, x='frame_per_bout', y='velocity_px_s', ax=ax3, legend=False, ci=95)

        for ax in axes:
            ax.axvline(median_dur, linestyle="--", color="gray", label="median duration")
            ax.axvline(low95, linestyle=":", color="gray", label="95% CI")
            ax.axvline(high95, linestyle=":", color="gray")

        ax1.set_title(f'Syllable {syllable} Feature Quantifications')
        ax1.set_ylabel('Heading (radians)')
        ax1.set_ylim(hmin, hmax)
        ax2.set_ylabel('Angular Velocity (radians/s)')
        ax2.set_ylim(avmin, avmax)
        ax3.set_ylabel('Velocity (px/s)')
        ax3.set_ylim(vmin, vmax)
        ax3.set_xlabel('Frame (0 = onset)')

        plt.tight_layout()
        plt.savefig(os.path.join(output, f'{syllable}_mean.png'), dpi=300, bbox_inches='tight')
        plt.close() 
    
    for syllable in sorted(df['syllable'].unique()):
        sub = df[df["syllable"] == syllable]  

        fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)

        durations = ( sub.groupby(["name", "bout_id"]).size() )  # length of each syllable bout
        median_dur = durations.median()
        low95, high95 = durations.quantile([0.025, 0.975]).values

        # --- NEW: plot every individual bout trace in light gray on each axis ---
        for ax, ycol in zip(axes, ['heading','angular_velocity','velocity_px_s']):
            for (_, g) in sub.groupby(["name","bout_id"]):
                ax.plot(
                    g["frame_per_bout"].values, g[ycol].values,
                    color="0.8", linewidth=0.7, alpha=0.4
                )
        # -----------------------------------------------------------------------

        ax1 = axes[0]
        sns.lineplot(data=sub, x='frame_per_bout', y='heading',
                    ax=ax1, legend=False, ci=None, color='black', linewidth=2)

        ax2 = axes[1]
        sns.lineplot(data=sub, x='frame_per_bout', y='angular_velocity',
                    ax=ax2, legend=False, ci=None, color='black', linewidth=2)

        ax3 = axes[2]
        sns.lineplot(data=sub, x='frame_per_bout', y='velocity_px_s',
                    ax=ax3, legend=False, ci=None, color='black', linewidth=2)

        for ax in axes:
            ax.axvline(median_dur, linestyle="--", color="gray", label="median duration")
            ax.axvline(low95, linestyle=":", color="gray", label="95% CI")
            ax.axvline(high95, linestyle=":", color="gray")

        ax1.set_title(f'Syllable {syllable} Feature Quantifications')
        ax1.set_ylabel('Heading (radians)')
        ax1.set_ylim(hmin, hmax)
        ax2.set_ylabel('Angular Velocity (radians/s)')
        ax2.set_ylim(avmin, avmax)
        ax3.set_ylabel('Velocity (px/s)')
        ax3.set_ylim(vmin, vmax)
        ax3.set_xlabel('Frame (0 = onset)')

        plt.tight_layout()
        plt.savefig(os.path.join(output, f'{syllable}_traces.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    ## plot syllables over normalised time

    bout_lengths = (df.groupby(["name", 'syllable', "bout_id"]).size().reset_index(name="bout_len"))
    df = df.merge(bout_lengths, on=["name", "syllable", "bout_id"], how="left")
    df["normalised_frames"] = np.where(df["bout_len"] > 1, df["frame_per_bout"] / (df["bout_len"] - 1),0.0)

    bounds = (
    bout_lengths.groupby("syllable")["bout_len"]
                .quantile([0.1, 0.9])
                .unstack())
    bounds.columns = ["low95", "high95"]

    low95  = df["syllable"].map(bounds["low95"])
    high95 = df["syllable"].map(bounds["high95"])

    df = df[(df["bout_len"] >= low95) & (df["bout_len"] <= high95)]


    # get per-syllable quantile limits
    for syllable in sorted(df['syllable'].unique()):
        sub = df[df["syllable"] == syllable]  

        fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)

        ax1 = axes[0]
        sns.lineplot(data=sub, x='normalised_frames', y='heading', ax=ax1, legend=False, ci=95)
        ax2 = axes[1]
        sns.lineplot(data=sub, x='normalised_frames', y='angular_velocity', ax=ax2, legend=False, ci=95)
        ax3 = axes[2]
        sns.lineplot(data=sub, x='normalised_frames', y='velocity_px_s', ax=ax3, legend=False, ci=95)


        ax1.set_title(f'Syllable {syllable} Feature Quantifications')
        ax1.set_ylabel('Heading (radians)')
        ax1.set_ylim(hmin, hmax)
        ax1.set_xlim(0,1)
        ax2.set_ylabel('Angular Velocity (radians/s)')
        ax2.set_ylim(avmin, avmax)
        ax2.set_xlim(0,1)
        ax3.set_ylabel('Velocity (px/s)')
        ax3.set_ylim(vmin, vmax)
        ax3.set_xlim(0,1)
        ax3.set_xlabel('Frame (0 = onset)')

        plt.tight_layout()
        plt.savefig(os.path.join(output, f'{syllable}_normalised.png'), dpi=300, bbox_inches='tight')
        plt.close() 


# --------------------------------------------------------
# COMPARING_MODELS: model comparison plots
# --------------------------------------------------------
def comparing_models(directory):

    dfs = []
    for folder in os.listdir(directory):
        if not folder.startswith('KEYPOINT'):
            continue 
        kappa = folder.split("KAPPA")[-1]
        path = os.path.join(directory, folder, 'moseq_df.csv')
        if not os.path.exists(path):
            print(f"Skipping {folder}: no moseq_df.csv found")
            continue
        df = pd.read_csv(path)
        df['kappa'] = int(kappa)
        dfs.append(df)

    output = os.path.join(directory, 'model_comparisons')
    if not os.path.exists(output):
        os.makedirs(output)
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(os.path.join(output, 'combined_moseq_df.csv'), index=False)

    dfs_stats = []
    for folder in os.listdir(directory):
        if not folder.startswith('KEYPOINT'):
            continue 
        kappa = folder.split("KAPPA")[-1]
        path = os.path.join(directory, folder, 'stats_df.csv')
        if not os.path.exists(path):
            print(f"Skipping {folder}: no stats_df.csv found")
            continue
        df_stat = pd.read_csv(path)
        df_stat['kappa'] = int(kappa)
        dfs_stats.append(df_stat)

    df_stat = pd.concat(dfs_stats, ignore_index=True)
    df_stat.to_csv(os.path.join(output, 'combined_summary_df.csv'), index=False)


    # 1. QUANTIFY 1 FRAME SYLLABLE OCCURRENCES

    def durations_by_onset(df):
        out = []
        for (kappa, name), g in df.groupby(["kappa", "name"], sort=False):
            g = g.reset_index(drop=False)  # keep original index as 'index'
            onset_rows = g[g["onset"] == True]

            start_idxs = onset_rows["index"].tolist()               # Frames: 0  1  2  3  4  5  6
            sylls = onset_rows["syllable"].tolist()                 # Onset:  T  F  F  T  F  T  F
            # add sentinel “end” at one past the last row index     # Sylls:  37 37 37 40 40 41 41
            end_sentinel = g["index"].iloc[-1] + 1                  # → Start indices: [0, 3, 5]
            next_starts = start_idxs[1:] + [end_sentinel]           # → Next starts:  [3, 5, 7]
                                                                    # → Durations:    [3, 2, 2]
            for s, n, sy in zip(start_idxs, next_starts, sylls):
                out.append({
                    'kappa': kappa,
                    "name": name,
                    "syllable": sy,
                    "start_idx": s,
                    "end_idx_exclusive": n,
                    "duration_frames": n - s})
        return pd.DataFrame(out)
    
    df = df.sort_values(['kappa', "name", "frame_index"]).reset_index(drop=True)
    durations = durations_by_onset(df)
    print(durations)

    for model in durations['kappa'].unique():
        sub = durations[durations['kappa'] == model]
        plt.figure(figsize=(8,6))
        sns.boxplot(data=sub, x='syllable', y='duration_frames')
        plt.title(f'{model} Syllable Duration Distribution')
        plt.xlabel('Syllable')
        plt.ylabel('Duration (frames)')
        plt.xticks(rotation=70)
        out = os.path.join(output, 'syllable_duration_distributions')
        if not os.path.exists(out):
            os.makedirs(out)
        plt.savefig(os.path.join(out, f'kappa_{model}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    


    ## per video 
    grouped_1_frame = durations[durations['duration_frames'] < 2].groupby(['kappa', 'name'])['duration_frames'].sum().reset_index(name='count_1_frame') # sum of frames 
    video_totals = durations.groupby(['kappa', 'name'])['duration_frames'].sum().reset_index(name='video_frame_count_total') # total frames per video
    video_level = video_totals.merge(grouped_1_frame, on=['kappa', 'name'], how='left')
    video_level['percent_1_frame'] = (video_level['count_1_frame'] / video_level['video_frame_count_total'] * 100)
    video_level.to_csv(os.path.join(output, 'percentage_1_frame_syllable_occurrences_per_video.csv'), index=False)

    plt.figure(figsize=(8,6))
    sns.barplot(data=video_level, x='kappa', y='percent_1_frame', ci='sd')
    plt.title('Percentage of 1 Frame Syllable Occurrences')
    plt.ylim(0, None)
    plt.xlabel('Kappa')
    plt.ylabel('Percentage of 1 Frame Syllable Occurrences (%)')
    plt.savefig(os.path.join(output, 'percentage_1_frame_syllable_occurrences_per_video.png'), dpi=300, bbox_inches='tight')
    plt.close()

    ## per syllable - identifying the % of syllables which are rubbish (dominated by 1 frame percentage is high)
    syllable_1_frame = durations[durations['duration_frames'] < 2].groupby(['kappa',  'syllable'])['duration_frames'].sum().reset_index(name='count_1_frame') 
    syllable_totals = durations.groupby(['kappa', 'syllable'])['duration_frames'].sum().reset_index(name='syllable_frame_count_total')
    syllable_level = syllable_totals.merge(syllable_1_frame, on=['kappa',  'syllable'], how='left')
    syllable_level['count_1_frame'] = syllable_level['count_1_frame'].fillna(0)
    syllable_level['percent_1_frame'] = (syllable_level['count_1_frame'] / syllable_level['syllable_frame_count_total'] * 100)
    syllable_level.to_csv(os.path.join(output, 'percentage_1_frame_syllable_occurrences_per_syllable.csv'), index=False)

    thresholds = [1, 5, 10, 30, 50, 70, 90]

    bad_syllables = []  
    for thresh in thresholds:
        syllable_level['bad_syllable'] = syllable_level['percent_1_frame'] > thresh #boolean- syllable above or over threshold
        summary = (syllable_level.groupby('kappa')['bad_syllable'].mean() * 100).reset_index(name=f'percent_junk_syllables')
        for _, row in summary.iterrows():
            bad_syllables.append({
                'kappa': int(row['kappa']),
                'threshold_percent': thresh,
                '1_frame_dominant_syllables_percent': row['percent_junk_syllables'],
            })
    
    bad_syllables_df = pd.DataFrame(bad_syllables)
    bad_syllables_df.to_csv(os.path.join(output, 'bad_syllables_summary.csv'), index=False)

    for threshold in bad_syllables_df['threshold_percent'].unique():

        sub = bad_syllables_df[bad_syllables_df['threshold_percent'] == threshold]
        plt.figure(figsize=(8,6))
        sns.barplot(data=sub, x='kappa', y='1_frame_dominant_syllables_percent')
        plt.title(f'Percentage of Syllables Dominated by 1 Frame (> {threshold}%)')
        plt.ylim(0, None)
        plt.xlabel('Kappa')
        plt.ylabel('Percentage of Junk Syllables (%)')
        plt.savefig(os.path.join(output, f'bad_syllables_percentage_threshold_{threshold}.png'), dpi=300, bbox_inches='tight')
        plt.close()


    # 2. QUANTIFY NUMBER OF SYLLABLES UNDER THRESHOLD (NOT INC IN SUMMARY GIFS)

    total_syllable_durations = durations.groupby(['kappa', 'syllable'])['duration_frames'].sum().reset_index(name='syllable_frame_count_total')

    kappa_rows = []

    for kappa in sorted(total_syllable_durations['kappa'].unique()):
        
        ## moseq_df syllables
        sub_total = total_syllable_durations[total_syllable_durations['kappa'] == kappa]
        total_syllable_number = sub_total['syllable'].nunique()
        total_frames = sub_total['syllable_frame_count_total'].sum()

        ## stats_df syllables
        df_stat['kappa'] = df_stat['kappa'].astype(int)
        sub_stats_df = df_stat[df_stat['kappa'] == int(kappa)]
        syllables_stats = sub_stats_df['syllable'].unique()

        ## identify hidden syllables 
        hidden = sub_total[~sub_total['syllable'].isin(syllables_stats)]

        n_hidden_syllables = hidden['syllable'].nunique()
        hidden_frames = hidden['syllable_frame_count_total'].sum()

        percent_hidden_frames = (
            100 * hidden_frames / total_frames if total_frames > 0 else 0.0)
        
        percentage_hidden_syllables = (
            100 * n_hidden_syllables / total_syllable_number if total_syllable_number > 0 else 0.0)

        kappa_rows.append({
            'kappa': int(kappa),
            'total_syllables': int(total_syllable_number),
            'no_underthreshold_syllables': int(n_hidden_syllables),
            'percent_underthreshold_syllables': percentage_hidden_syllables,
            'percent_frames_underthreshold_syllables': percent_hidden_frames,
        })
    
    underthreshold_syllables_df = pd.DataFrame(kappa_rows)
    underthreshold_syllables_df.to_csv(os.path.join(output, 'underthreshold_syllables_per_kappa.csv'),index=False)

    # % if syllables under threshold
    plt.figure(figsize=(8,6))
    sns.barplot(data=underthreshold_syllables_df, x='kappa', y='percent_underthreshold_syllables')
    plt.title('Percentage of Syllables Under Threshold')
    plt.ylim(0, None)
    plt.xlabel('Kappa')
    plt.ylabel('Percentage of Syllables Under Threshold (%)- good')
    plt.savefig(os.path.join(output, 'percentage_underthreshold_syllables.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # % frames belonging to syllables under threshold
    plt.figure(figsize=(8,6))
    sns.barplot(data=underthreshold_syllables_df, x='kappa', y='percent_frames_underthreshold_syllables')
    plt.title('Percentage of Frames Belonging to Syllables Under Threshold')
    plt.ylim(0, None)
    plt.xlabel('Kappa')
    plt.ylabel('Percentage of Frames Under Threshold (%)')
    plt.savefig(os.path.join(output, 'percentage_underthreshold_syllables_frames.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # plot raw number of syllables under threshold and total
    long = underthreshold_syllables_df.melt(
    id_vars='kappa',
    value_vars=['total_syllables', 'no_underthreshold_syllables'],
    var_name='syllable_type',
    value_name='count')

    plt.figure(figsize=(8,6))
    sns.barplot(data=long, x='kappa', y='count', hue='syllable_type')
    plt.title('Number of Irrelevant vs Total Syllables')
    plt.ylim(0, None)
    plt.xlabel('Kappa')
    plt.ylabel('Number of Syllables')
    plt.legend(title='Syllable Type', labels=['Total', 'Irrelevant'])
    plt.savefig(os.path.join(output, 'number_underthreshold_vs_total_syllables.png'), dpi=300, bbox_inches='tight')
    plt.close()


    # 3. QUANTIFY NUMBER OF FRAMES WHICH ARENT WITHIN THE 10-90th PERCENTILE OF DURATION OF A SYLLABLE

    quantiles = (
        durations
        .groupby(['kappa', 'syllable'])['duration_frames']
        .quantile([0.10, 0.90])
        .unstack()  # index: (kappa, syllable), columns: 0.10, 0.90
        .reset_index()
        .rename(columns={0.10: 'q10', 0.90: 'q90'}))
    
    durations_with_quantiles = durations.merge(quantiles, on=['kappa', 'syllable'], how='left')

    # durations_with_quantiles['inside_quantiles_bool'] = ((durations_with_quantiles['duration_frames'] >= durations_with_quantiles['q10']) & (durations_with_quantiles['duration_frames'] <= durations_with_quantiles['q90']))
    # durations_with_quantiles['frames_outside'] = np.where(durations_with_quantiles['inside_quantiles_bool'], 0, durations_with_quantiles['duration_frames'])
    # durations_with_quantiles['frames_inside'] = np.where(durations_with_quantiles['inside_quantiles_bool'], durations_with_quantiles['duration_frames'],0)

    dur = durations_with_quantiles['duration_frames']
    q10 = durations_with_quantiles['q10']
    q90 = durations_with_quantiles['q90']

    # Replace these two lines ONLY:
    # durations_with_quantiles['frames_outside'] = np.where(...)
    # durations_with_quantiles['frames_inside'] = np.where(...)

    # New logic:
    durations_with_quantiles['frames_outside'] = np.where(
        dur < q10,                                   # below q10 → whole duration outside
        dur,
        np.where(dur > q90, dur - q90, 0)            # above q90 → only excess outside
    )

    durations_with_quantiles['frames_inside'] = dur - durations_with_quantiles['frames_outside']


    summary_quantiles = (
    durations_with_quantiles.groupby(['kappa', 'syllable'], as_index=False)
    .agg(
        frames_inside=('frames_inside', 'sum'),
        frames_outside=('frames_outside', 'sum')))
    

    summary_quantiles['fraction_outside_quantiles'] = summary_quantiles['frames_outside'] / (summary_quantiles['frames_inside'] + summary_quantiles['frames_outside']) * 100
    summary_quantiles.to_csv(os.path.join(output, 'frames_outside_10-90th_percentile_per_syllable.csv'), index=False)

    plt.figure(figsize=(8,6))
    sns.barplot(data=summary_quantiles, x='kappa', y='fraction_outside_quantiles', ci='sd')
    plt.title('Fraction of Frames Outside 10-90th Percentile of Syllable Duration')
    plt.ylim(0, None)
    plt.xlabel('Kappa')
    plt.ylabel('Fraction of Frames Outside 10-90th Percentile(%)')
    plt.savefig(os.path.join(output, 'fraction_frames_outside_10-90th_percentile.png'), dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8,6))
    sns.barplot(data=summary_quantiles, x='kappa', y='frames_outside', ci='sd')
    plt.title('Number of Frames Relating to Syllables which are outside the 10-90th Percentile of Syllable Duration')
    plt.ylim(0, None)
    plt.xlabel('Kappa')
    plt.ylabel('Number of Frames Outside 10-90th Percentile')
    plt.savefig(os.path.join(output, 'number_frames_outside_10-90th_percentile.png'), dpi=300, bbox_inches='tight')
    plt.close() 

    # 4. QUANTIFICATION OF WELL USED SYLLABLES -- FRAMES
      # 1) syllables that pass the MoSeq ‘relevance’ threshold'; 
      # 2) have durations that are inside a reasonable band for that syllable?:
      # 3) are not dominated by 1-frame occurrences

    
    # Identify relevant syllables from stats_df
    relevant = (df_stat[['kappa', 'syllable']]
        .drop_duplicates()
        .assign(is_relevant=True)) # boolean column
    
    # Merge with durations_with_quantiles
    durations_with_quantiles = durations_with_quantiles.merge(
        relevant,
        on=['kappa', 'syllable'],
        how='left')
    durations_with_quantiles['is_relevant'] = durations_with_quantiles['is_relevant'].fillna(False)

    # Exclude syllables dominated by 1-frame occurrences
    durations_with_quantiles['is_good_syllable'] = (durations_with_quantiles['is_relevant'] & (durations_with_quantiles['duration_frames'] >= 2)) # boolean column

    durations_with_quantiles['total_frames'] = (durations_with_quantiles['frames_inside'] + durations_with_quantiles['frames_outside']) # total frames per syllable occurrence

    data = []

    for kappa in sorted(durations_with_quantiles['kappa'].unique()):
        sub = durations_with_quantiles[durations_with_quantiles['kappa'] == kappa]

        # frames in relevant syllables AND within 10–90% band
        good_frames = sub.loc[sub['is_good_syllable'], 'frames_inside'].sum()

        # total frames across ALL syllables
        total_frames_all = sub['total_frames'].sum()

       # total frames in GOOD syllables (relevant & duration >= 2), inside + outside
        total_frames_relevant = sub.loc[sub['is_good_syllable'], 'total_frames'].sum()

        data.append({
            'kappa': int(kappa),
            'total_frames': int(total_frames_all),
            'frames_in_relevant_syllables': int(total_frames_relevant),
            'good_frames': int(good_frames),
            'percent_good_frames': 100 * (good_frames / total_frames_all if total_frames_all > 0 else 0.0),
            'percent_good_frames_of_relevant_syllables': 100 * (good_frames / total_frames_relevant if total_frames_relevant > 0 else 0.0)
        })

    good_frames_df = pd.DataFrame(data)
    good_frames_df.to_csv(os.path.join(output, 'good_frames_in_relevant_syllables.csv'), index=False)


    plt.figure(figsize=(8,6))
    sns.barplot(data=good_frames_df, x='kappa', y='percent_good_frames')
    plt.title('Percentage of Good Frames (Relevant Syllables, within 10-90th Percentile, >1 frame)')
    plt.ylim(0, None)
    plt.xlabel('Kappa')
    plt.ylabel('Percentage of Good Frames (%)')
    plt.savefig(os.path.join(output, 'percentage_good_frames_in_relevant_syllables.png'), dpi=300, bbox_inches='tight')
    plt.close() 






def translate_rotate_syllables(df):

    df = df.sort_values(['name', 'frame_index']).reset_index(drop=True) # kappa shd be included if multiple models

    def translate_bout(bout):
        # first frame of this bout
        first = bout.iloc[0]
        x = first['centroid_x']
        y = first['centroid_y']

        # subtract from all frames in this bout
        bout['rel_centroid_x'] = bout['centroid_x'] - x
        bout['rel_centroid_y'] = bout['centroid_y'] - y
        return bout

    df = df.groupby('bout_id', group_keys=False).apply(translate_bout)

    ## ROTATION: USE HEADING TO ROTATE ALL COORDINATES SO THAT THE ANIMAL IS FACING 'UP' (Y AXIS POSITIVE) 
    # +1.57 radians is up so want the first frame heading to be this

    def rotate_bout(bout):
        # first frame of this bout
        first = bout.iloc[0]
        h0 = first['heading']          # starting heading
        target = np.pi / 2             # you want all bouts to start "up" (≈ 1.57) # +1.57 radians

        delta = target - h0            # how much to rotate this bout by
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)

        x = bout['rel_centroid_x'].to_numpy()
        y = bout['rel_centroid_y'].to_numpy()

        # rotate all points in this bout
        # cos θ = horizontal part of a direction
        # sin θ = vertical part of a direction
#         cos θ = “how much stays in its original direction”

        # sin θ = “how much is rotated into the perpendicular direction”

        # x' =  x*cosθ   -  y*sinθ
        # y' =  x*sinθ   +  y*cosθ

        x_rot = x * cos_d - y * sin_d
        y_rot = x * sin_d + y * cos_d

        bout['rotated_centroid_x'] = x_rot
        bout['rotated_centroid_y'] = y_rot

        # optional: rotated heading, so they all start near +pi/2
        bout['heading_rotated'] = bout['heading'] + delta

        return bout

    df = df.groupby('bout_id', group_keys=False).apply(rotate_bout)

    return df



# df = pd.read_csv('/Users/cochral/Desktop/MOSEQ/KEYPOINT-KAPPA100000/moseq_df.csv')
# output = '/Users/cochral/Desktop/MOSEQ/KEYPOINT-KAPPA100000/syllable_trajectories'
# translate_rotate_syllables(df, output, plot=True)
    


def mean_syllable_trajectories_per_kappa(directory):

    dfs = []
    for folder in os.listdir(directory):
        if not folder.startswith('KEYPOINT'):
            continue 
        kappa = folder.split("KAPPA")[-1]
        path = os.path.join(directory, folder, 'moseq_df.csv')
        if not os.path.exists(path):
            print(f"Skipping {folder}: no moseq_df.csv found")
            continue
        df = pd.read_csv(path)
        df['kappa'] = int(kappa)
        dfs.append(df)

    output = os.path.join(directory, 'model_comparisons', 'syllable_mean_trajectories')
    if not os.path.exists(output):
        os.makedirs(output)

    df = pd.concat(dfs, ignore_index=True)
    df['bout_id'] = df['onset'].astype(int).cumsum()

    ## work through each kappa model separately
    for kappa, g_kappa in df.groupby('kappa'):

        translated_df = translate_rotate_syllables(g_kappa)

        syllable_output = os.path.join(output, f'kappa_{kappa}')
        if not os.path.exists(syllable_output):
            os.makedirs(syllable_output)

        # 1) add relative frame index within each bout
        translated_df = translated_df.sort_values(['bout_id', 'frame_index'])
        translated_df['rel_frame'] = (
            translated_df.groupby('bout_id').cumcount()
        )

        # 3) per syllable, compute median / mean rotated trajectory
        for syll, g_syl in translated_df.groupby('syllable'):

            # group by relative frame index and summarise across bouts
            summary = (g_syl.groupby('rel_frame').agg(
                    median_x=('rotated_centroid_x', 'median'),
                    median_y=('rotated_centroid_y', 'median'),

                    mean_x=('rotated_centroid_x', 'mean'),
                    mean_y=('rotated_centroid_y', 'mean'),

                    sd_x=('rotated_centroid_x', 'std'),
                    sd_y=('rotated_centroid_y', 'std'),

                    p10_x=('rotated_centroid_x', lambda x: np.percentile(x, 10)),
                    p90_x=('rotated_centroid_x', lambda x: np.percentile(x, 90)),
                    p25_x=('rotated_centroid_x', lambda x: np.percentile(x, 25)),
                    p75_x=('rotated_centroid_x', lambda x: np.percentile(x, 75)),

                    p10_y=('rotated_centroid_y', lambda y: np.percentile(y, 10)),
                    p90_y=('rotated_centroid_y', lambda y: np.percentile(y, 90)),
                    p25_y=('rotated_centroid_y', lambda y: np.percentile(y, 25)),
                    p75_y=('rotated_centroid_y', lambda y: np.percentile(y, 75)),

                    n=('bout_id', 'nunique')
                ).reset_index())


            # save canonical trajectory for this syllable
            out_csv = os.path.join(
                syllable_output, 
                f'syllable_{syll}_canonical_trajectory.csv'
            )
            summary.to_csv(out_csv, index=False)

               # ---------- MEDIAN PLOT ----------
            # ---------- MEDIAN PLOT WITH PERCENTILE RIBBON ----------
            plt.figure(figsize=(4, 4))

            # shaded percentile region
            plt.fill_between(
                summary['median_x'],   # X-path is the median_x path
                summary['p10_y'],      # lower percentile
                summary['p90_y'],      # upper percentile
                alpha=0.2,
                color="gray"
            )

            # median trajectory
            sns.lineplot(
                x=summary['median_x'],
                y=summary['median_y'],
                linewidth=2,
                color='black'
            )

            plt.scatter([0], [0], s=20, zorder=10) 
            plt.gca().set_aspect('equal', 'box')
            plt.title(f"Kappa {kappa} – Syllable {syll}\nMedian Trajectory + 10–90% band")
            plt.xlabel("X (rotated)")
            plt.ylabel("Y (rotated)")
            out_png = os.path.join(syllable_output, f"syllable_{syll}_median.png")
            plt.savefig(out_png, dpi=300, bbox_inches='tight')
            plt.close()


            # ---------- MEAN PLOT ----------
            # ---------- MEAN PLOT WITH STANDARD DEVIATION RIBBON ----------
            plt.figure(figsize=(4, 4))

            plt.fill_between(
                summary['mean_x'],
                summary['mean_y'] - summary['sd_y'],
                summary['mean_y'] + summary['sd_y'],
                alpha=0.2,
                color='gray'
            )

            sns.lineplot(
                x=summary['mean_x'],
                y=summary['mean_y'],
                linewidth=2,
                color='blue'
            )

            plt.scatter([0], [0], s=20, zorder=10)
            plt.gca().set_aspect('equal', 'box')
            plt.title(f"Kappa {kappa} – Syllable {syll}\nMean Trajectory ± 1 SD")
            plt.xlabel("X (rotated)")
            plt.ylabel("Y (rotated)")
            out_png = os.path.join(syllable_output, f"syllable_{syll}_mean.png")
            plt.savefig(out_png, dpi=300, bbox_inches='tight')
            plt.close()

            # ---------- MEAN PLOT WITH RAW TRAJECTORIES ----------
            plt.figure(figsize=(4, 4))

            # 1) draw all individual bouts in light grey
            for bout_id, g_bout in g_syl.groupby('bout_id'):
                plt.plot(
                    g_bout['rotated_centroid_x'],
                    g_bout['rotated_centroid_y'],
                    color='gray',
                    alpha=0.15,
                    linewidth=0.7
                )

            # 2) overlay the mean trajectory in a dark line
            plt.plot(
                summary['mean_x'],
                summary['mean_y'],
                color='blue',
                linewidth=2.0,
                label='mean trajectory'
            )

            # 3) mark start
            plt.scatter([0], [0], s=20, zorder=10, color='black')

            plt.gca().set_aspect('equal', 'box')
            plt.title(f"Kappa {kappa} – Syllable {syll}\nRaw bouts + mean trajectory")
            plt.xlabel("X (rotated)")
            plt.ylabel("Y (rotated)")
            # plt.legend()  # optional

            out_png = os.path.join(syllable_output, f"syllable_{syll}_mean_with_raw.png")
            plt.savefig(out_png, dpi=300, bbox_inches='tight')
            plt.close()
        # --------------------------------------------------------




# directory = '/Users/cochral/Desktop/MOSEQ'
# mean_syllable_trajectories_per_kappa(directory)
    

def syllable_feature_quantifications(df, directory):

    # df = translate_rotate_syllables(df, output=None, plot=False)

    df['bout_id'] = df['onset'].astype(int).cumsum()

    dfs = Parallel(n_jobs=-1)(delayed(translate_rotate_syllables)(g.copy())for _, g in df.groupby('name'))
    df = pd.concat(dfs, ignore_index=True)
    print("Translated and rotated syllables")

    syllable_output = os.path.join(directory, 'syllable_feature_quantifications')
    if not os.path.exists(syllable_output):
        os.makedirs(syllable_output)

    df = df.sort_values(['bout_id', 'frame_index'])
    df['relative_frame'] = (df.groupby('bout_id').cumcount())

    bout_lengths = df.groupby('bout_id').size()
    good_bouts = bout_lengths[bout_lengths >= 2].index  # length ≥ 2 frames
    df = df[df['bout_id'].isin(good_bouts)].copy()

    df['normalised_frame'] = df.groupby('bout_id')['relative_frame'].transform(lambda x: x / (x.max())) #check 


     # 3) per syllable, compute median / mean rotated trajectory
    for syll, grouped in df.groupby('syllable'):

        # group by relative frame index and summarise across bouts
        summary = (grouped.groupby('relative_frame').agg(
                median_x=('rotated_centroid_x', 'median'),
                median_y=('rotated_centroid_y', 'median'),

                mean_x=('rotated_centroid_x', 'mean'),
                mean_y=('rotated_centroid_y', 'mean'),

                sd_x=('rotated_centroid_x', 'std'),
                sd_y=('rotated_centroid_y', 'std'),

                p10_x=('rotated_centroid_x', lambda x: np.percentile(x, 10)),
                p90_x=('rotated_centroid_x', lambda x: np.percentile(x, 90)),
                p25_x=('rotated_centroid_x', lambda x: np.percentile(x, 25)),
                p75_x=('rotated_centroid_x', lambda x: np.percentile(x, 75)),

                p10_y=('rotated_centroid_y', lambda y: np.percentile(y, 10)),
                p90_y=('rotated_centroid_y', lambda y: np.percentile(y, 90)),
                p25_y=('rotated_centroid_y', lambda y: np.percentile(y, 25)),
                p75_y=('rotated_centroid_y', lambda y: np.percentile(y, 75)),

                n=('bout_id', 'nunique') # per frame number of unique bouts
            ).reset_index())
        
        # ---------- MEAN PLOT WITH RAW TRAJECTORIES ----------
        plt.figure(figsize=(4, 4))

        # 1) draw all individual bouts in light grey
        for bout_id, g_bout in grouped.groupby('bout_id'):
            plt.plot(
                g_bout['rotated_centroid_x'],
                g_bout['rotated_centroid_y'],
                color='gray',
                alpha=0.15,
                linewidth=0.7
            )
        
        total_bouts = grouped['bout_id'].nunique()
        threshold_bouts = 0.05 * total_bouts
        mean_trace = summary[summary['n'] >= threshold_bouts]

        # 2) overlay the mean trajectory in a dark line
        plt.plot(
            mean_trace['mean_x'],
            mean_trace['mean_y'],
            color='blue',
            linewidth=2.0,
            label='mean trajectory')

        # 3) mark start
        plt.scatter([0], [0], s=20, zorder=10, color='black')

        plt.gca().set_aspect('equal', 'box')
        plt.title(f"{syll}\nRaw bouts + mean trajectory")
        plt.xlabel("X (rotated)")
        plt.ylabel("Y (rotated)")
        # plt.legend()  # optional

        out_png = os.path.join(syllable_output, f"syllable_{syll}.png")
        plt.savefig(out_png, dpi=300, bbox_inches='tight')
        plt.close()

        # ---------- MEAN PLOT WITH 1 SD PLOTTED ----------
        plt.figure(figsize=(4, 4))

        plt.plot(
            mean_trace['mean_x'],
            mean_trace['mean_y'],
            color='blue',
            linewidth=2.0,
            label='mean trajectory')
        
        plt.fill_between(
            mean_trace['mean_x'],
            mean_trace['mean_y'] - mean_trace['sd_y'],
            mean_trace['mean_y'] + mean_trace['sd_y'],
            alpha=0.2,
            color='blue'
        )
        

        ax = plt.gca()               # <-- define it
        ax.set_aspect('equal', 'box')
        plt.title(f"{syll}\nMean trajectory ± 1 SD")
        plt.xlabel("X (rotated)")
        plt.ylabel("Y (rotated)")
        plt.xlim(-50,50)
        # plt.legend()  # optional

        out_png = os.path.join(syllable_output, f"syllable_{syll}_mean.png")
        plt.savefig(out_png, dpi=300, bbox_inches='tight')
        plt.close()

        # ---------- RAW TRACES WITHIN 1 SD ----------


        # 1) attach mean / sd per relative_frame back to each frame
        stats_for_merge = summary[['relative_frame', 'mean_x', 'sd_x', 'mean_y', 'sd_y']]
        grouped = grouped.merge(stats_for_merge, on='relative_frame', how='left')

        # 2) per-frame: is this point within 1 SD of the mean (in both x and y)?
        grouped['within_1sd'] = (
            grouped['rotated_centroid_x'].between(
                grouped['mean_x'] - grouped['sd_x'],
                grouped['mean_x'] + grouped['sd_x']
            )
            &
            grouped['rotated_centroid_y'].between(
                grouped['mean_y'] - grouped['sd_y'],
                grouped['mean_y'] + grouped['sd_y']
            )
        )

        # 3) keep only bouts where *all* frames are within 1 SD
        #    (change .all() to .mean() >= 0.9 etc if you want a tolerance)
        good_bouts_mask = grouped.groupby('bout_id')['within_1sd'].transform('all')
        grouped_good = grouped[good_bouts_mask].copy()

        # 4) now plot ONLY these "good" raw traces
        plt.figure(figsize=(4, 4))

        for bout_id, g_bout in grouped_good.groupby('bout_id'):
            plt.plot(
                g_bout['rotated_centroid_x'],
                g_bout['rotated_centroid_y'],
                color='gray',
                alpha=0.15,
                linewidth=0.7
            )

        # overlay mean as before
        total_bouts = grouped['bout_id'].nunique()
        threshold_bouts = 0.05 * total_bouts
        mean_trace = summary[summary['n'] >= threshold_bouts]

        plt.plot(
            mean_trace['mean_x'],
            mean_trace['mean_y'],
            color='blue',
            linewidth=2.0,
            label='mean trajectory'
        )

        plt.scatter([0], [0], s=20, zorder=10, color='black')
        plt.gca().set_aspect('equal', 'box')
        plt.title(f"{syll}\nRaw bouts within 1 SD + mean")
        plt.xlabel("X (rotated)")
        plt.ylabel("Y (rotated)")

        out_png = os.path.join(syllable_output, f"syllable_{syll}_within_1sd.png")
        plt.savefig(out_png, dpi=300, bbox_inches='tight')
        plt.close()







        # ---------- VELOCITY AND HEADING QUANTIFICATION ----------
        total_bouts = grouped['bout_id'].nunique()
        threshold_bouts = 0.05 * total_bouts
        grouped['n'] = grouped.groupby(['syllable', 'relative_frame'])['bout_id'].transform('nunique')
        mean_trace = grouped[grouped['n'] >= threshold_bouts]
 

        fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)

        hmin, hmax   = -2, 2    # heading range (radians)
        avmin, avmax = -5, 5          # angular velocity
        vmin, vmax   = 0, 450          # speed (px/s)

        sns.lineplot(
            data=mean_trace,
            x='relative_frame',
            y='velocity_px_s',
            ax=axes[0],
            color='green')

        axes[0].set_title(f"Velocity")
        axes[0].set_ylabel("Velocity")
        axes[0].set_ylabel("")
        axes[0].set_ylim(vmin,vmax)

        sns.lineplot(
            data=mean_trace,
            x='relative_frame',
            y='heading_rotated',
            ax=axes[1],
            color='orange')     
        axes[1].set_title(f"Headingxw")
        axes[1].set_ylabel("Heading")
        axes[1].set_xlabel("")
        axes[1].set_ylim(hmin,hmax)

        sns.lineplot(
            data=mean_trace,
            x='relative_frame',
            y='angular_velocity',
            ax=axes[2],
            color='red')

        axes[2].set_title("Angular Velocity")
        axes[2].set_xlabel("Relative Frame")
        axes[2].set_ylabel("Angular Velocity")
        axes[2].set_ylim(avmin,avmax)

        plt.tight_layout()
        out_png = os.path.join(syllable_output, f"quantification_{syll}.png")
        plt.savefig(out_png, dpi=300, bbox_inches='tight')
        plt.close()


        # ---------- VELOCITY AND HEADING QUANTIFICATION, NORMALISED FRAME----------
        n_bins = 20
        grouped['norm_bin'] = (grouped['normalised_frame'] * (n_bins - 1)).round().astype(int)
        grouped['n_norm'] = grouped.groupby('norm_bin')['bout_id'].transform('nunique')
        mean_trace_norm = grouped[grouped['n_norm'] >= threshold_bouts].copy()

        # turn bins back into 0–1 for plotting
        mean_trace_norm['norm_t'] = mean_trace_norm['norm_bin'] / (n_bins - 1)

        

        fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)

        hmin, hmax   = -3, 3    # heading range (radians)
        avmin, avmax = -5, 5          # angular velocity
        vmin, vmax   = 0, 800         # speed (px/s)

        sns.lineplot(
            data=mean_trace_norm,
            x='norm_t',
            y='velocity_px_s',
            ax=axes[0],
            color='green')

        axes[0].set_title(f"Velocity")
        axes[0].set_ylabel("Velocity")
        axes[0].set_xlabel("")
        axes[0].set_ylim(vmin,vmax)

        sns.lineplot(
            data=mean_trace_norm,
            x='norm_t',
            y='heading_rotated',
            ax=axes[1],
            color='orange')     
        axes[1].set_title(f"Headingxw")
        axes[1].set_ylabel("Heading")
        axes[1].set_xlabel("")
        axes[1].set_ylim(hmin,hmax)

        sns.lineplot(
            data=mean_trace_norm,
            x='norm_t',
            y='angular_velocity',
            ax=axes[2],
            color='red')

        axes[2].set_title("Angular Velocity")
        axes[2].set_xlabel("Normalised Frame")
        axes[2].set_ylabel("Angular Velocity")
        axes[2].set_ylim(avmin,avmax)

        plt.tight_layout()
        out_png = os.path.join(syllable_output, f"quantification_normalised_{syll}.png")
        plt.savefig(out_png, dpi=300, bbox_inches='tight')
        plt.close()



def ethogram_plotting(df,  video, directory):

        # Filter only the animals you want (e.g., 'N10-GH')
    df_filtered = df[df['name'] == video].copy()

    tracks = df_filtered['name'].unique()

    # Make a syllable-to-color map using viridis
    syllables = sorted(df_filtered['syllable'].unique())
    palette = sns.color_palette('viridis', n_colors=len(syllables))
    syl2color = {s: palette[i] for i, s in enumerate(syllables)}

    # Start plot
    fig, ax = plt.subplots(figsize=(12, len(tracks) * 0.4))

    for i, name in enumerate(tracks):
        sub = df_filtered[df_filtered['name'] == name].sort_values('frame_index')

        for _, row in sub.iterrows():
            x = row['frame_index']
            y = i  # vertical stack by animal
            color = syl2color[row['syllable']]
            ax.add_patch(plt.Rectangle((x, y), 1, 1, color=color, linewidth=0))

    # Formatting
    ax.set_yticks(np.arange(len(tracks)) + 0.5)
    ax.set_yticklabels(tracks)
    ax.set_xlabel("Time")
    ax.set_ylabel("Track")
    ax.set_title("Syllable Ethogram N10-GH")
    ax.set_xlim(df_filtered['frame_index'].min(), df_filtered['frame_index'].max())
    ax.set_ylim(0, len(tracks))

    handles = [mpatches.Patch(color=c, label=s) for s, c in syl2color.items()]
    plt.legend(handles=handles, title="Syllables", bbox_to_anchor=(1.01, 1), loc='upper left')

    plt.tight_layout()
    output = os.path.join(directory, f'ethorgram-{video}.png')
    plt.savefig(output, dpi=300, bbox_inches='tight')
    plt.show()



## OEIGINAL WAY TO PLOT- CHATGPT SUGGESTED A FASTER WAY BELOW
# def analysis_main(df, output, ethogram=True):

#     df['group'] = df['name'].str.split('_').str[0]

#     if ethogram:

#         groups = sorted(df['group'].unique())

#         for group in groups:
#             df_filtered = df[df['group'] == group].copy()

#             tracks = df_filtered['name'].unique()

#             # Make a syllable-to-color map using viridis
#             syllables = sorted(df_filtered['syllable'].unique())
#             palette = sns.color_palette('viridis', n_colors=len(syllables))
#             syl2color = {s: palette[i] for i, s in enumerate(syllables)}

#             # Start plot
#             fig, ax = plt.subplots(figsize=(12, len(tracks) * 0.4))

#             for i, name in enumerate(tracks):
#                 sub = df_filtered[df_filtered['name'] == name].sort_values('frame_index')

#                 for _, row in sub.iterrows():
#                     x = row['frame_index']
#                     y = i  # vertical stack by animal
#                     color = syl2color[row['syllable']]
#                     ax.add_patch(plt.Rectangle((x, y), 1, 1, color=color, linewidth=0))

#             # Formatting
#             ax.set_yticks(np.arange(len(tracks)) + 0.5)
#             ax.set_yticklabels(tracks)
#             ax.set_xlabel("Time (frame index)")
#             ax.set_ylabel("Track")
#             ax.set_title(f"Syllable Ethogram {group}")

#             ax.set_xlim(df_filtered['frame_index'].min(), df_filtered['frame_index'].max())
#             ax.set_ylim(0, len(tracks))

#             handles = [mpatches.Patch(color=c, label=s) for s, c in syl2color.items()]
#             plt.legend(handles=handles, title="Syllables", bbox_to_anchor=(1.01, 1), loc='upper left')

#             plt.tight_layout()
#             outfile = os.path.join(output, f"ethogram-{group.lower()}.png")
#             plt.savefig(outfile, dpi=300, bbox_inches='tight')
#             plt.close(fig)
    
        




def analysis_main(df, stats_df, output, ethogram=True):

    def plot_ethogram_fast(df_group, group_name):
        # Convert track names to row indices
        tracks = sorted(df_group['name'].unique())
        track_to_row = {t: i for i, t in enumerate(tracks)}

        # Frame range
        min_f = df_group['frame_index'].min()
        max_f = df_group['frame_index'].max()

        width = max_f - min_f + 1

        # Create matrix: rows = tracks, columns = frames
        # mat = np.full((len(tracks), width), fill_value=-1)   # -1 means "no syllable"
        mat = np.full((len(tracks), width), fill_value=None, dtype=object)


        for _, row in df_group.iterrows():
            r = track_to_row[row['name']]
            c = row['frame_index'] - min_f
            # mat[r, c] = row['syllable']
            mat[r, c] = row['syllable_group']


        # Get syllables and colors
    # --- Make your exact color mapping ---
        # syllables = sorted(df_group['syllable'].unique())
        # palette = sns.color_palette('viridis', n_colors=len(syllables))
        # syl2color = {s: palette[i] for i, s in enumerate(syllables)}

        # # convert syllables → integer index image
        # idx_map = {s: i for i, s in enumerate(syllables)}
        # mat_idx = np.full_like(mat, -1)
        # for s, idx in idx_map.items():
        #     mat_idx[mat == s] = idx

        # # colormap = your palette + white for missing
        # cmap = ListedColormap(palette + [(1,1,1)])
        # mat_idx[mat_idx == -1] = len(palette)

        group_names = list(syllable_groups.keys())
        idx_map = {g: i for i, g in enumerate(group_names)}

        mat_idx = np.full(mat.shape, -1, dtype=int)
        for g, idx in idx_map.items():
            mat_idx[mat == g] = idx

        palette = [group_colors[g] for g in group_names]
        cmap = ListedColormap(palette + [(1, 1, 1)])
        mat_idx[mat_idx == -1] = len(palette)

        syl2color = {g: group_colors[g] for g in group_names}


        # ---- PLOT ----
        plt.figure(figsize=(12, len(tracks)*0.4))
        plt.imshow(mat_idx, aspect='auto', cmap=cmap, interpolation='nearest')


        plt.yticks(np.arange(len(tracks)), tracks)
        plt.xlabel("Frame index")
        plt.ylabel("Track")
        plt.title(f"Syllable Ethogram {group_name}")

   
        handles = [mpatches.Patch(color=c, label=s) for s, c in syl2color.items()]
        plt.legend(handles=handles, title="Syllables",
                bbox_to_anchor=(1.01, 1), loc='upper left')

        plt.tight_layout()
        plt.savefig(os.path.join(output, f'{group_name}-ethogram.png'), dpi=300, bbox_inches='tight')
        plt.close()

    ## unique bouts
    df = df.sort_values(['name', 'frame_index'])
    df['bout_num'] = df.groupby('name')['onset'].cumsum()
    df['bout_id'] = df.groupby(['name', 'bout_num']).ngroup()
    # df['bout_id'] = df['onset'].astype(int).cumsum()

    ## filter to only good bouts
    bout_lengths = df.groupby('bout_id').size()
    good_bouts = bout_lengths[bout_lengths >= 2].index  # length ≥ 2 frames
    df = df[df['bout_id'].isin(good_bouts)].copy()

    ## filter for good syllables
    good_syllables = stats_df['syllable'].unique()
    df = df[df['syllable'].isin(good_syllables)]


    df['syllable_group'] = df['syllable'].map(syll_to_group)
    df = df.dropna(subset=['syllable_group']).copy()

        # unique groups
    df['group'] = df['name'].str.split('_').str[0]
    df = df[df['frame_index'] <= 3600 ]  # ensure no negative frame indices


    # ============================================================
    # GROUPED SYLLABLE FREQUENCIES (IN PARALLEL TO RAW ONES)
    # ============================================================

    # build grouped frequency table
    grouped_frequencies = (
        df.groupby(['name', 'group', 'syllable_group'])
        .size()
        .reset_index(name='frequency')
    )


    # ensure consistent x-axis ordering
    group_order = list(syllable_groups.keys())
    grouped_frequencies['syllable_group'] = pd.Categorical(
        grouped_frequencies['syllable_group'],
        categories=group_order,
        ordered=True
    )

    def plot_grouped_freq(filter_mask, outname, title):
        sub = grouped_frequencies[filter_mask].copy()
        if sub.empty:
            print(f"[skip] {outname} (no data)")
            return

        plt.figure(figsize=(10, 6))
        sns.pointplot(
            data=sub,
            x='syllable_group',
            y='frequency',
            hue='group',
            order=group_order
        )
        plt.title(title)
        plt.xlabel("Syllable group")
        plt.ylabel("Frequency")
        plt.legend(title="Group")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output, outname), dpi=300, bbox_inches='tight')
        plt.close()

    # ---- GH1 vs SI1 ----
    # ---- GH1 vs SI1 (N1) ----
    plot_grouped_freq(
        grouped_frequencies['group'].str.match(r'^N1-(GH|SI)', na=False),
        'grouped_syllable_frequencies_GH1_vs_SI1.png',
        'Grouped syllable frequencies: GH1 vs SI1'
    )

    plot_grouped_freq(
    grouped_frequencies['group'].str.match(r'^N2-(GH|SI)', na=False),
    'grouped_syllable_frequencies_GH2_vs_SI2.png',
    'Grouped syllable frequencies: GH2 vs SI2'
)



    plot_grouped_freq(
        grouped_frequencies['group'].str.match(r'^N10-(GH|SI)', na=False),
        'grouped_syllable_frequencies_GH10_vs_SI10.png',
        'Grouped syllable frequencies: GH10 vs SI10'
    )



    # ---- ALL GH ----
    plot_grouped_freq(
        grouped_frequencies['group'].str.contains('GH', na=False),
        'grouped_syllable_frequencies_all_GH.png',
        'Grouped syllable frequencies: all GH'
    )

    # ---- ALL SI ----
    plot_grouped_freq(
        grouped_frequencies['group'].str.contains('SI', na=False),
        'grouped_syllable_frequencies_all_SI.png',
        'Grouped syllable frequencies: all SI'
    )

    # ============================================================













    syllable_frequencies = (df.groupby(['group', 'syllable'])
                            .size()
                            .unstack(fill_value=0)
                            .stack()
                            .reset_index(name='frequency'))
    


    plt.figure(figsize=(10,6))
    sns.pointplot(
        data=syllable_frequencies[syllable_frequencies['group'].str.startswith('N10')],
        x='syllable',
        y='frequency',
        hue='group'
    )

    plt.title('Syllable Frequencies by Group')
    plt.xlabel('Syllable')
    plt.ylabel('Frequency')
    plt.legend(title='Group')
    plt.savefig(os.path.join(output, 'syllable_frequencies_n10.png'), dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10,6))
    sns.pointplot(
        data=syllable_frequencies[syllable_frequencies['group'].str.startswith('N2')],
        x='syllable',
        y='frequency',
        hue='group'
    )

    plt.title('Syllable Frequencies by Group')
    plt.xlabel('Syllable')
    plt.ylabel('Frequency')
    plt.legend(title='Group')
    plt.savefig(os.path.join(output, 'syllable_frequencies_n2.png'), dpi=300, bbox_inches='tight')
    plt.close()


    plt.figure(figsize=(10,6))
    sns.pointplot(
        data=syllable_frequencies[syllable_frequencies['group'].str.match(r'^N1-')],
        x='syllable',
        y='frequency',
        hue='group'
    )

    plt.title('Syllable Frequencies by Group')
    plt.xlabel('Syllable')
    plt.ylabel('Frequency')
    plt.legend(title='Group')
    plt.savefig(os.path.join(output, 'syllable_frequencies_n1.png'), dpi=300, bbox_inches='tight')
    plt.close()


    plt.figure(figsize=(10,6))
    sns.pointplot(
        data=syllable_frequencies[syllable_frequencies['group'].str.contains('GH')],
        x='syllable',
        y='frequency',
        hue='group'
    )

    plt.title('Syllable Frequencies by Group')
    plt.xlabel('Syllable')
    plt.ylabel('Frequency')
    plt.legend(title='Group')
    plt.savefig(os.path.join(output, 'syllable_frequencies_gh.png'), dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10,6))
    sns.pointplot(
        data=syllable_frequencies[syllable_frequencies['group'].str.contains('SI')],
        x='syllable',
        y='frequency',
        hue='group'
    )

    plt.title('Syllable Frequencies by Group')
    plt.xlabel('Syllable')
    plt.ylabel('Frequency')
    plt.legend(title='Group')
    plt.savefig(os.path.join(output, 'syllable_frequencies_si.png'), dpi=300, bbox_inches='tight')
    plt.close()




    if ethogram:
        groups = sorted(df['group'].unique())
        for g in groups:
            df_group = df[df['group'] == g].copy()
            plot_ethogram_fast(df_group, g)


    ## proximity to other larvae

    def extract_track(name):
        match = re.search(r'track(\d+)', name)
        if match:
            return int(match.group(1))
        else:
            return 0
    

    def extract_video_id(name):
        match = re.search(r'_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_td\d+)', name)
        if match:
            return match.group(1)
        else:
            return None  # or "" if you prefer

    df['video_id'] = df['name'].apply(extract_video_id)
    df['track'] = df['name'].apply(extract_track)


    proximity_list = []

        # 1) loop over videos
    for video_id, df_v in df.groupby('video_id'):
        df_v = df_v.sort_values('frame_index')

        # 2) all onset events in this video
        df_onsets = df_v[df_v['onset']]

        for onset_row in df_onsets.itertuples():
            group = onset_row.group
            frame = onset_row.frame_index
            track = onset_row.track
            syll = onset_row.syllable
            x = onset_row.centroid_x
            y = onset_row.centroid_y

            # 4) other larvae in the SAME frame, different track
            others = df_v[(df_v['frame_index'] == frame) & (df_v['track'] != track)]

            # loop over others with itertuples as well
            for other in others.itertuples():
                d = np.hypot(other.centroid_x - x, other.centroid_y - y)

                proximity_list.append({
                    'video_id': video_id,
                    'group': group,
                    'frame_index': frame,
                    'track': track,
                    'syllable': syll,
                    'track_other': other.track,
                    'syllable_other': other.syllable,
                    'distance': d,
                })


    proximal_df = pd.DataFrame(proximity_list)
    proximal_df.to_csv(os.path.join(output, 'syllable_proximity_analysis.csv'), index=False)

    onset_summary = (proximal_df
    .groupby(['group', 'video_id', 'frame_index', 'track'])
    .agg(
        syllable=('syllable', 'first'),     # same for all rows in the onset
        mean_distance=('distance', 'mean'),
        min_distance=('distance', 'min')
    )
    .reset_index())

    onset_summary.to_csv(os.path.join(output, 'syllable_proximity_onset_summary.csv'), index=False)


    onset_summary = onset_summary[onset_summary['group'].str.contains('N10')]

    plt.figure(figsize=(10,6))
    sns.pointplot(data=onset_summary, x='syllable', y='mean_distance')
    plt.title('Mean Distance to Other Larvae at Syllable Onset')
    plt.xlabel('Syllable')
    plt.ylabel('Mean Distance (pixels)')
    plt.savefig(os.path.join(output, 'syllable_mean_proximity.png'), dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10,6))
    sns.pointplot(data=onset_summary, x='syllable', y='min_distance')
    plt.title('Min Distance to Other Larvae at Syllable Onset')
    plt.xlabel('Syllable')
    plt.ylabel('Min Distance (pixels)')
    plt.savefig(os.path.join(output, 'syllable_min_proximity.png'), dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10,6))
    sns.pointplot(data=onset_summary, x='syllable', y='mean_distance', hue='group')
    plt.title('Mean Distance to Other Larvae at Syllable Onset')
    plt.xlabel('Syllable')
    plt.ylabel('Mean Distance (pixels)')
    plt.savefig(os.path.join(output, 'syllable_mean_proximity_per_group.png'), dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10,6))
    sns.pointplot(data=onset_summary, x='syllable', y='min_distance', hue='group')
    plt.title('Min Distance to Other Larvae at Syllable Onset')
    plt.xlabel('Syllable')
    plt.ylabel('Min Distance (pixels)')
    plt.savefig(os.path.join(output, 'syllable_min_proximity_per_group.png'), dpi=300, bbox_inches='tight')
    plt.close()


    onsets = df[df['onset']]
    for id, syllable in onsets.groupby('syllable'):
        plt.figure(figsize=(10,6))
        sns.scatterplot(data=syllable, x='centroid_x', y='centroid_y', alpha=0.3)
        plt.title(f'Syllable {id} Onset Times')
        plt.xlabel('Centroid X')
        plt.ylabel('Centroid Y')
        plt.savefig(os.path.join(output, f'spatial_plot_{id}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    


    syllable_ids = sorted(onsets['syllable'].unique())
    n = len(syllable_ids)

    ncols = 10
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*3, nrows*3))
    axes = axes.flatten()

    for i, sid in enumerate(syllable_ids):
        ax = axes[i]
        sub = onsets[onsets['syllable'] == sid]
        ax.scatter(sub['centroid_x'], sub['centroid_y'], s=2, alpha=0.3)
        ax.set_title(f"Syll {sid}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    # turn off any empty panels at the end
    for j in range(i+1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output, "all_syllables_spatial_grid.png"), dpi=300, bbox_inches="tight")
    plt.close()






# df = pd.read_csv('/Users/cochral/Desktop/MOSEQ/KEYPOINT-KAPPA1500/moseq_df.csv')
# stats_df = pd.read_csv('/Users/cochral/Desktop/MOSEQ/KEYPOINT-KAPPA1500/stats_df.csv')
# output = '/Users/cochral/Desktop/MOSEQ/KEYPOINT-KAPPA1500/plots/ethograms'
# if not os.path.exists(output):
#     os.makedirs(output)
# analysis_main(df, stats_df, output)




def interaction_syllable(interactions, cluster, moseq, stat, output):

    interactions = pd.merge(
            interactions, 
            cluster[['interaction_id', 'Yhat.idt.pca']], 
            on='interaction_id', 
            how='inner')
     
    print("Merged interactions with cluster data")

    interactions['video_id'] = interactions['file'].str.replace('.mp4', '', regex=False)

    # quick dirty method- just keep video, frame and track involved in interaction 

    keep = ['file', 'video_id', 'Frame', 'Interaction Number', 'Normalized Frame', 'Interaction Pair', 'Yhat.idt.pca']

    interactions['Interaction Pair'] = interactions['Interaction Pair'].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x)


    interaction_tracks = (
        interactions[keep]
        .assign(track=interactions['Interaction Pair'])
        .explode('track')
        .drop(columns='Interaction Pair')
        .reset_index(drop=True))
    
    print("Extracted interaction tracks")



    moseq['bout_id'] = moseq['onset'].astype(int).cumsum()
    bout_lengths = moseq.groupby('bout_id').size()
    good_bouts = bout_lengths[bout_lengths >= 2].index  # length ≥ 2 frames
    moseq = moseq[moseq['bout_id'].isin(good_bouts)].copy()


    good_syllables = stat['syllable'].unique()
    moseq = moseq[moseq['syllable'].isin(good_syllables)]


    moseq['group'] = moseq['name'].str.split('_').str[0]
    moseq = moseq[moseq['frame_index'] <= 3600 ]  # ensure no negative frame indices

    def extract_track(name):
        match = re.search(r'track(\d+)', name)
        if match:
            return int(match.group(1))
        else:
            return 0
    

    def extract_video_id(name):
        match = re.search(r'_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_td\d+)', name)
        if match:
            return match.group(1)
        else:
            return None  # or "" if you prefer

    moseq['video_id'] = moseq['name'].apply(extract_video_id)
    moseq['track'] = moseq['name'].apply(extract_track)
    moseq = moseq.rename(columns={'frame_index': 'Frame'})


    interactions_with_syllables = interaction_tracks.merge(moseq[['video_id', 'Frame', 'track', 'syllable', 'onset']], on=['video_id', 'Frame', 'track'], how='left')
    interactions_with_syllables['syllable'] = interactions_with_syllables['syllable'].astype('Int64')  # capital I
    interactions_with_syllables = interactions_with_syllables.rename(columns={'Yhat.idt.pca': 'cluster'})
    interactions_with_syllables['Normalized Frame'] = interactions_with_syllables['Normalized Frame'].astype(int)
    interactions_with_syllables['unique_track_id'] = interactions_with_syllables['video_id'] + '_track' + interactions_with_syllables['track'].astype(str)
    interactions_with_syllables['period'] = np.where(interactions_with_syllables['Normalized Frame'] < 0, 'pre', 'post')


    print("Merged interactions with MOSEQ syllable data")
    interactions_with_syllables.to_csv(os.path.join(output, 'interactions_with_syllables.csv'), index=False)
    print(interactions_with_syllables.head())



    def plot_ethogram_fast(group, group_name):

        # Convert track names to row indices
        tracks = sorted(group['unique_track_id'].unique())
        track_to_row = {t: i for i, t in enumerate(tracks)}

        # Frame range
        min_f = group['Normalized Frame'].min()
        max_f = group['Normalized Frame'].max()

        width = max_f - min_f + 1

        # Create matrix: rows = tracks, columns = frames
        mat = np.full((len(tracks), width), fill_value=-1)   # -1 means "no syllable"

        for _, row in group.iterrows():
            r = track_to_row[row['unique_track_id']]
            c = row['Normalized Frame'] - min_f
            mat[r, c] = row['syllable']

        # Get syllables and colors
    # --- Make your exact color mapping ---
        syllables = sorted(group['syllable'].unique())
        palette = sns.color_palette('viridis', n_colors=len(syllables))
        syl2color = {s: palette[i] for i, s in enumerate(syllables)}

        # convert syllables → integer index image
        idx_map = {s: i for i, s in enumerate(syllables)}
        mat_idx = np.full_like(mat, -1)
        for s, idx in idx_map.items():
            mat_idx[mat == s] = idx

        # colormap = your palette + white for missing
        cmap = ListedColormap(palette + [(1,1,1)])
        mat_idx[mat_idx == -1] = len(palette)

        # ---- PLOT ----
        plt.figure(figsize=(12, len(tracks)*0.4))
        plt.imshow(mat_idx, aspect='auto', cmap=cmap, interpolation='nearest')


        plt.yticks(np.arange(len(tracks)), tracks)
        plt.xlabel("Frame index")
        plt.ylabel("Track")
        plt.title(f"Syllable Ethogram {group_name}")

   
        handles = [mpatches.Patch(color=c, label=s) for s, c in syl2color.items()]
        plt.legend(handles=handles, title="Syllables",
                bbox_to_anchor=(1.01, 1), loc='upper left')

        plt.tight_layout()
        plt.savefig(os.path.join(output, f'{group_name}-ethogram.png'), dpi=300, bbox_inches='tight')
        plt.close()
    

    groups = sorted(interactions_with_syllables['cluster'].unique())
    for group_id in groups:
        df_group = interactions_with_syllables[interactions_with_syllables['cluster'] == group_id].copy()
        df_group = df_group.dropna(subset=['syllable'])  # after merge certain rows dont have syllables
        plot_ethogram_fast(df_group, group_id)


    onsets = interactions_with_syllables[interactions_with_syllables['onset'] == True]

    freq = (
    onsets
    .groupby(['syllable', 'cluster'])
    .size()
    .reset_index(name='frequency'))

    cluster_totals = (
    onsets
    .groupby('cluster')
    .size()
    .reset_index(name='total')
)
    freq = freq.merge(cluster_totals, on='cluster')
    freq['relative_frequency'] = freq['frequency'] / freq['total']   # fraction
    # or percent:
    freq['relative_frequency_percent'] = freq['relative_frequency'] * 100


    syllable_ids = sorted(onsets['syllable'].unique())
    n = len(syllable_ids)

    ncols = 10
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*3, nrows*3))
    axes = axes.flatten()

    for i, sid in enumerate(syllable_ids):
        ax = axes[i]
        sub = freq[freq['syllable'] == sid]
        ax.bar(sub['cluster'], sub['relative_frequency'])
        ax.set_title(f"Syll {sid}", fontsize=8)
        # ax.set_xticks([])
        # ax.set_yticks([])
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Frequency")

    # turn off any empty panels at the end
    for j in range(i+1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output, "syllable_relative_frequency.png"), dpi=300, bbox_inches="tight")
    plt.close()


    freq = (
        onsets
        .groupby(['syllable', 'cluster', 'period'])
        .size()
        .reset_index(name='count')
    )

    # 4) relative frequency "within each syllable + period"
    #    (so for a given syllable, pre bars across clusters sum to 1; same for post)
    totals = (
        freq
        .groupby(['syllable', 'period'])['count']
        .sum()
        .reset_index(name='total')
    )

    freq = freq.merge(totals, on=['syllable', 'period'])
    freq['rel_freq'] = freq['count'] / freq['total']


    # ---------- PLOT: one subplot per syllable, pre vs post bars per cluster ----------
    syllable_ids = sorted(freq['syllable'].dropna().unique())
    clusters_sorted = sorted(freq['cluster'].dropna().unique())

    n = len(syllable_ids)
    ncols = 10
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*3, nrows*3))
    axes = axes.flatten()

    for i, sid in enumerate(syllable_ids):
        ax = axes[i]
        sub = freq[freq['syllable'] == sid]

        pre = sub[sub['period'] == 'pre'].set_index('cluster').reindex(clusters_sorted, fill_value=0)
        post = sub[sub['period'] == 'post'].set_index('cluster').reindex(clusters_sorted, fill_value=0)

        x = np.arange(len(clusters_sorted))
        w = 0.4

        ax.bar(x - w/2, pre['rel_freq'], width=w, label='pre')
        ax.bar(x + w/2, post['rel_freq'], width=w, label='post')

        ax.set_title(f"Syll {sid}", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(clusters_sorted, rotation=90, fontsize=6)
        ax.set_yticks([])
        ax.set_ylim(0, max(pre['rel_freq'].max(), post['rel_freq'].max(), 1e-6) * 1.1)

    # turn off empty panels
    for j in range(i+1, len(axes)):
        axes[j].axis("off")

    # one legend for whole figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(output, "syllable_pre_post_per_cluster.png"),
                dpi=300, bbox_inches="tight")
    plt.close()







def anchor_partner(df):


        # make sure Interaction Pair is a real tuple
    df['Interaction Pair'] = df['Interaction Pair'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    # because (left,right) = (Track_1_id, Track_2_id)
    df['track1_id'] = df['Interaction Pair'].str[0]
    df['track2_id'] = df['Interaction Pair'].str[1]

    # prep columns
    df['anchor_track_id'] = np.nan
    df['partner_track_id'] = np.nan

    ## Returns a straightness score 
    def compute_pca_axis(points):
        pca = PCA(n_components=2).fit(points)
        axis = pca.components_[0]
        score = pca.explained_variance_ratio_[0]
        # ensure the axis points upward
        return (axis if axis[1] >= 0 else -axis), score
    
    ## == Align the tracks (anchor 0,0) and rotate partner accordingly (on the right)
    def align_and_flip(track, anchor_axis, anchor_start):
        X = track - anchor_start
        phi = np.arctan2(anchor_axis[1], anchor_axis[0])  # angle of axis
        alpha = np.pi/2 - phi                            # rotate to +y
        R = np.array([[np.cos(alpha), -np.sin(alpha)],
                    [np.sin(alpha),  np.cos(alpha)]])
        X_rot = X.dot(R.T)
        return X_rot
    
    df['anchor x_body'] = np.nan
    df['anchor y_body'] = np.nan
    df['partner x_body'] = np.nan
    df['partner y_body'] = np.nan

    ## == Generate the anchor and partner x,y coordinates for future

    for interaction_id, group in df.groupby('interaction_id'):
        group = group.sort_values('Frame')
        coords1 = group[['Track_1 x_body','Track_1 y_body']].values
        coords2 = group[['Track_2 x_body','Track_2 y_body']].values
        if len(coords1) < 2 or len(coords2) < 2:
            continue
        # Compute PCA axes & scores
        axis1, s1 = compute_pca_axis(coords1)
        axis2, s2 = compute_pca_axis(coords2)
        # Choose anchor and partner
        if s1 >= s2:
            winner = 1
            anchor_pts, partner_pts, anchor_axis = coords1, coords2, axis1
        else:
            winner = 2
            anchor_pts, partner_pts, anchor_axis = coords2, coords1, axis2

        # Align both
        start = anchor_pts[0]
        A_al = align_and_flip(anchor_pts, anchor_axis, start)
        B_al = align_and_flip(partner_pts, anchor_axis, start)

        # --- NEW: align head/tail using the SAME reference (before flips) ---
        h1 = group[['Track_1 x_head','Track_1 y_head']].dropna().values
        t1 = group[['Track_1 x_tail','Track_1 y_tail']].dropna().values
        h2 = group[['Track_2 x_head','Track_2 y_head']].dropna().values
        t2 = group[['Track_2 x_tail','Track_2 y_tail']].dropna().values

        A_head = align_and_flip(h1 if winner == 1 else h2, anchor_axis, start) if (len(h1) or len(h2)) else np.empty((0,2))
        A_tail = align_and_flip(t1 if winner == 1 else t2, anchor_axis, start) if (len(t1) or len(t2)) else np.empty((0,2))
        B_head = align_and_flip(h2 if winner == 1 else h1, anchor_axis, start) if (len(h1) or len(h2)) else np.empty((0,2))
        B_tail = align_and_flip(t2 if winner == 1 else t1, anchor_axis, start) if (len(t1) or len(t2)) else np.empty((0,2))
    # --------------------------------------------------------------------

        # Horizontal flip if partner is left
        # if np.median(B_al[:,0]) < 0:
        #     A_al[:,0] *= -1
        #     B_al[:,0] *= -1

        # Horizontal flip if partner starts on the left
        if B_al[0, 0] < 0:
            A_al[:, 0] *= -1
            B_al[:, 0] *= -1
            # --- NEW: apply same horizontal flip to head/tail
            if A_head.size: A_head[:, 0] *= -1
            if A_tail.size: A_tail[:, 0] *= -1
            if B_head.size: B_head[:, 0] *= -1
            if B_tail.size: B_tail[:, 0] *= -1

        # Vertical flip if anchor is predominantly down
        if np.mean(A_al[:,1]) < 0:
            A_al[:,1] *= -1
            B_al[:,1] *= -1
                # --- NEW: apply same vertical flip to head/tail
            if A_head.size: A_head[:, 1] *= -1
            if A_tail.size: A_tail[:, 1] *= -1
            if B_head.size: B_head[:, 1] *= -1
            if B_tail.size: B_tail[:, 1] *= -1


        # Assign back to DataFrame
        # idx = group.index[:len(A_al)]

        idx = group.index  # safer- why 


        df.loc[idx, ['anchor x_body','anchor y_body']]  = A_al
        df.loc[idx, ['partner x_body','partner y_body']] = B_al# Initialize aligned columns

        # --- NEW: write aligned head/tail back (each uses its own length) ---
        if A_head.size:
            df.loc[group.index[:len(A_head)], ['anchor x_head','anchor y_head']] = A_head
        if A_tail.size:
            df.loc[group.index[:len(A_tail)], ['anchor x_tail','anchor y_tail']] = A_tail
        if B_head.size:
            df.loc[group.index[:len(B_head)], ['partner x_head','partner y_head']] = B_head
        if B_tail.size:
            df.loc[group.index[:len(B_tail)], ['partner x_tail','partner y_tail']] = B_tail
        # --------------------------------------------------------------------

        # → tag which original track was anchor (1 or 2)
        df.loc[idx, 'anchor_track']  = winner
        df.loc[idx, 'partner_track'] = 3 - winner

        # --- THIS is where the anchor/partner REAL ids go ---
        anchor_id  = group['track1_id'].iloc[0] if winner == 1 else group['track2_id'].iloc[0]
        partner_id = group['track2_id'].iloc[0] if winner == 1 else group['track1_id'].iloc[0]

        df.loc[idx, 'anchor_track_id']  = anchor_id
        df.loc[idx, 'partner_track_id'] = partner_id






    # === HEADING ANGLE CHANGE ===
    df['track1_heading_angle_change'] = df.groupby("interaction_id")["track1_angle"].diff().abs()
    df['track2_heading_angle_change'] = df.groupby("interaction_id")["track2_angle"].diff().abs()

    # === APPROACH ANGLE CHANGE ===
    df['track1_approach_angle_change'] = df.groupby("interaction_id")["track1_approach_angle"].diff().abs()
    df['track2_approach_angle_change'] = df.groupby("interaction_id")["track2_approach_angle"].diff().abs()

    metrics = [
    'speed',
    'acceleration',
    'angle',
    'approach_angle']

    for m in metrics:
        t1 = df[f'track1_{m}']
        t2 = df[f'track2_{m}']
        df[f'anchor_{m}']  = np.where(df['anchor_track']==1, t1, t2)
        df[f'partner_{m}'] = np.where(df['anchor_track']==1, t2, t1)

        # === Assign anchor/partner versions
        df['anchor_heading_angle_change']  = np.where(df['anchor_track'] == 1, df['track1_heading_angle_change'], df['track2_heading_angle_change'])
        df['partner_heading_angle_change'] = np.where(df['anchor_track'] == 1, df['track2_heading_angle_change'], df['track1_heading_angle_change'])

        df['anchor_approach_angle_change']  = np.where(df['anchor_track'] == 1, df['track1_approach_angle_change'], df['track2_approach_angle_change'])
        df['partner_approach_angle_change'] = np.where(df['anchor_track'] == 1, df['track2_approach_angle_change'], df['track1_approach_angle_change'])

    
    return df





def interaction_syllable_partner(interactions, cluster, moseq, stat, output):

    interactions = pd.merge(
            interactions, 
            cluster[['interaction_id', 'Yhat.idt.pca']], 
            on='interaction_id', 
            how='inner')
     
    print("Merged interactions with cluster data")

    interactions = anchor_partner(interactions)
    print("Computed anchor and partner tracks")
    interactions.to_csv(os.path.join(output, 'interactions_with_anchor_partner.csv'), index=False)


    interactions['video_id'] = interactions['file'].str.replace('.mp4', '', regex=False)

    # quick dirty method- just keep video, frame and track involved in interaction 

    # keep = ['file', 'video_id', 'Frame', 'Interaction Number', 'Normalized Frame', 'Interaction Pair', 'Yhat.idt.pca']

    # interactions['Interaction Pair'] = interactions['Interaction Pair'].apply(
    # lambda x: ast.literal_eval(x) if isinstance(x, str) else x)


    # interaction_tracks = (
    #     interactions[keep]
    #     .assign(track=interactions['Interaction Pair'])
    #     .explode('track')
    #     .drop(columns='Interaction Pair')
    #     .reset_index(drop=True))
    

    # print("Extracted interaction tracks")

    # columns you want to keep the same for both roles
# 1) columns you want to keep the same for both
    base_cols = ['file', 'video_id', 'Frame', 'Interaction Number',
                'Normalized Frame', 'Yhat.idt.pca']

    # 2) make ANCHOR df
    anchor_df = interactions[base_cols].copy()
    anchor_df['role'] = 'anchor'
    anchor_df['track'] = interactions['anchor_track_id'].values

    # add anchor features, but strip the prefix
    for c in [c for c in interactions.columns if c.startswith('anchor_')
            and c not in ['anchor_track','anchor_track_id']]:
        anchor_df[c.replace('anchor_', '')] = interactions[c].values


    # 3) make PARTNER df
    partner_df = interactions[base_cols].copy()
    partner_df['role'] = 'partner'
    partner_df['track'] = interactions['partner_track_id'].values

    for c in [c for c in interactions.columns if c.startswith('partner_')
            and c not in ['partner_track','partner_track_id']]:
        partner_df[c.replace('partner_', '')] = interactions[c].values


    # 4) stack them
    interaction_tracks = pd.concat([anchor_df, partner_df], ignore_index=True)
    interaction_tracks = interaction_tracks.sort_values(['file', 'Interaction Number', 'Normalized Frame', 'role']).reset_index(drop=True)

    print(interaction_tracks['role'].unique())
    interaction_tracks['track'] = interaction_tracks['track'].astype('Int64')


    print("Extracted interaction tracks (anchor/partner).")
    interaction_tracks.to_csv(os.path.join(output, 'interaction_tracks_anchor_partner.csv'), index=False)


    moseq = moseq.sort_values(['name','frame_index']).copy()
    moseq['bout_id'] = moseq.groupby(['name'])['onset'].cumsum()

    bout_lengths = moseq.groupby(['name','bout_id']).size()
    good = bout_lengths[bout_lengths >= 2].reset_index()[['name','bout_id']]

    moseq = moseq.merge(good, on=['name','bout_id'], how='inner')


    # moseq['bout_id'] = moseq['onset'].astype(int).cumsum()
    # bout_lengths = moseq.groupby('bout_id').size()
    # good_bouts = bout_lengths[bout_lengths >= 2].index  # length ≥ 2 frames
    # moseq = moseq[moseq['bout_id'].isin(good_bouts)].copy()


    good_syllables = stat['syllable'].unique()
    moseq = moseq[moseq['syllable'].isin(good_syllables)]


    moseq['group'] = moseq['name'].str.split('_').str[0]
    moseq = moseq[moseq['frame_index'] <= 3600 ]  # ensure no negative frame indices

    def extract_track(name):
        match = re.search(r'track(\d+)', name)
        if match:
            return int(match.group(1))
        else:
            return 0
    

    def extract_video_id(name):
        match = re.search(r'_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_td\d+)', name)
        if match:
            return match.group(1)
        else:
            return None  # or "" if you prefer

    moseq['video_id'] = moseq['name'].apply(extract_video_id)
    moseq['track'] = moseq['name'].apply(extract_track)
    moseq = moseq.rename(columns={'frame_index': 'Frame'})
    moseq['track'] = moseq['track'].astype('Int64')


    interactions_with_syllables = interaction_tracks.merge(moseq[['video_id', 'Frame', 'track', 'syllable', 'onset']], on=['video_id', 'Frame', 'track'], how='left')
    interactions_with_syllables['syllable'] = interactions_with_syllables['syllable'].astype('Int64')  # capital I
    interactions_with_syllables = interactions_with_syllables.rename(columns={'Yhat.idt.pca': 'cluster'})
    interactions_with_syllables['Normalized Frame'] = interactions_with_syllables['Normalized Frame'].astype(int)
    interactions_with_syllables['unique_track_id'] = interactions_with_syllables['video_id'] + '_track' + interactions_with_syllables['track'].astype(str)
    interactions_with_syllables['period'] = np.where(interactions_with_syllables['Normalized Frame'] < 0, 'pre', 'post')
    interactions_with_syllables['track'] = interactions_with_syllables['track'].astype('Int64')


    print("Merged interactions with MOSEQ syllable data")
    interactions_with_syllables.to_csv(os.path.join(output, 'interactions_with_syllables.csv'), index=False)
    print(interactions_with_syllables.head())


    # partner rows, RAW (keep everything)
    partners_only_raw = interactions_with_syllables[interactions_with_syllables['role'] == 'partner'].copy()

    # add grouped label (but do NOT drop anything here)
    partners_only_raw['syllable_group'] = partners_only_raw['syllable'].map(syll_to_group)

    # grouped-only df (only rows that map to a group)
    partners_only_grouped = partners_only_raw.dropna(subset=['syllable_group']).copy()

    # ordering for grouped plots only
    group_order = list(syllable_groups.keys())
    partners_only_grouped['syllable_group'] = pd.Categorical(
        partners_only_grouped['syllable_group'],
        categories=group_order,
        ordered=True
    )



    # def plot_ethogram_fast(group, group_name):

    #     # Convert track names to row indices
    #     tracks = sorted(group['unique_track_id'].unique())
    #     track_to_row = {t: i for i, t in enumerate(tracks)}

    #     # Frame range
    #     min_f = group['Normalized Frame'].min()
    #     max_f = group['Normalized Frame'].max()

    #     width = max_f - min_f + 1

    #     # Create matrix: rows = tracks, columns = frames
    #     mat = np.full((len(tracks), width), fill_value=-1)   # -1 means "no syllable"

    #     for _, row in group.iterrows():
    #         r = track_to_row[row['unique_track_id']]
    #         c = row['Normalized Frame'] - min_f
    #         mat[r, c] = row['syllable']

    #     # Get syllables and colors
    # # --- Make your exact color mapping ---
    #     syllables = sorted(group['syllable'].unique())
    #     palette = sns.color_palette('viridis', n_colors=len(syllables))
    #     syl2color = {s: palette[i] for i, s in enumerate(syllables)}

    #     # convert syllables → integer index image
    #     idx_map = {s: i for i, s in enumerate(syllables)}
    #     mat_idx = np.full_like(mat, -1)
    #     for s, idx in idx_map.items():
    #         mat_idx[mat == s] = idx

    #     # colormap = your palette + white for missing
    #     cmap = ListedColormap(palette + [(1,1,1)])
    #     mat_idx[mat_idx == -1] = len(palette)

    #     # ---- PLOT ----
    #     plt.figure(figsize=(12, len(tracks)*0.4))
    #     plt.imshow(mat_idx, aspect='auto', cmap=cmap, interpolation='nearest')


    #     plt.yticks(np.arange(len(tracks)), tracks)
    #     plt.xlabel("Frame index")
    #     plt.ylabel("Track")
    #     plt.title(f"Syllable Ethogram {group_name}")

   
    #     handles = [mpatches.Patch(color=c, label=s) for s, c in syl2color.items()]
    #     plt.legend(handles=handles, title="Syllables",
    #             bbox_to_anchor=(1.01, 1), loc='upper left')

    #     plt.tight_layout()
    #     plt.savefig(os.path.join(output, f'{group_name}-ethogram.png'), dpi=300, bbox_inches='tight')
    #     plt.close()


    def plot_ethogram_fast(group, group_name):

        tracks = sorted(group['unique_track_id'].unique())

        # ---- sort tracks by dominant PRE syllable_group (Normalized Frame < 0) ----
        pre = group[group['Normalized Frame'] < 0].copy()

        # dominant PRE group per track
        dominant_pre = (
            pre.groupby(['unique_track_id', 'syllable_group'])
            .size()
            .reset_index(name='n')
            .sort_values(['unique_track_id', 'n'], ascending=[True, False])
            .drop_duplicates('unique_track_id')
            .set_index('unique_track_id')['syllable_group']
        )

        # ordering: dominant_pre (categorical order if you set it), then track id
        tracks = (
            group[['unique_track_id']]
            .drop_duplicates()
            .assign(dominant_pre=lambda d: d['unique_track_id'].map(dominant_pre))
            .sort_values(['dominant_pre', 'unique_track_id'])
            ['unique_track_id']
            .tolist()
        )

        track_to_row = {t: i for i, t in enumerate(tracks)}

        # track_to_row = {t: i for i, t in enumerate(tracks)}

        min_f = group['Normalized Frame'].min()
        max_f = group['Normalized Frame'].max()
        width = max_f - min_f + 1

        mat = np.full((len(tracks), width), fill_value=None, dtype=object)

        for _, row in group.iterrows():
            r = track_to_row[row['unique_track_id']]
            c = row['Normalized Frame'] - min_f
            mat[r, c] = row['syllable_group']

        # map group labels -> integer image
        group_names = list(syllable_groups.keys())
        idx_map = {g: i for i, g in enumerate(group_names)}

        mat_idx = np.full(mat.shape, -1, dtype=int)
        for g, idx in idx_map.items():
            mat_idx[mat == g] = idx

        palette = [group_colors[g] for g in group_names]
        cmap = ListedColormap(palette + [(1, 1, 1)])   # white for missing
        mat_idx[mat_idx == -1] = len(palette)

        plt.figure(figsize=(12, len(tracks) * 0.4))
        plt.imshow(mat_idx, aspect='auto', cmap=cmap, interpolation='nearest')

        plt.yticks(np.arange(len(tracks)), tracks)
        plt.xlabel("Normalized Frame")
        plt.ylabel("Track")
        plt.title(f"Grouped Syllable Ethogram {group_name}")

        handles = [mpatches.Patch(color=group_colors[g], label=g) for g in group_names]
        plt.legend(handles=handles, title="Syllable groups",
                bbox_to_anchor=(1.01, 1), loc='upper left')

        plt.tight_layout()
        plt.savefig(os.path.join(output, f'{group_name}-ethogram_grouped.png'), dpi=300, bbox_inches='tight')
        plt.close()

    


    
    # partners_only = interactions_with_syllables[interactions_with_syllables['role'] == 'partner'].copy()


    # groups = sorted(partners_only['cluster'].unique())
    # for group_id in groups:
    #     df_group = partners_only[partners_only['cluster'] == group_id].copy()
    #     df_group = df_group.dropna(subset=['syllable_group'])  # after merge certain rows dont have syllables
    #     plot_ethogram_fast(df_group, group_id)

    groups = sorted(partners_only_grouped['cluster'].unique())
    for group_id in groups:
        df_group = partners_only_grouped[partners_only_grouped['cluster'] == group_id].copy()
        df_group = df_group.dropna(subset=['syllable_group'])
        plot_ethogram_fast(df_group, group_id)




    # onsets = partners_only[partners_only['onset'] == True]
    onsets = partners_only_raw[partners_only_raw['onset'] == True]


    freq = (
    onsets
    .groupby(['syllable', 'cluster'])
    .size()
    .reset_index(name='frequency'))

    cluster_totals = (
    onsets
    .groupby('cluster')
    .size()
    .reset_index(name='total')
)
    freq = freq.merge(cluster_totals, on='cluster')
    freq['relative_frequency'] = freq['frequency'] / freq['total']   # fraction
    # or percent:
    freq['relative_frequency_percent'] = freq['relative_frequency'] * 100


    syllable_ids = sorted(onsets['syllable'].unique())
    n = len(syllable_ids)

    ncols = 10
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*3, nrows*3))
    axes = axes.flatten()

    for i, sid in enumerate(syllable_ids):
        ax = axes[i]
        sub = freq[freq['syllable'] == sid]
        ax.bar(sub['cluster'], sub['relative_frequency'])
        ax.set_title(f"Syll {sid}", fontsize=8)
        # ax.set_xticks([])
        # ax.set_yticks([])
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Frequency")

    # turn off any empty panels at the end
    for j in range(i+1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output, "syllable_relative_frequency.png"), dpi=300, bbox_inches="tight")
    plt.close()


    freq = (
        onsets
        .groupby(['syllable', 'cluster', 'period'])
        .size()
        .reset_index(name='count')
    )

    # 4) relative frequency "within each syllable + period"
    #    (so for a given syllable, pre bars across clusters sum to 1; same for post)
    totals = (
        freq
        .groupby(['syllable', 'period'])['count']
        .sum()
        .reset_index(name='total')
    )

    freq = freq.merge(totals, on=['syllable', 'period'])
    freq['rel_freq'] = freq['count'] / freq['total']


    # ---------- PLOT: one subplot per syllable, pre vs post bars per cluster ----------
    syllable_ids = sorted(freq['syllable'].dropna().unique())
    clusters_sorted = sorted(freq['cluster'].dropna().unique())

    n = len(syllable_ids)
    ncols = 10
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*3, nrows*3))
    axes = axes.flatten()

    for i, sid in enumerate(syllable_ids):
        ax = axes[i]
        sub = freq[freq['syllable'] == sid]

        pre = sub[sub['period'] == 'pre'].set_index('cluster').reindex(clusters_sorted, fill_value=0)
        post = sub[sub['period'] == 'post'].set_index('cluster').reindex(clusters_sorted, fill_value=0)

        x = np.arange(len(clusters_sorted))
        w = 0.4

        ax.bar(x - w/2, pre['rel_freq'], width=w, label='pre')
        ax.bar(x + w/2, post['rel_freq'], width=w, label='post')

        ax.set_title(f"Syll {sid}", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(clusters_sorted, rotation=90, fontsize=6)
        ax.set_yticks([])
        ax.set_ylim(0, max(pre['rel_freq'].max(), post['rel_freq'].max(), 1e-6) * 1.1)

    # turn off empty panels
    for j in range(i+1, len(axes)):
        axes[j].axis("off")

    # one legend for whole figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(output, "syllable_pre_post_per_cluster.png"),
                dpi=300, bbox_inches="tight")
    plt.close()


    # ============================================================
    # GROUPED SYLLABLE FREQUENCY PLOTS (ADDITIONAL, keep raw ones)
    # ============================================================

    # Use ONLY rows where we have a valid grouped label
    onsets_g = onsets.dropna(subset=['syllable_group']).copy()

    # --- 1) Grouped relative frequency per cluster ---
    freq_g = (
        onsets_g
        .groupby(['syllable_group', 'cluster'])
        .size()
        .reset_index(name='frequency')
    )

    cluster_totals_g = (
        onsets_g
        .groupby('cluster')
        .size()
        .reset_index(name='total')
    )

    freq_g = freq_g.merge(cluster_totals_g, on='cluster', how='left')
    freq_g['relative_frequency'] = freq_g['frequency'] / freq_g['total']
    freq_g['relative_frequency_percent'] = freq_g['relative_frequency'] * 100


    # Plot grid (one panel per grouped syllable)
    syllable_groups_order = list(syllable_groups.keys())
    clusters_sorted = sorted(freq_g['cluster'].dropna().unique())

    n = len(syllable_groups_order)
    ncols = 6
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*3, nrows*3))
    axes = axes.flatten()

    for i, sg in enumerate(syllable_groups_order):
        ax = axes[i]
        sub = (
            freq_g[freq_g['syllable_group'] == sg]
            .set_index('cluster')
            .reindex(clusters_sorted, fill_value=0)
            .reset_index()
        )

        ax.bar(sub['cluster'], sub['relative_frequency'])
        ax.set_title(str(sg), fontsize=9)
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Rel freq")

    for j in range(i+1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output, "grouped_syllable_relative_frequency.png"),
                dpi=300, bbox_inches="tight")
    plt.close()


    # --- 2) Grouped PRE vs POST per cluster ---
    freq_gp = (
        onsets_g
        .groupby(['syllable_group', 'cluster', 'period'])
        .size()
        .reset_index(name='count')
    )

    totals_gp = (
        freq_gp
        .groupby(['syllable_group', 'period'])['count']
        .sum()
        .reset_index(name='total')
    )

    freq_gp = freq_gp.merge(totals_gp, on=['syllable_group', 'period'], how='left')
    freq_gp['rel_freq'] = freq_gp['count'] / freq_gp['total']


    # Plot grid: one subplot per grouped syllable, pre/post bars per cluster
    syllable_ids = syllable_groups_order
    clusters_sorted = sorted(freq_gp['cluster'].dropna().unique())

    n = len(syllable_ids)
    ncols = 6
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*3, nrows*3))
    axes = axes.flatten()

    for i, sg in enumerate(syllable_ids):
        ax = axes[i]
        sub = freq_gp[freq_gp['syllable_group'] == sg]

        pre = (sub[sub['period'] == 'pre']
            .set_index('cluster')
            .reindex(clusters_sorted, fill_value=0))
        post = (sub[sub['period'] == 'post']
                .set_index('cluster')
                .reindex(clusters_sorted, fill_value=0))

        x = np.arange(len(clusters_sorted))
        w = 0.4
        ax.bar(x - w/2, pre['rel_freq'], width=w, label='pre')
        ax.bar(x + w/2, post['rel_freq'], width=w, label='post')

        ax.set_title(str(sg), fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(clusters_sorted, rotation=90, fontsize=6)
        ax.set_yticks([])
        ax.set_ylim(0, max(pre['rel_freq'].max(), post['rel_freq'].max(), 1e-6) * 1.1)

    for j in range(i+1, len(axes)):
        axes[j].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(output, "grouped_syllable_pre_post_per_cluster.png"),
                dpi=300, bbox_inches="tight")
    plt.close()















"""------------- INTERACTIONS ------------"""
# interactions = pd.read_csv('/Users/cochral/Desktop/MOSEQ/KEYPOINT-KAPPA1500/plots/interactions-n10/cropped_interactions.csv')
# cluster = pd.read_csv('/Users/cochral/Desktop/MOSEQ/KEYPOINT-KAPPA1500/plots/interactions-n10/pca-data2-F18.csv')
# moseq = pd.read_csv('/Users/cochral/Desktop/MOSEQ/KEYPOINT-KAPPA1500/moseq_df.csv')
# stat = pd.read_csv('/Users/cochral/Desktop/MOSEQ/KEYPOINT-KAPPA1500/stats_df.csv')
# output = '/Users/cochral/Desktop/MOSEQ/KEYPOINT-KAPPA1500/plots/grouped-dendrogram/interaction-partner'


# interaction_syllable_partner(interactions, cluster, moseq, stat, output)







# --------- ETHOGRAMS OVER TIME ------------- #
df = pd.read_csv('/Users/cochral/Desktop/MOSEQ/KEYPOINT-KAPPA1500/moseq_df.csv')
stats_df = pd.read_csv('/Users/cochral/Desktop/MOSEQ/KEYPOINT-KAPPA1500/stats_df.csv')
output = '/Users/cochral/Desktop/MOSEQ/KEYPOINT-KAPPA1500/plots/grouped-dendrogram/ethograms'
os.makedirs(output, exist_ok=True)

analysis_main(df, stats_df, output, ethogram=True)






# df = pd.read_csv('/Users/cochral/Desktop/MOSEQ/KEYPOINT-KAPPA1500/moseq_df.csv')
# directory = '/Users/cochral/Desktop/MOSEQ/KEYPOINT-KAPPA1500/plots'
# syllable_overlay(df, directory, 'N1-GH_2025-02-24_15-16-50_td7') #video_track_name = 'N1-GH_2025-02-24_15-16-50_td7'
# syllable_feature_quantifications(df, directory)



        

   




# directory = '/Users/cochral/Desktop/MOSEQ'
# comparing_models(directory)

    


## plot syllables over time - and also mean line but idk yes frequency 

## can i add in the testing 1 video overlay yes


# df_moseq = pd.read_csv('/Users/cochral/Desktop/MOSEQ/KEYPOINT-KAPPA3600/moseq_df.csv')
# df_stat = pd.read_csv('/Users/cochral/Desktop/MOSEQ/KEYPOINT-KAPPA3600/stats_df.csv')


# output = '/Users/cochral/Desktop/MOSEQ/KEYPOINT-KAPPA3600/testing' 
# if not os.path.exists(output):
#     os.makedirs(output)


# basic_stats(df_stat, output)
# durations(df_moseq, output)
# syllable_overlay(df_moseq, output, 'N1-GH_2025-02-24_15-16-50_td7') #video_track_name = 'N1-GH_2025-02-24_15-16-50_td7'  
# syllable_features(df_moseq, output)







