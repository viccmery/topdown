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
from matplotlib.collections import LineCollection
import matplotlib.image as mpimg
from PIL import Image
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from scipy.stats import kruskal
from statsmodels.stats.multitest import multipletests


""" ANALYSIS FOR MOSEQ DATA WITH ONLY 10 LARVAE MODEL """

###############################################
# 1. DEFINE SYLLABLE GROUPS FROM THE DENDROGRAM
###############################################


## RANDOM CHECK black line
syllable_groups = {
    "group1": [1], 
    "group2": [25], 
    "group3": [0], 
    "group4": [20], 
    "group5": [2], 
    'group6': [23], 
    'group7': [4],
    'group8': [19],
    'group9': [ 7, 12, 29],
    'group10': [15, 32],
    'group11': [8, 26, 28, 21, 31, 13, 9, 24, 17, 11, 14],
    'group12': [6, 16, 30, 5, 22, 18, 27],
    'group13': [3],
    'group14': [10],

}


group_colors = {
    "group1": 'blue',
    "group2": 'lightblue',
    "group3": 'darkred',
    "group4": 'firebrick',
    "group5": 'chocolate',
    "group6": 'sandybrown',
    "group7": 'mediumseagreen',
    'group8': 'mediumvioletred',
    'group9': 'deeppink',
    'group10': 'hotpink',
    'group11': 'green',
    'group12': 'limegreen',
    'group13': 'darkgreen',
    'group14': 'mediumseagreen',
}

# map syllable id -> group name
syll_to_group = {
    s: g for g, syls in syllable_groups.items() for s in syls
}

###########################################
# 2. CALL DIFFERENT FUNCTIONS FOR ANALYSIS 
###########################################

# --------------------------------------------------------
# BASIC_STATS: STATS_DF DURATION N FREQUENCY BY SYLLABLE
# --------------------------------------------------------
def stats_df(df, output):
    
    plt.figure(figsize=(8,6))
    sns.barplot(data=df, x='syllable', y='duration',  ci='sd')
    plt.title('Syllable Duration by Condition')
    plt.ylim(0, None)
    plt.xticks(rotation=90)
    plt.savefig(os.path.join(output, 'syllable_duration.png'), dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8,6))
    sns.pointplot(
        data=df,
        x='syllable',
        y='frequency',
        hue='group',      # or 'condition' depending on your column name
        errorbar=('ci', 95)
    )
    plt.title('Syllable Frequency by Condition')
    plt.ylim(0, None)
    plt.xticks(rotation=90)
    plt.savefig(os.path.join(output, 'syllable_frequency.png'), dpi=300, bbox_inches='tight')
    plt.close()

# --------------------------------------------------------
# DURATIONS: test durations of syllables
# --------------------------------------------------------
def fraction_one_frame_duration(df, output):

    def durations_by_onset(df):

        out = []
        for name, g in df.groupby("name", sort=False):
            g = g.reset_index(drop=False)
            onset_rows = g[g["onset"] == True]

            start_idxs = onset_rows["index"].tolist()
            sylls = onset_rows["syllable"].tolist()

            end_sentinel = g["index"].iloc[-1] + 1
            next_starts = start_idxs[1:] + [end_sentinel]

            for s, n, sy in zip(start_idxs, next_starts, sylls):
                out.append({
                    "name": name,
                    "syllable": sy,
                    "duration_frames": n - s
                })

        return pd.DataFrame(out)


    # --- compute bout durations ---
    df = df.sort_values(["name", "frame_index"]).reset_index(drop=True)
    durations = durations_by_onset(df)

    # --- count 1-frame bouts ---
    grouped_1_frame = (
        durations[durations["duration_frames"] < 2]
        .groupby(["syllable", "name"])
        .size()
        .reset_index(name="count_1_frame")
    )

    # --- total bouts ---
    total_counts = (
        durations.groupby(["syllable", "name"])
        .size()
        .reset_index(name="total_bouts")
    )

    # --- merge and compute fraction ---
    freqs = total_counts.merge(grouped_1_frame, on=["name", "syllable"], how="left").fillna(0)
    freqs["fraction_1_frame"] = freqs["count_1_frame"] / freqs["total_bouts"]

    # --- plot ---
    plt.figure(figsize=(8,6))
    sns.barplot(data=freqs, x="syllable", y="fraction_1_frame", ci="sd")
    plt.title("Fraction of 1-Frame Syllable Occurrences")
    plt.ylabel("Fraction of Bouts That Are 1 Frame")
    plt.xlabel("Syllable")
    plt.ylim(0, 1)
    plt.xticks(rotation=90)

    plt.savefig(
        os.path.join(output, "fraction_1_frame_syllable_occurrences.png"),
        dpi=300,
        bbox_inches="tight"
    )
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




def syllables_with_traces(df, output_directory):

    """
    If < 5% of bouts even have data at that frame → drop it
    - acc no - A raw trace is kept only if every frame of that bout stays within ±1 SD of the mean trajectory at that relative frame.- raw trace
    
    """

    def plot_time_colored_mean(ax, x, y, lw=2):
        x = np.asarray(x)
        y = np.asarray(y)

        # build line segments
        pts = np.column_stack([x, y]).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)

        t = np.linspace(0, 1, len(x) - 1)

        lc = LineCollection(segs, cmap='plasma')
        lc.set_array(t)  
        # lc.set_array(t[:-1])     # colour each segment in order
        lc.set_clim(0, 1)        # force full start→finish range
        lc.set_linewidth(lw)

        ax.add_collection(lc)
        return lc

    os.makedirs(output_directory, exist_ok=True)

    df = df.copy()
    df['bout_id'] = df['onset'].astype(int).cumsum()

    # --- translate + rotate ---
    translated_df = translate_rotate_syllables(df)

    # --- relative frame within each bout ---
    translated_df = translated_df.sort_values(['bout_id', 'frame_index'])
    translated_df['rel_frame'] = translated_df.groupby('bout_id').cumcount()

    # --- per syllable: compute mean trace + plot ---
    for syll, g_syl in translated_df.groupby('syllable', sort=True):

        # canonical (mean) trajectory over time within bout
        summary = (g_syl.groupby('rel_frame').agg(
            mean_x=('rotated_centroid_x', 'mean'),
            mean_y=('rotated_centroid_y', 'mean'),
            sd_x=('rotated_centroid_x', 'std'),     # <-- ADDED (needed for within 1SD)
            sd_y=('rotated_centroid_y', 'std'),     # <-- ADDED (needed for within 1SD)
            n_bouts=('bout_id', 'nunique'),
            n_frames=('bout_id', 'size'),
        ).reset_index())

        # ---------------- ADDED: mean_trace threshold (>= 5% of bouts contribute) ----------------
        total_bouts = g_syl['bout_id'].nunique()
        threshold_bouts = max(1, int(np.ceil(0.05 * total_bouts)))   # 5% rule
        mean_trace = summary[summary['n_bouts'] >= threshold_bouts].copy()
        # ---------------------------------------------------------------------------------------

        # save canonical CSV (clean name)
        out_csv = os.path.join(output_directory, f"syllable_{syll}.csv")
        summary.to_csv(out_csv, index=False)

        # ---------------- ADDED: optionally filter raw bouts within ±1 SD ----------------
        filter_raw_within_1sd = False   # <-- set False if you want *all* raw traces instead

        if filter_raw_within_1sd:
            stats_for_merge = summary[['rel_frame', 'mean_x', 'mean_y', 'sd_x', 'sd_y']].copy()
            g_tmp = g_syl.merge(stats_for_merge, on='rel_frame', how='left')

            g_tmp['within_1sd'] = (
                g_tmp['rotated_centroid_x'].between(g_tmp['mean_x'] - g_tmp['sd_x'],
                                                    g_tmp['mean_x'] + g_tmp['sd_x'])
                &
                g_tmp['rotated_centroid_y'].between(g_tmp['mean_y'] - g_tmp['sd_y'],
                                                    g_tmp['mean_y'] + g_tmp['sd_y'])
            )

            good_bouts_mask = g_tmp.groupby('bout_id')['within_1sd'].transform('all')
            g_plot = g_tmp[good_bouts_mask].copy()
        else:
            g_plot = g_syl
        # -------------------------------------------------------------------------------

        # plot (raw + mean)
        fig, ax = plt.subplots(figsize=(4, 4))

        # raw traces (either filtered or all)
        for bout_id, g_bout in g_plot.groupby('bout_id'):
            ax.plot(
                g_bout['rotated_centroid_x'],
                g_bout['rotated_centroid_y'],
                color='gray',
                alpha=0.15,
                linewidth=0.7
            )

        # mean trace (coverage-thresholded)
        # ax.plot(
        #     mean_trace['mean_x'],
        #     mean_trace['mean_y'],
        #     color='blue',
        #     linewidth=2.0,
        #     label='mean'
        # )

        plot_time_colored_mean(ax, mean_trace['mean_x'], mean_trace['mean_y'], lw=2.0)

        # mark start
        ax.scatter([0], [0], s=10, zorder=10, color='black')

        ax.set_aspect('equal', 'box')
        ax.set_title(f"{syll}")
        ax.set_xlabel("")
        ax.set_ylabel("")

        out_pdf = os.path.join(output_directory, f"syllable_{syll}.pdf")
        fig.savefig(out_pdf, bbox_inches='tight')
        plt.close(fig)



def grouped_syllables_with_traces(df, output_directory):
    """
    Plot raw aligned traces + mean trace for grouped syllables.

    Each grouped syllable is defined by syllable_groups / syll_to_group.
    Saves one PDF + CSV per grouped syllable.
    """

    def plot_time_colored_mean(ax, x, y, lw=2):
        x = np.asarray(x)
        y = np.asarray(y)

        if len(x) < 2:
            ax.plot(x, y, linewidth=lw)
            return

        pts = np.column_stack([x, y]).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)

        t = np.linspace(0, 1, len(x) - 1)

        lc = LineCollection(segs, cmap='plasma')
        lc.set_array(t)
        lc.set_clim(0, 1)
        lc.set_linewidth(lw)
        ax.add_collection(lc)

    os.makedirs(output_directory, exist_ok=True)

    df = df.copy()
    df = df.sort_values(['name', 'frame_index']).reset_index(drop=True)

    # safer bout ids within each recording
    df['bout_num'] = df.groupby('name')['onset'].cumsum()
    df['bout_id'] = df.groupby(['name', 'bout_num']).ngroup()

    # map raw syllables to grouped syllables
    df['syllable_group'] = df['syllable'].map(syll_to_group)
    df = df.dropna(subset=['syllable_group']).copy()

    # translate + rotate all bouts
    translated_df = translate_rotate_syllables(df)

    # relative frame within each bout
    translated_df = translated_df.sort_values(['bout_id', 'frame_index'])
    translated_df['rel_frame'] = translated_df.groupby('bout_id').cumcount()

    group_order = list(syllable_groups.keys())

    for group_name in group_order:
        g_syl = translated_df[translated_df['syllable_group'] == group_name].copy()

        if g_syl.empty:
            continue

        # canonical mean trajectory across all bouts in this grouped syllable
        summary = (
            g_syl.groupby('rel_frame')
            .agg(
                mean_x=('rotated_centroid_x', 'mean'),
                mean_y=('rotated_centroid_y', 'mean'),
                sd_x=('rotated_centroid_x', 'std'),
                sd_y=('rotated_centroid_y', 'std'),
                n_bouts=('bout_id', 'nunique'),
                n_frames=('bout_id', 'size'),
            )
            .reset_index()
        )

        total_bouts = g_syl['bout_id'].nunique()
        threshold_bouts = max(1, int(np.ceil(0.05 * total_bouts)))
        mean_trace = summary[summary['n_bouts'] >= threshold_bouts].copy()

        out_csv = os.path.join(output_directory, f"{group_name}.csv")
        summary.to_csv(out_csv, index=False)

        filter_raw_within_1sd = False

        if filter_raw_within_1sd:
            stats_for_merge = summary[['rel_frame', 'mean_x', 'mean_y', 'sd_x', 'sd_y']].copy()
            g_tmp = g_syl.merge(stats_for_merge, on='rel_frame', how='left')

            g_tmp['within_1sd'] = (
                g_tmp['rotated_centroid_x'].between(
                    g_tmp['mean_x'] - g_tmp['sd_x'],
                    g_tmp['mean_x'] + g_tmp['sd_x']
                )
                &
                g_tmp['rotated_centroid_y'].between(
                    g_tmp['mean_y'] - g_tmp['sd_y'],
                    g_tmp['mean_y'] + g_tmp['sd_y']
                )
            )

            good_bouts_mask = g_tmp.groupby('bout_id')['within_1sd'].transform('all')
            g_plot = g_tmp[good_bouts_mask].copy()
        else:
            g_plot = g_syl

        fig, ax = plt.subplots(figsize=(4, 4))

        for bout_id, g_bout in g_plot.groupby('bout_id'):
            ax.plot(
                g_bout['rotated_centroid_x'],
                g_bout['rotated_centroid_y'],
                color='gray',
                alpha=0.15,
                linewidth=0.7
            )

        if not mean_trace.empty:
            plot_time_colored_mean(ax, mean_trace['mean_x'], mean_trace['mean_y'], lw=2.0)

        ax.scatter([0], [0], s=10, zorder=10, color='black')
        ax.set_aspect('equal', 'box')
        ax.set_title(group_name)
        ax.set_xlabel("")
        ax.set_ylabel("")

        out_pdf = os.path.join(output_directory, f"{group_name}.pdf")
        fig.savefig(out_pdf, bbox_inches='tight')
        plt.close(fig)






def syllables_just_mean(df, output_directory):

    def plot_time_colored_mean(ax, x, y, lw=2):
        x = np.asarray(x)
        y = np.asarray(y)

        # build line segments
        pts = np.column_stack([x, y]).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)

        t = np.linspace(0, 1, len(x) - 1)

        lc = LineCollection(segs, cmap='viridis')
        lc.set_array(t)  
        # lc.set_array(t[:-1])     # colour each segment in order
        lc.set_clim(0, 1)        # force full start→finish range
        lc.set_linewidth(lw)

        ax.add_collection(lc)
        return lc

    os.makedirs(output_directory, exist_ok=True)

    df = df.copy()
    df['bout_id'] = df['onset'].astype(int).cumsum()

    # --- translate + rotate ---
    translated_df = translate_rotate_syllables(df)

    # --- relative frame within each bout ---
    translated_df = translated_df.sort_values(['bout_id', 'frame_index'])
    translated_df['rel_frame'] = translated_df.groupby('bout_id').cumcount()

    # --- per syllable: compute mean trace + plot ---
    for syll, g_syl in translated_df.groupby('syllable', sort=True):

        summary = (g_syl.groupby('rel_frame').agg(
            mean_x=('rotated_centroid_x', 'mean'),
            mean_y=('rotated_centroid_y', 'mean'),
            n_bouts=('bout_id', 'nunique'),
            n_frames=('bout_id', 'size'),
        ).reset_index())

        # ---------------- ADDED: mean_trace threshold (>= 5% of bouts contribute) ----------------
        total_bouts = g_syl['bout_id'].nunique()
        threshold_bouts = max(1, int(np.ceil(0.05 * total_bouts)))   # 5% rule
        mean_trace = summary[summary['n_bouts'] >= threshold_bouts].copy()
        # ---------------------------------------------------------------------------------------

        # save canonical CSV (clean name)
        out_csv = os.path.join(output_directory, f"syllable_{syll}.csv")
        summary.to_csv(out_csv, index=False)

        # plot (raw + mean)
        fig, ax = plt.subplots(figsize=(4, 4))

        plot_time_colored_mean(ax, mean_trace['mean_x'], mean_trace['mean_y'], lw=2.0)

        # mark start
        ax.scatter([0], [0], s=10, zorder=10, color='black')

        ax.set_aspect('equal', 'box')
        ax.set_title(f"{syll}")
        ax.set_xlabel("")
        ax.set_ylabel("")
        plt.ylim(-20, 420)
        plt.xlim(-100, 100)

        out_png = os.path.join(output_directory, f"syllable_{syll}.pdf")
        fig.savefig(out_png, bbox_inches='tight')
        plt.close(fig)



def plot_syllable_row(syllable_folder, syllable_order):
    # output saved inside the same folder
    out_png = os.path.join(syllable_folder, "syllable_order_row.png")

    n = len(syllable_order)
    fig, axes = plt.subplots(1, n, figsize=(n * 2.2, 2.2))  # one row

    # when n==1, axes isn't a list
    if n == 1:
        axes = [axes]

    for ax, syll in zip(axes, syllable_order):
        path = os.path.join(syllable_folder, f"syllable_{syll}.png")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing: {path}")

        img = mpimg.imread(path)
        ax.imshow(img)
        ax.axis("off")

    plt.tight_layout(pad=0.05)
    out_png = os.path.join(syllable_folder, "syllable_order_row.png")
    out_pdf = os.path.join(syllable_folder, "syllable_order_row.pdf")

    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=600, bbox_inches="tight")  # PDF
    plt.close(fig)


def plot_syllable_row_gif(syllable_folder, syllable_order):
    out_gif = os.path.join(syllable_folder, "syllable_order_row.gif")

    # load gifs
    gifs = []
    for syll in syllable_order:
        path = os.path.join(syllable_folder, f"Syllable{syll}.gif")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing: {path}")
        gifs.append(Image.open(path))

    # assume all same frame count + duration
    n_frames = getattr(gifs[0], "n_frames", 1)
    w, h = gifs[0].size

    # duration per frame (ms) — keep from first gif if present
    frame0 = gifs[0].copy()
    duration = frame0.info.get("duration", 40)  # fallback ~25fps

    frames_out = []
    for i in range(n_frames):
        row = Image.new("RGBA", (w * len(gifs), h), (255, 255, 255, 0))

        for j, im in enumerate(gifs):
            im.seek(i)
            frame = im.convert("RGBA")
            row.paste(frame, (j * w, 0), frame)

        frames_out.append(row)

    # save animated gif
    frames_out[0].save(
        out_gif,
        save_all=True,
        append_images=frames_out[1:],
        duration=duration,
        loop=0,
        disposal=2,
    )

    # close
    for im in gifs:
        im.close()




def plot_raw(df, output_directory, n_examples=100, ncols=10, random_state=42):
    """
    For each syllable, plot up to n_examples aligned raw bouts in a grid.
    Each subplot contains one aligned bout.

    Saves one PDF per syllable into:
        <output_directory>/raw/syllable_<id>.pdf
    """

    os.makedirs(output_directory, exist_ok=True)
    raw_output = os.path.join(output_directory, "raw")
    os.makedirs(raw_output, exist_ok=True)

    df = df.sort_values(["name", "frame_index"]).copy()

    # bout ids within each recording
    df["bout_num"] = df.groupby("name")["onset"].cumsum()
    df["bout_id"] = df.groupby(["name", "bout_num"]).ngroup()

    # align all bouts exactly like the other functions
    translated_df = translate_rotate_syllables(df)

    rng = np.random.default_rng(random_state)
    nrows = int(np.ceil(n_examples / ncols))

    for syll, g_syl in translated_df.groupby("syllable", sort=True):

        bout_ids = g_syl["bout_id"].dropna().unique()
        if len(bout_ids) == 0:
            continue

        n_take = min(n_examples, len(bout_ids))
        sampled_bout_ids = rng.choice(bout_ids, size=n_take, replace=False)

        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
        axes = np.atleast_1d(axes).flatten()

        # consistent axis limits across all sampled bouts for this syllable
        g_plot = g_syl[g_syl["bout_id"].isin(sampled_bout_ids)].copy()
        xmin = g_plot["rotated_centroid_x"].min()
        xmax = g_plot["rotated_centroid_x"].max()
        ymin = g_plot["rotated_centroid_y"].min()
        ymax = g_plot["rotated_centroid_y"].max()

        # add a little padding
        xpad = (xmax - xmin) * 0.05 if xmax > xmin else 1
        ypad = (ymax - ymin) * 0.05 if ymax > ymin else 1

        for ax, bout_id in zip(axes, sampled_bout_ids):
            g_bout = g_syl[g_syl["bout_id"] == bout_id].sort_values("frame_index")

            ax.plot(
                g_bout["rotated_centroid_x"],
                g_bout["rotated_centroid_y"],
                color="gray",
                linewidth=1.0
            )
            ax.scatter([0], [0], s=8, color="black", zorder=10)

            ax.set_aspect("equal", "box")
            ax.set_xlim(xmin - xpad, xmax + xpad)
            ax.set_ylim(ymin - ypad, ymax + ypad)
            ax.set_xticks([])
            ax.set_yticks([])

        # turn off unused panels
        for ax in axes[n_take:]:
            ax.axis("off")

        fig.suptitle(f"Syllable {syll}", y=0.995)
        plt.tight_layout()

        out_pdf = os.path.join(raw_output, f"syllable_{syll}.pdf")
        fig.savefig(out_pdf, bbox_inches="tight")
        plt.close(fig)



def grouped_syllable_frequencies(df, output):
    """
    Use stats_df to group syllables according to syllable_groups
    and plot grouped frequency for GH vs SI.
    """

    os.makedirs(output, exist_ok=True)

    df = df.copy()

    # map syllable -> group
    df["syllable_group"] = df["syllable"].map(syll_to_group)
    df = df.dropna(subset=["syllable_group"])

    # sum frequencies within grouped syllables per recording
    grouped = (
        df.groupby(["group", "name", "syllable_group"], as_index=False)["frequency"]
        .sum()
    )

    # preserve group order from your dict
    group_order = list(syllable_groups.keys())
    grouped["syllable_group"] = pd.Categorical(
        grouped["syllable_group"],
        categories=group_order,
        ordered=True
    )

    plt.figure(figsize=(10,6))

    sns.pointplot(
        data=grouped,
        x="syllable_group",
        y="frequency",
        hue="group",
        order=group_order,
        errorbar="sd"
    )

    plt.xlabel("Grouped syllable")
    plt.ylabel("Frequency")
    plt.title("Grouped Syllable Frequency (GH vs SI)")
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    plt.savefig(
        os.path.join(output, "grouped_syllable_frequency_stats_df.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # -----------------------
    # Mann-Whitney per group
    # -----------------------
    results = []

    for sg in group_order:
        sub = grouped[grouped["syllable_group"] == sg].copy()

        gh = sub.loc[sub["group"] == "GH", "frequency"].dropna()
        si = sub.loc[sub["group"] == "SI", "frequency"].dropna()

        # skip if either condition missing
        if len(gh) == 0 or len(si) == 0:
            results.append({
                "syllable_group": sg,
                "n_GH": len(gh),
                "n_SI": len(si),
                "GH_mean": gh.mean() if len(gh) > 0 else np.nan,
                "SI_mean": si.mean() if len(si) > 0 else np.nan,
                "GH_median": gh.median() if len(gh) > 0 else np.nan,
                "SI_median": si.median() if len(si) > 0 else np.nan,
                "u_stat": np.nan,
                "p_value": np.nan
            })
            continue

        u_stat, p_value = mannwhitneyu(gh, si, alternative="two-sided")

        results.append({
            "syllable_group": sg,
            "n_GH": len(gh),
            "n_SI": len(si),
            "GH_mean": gh.mean(),
            "SI_mean": si.mean(),
            "GH_median": gh.median(),
            "SI_median": si.median(),
            "u_stat": u_stat,
            "p_value": p_value
        })

    results_df = pd.DataFrame(results)

    # FDR correction across grouped syllables
    valid = results_df["p_value"].notna()
    results_df["p_fdr"] = np.nan
    results_df["significant_fdr_0.05"] = False

    if valid.sum() > 0:
        reject, p_fdr, _, _ = multipletests(
            results_df.loc[valid, "p_value"],
            method="fdr_bh"
        )
        results_df.loc[valid, "p_fdr"] = p_fdr
        results_df.loc[valid, "significant_fdr_0.05"] = reject

    results_df.to_csv(
        os.path.join(output, "grouped_syllable_frequency_mannwhitney.csv"),
        index=False
    )


    return grouped






def grouped_ethogram_by_condition(df, output, max_frame=3600):
    """
    Plot grouped syllables over time for each individual track.
    Makes two plots:
        - GH_grouped_ethogram.png
        - SI_grouped_ethogram.png
    """

    os.makedirs(output, exist_ok=True)

    df = df.copy()
    df = df.sort_values(['name', 'frame_index'])

    # keep only grouped syllables
    # df['syllable_group'] = df['syllable'].map(syll_to_group)
    
    # df = df.dropna(subset=['syllable_group']).copy()

    df['syllable_group'] = df['syllable'].map(syll_to_group).fillna('unknown')

    # keep only desired time window
    df = df[(df['frame_index'] >= 0) & (df['frame_index'] <= max_frame)].copy()

    # extract condition from name
    df['condition'] = df['name'].str.extract(r'(GH|SI)')

    group_names = list(syllable_groups.keys())
    idx_map = {g: i for i, g in enumerate(group_names)}
    palette = [group_colors[g] for g in group_names]
    cmap = ListedColormap(palette + [(1, 1, 1)])  # white for missing

    def plot_condition(df_cond, condition_name):
        if df_cond.empty:
            print(f'No data for {condition_name}')
            return

        tracks = sorted(df_cond['name'].unique())
        track_to_row = {t: i for i, t in enumerate(tracks)}

        width = max_frame + 1
        mat = np.full((len(tracks), width), fill_value=None, dtype=object)

        for _, row in df_cond.iterrows():
            r = track_to_row[row['name']]
            c = int(row['frame_index'])
            if 0 <= c <= max_frame:
                mat[r, c] = row['syllable_group']

        mat_idx = np.full(mat.shape, -1, dtype=int)
        for g, idx in idx_map.items():
            mat_idx[mat == g] = idx
        mat_idx[mat_idx == -1] = len(palette)

        plt.figure(figsize=(16, max(6, len(tracks) * 0.25)))
        plt.imshow(mat_idx, aspect='auto', cmap=cmap, interpolation='nearest')

        plt.yticks(np.arange(len(tracks)), tracks, fontsize=6)
        plt.xlabel("Frame index")
        plt.ylabel("Track")
        plt.title(f"Grouped syllables over time: {condition_name}")
        plt.xlim(0, max_frame)

        handles = [mpatches.Patch(color=group_colors[g], label=g) for g in group_names]
        plt.legend(handles=handles, title="Syllable groups",
                   bbox_to_anchor=(1.01, 1), loc='upper left')

        plt.tight_layout()
        plt.savefig(
            os.path.join(output, f"{condition_name}_grouped_ethogram.png"),
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()

    plot_condition(df[df['condition'] == 'GH'].copy(), 'GH')
    plot_condition(df[df['condition'] == 'SI'].copy(), 'SI')




def interaction_grouped_ethogram(interactions, cluster, moseq, stat, output):

    interactions = pd.merge(
        interactions,
        cluster[['interaction_id', 'Yhat.idt.pca']],
        on='interaction_id',
        how='inner'
    )

    print("Merged interactions with cluster data")

    interactions['video_id'] = interactions['file'].str.replace('.mp4', '', regex=False)

    keep = ['file', 'video_id', 'Frame', 'Interaction Number', 'Normalized Frame', 'Interaction Pair', 'Yhat.idt.pca']

    interactions['Interaction Pair'] = interactions['Interaction Pair'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    interaction_tracks = (
        interactions[keep]
        .assign(track=interactions['Interaction Pair'])
        .explode('track')
        .drop(columns='Interaction Pair')
        .reset_index(drop=True)
    )

    print("Extracted interaction tracks")

    moseq = moseq.sort_values(['name', 'frame_index']).copy()
    moseq['bout_id'] = moseq.groupby(['name'])['onset'].cumsum()

    bout_lengths = moseq.groupby(['name', 'bout_id']).size()
    good = bout_lengths[bout_lengths >= 2].reset_index()[['name', 'bout_id']]
    moseq = moseq.merge(good, on=['name', 'bout_id'], how='inner')

    good_syllables = stat['syllable'].unique()
    moseq = moseq[moseq['syllable'].isin(good_syllables)].copy()

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
            return None

    moseq['video_id'] = moseq['name'].apply(extract_video_id)
    moseq['track'] = moseq['name'].apply(extract_track)
    moseq = moseq.rename(columns={'frame_index': 'Frame'})

    interaction_tracks['track'] = pd.to_numeric(interaction_tracks['track'], errors='coerce').astype('Int64')
    moseq['track'] = pd.to_numeric(moseq['track'], errors='coerce').astype('Int64')

    interactions_with_syllables = interaction_tracks.merge(
        moseq[['video_id', 'Frame', 'track', 'syllable', 'onset']],
        on=['video_id', 'Frame', 'track'],
        how='left'
    )

    interactions_with_syllables['syllable'] = interactions_with_syllables['syllable'].astype('Int64')
    interactions_with_syllables = interactions_with_syllables.rename(columns={'Yhat.idt.pca': 'cluster'})
    interactions_with_syllables['Normalized Frame'] = interactions_with_syllables['Normalized Frame'].astype(int)
    interactions_with_syllables['unique_track_id'] = (
        interactions_with_syllables['video_id'] + '_track' + interactions_with_syllables['track'].astype(str)
    )

    interactions_with_syllables['syllable_group'] = interactions_with_syllables['syllable'].map(syll_to_group)

    interactions_with_syllables.to_csv(
        os.path.join(output, 'interactions_with_grouped_syllables.csv'),
        index=False
    )

    print("Merged interactions with MOSEQ grouped syllables")

    grouped_only = interactions_with_syllables.dropna(subset=['syllable_group']).copy()

    group_order = list(syllable_groups.keys())
    grouped_only['syllable_group'] = pd.Categorical(
        grouped_only['syllable_group'],
        categories=group_order,
        ordered=True
    )

    def plot_ethogram_fast(group, group_name):

        tracks = sorted(group['unique_track_id'].unique())
        track_to_row = {t: i for i, t in enumerate(tracks)}

        min_f = group['Normalized Frame'].min()
        max_f = group['Normalized Frame'].max()
        width = max_f - min_f + 1

        mat = np.full((len(tracks), width), fill_value=None, dtype=object)

        for _, row in group.iterrows():
            r = track_to_row[row['unique_track_id']]
            c = row['Normalized Frame'] - min_f
            mat[r, c] = row['syllable_group']

        group_names = list(syllable_groups.keys())
        idx_map = {g: i for i, g in enumerate(group_names)}

        mat_idx = np.full(mat.shape, -1, dtype=int)
        for g, idx in idx_map.items():
            mat_idx[mat == g] = idx

        palette = [group_colors[g] for g in group_names]
        cmap = ListedColormap(palette + [(1, 1, 1)])
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
        plt.savefig(
            os.path.join(output, f'{group_name}-grouped-ethogram.png'),
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()

    groups = sorted(grouped_only['cluster'].dropna().unique())
    for group_id in groups:
        df_group = grouped_only[grouped_only['cluster'] == group_id].copy()
        plot_ethogram_fast(df_group, group_id)



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





def partner_interaction_ethogram(interactions, cluster, moseq, stat, output):
    """
    Merge interaction data with MoSeq syllables, keep only PARTNER rows,
    map to grouped syllables, and plot one grouped ethogram per cluster.
    """

    os.makedirs(output, exist_ok=True)

    # --------------------------------------------------
    # 1) merge interaction clusters
    # --------------------------------------------------
    interactions = pd.merge(
        interactions,
        cluster[['interaction_id', 'Yhat.idt.pca']],
        on='interaction_id',
        how='inner'
    )

    print("Merged interactions with cluster data")

    # --------------------------------------------------
    # 2) assign anchor / partner
    # --------------------------------------------------
    interactions = anchor_partner(interactions)
    print("Computed anchor and partner tracks")

    interactions['video_id'] = interactions['file'].str.replace('.mp4', '', regex=False)

    # --------------------------------------------------
    # 3) build long df with one row for anchor and one for partner
    # --------------------------------------------------
    base_cols = ['file', 'video_id', 'Frame', 'Interaction Number',
                 'Normalized Frame', 'Yhat.idt.pca']

    anchor_df = interactions[base_cols].copy()
    anchor_df['role'] = 'anchor'
    anchor_df['track'] = interactions['anchor_track_id'].values

    partner_df = interactions[base_cols].copy()
    partner_df['role'] = 'partner'
    partner_df['track'] = interactions['partner_track_id'].values

    interaction_tracks = pd.concat([anchor_df, partner_df], ignore_index=True)
    interaction_tracks = interaction_tracks.sort_values(
        ['file', 'Interaction Number', 'Normalized Frame', 'role']
    ).reset_index(drop=True)

    interaction_tracks['track'] = interaction_tracks['track'].astype('Int64')

    print("Extracted interaction tracks (anchor/partner)")

    # --------------------------------------------------
    # 4) prep moseq df exactly like before
    # --------------------------------------------------
    moseq = moseq.sort_values(['name', 'frame_index']).copy()
    moseq['bout_id'] = moseq.groupby('name')['onset'].cumsum()

    bout_lengths = moseq.groupby(['name', 'bout_id']).size()
    good = bout_lengths[bout_lengths >= 2].reset_index()[['name', 'bout_id']]
    moseq = moseq.merge(good, on=['name', 'bout_id'], how='inner')

    good_syllables = stat['syllable'].unique()
    moseq = moseq[moseq['syllable'].isin(good_syllables)].copy()

    moseq['group'] = moseq['name'].str.split('_').str[0]
    moseq = moseq[moseq['frame_index'] <= 3600].copy()

    def extract_track(name):
        match = re.search(r'track(\d+)', name)
        if match:
            return int(match.group(1))
        return 0

    def extract_video_id(name):
        match = re.search(r'_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_td\d+)', name)
        if match:
            return match.group(1)
        return None

    moseq['video_id'] = moseq['name'].apply(extract_video_id)
    moseq['track'] = moseq['name'].apply(extract_track).astype('Int64')
    moseq = moseq.rename(columns={'frame_index': 'Frame'})

    # --------------------------------------------------
    # 5) merge interaction rows with moseq syllables
    # --------------------------------------------------
    interactions_with_syllables = interaction_tracks.merge(
        moseq[['video_id', 'Frame', 'track', 'syllable', 'onset']],
        on=['video_id', 'Frame', 'track'],
        how='left'
    )

    interactions_with_syllables['syllable'] = interactions_with_syllables['syllable'].astype('Int64')
    interactions_with_syllables = interactions_with_syllables.rename(
        columns={'Yhat.idt.pca': 'cluster'}
    )
    interactions_with_syllables['Normalized Frame'] = interactions_with_syllables['Normalized Frame'].astype(int)
    interactions_with_syllables['unique_track_id'] = (
        interactions_with_syllables['video_id'] + '_track' +
        interactions_with_syllables['track'].astype(str)
    )

    print("Merged interactions with MOSEQ syllable data")
    interactions_with_syllables.to_csv(
        os.path.join(output, 'partner_interactions_with_syllables.csv'),
        index=False
    )

    # --------------------------------------------------
    # 6) keep ONLY partner rows
    # --------------------------------------------------
    partners_only = interactions_with_syllables[
        interactions_with_syllables['role'] == 'partner'
    ].copy()

    # map grouped syllables
    partners_only['syllable_group'] = partners_only['syllable'].map(syll_to_group)

    # keep only grouped rows
    partners_only = partners_only.dropna(subset=['syllable_group']).copy()

    group_order = list(syllable_groups.keys())
    partners_only['syllable_group'] = pd.Categorical(
        partners_only['syllable_group'],
        categories=group_order,
        ordered=True
    )

    # --------------------------------------------------
    # 7) plotting function
    # --------------------------------------------------
    def plot_ethogram_fast(group, group_name):
        if group.empty:
            return

        tracks = sorted(group['unique_track_id'].unique())

        pre = group[group['Normalized Frame'] < 0].copy()

        dominant_pre = (
            pre.groupby(['unique_track_id', 'syllable_group'])
               .size()
               .reset_index(name='n')
               .sort_values(['unique_track_id', 'n'], ascending=[True, False])
               .drop_duplicates('unique_track_id')
               .set_index('unique_track_id')['syllable_group']
        )

        tracks = (
            group[['unique_track_id']]
            .drop_duplicates()
            .assign(dominant_pre=lambda d: d['unique_track_id'].map(dominant_pre))
            .sort_values(['dominant_pre', 'unique_track_id'])
            ['unique_track_id']
            .tolist()
        )

        track_to_row = {t: i for i, t in enumerate(tracks)}

        min_f = group['Normalized Frame'].min()
        max_f = group['Normalized Frame'].max()
        width = max_f - min_f + 1

        mat = np.full((len(tracks), width), fill_value=None, dtype=object)

        for _, row in group.iterrows():
            r = track_to_row[row['unique_track_id']]
            c = row['Normalized Frame'] - min_f
            mat[r, c] = row['syllable_group']

        idx_map = {g: i for i, g in enumerate(group_order)}

        mat_idx = np.full(mat.shape, -1, dtype=int)
        for g, idx in idx_map.items():
            mat_idx[mat == g] = idx

        palette = [group_colors[g] for g in group_order]
        cmap = ListedColormap(palette + [(1, 1, 1)])
        mat_idx[mat_idx == -1] = len(palette)

        plt.figure(figsize=(12, len(tracks) * 0.4))
        plt.imshow(mat_idx, aspect='auto', cmap=cmap, interpolation='nearest')

        plt.yticks(np.arange(len(tracks)), tracks)
        plt.xlabel("Normalized Frame")
        plt.ylabel("Partner track")
        plt.title(f"Partner grouped syllable ethogram {group_name}")

        handles = [mpatches.Patch(color=group_colors[g], label=g) for g in group_order]
        plt.legend(
            handles=handles,
            title="Syllable groups",
            bbox_to_anchor=(1.01, 1),
            loc='upper left'
        )

        plt.tight_layout()
        plt.savefig(
            os.path.join(output, f'{group_name}_partner_ethogram_grouped.png'),
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()

    # --------------------------------------------------
    # 8) one ethogram per cluster
    # --------------------------------------------------
    groups = sorted(partners_only['cluster'].dropna().unique())

    for group_id in groups:
        df_group = partners_only[partners_only['cluster'] == group_id].copy()
        plot_ethogram_fast(df_group, group_id)




from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests

def interaction_syllable_frequency(interactions, cluster, moseq, stat, output):
    """
    Map MoSeq grouped syllables onto interaction frames/tracks and quantify:

    1) grouped syllable frequency across interaction clusters
    2) grouped syllable frequency across interaction clusters split by pre/post
       where pre = Normalized Frame < 0
             post = Normalized Frame >= 0

    Saves:
    - merged csv
    - per-grouped-syllable barplots across clusters
    - per-grouped-syllable pre/post barplots across clusters
    - stats tables testing enrichment across clusters
    """

    os.makedirs(output, exist_ok=True)

    # --------------------------------------------------
    # 1. merge interaction clusters
    # --------------------------------------------------
    interactions = pd.merge(
        interactions,
        cluster[['interaction_id', 'Yhat.idt.pca']],
        on='interaction_id',
        how='inner'
    )

    print("Merged interactions with cluster data")

    interactions['video_id'] = interactions['file'].str.replace('.mp4', '', regex=False)

    keep = ['file', 'video_id', 'Frame', 'Interaction Number',
            'Normalized Frame', 'Interaction Pair', 'Yhat.idt.pca']

    interactions['Interaction Pair'] = interactions['Interaction Pair'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    interaction_tracks = (
        interactions[keep]
        .assign(track=interactions['Interaction Pair'])
        .explode('track')
        .drop(columns='Interaction Pair')
        .reset_index(drop=True)
    )

    interaction_tracks['track'] = interaction_tracks['track'].astype('Int64')

    print("Extracted interaction tracks")

    # --------------------------------------------------
    # 2. prep moseq the same way as before
    # --------------------------------------------------
    moseq = moseq.sort_values(['name', 'frame_index']).copy()
    moseq['bout_id'] = moseq.groupby('name')['onset'].cumsum()

    bout_lengths = moseq.groupby(['name', 'bout_id']).size()
    good = bout_lengths[bout_lengths >= 2].reset_index()[['name', 'bout_id']]
    moseq = moseq.merge(good, on=['name', 'bout_id'], how='inner')

    good_syllables = stat['syllable'].unique()
    moseq = moseq[moseq['syllable'].isin(good_syllables)].copy()

    moseq['group'] = moseq['name'].str.split('_').str[0]
    moseq = moseq[moseq['frame_index'] <= 3600].copy()

    def extract_track(name):
        match = re.search(r'track(\d+)', name)
        if match:
            return int(match.group(1))
        return 0

    def extract_video_id(name):
        match = re.search(r'_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_td\d+)', name)
        if match:
            return match.group(1)
        return None

    moseq['video_id'] = moseq['name'].apply(extract_video_id)
    moseq['track'] = moseq['name'].apply(extract_track).astype('Int64')
    moseq = moseq.rename(columns={'frame_index': 'Frame'})

    # --------------------------------------------------
    # 3. merge interaction rows with moseq syllables
    # --------------------------------------------------
    merged = interaction_tracks.merge(
        moseq[['video_id', 'Frame', 'track', 'syllable', 'onset']],
        on=['video_id', 'Frame', 'track'],
        how='left'
    )

    merged['syllable'] = merged['syllable'].astype('Int64')
    merged = merged.rename(columns={'Yhat.idt.pca': 'cluster'})
    merged['Normalized Frame'] = merged['Normalized Frame'].astype(int)
    merged['period'] = np.where(merged['Normalized Frame'] < 0, 'pre', 'post')

    merged['syllable_group'] = merged['syllable'].map(syll_to_group)

    merged.to_csv(os.path.join(output, 'interactions_with_grouped_syllables.csv'), index=False)
    print("Merged interactions with MOSEQ syllable data")

    # --------------------------------------------------
    # 4. keep onset rows only for frequency analysis
    # --------------------------------------------------
    onsets = merged[merged['onset'] == True].copy()
    onsets = onsets.dropna(subset=['syllable_group', 'cluster']).copy()

    group_order = list(syllable_groups.keys())
    cluster_order = sorted(onsets['cluster'].dropna().unique())

    onsets['syllable_group'] = pd.Categorical(
        onsets['syllable_group'],
        categories=group_order,
        ordered=True
    )

    # --------------------------------------------------
    # 5. grouped syllable frequency across clusters
    # --------------------------------------------------
    freq = (
        onsets.groupby(['syllable_group', 'cluster'])
        .size()
        .reset_index(name='count')
    )

    cluster_totals = (
        onsets.groupby('cluster')
        .size()
        .reset_index(name='cluster_total')
    )

    freq = freq.merge(cluster_totals, on='cluster', how='left')
    freq['relative_frequency'] = freq['count'] / freq['cluster_total']

    freq.to_csv(os.path.join(output, 'grouped_syllable_frequency_per_cluster.csv'), index=False)

    # one subplot per grouped syllable
    n = len(group_order)
    ncols = 4
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    axes = axes.flatten()

    for i, sg in enumerate(group_order):
        ax = axes[i]
        sub = (
            freq[freq['syllable_group'] == sg]
            .set_index('cluster')
            .reindex(cluster_order, fill_value=0)
            .reset_index()
        )

        ax.bar(sub['cluster'], sub['relative_frequency'])
        ax.set_title(str(sg), fontsize=10)
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Rel. freq")
        ax.set_xticks(cluster_order)
        ax.set_xticklabels(cluster_order, rotation=90, fontsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(
        os.path.join(output, "grouped_syllable_frequency_per_cluster.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # --------------------------------------------------
    # 6. stats: does each grouped syllable depend on cluster?
    #    2 x N table: this syllable vs all others across clusters
    # --------------------------------------------------
    stats_rows = []

    for sg in group_order:
        sub = onsets.copy()
        sub['is_this_group'] = sub['syllable_group'] == sg

        contingency = pd.crosstab(sub['is_this_group'], sub['cluster'])

        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            stats_rows.append({
                'syllable_group': sg,
                'chi2': np.nan,
                'p_value': np.nan,
                'dof': np.nan
            })
            continue

        chi2, p, dof, expected = chi2_contingency(contingency)

        stats_rows.append({
            'syllable_group': sg,
            'chi2': chi2,
            'p_value': p,
            'dof': dof
        })

    stats_df = pd.DataFrame(stats_rows)
    stats_df['p_fdr'] = np.nan
    stats_df['significant_fdr_0.05'] = False

    valid = stats_df['p_value'].notna()
    if valid.sum() > 0:
        reject, p_fdr, _, _ = multipletests(stats_df.loc[valid, 'p_value'], method='fdr_bh')
        stats_df.loc[valid, 'p_fdr'] = p_fdr
        stats_df.loc[valid, 'significant_fdr_0.05'] = reject

    stats_df.to_csv(
        os.path.join(output, 'grouped_syllable_cluster_stats.csv'),
        index=False
    )

    # --------------------------------------------------
    # 7. pre/post version
    # --------------------------------------------------
    freq_pp = (
        onsets.groupby(['syllable_group', 'cluster', 'period'])
        .size()
        .reset_index(name='count')
    )

    totals_pp = (
        onsets.groupby(['cluster', 'period'])
        .size()
        .reset_index(name='cluster_period_total')
    )

    freq_pp = freq_pp.merge(totals_pp, on=['cluster', 'period'], how='left')
    freq_pp['relative_frequency'] = freq_pp['count'] / freq_pp['cluster_period_total']

    freq_pp.to_csv(
        os.path.join(output, 'grouped_syllable_frequency_per_cluster_pre_post.csv'),
        index=False
    )

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    axes = axes.flatten()

    for i, sg in enumerate(group_order):
        ax = axes[i]
        sub = freq_pp[freq_pp['syllable_group'] == sg]

        pre = (
            sub[sub['period'] == 'pre']
            .set_index('cluster')
            .reindex(cluster_order, fill_value=0)
        )
        post = (
            sub[sub['period'] == 'post']
            .set_index('cluster')
            .reindex(cluster_order, fill_value=0)
        )

        x = np.arange(len(cluster_order))
        w = 0.4

        ax.bar(x - w/2, pre['relative_frequency'], width=w, label='pre')
        ax.bar(x + w/2, post['relative_frequency'], width=w, label='post')

        ax.set_title(str(sg), fontsize=10)
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Rel. freq")
        ax.set_xticks(x)
        ax.set_xticklabels(cluster_order, rotation=90, fontsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')

    plt.tight_layout()
    plt.savefig(
        os.path.join(output, "grouped_syllable_frequency_per_cluster_pre_post.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # --------------------------------------------------
    # 8. stats for pre/post separately
    # --------------------------------------------------
    stats_rows_pp = []

    for period in ['pre', 'post']:
        sub_period = onsets[onsets['period'] == period].copy()

        for sg in group_order:
            sub = sub_period.copy()
            sub['is_this_group'] = sub['syllable_group'] == sg

            contingency = pd.crosstab(sub['is_this_group'], sub['cluster'])

            if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                stats_rows_pp.append({
                    'period': period,
                    'syllable_group': sg,
                    'chi2': np.nan,
                    'p_value': np.nan,
                    'dof': np.nan
                })
                continue

            chi2, p, dof, expected = chi2_contingency(contingency)

            stats_rows_pp.append({
                'period': period,
                'syllable_group': sg,
                'chi2': chi2,
                'p_value': p,
                'dof': dof
            })

    stats_pp_df = pd.DataFrame(stats_rows_pp)
    stats_pp_df['p_fdr'] = np.nan
    stats_pp_df['significant_fdr_0.05'] = False

    for period in ['pre', 'post']:
        mask = (stats_pp_df['period'] == period) & stats_pp_df['p_value'].notna()
        if mask.sum() > 0:
            reject, p_fdr, _, _ = multipletests(
                stats_pp_df.loc[mask, 'p_value'],
                method='fdr_bh'
            )
            stats_pp_df.loc[mask, 'p_fdr'] = p_fdr
            stats_pp_df.loc[mask, 'significant_fdr_0.05'] = reject

    stats_pp_df.to_csv(
        os.path.join(output, 'grouped_syllable_cluster_stats_pre_post.csv'),
        index=False
    )

    return merged, freq, stats_df, freq_pp, stats_pp_df



def interaction_syllable_frame_coverage(interactions, cluster, moseq, stat, output):
    """
    Same structure as your interaction syllable functions,
    but uses grouped syllable FRAME COVERAGE instead of onset frequency.

    Also does the same split into pre/post using Normalized Frame.
    """

    os.makedirs(output, exist_ok=True)

    interactions = pd.merge(
        interactions,
        cluster[['interaction_id', 'Yhat.idt.pca']],
        on='interaction_id',
        how='inner'
    )

    print("Merged interactions with cluster data")

    interactions['video_id'] = interactions['file'].str.replace('.mp4', '', regex=False)

    keep = ['file', 'video_id', 'Frame', 'Interaction Number', 'Normalized Frame', 'Interaction Pair', 'Yhat.idt.pca']

    interactions['Interaction Pair'] = interactions['Interaction Pair'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    interaction_tracks = (
        interactions[keep]
        .assign(track=interactions['Interaction Pair'])
        .explode('track')
        .drop(columns='Interaction Pair')
        .reset_index(drop=True)
    )

    print("Extracted interaction tracks")

    moseq = moseq.copy()
    moseq['bout_id'] = moseq['onset'].astype(int).cumsum()

    bout_lengths = moseq.groupby('bout_id').size()
    good_bouts = bout_lengths[bout_lengths >= 2].index
    moseq = moseq[moseq['bout_id'].isin(good_bouts)].copy()

    good_syllables = stat['syllable'].unique()
    moseq = moseq[moseq['syllable'].isin(good_syllables)].copy()

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
            return None

    moseq['video_id'] = moseq['name'].apply(extract_video_id)
    moseq['track'] = moseq['name'].apply(extract_track)
    moseq = moseq.rename(columns={'frame_index': 'Frame'})

    interactions_with_syllables = interaction_tracks.merge(
        moseq[['video_id', 'Frame', 'track', 'syllable', 'onset']],
        on=['video_id', 'Frame', 'track'],
        how='left'
    )

    interactions_with_syllables['syllable'] = interactions_with_syllables['syllable'].astype('Int64')
    interactions_with_syllables = interactions_with_syllables.rename(columns={'Yhat.idt.pca': 'cluster'})
    interactions_with_syllables['Normalized Frame'] = interactions_with_syllables['Normalized Frame'].astype(int)
    interactions_with_syllables['period'] = np.where(
        interactions_with_syllables['Normalized Frame'] < 0,
        'pre',
        'post'
    )

    # grouped syllables
    interactions_with_syllables['syllable_group'] = interactions_with_syllables['syllable'].map(syll_to_group)
    grouped = interactions_with_syllables.dropna(subset=['syllable_group']).copy()

    grouped['syllable_group'] = pd.Categorical(
        grouped['syllable_group'],
        categories=list(syllable_groups.keys()),
        ordered=True
    )

    grouped.to_csv(os.path.join(output, 'interactions_with_grouped_syllables.csv'), index=False)

    # =========================================================
    # 1. FRAME COVERAGE PER GROUPED SYLLABLE PER CLUSTER
    # =========================================================
    coverage = (
        grouped.groupby(['cluster', 'syllable_group'])
        .size()
        .reset_index(name='n_frames')
    )

    cluster_totals = (
        grouped.groupby('cluster')
        .size()
        .reset_index(name='total_frames')
    )

    coverage = coverage.merge(cluster_totals, on='cluster', how='left')
    coverage['frame_coverage'] = coverage['n_frames'] / coverage['total_frames']
    coverage['frame_coverage_percent'] = coverage['frame_coverage'] * 100

    coverage.to_csv(
        os.path.join(output, 'grouped_syllable_frame_coverage.csv'),
        index=False
    )

    syllable_groups_order = list(syllable_groups.keys())
    clusters_sorted = sorted(coverage['cluster'].dropna().unique())

    n = len(syllable_groups_order)
    ncols = 6
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    axes = axes.flatten()

    for i, sg in enumerate(syllable_groups_order):
        ax = axes[i]
        sub = (
            coverage[coverage['syllable_group'] == sg]
            .set_index('cluster')
            .reindex(clusters_sorted, fill_value=0)
            .reset_index()
        )

        ax.bar(sub['cluster'], sub['frame_coverage'])
        ax.set_title(str(sg), fontsize=9)
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Coverage")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(
        os.path.join(output, "grouped_syllable_frame_coverage_per_cluster.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # =========================================================
    # 2. FRAME COVERAGE PRE / POST PER GROUPED SYLLABLE
    # =========================================================
    coverage_pp = (
        grouped.groupby(['syllable_group', 'cluster', 'period'])
        .size()
        .reset_index(name='n_frames')
    )

    totals_pp = (
        grouped.groupby(['cluster', 'period'])
        .size()
        .reset_index(name='total_frames')
    )

    coverage_pp = coverage_pp.merge(totals_pp, on=['cluster', 'period'], how='left')
    coverage_pp['frame_coverage'] = coverage_pp['n_frames'] / coverage_pp['total_frames']
    coverage_pp['frame_coverage_percent'] = coverage_pp['frame_coverage'] * 100

    coverage_pp.to_csv(
        os.path.join(output, 'grouped_syllable_frame_coverage_pre_post.csv'),
        index=False
    )

    syllable_ids = syllable_groups_order
    clusters_sorted = sorted(coverage_pp['cluster'].dropna().unique())

    n = len(syllable_ids)
    ncols = 6
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    axes = axes.flatten()

    for i, sg in enumerate(syllable_ids):
        ax = axes[i]
        sub = coverage_pp[coverage_pp['syllable_group'] == sg]

        pre = (
            sub[sub['period'] == 'pre']
            .set_index('cluster')
            .reindex(clusters_sorted, fill_value=0)
        )
        post = (
            sub[sub['period'] == 'post']
            .set_index('cluster')
            .reindex(clusters_sorted, fill_value=0)
        )

        x = np.arange(len(clusters_sorted))
        w = 0.4

        ax.bar(x - w/2, pre['frame_coverage'], width=w, label='pre')
        ax.bar(x + w/2, post['frame_coverage'], width=w, label='post')

        ax.set_title(str(sg), fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(clusters_sorted, rotation=90, fontsize=6)
        ax.set_yticks([])

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')

    plt.tight_layout()
    plt.savefig(
        os.path.join(output, "grouped_syllable_frame_coverage_pre_post_per_cluster.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()




def interaction_syllable_frequency(interactions, cluster, moseq, stat, output):
    """
    Map MoSeq grouped syllables onto interaction frames/tracks and quantify:

    1) grouped syllable ONSET frequency across interaction clusters
       using video_id x cluster as the replicate level

    2) grouped syllable ONSET frequency across interaction clusters split by pre/post
       where pre  = Normalized Frame < 0
             post = Normalized Frame >= 0
       again using video_id x cluster x period as the replicate level

    Saves:
    - merged csv
    - replicate-level csv tables
    - mean +/- SD plots across videos
    - stats tables
    """

    os.makedirs(output, exist_ok=True)

    # --------------------------------------------------
    # 1. merge interaction clusters
    # --------------------------------------------------
    interactions = pd.merge(
        interactions,
        cluster[['interaction_id', 'Yhat.idt.pca']],
        on='interaction_id',
        how='inner'
    )

    print("Merged interactions with cluster data")

    interactions['video_id'] = interactions['file'].str.replace('.mp4', '', regex=False)

    keep = [
        'file', 'video_id', 'Frame', 'Interaction Number',
        'Normalized Frame', 'Interaction Pair', 'Yhat.idt.pca'
    ]

    interactions['Interaction Pair'] = interactions['Interaction Pair'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    interaction_tracks = (
        interactions[keep]
        .assign(track=interactions['Interaction Pair'])
        .explode('track')
        .drop(columns='Interaction Pair')
        .reset_index(drop=True)
    )

    interaction_tracks['track'] = interaction_tracks['track'].astype('Int64')

    print("Extracted interaction tracks")

    # --------------------------------------------------
    # 2. prep moseq same way as before
    # --------------------------------------------------
    moseq = moseq.sort_values(['name', 'frame_index']).copy()
    moseq['bout_id'] = moseq.groupby('name')['onset'].cumsum()

    bout_lengths = moseq.groupby(['name', 'bout_id']).size()
    good = bout_lengths[bout_lengths >= 2].reset_index()[['name', 'bout_id']]
    moseq = moseq.merge(good, on=['name', 'bout_id'], how='inner')

    good_syllables = stat['syllable'].unique()
    moseq = moseq[moseq['syllable'].isin(good_syllables)].copy()

    def extract_track(name):
        match = re.search(r'track(\d+)', name)
        if match:
            return int(match.group(1))
        return 0

    def extract_video_id(name):
        match = re.search(r'_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_td\d+)', name)
        if match:
            return match.group(1)
        return None

    moseq['video_id'] = moseq['name'].apply(extract_video_id)
    moseq['track'] = moseq['name'].apply(extract_track).astype('Int64')
    moseq = moseq.rename(columns={'frame_index': 'Frame'})

    # --------------------------------------------------
    # 3. merge interaction rows with moseq syllables
    # --------------------------------------------------
    merged = interaction_tracks.merge(
        moseq[['video_id', 'Frame', 'track', 'syllable', 'onset']],
        on=['video_id', 'Frame', 'track'],
        how='left'
    )

    merged['syllable'] = merged['syllable'].astype('Int64')
    merged = merged.rename(columns={'Yhat.idt.pca': 'cluster'})
    merged['Normalized Frame'] = merged['Normalized Frame'].astype(int)
    merged['period'] = np.where(merged['Normalized Frame'] < 0, 'pre', 'post')
    merged['syllable_group'] = merged['syllable'].map(syll_to_group)

    merged.to_csv(
        os.path.join(output, 'interactions_with_grouped_syllables.csv'),
        index=False
    )
    print("Merged interactions with MOSEQ syllable data")

    # --------------------------------------------------
    # 4. keep onset rows only
    # --------------------------------------------------
    onsets = merged[merged['onset'] == True].copy()
    onsets = onsets.dropna(subset=['syllable_group', 'cluster', 'video_id']).copy()

    group_order = list(syllable_groups.keys())
    cluster_order = sorted(onsets['cluster'].dropna().unique())

    onsets['syllable_group'] = pd.Categorical(
        onsets['syllable_group'],
        categories=group_order,
        ordered=True
    )

    # --------------------------------------------------
    # 5. replicate-level onset frequency per video x cluster
    # --------------------------------------------------
    counts = (
        onsets.groupby(['video_id', 'cluster', 'syllable_group'])
        .size()
        .reset_index(name='count')
    )

    totals = (
        onsets.groupby(['video_id', 'cluster'])
        .size()
        .reset_index(name='cluster_total')
    )

    freq_video = counts.merge(totals, on=['video_id', 'cluster'], how='left')
    freq_video['relative_frequency'] = freq_video['count'] / freq_video['cluster_total']

    # complete missing grouped syllables as zeros within each video x cluster
    all_video_cluster = totals[['video_id', 'cluster']].drop_duplicates()
    full_index = pd.MultiIndex.from_product(
        [all_video_cluster.index, group_order],
        names=['vc_row', 'syllable_group']
    )

    vc_lookup = all_video_cluster.reset_index().rename(columns={'index': 'vc_row'})
    freq_video = (
        vc_lookup.merge(
            pd.DataFrame(index=full_index).reset_index(),
            on='vc_row',
            how='right'
        )
        .merge(freq_video, on=['video_id', 'cluster', 'syllable_group'], how='left')
    )

    freq_video['cluster_total'] = freq_video['cluster_total'].fillna(
        freq_video.groupby(['video_id', 'cluster'])['cluster_total'].transform('max')
    )
    freq_video['count'] = freq_video['count'].fillna(0)
    freq_video['relative_frequency'] = freq_video['relative_frequency'].fillna(0)

    freq_video.to_csv(
        os.path.join(output, 'grouped_syllable_frequency_per_video_cluster.csv'),
        index=False
    )

    # summary for plotting
    freq_summary = (
        freq_video.groupby(['syllable_group', 'cluster'], as_index=False)
        .agg(
            mean_relative_frequency=('relative_frequency', 'mean'),
            sd_relative_frequency=('relative_frequency', 'std'),
            n_videos=('video_id', 'nunique')
        )
    )

    freq_summary.to_csv(
        os.path.join(output, 'grouped_syllable_frequency_per_cluster_summary.csv'),
        index=False
    )

    # plot one subplot per grouped syllable
    n = len(group_order)
    ncols = 4
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    axes = axes.flatten()

    for i, sg in enumerate(group_order):
        ax = axes[i]
        sub = (
            freq_summary[freq_summary['syllable_group'] == sg]
            .set_index('cluster')
            .reindex(cluster_order)
            .reset_index()
        )

        x = np.arange(len(cluster_order))
        y = sub['mean_relative_frequency'].fillna(0).to_numpy()
        yerr = sub['sd_relative_frequency'].fillna(0).to_numpy()

        ax.bar(x, y, yerr=yerr, capsize=3)
        ax.set_title(str(sg), fontsize=10)
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Rel. freq")
        ax.set_xticks(x)
        ax.set_xticklabels(cluster_order, rotation=90, fontsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(
        os.path.join(output, "grouped_syllable_frequency_per_cluster.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # --------------------------------------------------
    # 6. stats across clusters using video-level replicates
    # --------------------------------------------------
    stats_rows = []

    for sg in group_order:
        sub = freq_video[freq_video['syllable_group'] == sg].copy()

        if sub.empty or sub['cluster'].nunique() < 2:
            stats_rows.append({
                'syllable_group': sg,
                'test': 'kruskal',
                'statistic': np.nan,
                'p_value': np.nan
            })
            continue

        samples = [
            g['relative_frequency'].dropna().values
            for _, g in sub.groupby('cluster')
        ]
        samples = [x for x in samples if len(x) > 0]

        if len(samples) < 2:
            stats_rows.append({
                'syllable_group': sg,
                'test': 'kruskal',
                'statistic': np.nan,
                'p_value': np.nan
            })
            continue

        stat_out, p_out = kruskal(*samples)

        stats_rows.append({
            'syllable_group': sg,
            'test': 'kruskal',
            'statistic': stat_out,
            'p_value': p_out
        })

    stats_df = pd.DataFrame(stats_rows)
    stats_df['p_fdr'] = np.nan
    stats_df['significant_fdr_0.05'] = False

    valid = stats_df['p_value'].notna()
    if valid.sum() > 0:
        reject, p_fdr, _, _ = multipletests(stats_df.loc[valid, 'p_value'], method='fdr_bh')
        stats_df.loc[valid, 'p_fdr'] = p_fdr
        stats_df.loc[valid, 'significant_fdr_0.05'] = reject

    stats_df.to_csv(
        os.path.join(output, 'grouped_syllable_cluster_stats.csv'),
        index=False
    )

    # --------------------------------------------------
    # 7. pre/post replicate-level onset frequency
    # --------------------------------------------------
    counts_pp = (
        onsets.groupby(['video_id', 'cluster', 'period', 'syllable_group'])
        .size()
        .reset_index(name='count')
    )

    totals_pp = (
        onsets.groupby(['video_id', 'cluster', 'period'])
        .size()
        .reset_index(name='cluster_period_total')
    )

    freq_video_pp = counts_pp.merge(
        totals_pp,
        on=['video_id', 'cluster', 'period'],
        how='left'
    )
    freq_video_pp['relative_frequency'] = freq_video_pp['count'] / freq_video_pp['cluster_period_total']

    # complete missing grouped syllables as zeros within each video x cluster x period
    all_vcp = totals_pp[['video_id', 'cluster', 'period']].drop_duplicates()
    full_index_pp = pd.MultiIndex.from_product(
        [all_vcp.index, group_order],
        names=['vcp_row', 'syllable_group']
    )

    vcp_lookup = all_vcp.reset_index().rename(columns={'index': 'vcp_row'})
    freq_video_pp = (
        vcp_lookup.merge(
            pd.DataFrame(index=full_index_pp).reset_index(),
            on='vcp_row',
            how='right'
        )
        .merge(freq_video_pp, on=['video_id', 'cluster', 'period', 'syllable_group'], how='left')
    )

    freq_video_pp['cluster_period_total'] = freq_video_pp['cluster_period_total'].fillna(
        freq_video_pp.groupby(['video_id', 'cluster', 'period'])['cluster_period_total'].transform('max')
    )
    freq_video_pp['count'] = freq_video_pp['count'].fillna(0)
    freq_video_pp['relative_frequency'] = freq_video_pp['relative_frequency'].fillna(0)

    freq_video_pp.to_csv(
        os.path.join(output, 'grouped_syllable_frequency_per_video_cluster_pre_post.csv'),
        index=False
    )

    freq_pp_summary = (
        freq_video_pp.groupby(['syllable_group', 'cluster', 'period'], as_index=False)
        .agg(
            mean_relative_frequency=('relative_frequency', 'mean'),
            sd_relative_frequency=('relative_frequency', 'std'),
            n_videos=('video_id', 'nunique')
        )
    )

    freq_pp_summary.to_csv(
        os.path.join(output, 'grouped_syllable_frequency_per_cluster_pre_post_summary.csv'),
        index=False
    )

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    axes = axes.flatten()

    for i, sg in enumerate(group_order):
        ax = axes[i]
        sub = freq_pp_summary[freq_pp_summary['syllable_group'] == sg]

        pre = (
            sub[sub['period'] == 'pre']
            .set_index('cluster')
            .reindex(cluster_order)
            .reset_index()
        )
        post = (
            sub[sub['period'] == 'post']
            .set_index('cluster')
            .reindex(cluster_order)
            .reset_index()
        )

        x = np.arange(len(cluster_order))
        w = 0.4

        pre_y = pre['mean_relative_frequency'].fillna(0).to_numpy()
        pre_err = pre['sd_relative_frequency'].fillna(0).to_numpy()

        post_y = post['mean_relative_frequency'].fillna(0).to_numpy()
        post_err = post['sd_relative_frequency'].fillna(0).to_numpy()

        ax.bar(x - w/2, pre_y, width=w, yerr=pre_err, capsize=3, label='pre')
        ax.bar(x + w/2, post_y, width=w, yerr=post_err, capsize=3, label='post')

        ax.set_title(str(sg), fontsize=10)
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Rel. freq")
        ax.set_xticks(x)
        ax.set_xticklabels(cluster_order, rotation=90, fontsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')

    plt.tight_layout()
    plt.savefig(
        os.path.join(output, "grouped_syllable_frequency_per_cluster_pre_post.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # --------------------------------------------------
    # 8. stats for pre/post separately using video-level replicates
    # --------------------------------------------------
    stats_rows_pp = []

    for period in ['pre', 'post']:
        sub_period = freq_video_pp[freq_video_pp['period'] == period].copy()

        for sg in group_order:
            sub = sub_period[sub_period['syllable_group'] == sg].copy()

            if sub.empty or sub['cluster'].nunique() < 2:
                stats_rows_pp.append({
                    'period': period,
                    'syllable_group': sg,
                    'test': 'kruskal',
                    'statistic': np.nan,
                    'p_value': np.nan
                })
                continue

            samples = [
                g['relative_frequency'].dropna().values
                for _, g in sub.groupby('cluster')
            ]
            samples = [x for x in samples if len(x) > 0]

            if len(samples) < 2:
                stats_rows_pp.append({
                    'period': period,
                    'syllable_group': sg,
                    'test': 'kruskal',
                    'statistic': np.nan,
                    'p_value': np.nan
                })
                continue

            stat_out, p_out = kruskal(*samples)

            stats_rows_pp.append({
                'period': period,
                'syllable_group': sg,
                'test': 'kruskal',
                'statistic': stat_out,
                'p_value': p_out
            })

    stats_pp_df = pd.DataFrame(stats_rows_pp)
    stats_pp_df['p_fdr'] = np.nan
    stats_pp_df['significant_fdr_0.05'] = False

    for period in ['pre', 'post']:
        mask = (stats_pp_df['period'] == period) & stats_pp_df['p_value'].notna()
        if mask.sum() > 0:
            reject, p_fdr, _, _ = multipletests(
                stats_pp_df.loc[mask, 'p_value'],
                method='fdr_bh'
            )
            stats_pp_df.loc[mask, 'p_fdr'] = p_fdr
            stats_pp_df.loc[mask, 'significant_fdr_0.05'] = reject

    stats_pp_df.to_csv(
        os.path.join(output, 'grouped_syllable_cluster_stats_pre_post.csv'),
        index=False
    )

    return merged, freq_video, freq_summary, stats_df, freq_video_pp, freq_pp_summary, stats_pp_df


def interaction_syllable_frame_coverage(interactions, cluster, moseq, stat, output):
    """
    Same structure as interaction_syllable_frequency,
    but uses grouped syllable FRAME COVERAGE instead of onset frequency.

    Replicate level:
    - video_id x cluster
    - video_id x cluster x period
    """

    os.makedirs(output, exist_ok=True)

    # --------------------------------------------------
    # 1. merge interaction clusters
    # --------------------------------------------------
    interactions = pd.merge(
        interactions,
        cluster[['interaction_id', 'Yhat.idt.pca']],
        on='interaction_id',
        how='inner'
    )

    print("Merged interactions with cluster data")

    interactions['video_id'] = interactions['file'].str.replace('.mp4', '', regex=False)

    keep = [
        'file', 'video_id', 'Frame', 'Interaction Number',
        'Normalized Frame', 'Interaction Pair', 'Yhat.idt.pca'
    ]

    interactions['Interaction Pair'] = interactions['Interaction Pair'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    interaction_tracks = (
        interactions[keep]
        .assign(track=interactions['Interaction Pair'])
        .explode('track')
        .drop(columns='Interaction Pair')
        .reset_index(drop=True)
    )

    interaction_tracks['track'] = interaction_tracks['track'].astype('Int64')

    print("Extracted interaction tracks")

    # --------------------------------------------------
    # 2. prep moseq the same way as before
    # --------------------------------------------------
    moseq = moseq.sort_values(['name', 'frame_index']).copy()
    moseq['bout_id'] = moseq.groupby('name')['onset'].cumsum()

    bout_lengths = moseq.groupby(['name', 'bout_id']).size()
    good = bout_lengths[bout_lengths >= 2].reset_index()[['name', 'bout_id']]
    moseq = moseq.merge(good, on=['name', 'bout_id'], how='inner')

    good_syllables = stat['syllable'].unique()
    moseq = moseq[moseq['syllable'].isin(good_syllables)].copy()

    def extract_track(name):
        match = re.search(r'track(\d+)', name)
        if match:
            return int(match.group(1))
        return 0

    def extract_video_id(name):
        match = re.search(r'_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_td\d+)', name)
        if match:
            return match.group(1)
        return None

    moseq['video_id'] = moseq['name'].apply(extract_video_id)
    moseq['track'] = moseq['name'].apply(extract_track).astype('Int64')
    moseq = moseq.rename(columns={'frame_index': 'Frame'})

    # --------------------------------------------------
    # 3. merge interaction rows with moseq syllables
    # --------------------------------------------------
    merged = interaction_tracks.merge(
        moseq[['video_id', 'Frame', 'track', 'syllable', 'onset']],
        on=['video_id', 'Frame', 'track'],
        how='left'
    )

    merged['syllable'] = merged['syllable'].astype('Int64')
    merged = merged.rename(columns={'Yhat.idt.pca': 'cluster'})
    merged['Normalized Frame'] = merged['Normalized Frame'].astype(int)
    merged['period'] = np.where(merged['Normalized Frame'] < 0, 'pre', 'post')
    merged['syllable_group'] = merged['syllable'].map(syll_to_group)

    grouped = merged.dropna(subset=['syllable_group', 'cluster', 'video_id']).copy()

    grouped['syllable_group'] = pd.Categorical(
        grouped['syllable_group'],
        categories=list(syllable_groups.keys()),
        ordered=True
    )

    grouped.to_csv(
        os.path.join(output, 'interactions_with_grouped_syllables.csv'),
        index=False
    )

    group_order = list(syllable_groups.keys())
    cluster_order = sorted(grouped['cluster'].dropna().unique())

    # --------------------------------------------------
    # 4. replicate-level frame coverage per video x cluster
    # --------------------------------------------------
    counts = (
        grouped.groupby(['video_id', 'cluster', 'syllable_group'])
        .size()
        .reset_index(name='n_frames')
    )

    totals = (
        grouped.groupby(['video_id', 'cluster'])
        .size()
        .reset_index(name='cluster_total_frames')
    )

    coverage_video = counts.merge(totals, on=['video_id', 'cluster'], how='left')
    coverage_video['frame_coverage'] = coverage_video['n_frames'] / coverage_video['cluster_total_frames']

    # complete missing grouped syllables as zeros within each video x cluster
    all_video_cluster = totals[['video_id', 'cluster']].drop_duplicates()
    full_index = pd.MultiIndex.from_product(
        [all_video_cluster.index, group_order],
        names=['vc_row', 'syllable_group']
    )

    vc_lookup = all_video_cluster.reset_index().rename(columns={'index': 'vc_row'})
    coverage_video = (
        vc_lookup.merge(
            pd.DataFrame(index=full_index).reset_index(),
            on='vc_row',
            how='right'
        )
        .merge(coverage_video, on=['video_id', 'cluster', 'syllable_group'], how='left')
    )

    coverage_video['cluster_total_frames'] = coverage_video['cluster_total_frames'].fillna(
        coverage_video.groupby(['video_id', 'cluster'])['cluster_total_frames'].transform('max')
    )
    coverage_video['n_frames'] = coverage_video['n_frames'].fillna(0)
    coverage_video['frame_coverage'] = coverage_video['frame_coverage'].fillna(0)

    coverage_video.to_csv(
        os.path.join(output, 'grouped_syllable_frame_coverage_per_video_cluster.csv'),
        index=False
    )

    coverage_summary = (
        coverage_video.groupby(['syllable_group', 'cluster'], as_index=False)
        .agg(
            mean_frame_coverage=('frame_coverage', 'mean'),
            sd_frame_coverage=('frame_coverage', 'std'),
            n_videos=('video_id', 'nunique')
        )
    )

    coverage_summary.to_csv(
        os.path.join(output, 'grouped_syllable_frame_coverage_per_cluster_summary.csv'),
        index=False
    )

    # plot one subplot per grouped syllable
    n = len(group_order)
    ncols = 4
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    axes = axes.flatten()

    for i, sg in enumerate(group_order):
        ax = axes[i]
        sub = (
            coverage_summary[coverage_summary['syllable_group'] == sg]
            .set_index('cluster')
            .reindex(cluster_order)
            .reset_index()
        )

        x = np.arange(len(cluster_order))
        y = sub['mean_frame_coverage'].fillna(0).to_numpy()
        yerr = sub['sd_frame_coverage'].fillna(0).to_numpy()

        ax.bar(x, y, yerr=yerr, capsize=3)
        ax.set_title(str(sg), fontsize=10)
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Coverage")
        ax.set_xticks(x)
        ax.set_xticklabels(cluster_order, rotation=90, fontsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(
        os.path.join(output, "grouped_syllable_frame_coverage_per_cluster.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # --------------------------------------------------
    # 5. stats across clusters using video-level replicates
    # --------------------------------------------------
    stats_rows = []

    for sg in group_order:
        sub = coverage_video[coverage_video['syllable_group'] == sg].copy()

        if sub.empty or sub['cluster'].nunique() < 2:
            stats_rows.append({
                'syllable_group': sg,
                'test': 'kruskal',
                'statistic': np.nan,
                'p_value': np.nan
            })
            continue

        samples = [
            g['frame_coverage'].dropna().values
            for _, g in sub.groupby('cluster')
        ]
        samples = [x for x in samples if len(x) > 0]

        if len(samples) < 2:
            stats_rows.append({
                'syllable_group': sg,
                'test': 'kruskal',
                'statistic': np.nan,
                'p_value': np.nan
            })
            continue

        stat_out, p_out = kruskal(*samples)

        stats_rows.append({
            'syllable_group': sg,
            'test': 'kruskal',
            'statistic': stat_out,
            'p_value': p_out
        })

    stats_df = pd.DataFrame(stats_rows)
    stats_df['p_fdr'] = np.nan
    stats_df['significant_fdr_0.05'] = False

    valid = stats_df['p_value'].notna()
    if valid.sum() > 0:
        reject, p_fdr, _, _ = multipletests(stats_df.loc[valid, 'p_value'], method='fdr_bh')
        stats_df.loc[valid, 'p_fdr'] = p_fdr
        stats_df.loc[valid, 'significant_fdr_0.05'] = reject

    stats_df.to_csv(
        os.path.join(output, 'grouped_syllable_frame_coverage_stats.csv'),
        index=False
    )

    # --------------------------------------------------
    # 6. replicate-level frame coverage per video x cluster x period
    # --------------------------------------------------
    counts_pp = (
        grouped.groupby(['video_id', 'cluster', 'period', 'syllable_group'])
        .size()
        .reset_index(name='n_frames')
    )

    totals_pp = (
        grouped.groupby(['video_id', 'cluster', 'period'])
        .size()
        .reset_index(name='cluster_period_total_frames')
    )

    coverage_video_pp = counts_pp.merge(
        totals_pp,
        on=['video_id', 'cluster', 'period'],
        how='left'
    )
    coverage_video_pp['frame_coverage'] = (
        coverage_video_pp['n_frames'] / coverage_video_pp['cluster_period_total_frames']
    )

    # complete missing grouped syllables as zeros within each video x cluster x period
    all_vcp = totals_pp[['video_id', 'cluster', 'period']].drop_duplicates()
    full_index_pp = pd.MultiIndex.from_product(
        [all_vcp.index, group_order],
        names=['vcp_row', 'syllable_group']
    )

    vcp_lookup = all_vcp.reset_index().rename(columns={'index': 'vcp_row'})
    coverage_video_pp = (
        vcp_lookup.merge(
            pd.DataFrame(index=full_index_pp).reset_index(),
            on='vcp_row',
            how='right'
        )
        .merge(coverage_video_pp, on=['video_id', 'cluster', 'period', 'syllable_group'], how='left')
    )

    coverage_video_pp['cluster_period_total_frames'] = coverage_video_pp['cluster_period_total_frames'].fillna(
        coverage_video_pp.groupby(['video_id', 'cluster', 'period'])['cluster_period_total_frames'].transform('max')
    )
    coverage_video_pp['n_frames'] = coverage_video_pp['n_frames'].fillna(0)
    coverage_video_pp['frame_coverage'] = coverage_video_pp['frame_coverage'].fillna(0)

    coverage_video_pp.to_csv(
        os.path.join(output, 'grouped_syllable_frame_coverage_per_video_cluster_pre_post.csv'),
        index=False
    )

    coverage_pp_summary = (
        coverage_video_pp.groupby(['syllable_group', 'cluster', 'period'], as_index=False)
        .agg(
            mean_frame_coverage=('frame_coverage', 'mean'),
            sd_frame_coverage=('frame_coverage', 'std'),
            n_videos=('video_id', 'nunique')
        )
    )

    coverage_pp_summary.to_csv(
        os.path.join(output, 'grouped_syllable_frame_coverage_pre_post_summary.csv'),
        index=False
    )

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    axes = axes.flatten()

    for i, sg in enumerate(group_order):
        ax = axes[i]
        sub = coverage_pp_summary[coverage_pp_summary['syllable_group'] == sg]

        pre = (
            sub[sub['period'] == 'pre']
            .set_index('cluster')
            .reindex(cluster_order)
            .reset_index()
        )
        post = (
            sub[sub['period'] == 'post']
            .set_index('cluster')
            .reindex(cluster_order)
            .reset_index()
        )

        x = np.arange(len(cluster_order))
        w = 0.4

        pre_y = pre['mean_frame_coverage'].fillna(0).to_numpy()
        pre_err = pre['sd_frame_coverage'].fillna(0).to_numpy()

        post_y = post['mean_frame_coverage'].fillna(0).to_numpy()
        post_err = post['sd_frame_coverage'].fillna(0).to_numpy()

        ax.bar(x - w/2, pre_y, width=w, yerr=pre_err, capsize=3, label='pre')
        ax.bar(x + w/2, post_y, width=w, yerr=post_err, capsize=3, label='post')

        ax.set_title(str(sg), fontsize=10)
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Coverage")
        ax.set_xticks(x)
        ax.set_xticklabels(cluster_order, rotation=90, fontsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')

    plt.tight_layout()
    plt.savefig(
        os.path.join(output, "grouped_syllable_frame_coverage_pre_post_per_cluster.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # --------------------------------------------------
    # 7. stats for pre/post separately using video-level replicates
    # --------------------------------------------------
    stats_rows_pp = []

    for period in ['pre', 'post']:
        sub_period = coverage_video_pp[coverage_video_pp['period'] == period].copy()

        for sg in group_order:
            sub = sub_period[sub_period['syllable_group'] == sg].copy()

            if sub.empty or sub['cluster'].nunique() < 2:
                stats_rows_pp.append({
                    'period': period,
                    'syllable_group': sg,
                    'test': 'kruskal',
                    'statistic': np.nan,
                    'p_value': np.nan
                })
                continue

            samples = [
                g['frame_coverage'].dropna().values
                for _, g in sub.groupby('cluster')
            ]
            samples = [x for x in samples if len(x) > 0]

            if len(samples) < 2:
                stats_rows_pp.append({
                    'period': period,
                    'syllable_group': sg,
                    'test': 'kruskal',
                    'statistic': np.nan,
                    'p_value': np.nan
                })
                continue

            stat_out, p_out = kruskal(*samples)

            stats_rows_pp.append({
                'period': period,
                'syllable_group': sg,
                'test': 'kruskal',
                'statistic': stat_out,
                'p_value': p_out
            })

    stats_pp_df = pd.DataFrame(stats_rows_pp)
    stats_pp_df['p_fdr'] = np.nan
    stats_pp_df['significant_fdr_0.05'] = False

    for period in ['pre', 'post']:
        mask = (stats_pp_df['period'] == period) & stats_pp_df['p_value'].notna()
        if mask.sum() > 0:
            reject, p_fdr, _, _ = multipletests(
                stats_pp_df.loc[mask, 'p_value'],
                method='fdr_bh'
            )
            stats_pp_df.loc[mask, 'p_fdr'] = p_fdr
            stats_pp_df.loc[mask, 'significant_fdr_0.05'] = reject

    stats_pp_df.to_csv(
        os.path.join(output, 'grouped_syllable_frame_coverage_stats_pre_post.csv'),
        index=False
    )

    return merged, coverage_video, coverage_summary, stats_df, coverage_video_pp, coverage_pp_summary, stats_pp_df





def interaction_syllable_frequency(interactions, cluster, moseq, stat, output):
    """
    Grouped syllable ONSET frequency within interaction clusters.

    Correct orientation:
        - subplot = cluster
        - x-axis = grouped syllable
        - y-axis = relative frequency

    Also makes the same plot split by pre/post:
        pre  = Normalized Frame < 0
        post = Normalized Frame >= 0

    Replicates are VIDEO within each cluster, so error bars come from
    variation across videos, not pooled raw counts.
    """
    from scipy.stats import chi2_contingency
    from statsmodels.stats.multitest import multipletests

    os.makedirs(output, exist_ok=True)

    # --------------------------------------------------
    # 1. merge interaction clusters
    # --------------------------------------------------
    interactions = pd.merge(
        interactions,
        cluster[['interaction_id', 'Yhat.idt.pca']],
        on='interaction_id',
        how='inner'
    ).copy()

    interactions['video_id'] = interactions['file'].str.replace('.mp4', '', regex=False)

    keep = [
        'file', 'video_id', 'Frame', 'Interaction Number',
        'Normalized Frame', 'Interaction Pair', 'Yhat.idt.pca'
    ]

    interactions['Interaction Pair'] = interactions['Interaction Pair'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    interaction_tracks = (
        interactions[keep]
        .assign(track=interactions['Interaction Pair'])
        .explode('track')
        .drop(columns='Interaction Pair')
        .reset_index(drop=True)
    )
    interaction_tracks['track'] = interaction_tracks['track'].astype('Int64')

    print("Merged interactions with cluster data and exploded tracks")

    # --------------------------------------------------
    # 2. prep moseq exactly like before
    # --------------------------------------------------
    moseq = moseq.sort_values(['name', 'frame_index']).copy()
    moseq['bout_id'] = moseq.groupby('name')['onset'].cumsum()

    bout_lengths = moseq.groupby(['name', 'bout_id']).size()
    good = bout_lengths[bout_lengths >= 2].reset_index()[['name', 'bout_id']]
    moseq = moseq.merge(good, on=['name', 'bout_id'], how='inner')

    # keep only "good" syllables exactly like your older code
    good_syllables = stat['syllable'].unique()
    moseq = moseq[moseq['syllable'].isin(good_syllables)].copy()

    def extract_track(name):
        match = re.search(r'track(\d+)', name)
        return int(match.group(1)) if match else 0

    def extract_video_id(name):
        match = re.search(r'_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_td\d+)', name)
        return match.group(1) if match else None

    moseq['video_id'] = moseq['name'].apply(extract_video_id)
    moseq['track'] = moseq['name'].apply(extract_track).astype('Int64')
    moseq = moseq.rename(columns={'frame_index': 'Frame'})

    # --------------------------------------------------
    # 3. merge interaction rows with moseq syllables
    # --------------------------------------------------
    merged = interaction_tracks.merge(
        moseq[['video_id', 'Frame', 'track', 'syllable', 'onset']],
        on=['video_id', 'Frame', 'track'],
        how='left'
    )

    merged['syllable'] = merged['syllable'].astype('Int64')
    merged = merged.rename(columns={'Yhat.idt.pca': 'cluster'})
    merged['Normalized Frame'] = merged['Normalized Frame'].astype(int)
    merged['period'] = np.where(merged['Normalized Frame'] < 0, 'pre', 'post')
    merged['syllable_group'] = merged['syllable'].map(syll_to_group)

    merged.to_csv(
        os.path.join(output, 'interactions_with_grouped_syllables.csv'),
        index=False
    )
    print("Mapped grouped syllables onto interaction frames")

    # --------------------------------------------------
    # 4. onset rows only
    # --------------------------------------------------
    onsets = merged[merged['onset'] == True].copy()
    onsets = onsets.dropna(subset=['syllable_group', 'cluster', 'video_id']).copy()

    group_order = list(syllable_groups.keys())
    cluster_order = sorted(onsets['cluster'].dropna().unique())

    onsets['syllable_group'] = pd.Categorical(
        onsets['syllable_group'],
        categories=group_order,
        ordered=True
    )

    # --------------------------------------------------
    # 5. per VIDEO within CLUSTER frequencies
    # --------------------------------------------------
    counts_video = (
        onsets.groupby(['cluster', 'video_id', 'syllable_group'])
        .size()
        .reset_index(name='count')
    )

    totals_video = (
        onsets.groupby(['cluster', 'video_id'])
        .size()
        .reset_index(name='video_total_onsets')
    )

    freq_video = counts_video.merge(
        totals_video,
        on=['cluster', 'video_id'],
        how='left'
    )
    freq_video['relative_frequency'] = (
        freq_video['count'] / freq_video['video_total_onsets']
    )

    # complete missing syllable groups as zeros per cluster-video
    complete_index = pd.MultiIndex.from_product(
        [cluster_order, sorted(freq_video['video_id'].unique()), group_order],
        names=['cluster', 'video_id', 'syllable_group']
    )

    freq_video = (
        freq_video.set_index(['cluster', 'video_id', 'syllable_group'])
        .reindex(complete_index, fill_value=0)
        .reset_index()
    )

    # reattach totals for rows that were absent
    totals_video_lookup = totals_video.set_index(['cluster', 'video_id'])['video_total_onsets']
    freq_video['video_total_onsets'] = [
        totals_video_lookup.get((c, v), 0)
        for c, v in zip(freq_video['cluster'], freq_video['video_id'])
    ]
    freq_video['relative_frequency'] = np.where(
        freq_video['video_total_onsets'] > 0,
        freq_video['count'] / freq_video['video_total_onsets'],
        0
    )

    freq_video.to_csv(
        os.path.join(output, 'grouped_syllable_frequency_per_cluster_per_video.csv'),
        index=False
    )

    # summary for export
    freq_summary = (
        freq_video.groupby(['cluster', 'syllable_group'], as_index=False)
        .agg(
            mean_relative_frequency=('relative_frequency', 'mean'),
            sd_relative_frequency=('relative_frequency', 'std'),
            n_videos=('video_id', 'nunique')
        )
    )
    freq_summary.to_csv(
        os.path.join(output, 'grouped_syllable_frequency_per_cluster_summary.csv'),
        index=False
    )

    # --------------------------------------------------
    # 6. plot: subplot = cluster, x = grouped syllable
    # --------------------------------------------------
    n = len(cluster_order)
    ncols = 4
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3.5), sharey=True)
    axes = np.atleast_1d(axes).flatten()

    for i, cl in enumerate(cluster_order):
        ax = axes[i]
        sub = freq_video[freq_video['cluster'] == cl].copy()

        sns.barplot(
            data=sub,
            x='syllable_group',
            y='relative_frequency',
            order=group_order,
            errorbar='sd',
            ax=ax,
            color='steelblue'
        )

        ax.set_title(f"Cluster {cl}")
        ax.set_xlabel("")
        ax.set_ylabel("Relative onset frequency")
        ax.tick_params(axis='x', rotation=45)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(
        os.path.join(output, "grouped_syllable_frequency_clusters_as_subplots.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # --------------------------------------------------
    # 7. stats: grouped syllable association with cluster
    # --------------------------------------------------
    contingency = pd.crosstab(onsets['cluster'], onsets['syllable_group'])
    chi2, p, dof, expected = chi2_contingency(contingency)

    overall_stats = pd.DataFrame([{
        'chi2': chi2,
        'p_value': p,
        'dof': dof
    }])
    overall_stats.to_csv(
        os.path.join(output, 'grouped_syllable_cluster_association_overall.csv'),
        index=False
    )

    per_group_rows = []
    for sg in group_order:
        tmp = onsets.copy()
        tmp['is_this_group'] = tmp['syllable_group'] == sg
        cont = pd.crosstab(tmp['cluster'], tmp['is_this_group'])

        if cont.shape[0] < 2 or cont.shape[1] < 2:
            per_group_rows.append({
                'syllable_group': sg,
                'chi2': np.nan,
                'p_value': np.nan,
                'dof': np.nan
            })
            continue

        chi2_sg, p_sg, dof_sg, _ = chi2_contingency(cont)
        per_group_rows.append({
            'syllable_group': sg,
            'chi2': chi2_sg,
            'p_value': p_sg,
            'dof': dof_sg
        })

    stats_df = pd.DataFrame(per_group_rows)
    stats_df['p_fdr'] = np.nan
    stats_df['significant_fdr_0.05'] = False

    valid = stats_df['p_value'].notna()
    if valid.sum() > 0:
        reject, p_fdr, _, _ = multipletests(stats_df.loc[valid, 'p_value'], method='fdr_bh')
        stats_df.loc[valid, 'p_fdr'] = p_fdr
        stats_df.loc[valid, 'significant_fdr_0.05'] = reject

    stats_df.to_csv(
        os.path.join(output, 'grouped_syllable_cluster_association_per_group.csv'),
        index=False
    )

    # --------------------------------------------------
    # 8. pre/post per VIDEO within CLUSTER
    # --------------------------------------------------
    counts_video_pp = (
        onsets.groupby(['cluster', 'video_id', 'period', 'syllable_group'])
        .size()
        .reset_index(name='count')
    )

    totals_video_pp = (
        onsets.groupby(['cluster', 'video_id', 'period'])
        .size()
        .reset_index(name='video_period_total_onsets')
    )

    freq_video_pp = counts_video_pp.merge(
        totals_video_pp,
        on=['cluster', 'video_id', 'period'],
        how='left'
    )
    freq_video_pp['relative_frequency'] = (
        freq_video_pp['count'] / freq_video_pp['video_period_total_onsets']
    )

    complete_index_pp = pd.MultiIndex.from_product(
        [cluster_order, sorted(onsets['video_id'].unique()), ['pre', 'post'], group_order],
        names=['cluster', 'video_id', 'period', 'syllable_group']
    )

    freq_video_pp = (
        freq_video_pp.set_index(['cluster', 'video_id', 'period', 'syllable_group'])
        .reindex(complete_index_pp, fill_value=0)
        .reset_index()
    )

    totals_video_pp_lookup = totals_video_pp.set_index(['cluster', 'video_id', 'period'])['video_period_total_onsets']
    freq_video_pp['video_period_total_onsets'] = [
        totals_video_pp_lookup.get((c, v, p), 0)
        for c, v, p in zip(
            freq_video_pp['cluster'],
            freq_video_pp['video_id'],
            freq_video_pp['period']
        )
    ]
    freq_video_pp['relative_frequency'] = np.where(
        freq_video_pp['video_period_total_onsets'] > 0,
        freq_video_pp['count'] / freq_video_pp['video_period_total_onsets'],
        0
    )

    freq_video_pp.to_csv(
        os.path.join(output, 'grouped_syllable_frequency_per_cluster_pre_post_per_video.csv'),
        index=False
    )

    # --------------------------------------------------
    # 9. plot pre/post: subplot = cluster
    # --------------------------------------------------
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3.5), sharey=True)
    axes = np.atleast_1d(axes).flatten()

    for i, cl in enumerate(cluster_order):
        ax = axes[i]
        sub = freq_video_pp[freq_video_pp['cluster'] == cl].copy()

        sns.barplot(
            data=sub,
            x='syllable_group',
            y='relative_frequency',
            hue='period',
            order=group_order,
            errorbar='sd',
            ax=ax
        )

        ax.set_title(f"Cluster {cl}")
        ax.set_xlabel("")
        ax.set_ylabel("Relative onset frequency")
        ax.tick_params(axis='x', rotation=45)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    for ax in axes[:len(cluster_order)]:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

    plt.tight_layout()
    plt.savefig(
        os.path.join(output, "grouped_syllable_frequency_clusters_as_subplots_pre_post.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # --------------------------------------------------
    # 10. pre/post stats separately
    # --------------------------------------------------
    stats_rows_pp = []

    for period in ['pre', 'post']:
        sub_period = onsets[onsets['period'] == period].copy()
        if sub_period.empty:
            continue

        contingency_pp = pd.crosstab(sub_period['cluster'], sub_period['syllable_group'])
        if contingency_pp.shape[0] >= 2 and contingency_pp.shape[1] >= 2:
            chi2_pp, p_pp, dof_pp, _ = chi2_contingency(contingency_pp)
        else:
            chi2_pp, p_pp, dof_pp = np.nan, np.nan, np.nan

        stats_rows_pp.append({
            'period': period,
            'syllable_group': 'ALL',
            'chi2': chi2_pp,
            'p_value': p_pp,
            'dof': dof_pp
        })

        for sg in group_order:
            tmp = sub_period.copy()
            tmp['is_this_group'] = tmp['syllable_group'] == sg
            cont = pd.crosstab(tmp['cluster'], tmp['is_this_group'])

            if cont.shape[0] < 2 or cont.shape[1] < 2:
                stats_rows_pp.append({
                    'period': period,
                    'syllable_group': sg,
                    'chi2': np.nan,
                    'p_value': np.nan,
                    'dof': np.nan
                })
                continue

            chi2_sg, p_sg, dof_sg, _ = chi2_contingency(cont)
            stats_rows_pp.append({
                'period': period,
                'syllable_group': sg,
                'chi2': chi2_sg,
                'p_value': p_sg,
                'dof': dof_sg
            })

    stats_pp_df = pd.DataFrame(stats_rows_pp)
    stats_pp_df['p_fdr'] = np.nan
    stats_pp_df['significant_fdr_0.05'] = False

    for period in ['pre', 'post']:
        mask = (
            (stats_pp_df['period'] == period) &
            (stats_pp_df['syllable_group'] != 'ALL') &
            stats_pp_df['p_value'].notna()
        )
        if mask.sum() > 0:
            reject, p_fdr, _, _ = multipletests(
                stats_pp_df.loc[mask, 'p_value'],
                method='fdr_bh'
            )
            stats_pp_df.loc[mask, 'p_fdr'] = p_fdr
            stats_pp_df.loc[mask, 'significant_fdr_0.05'] = reject

    stats_pp_df.to_csv(
        os.path.join(output, 'grouped_syllable_cluster_association_pre_post.csv'),
        index=False
    )

    return merged, freq_video, stats_df, freq_video_pp, stats_pp_df


def interaction_syllable_frame_coverage(interactions, cluster, moseq, stat, output):
    """
    Same as interaction_syllable_frequency, but using grouped syllable FRAME COVERAGE
    instead of onset frequency.

    Correct orientation:
        - subplot = cluster
        - x-axis = grouped syllable
        - y-axis = relative frame coverage

    Also makes the same pre/post split.
    Replicates are VIDEO within each cluster, so error bars come from variation across videos.
    """
    from scipy.stats import chi2_contingency
    from statsmodels.stats.multitest import multipletests

    os.makedirs(output, exist_ok=True)

    # --------------------------------------------------
    # 1. merge interaction clusters
    # --------------------------------------------------
    interactions = pd.merge(
        interactions,
        cluster[['interaction_id', 'Yhat.idt.pca']],
        on='interaction_id',
        how='inner'
    ).copy()

    interactions['video_id'] = interactions['file'].str.replace('.mp4', '', regex=False)

    keep = [
        'file', 'video_id', 'Frame', 'Interaction Number',
        'Normalized Frame', 'Interaction Pair', 'Yhat.idt.pca'
    ]

    interactions['Interaction Pair'] = interactions['Interaction Pair'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    interaction_tracks = (
        interactions[keep]
        .assign(track=interactions['Interaction Pair'])
        .explode('track')
        .drop(columns='Interaction Pair')
        .reset_index(drop=True)
    )
    interaction_tracks['track'] = interaction_tracks['track'].astype('Int64')

    print("Merged interactions with cluster data and exploded tracks")

    # --------------------------------------------------
    # 2. prep moseq the same way as before
    # --------------------------------------------------
    moseq = moseq.sort_values(['name', 'frame_index']).copy()
    moseq['bout_id'] = moseq.groupby('name')['onset'].cumsum()

    bout_lengths = moseq.groupby(['name', 'bout_id']).size()
    good = bout_lengths[bout_lengths >= 2].reset_index()[['name', 'bout_id']]
    moseq = moseq.merge(good, on=['name', 'bout_id'], how='inner')

    good_syllables = stat['syllable'].unique()
    moseq = moseq[moseq['syllable'].isin(good_syllables)].copy()

    def extract_track(name):
        match = re.search(r'track(\d+)', name)
        return int(match.group(1)) if match else 0

    def extract_video_id(name):
        match = re.search(r'_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_td\d+)', name)
        return match.group(1) if match else None

    moseq['video_id'] = moseq['name'].apply(extract_video_id)
    moseq['track'] = moseq['name'].apply(extract_track).astype('Int64')
    moseq = moseq.rename(columns={'frame_index': 'Frame'})

    # --------------------------------------------------
    # 3. merge interaction rows with moseq syllables
    # --------------------------------------------------
    merged = interaction_tracks.merge(
        moseq[['video_id', 'Frame', 'track', 'syllable', 'onset']],
        on=['video_id', 'Frame', 'track'],
        how='left'
    )

    merged['syllable'] = merged['syllable'].astype('Int64')
    merged = merged.rename(columns={'Yhat.idt.pca': 'cluster'})
    merged['Normalized Frame'] = merged['Normalized Frame'].astype(int)
    merged['period'] = np.where(merged['Normalized Frame'] < 0, 'pre', 'post')
    merged['syllable_group'] = merged['syllable'].map(syll_to_group)

    merged.to_csv(
        os.path.join(output, 'interactions_with_grouped_syllables.csv'),
        index=False
    )
    print("Mapped grouped syllables onto interaction frames")

    # --------------------------------------------------
    # 4. all frames, not onsets
    # --------------------------------------------------
    frames = merged.dropna(subset=['syllable_group', 'cluster', 'video_id']).copy()

    group_order = list(syllable_groups.keys())
    cluster_order = sorted(frames['cluster'].dropna().unique())

    frames['syllable_group'] = pd.Categorical(
        frames['syllable_group'],
        categories=group_order,
        ordered=True
    )

    # --------------------------------------------------
    # 5. per VIDEO within CLUSTER frame coverage
    # --------------------------------------------------
    counts_video = (
        frames.groupby(['cluster', 'video_id', 'syllable_group'])
        .size()
        .reset_index(name='n_frames')
    )

    totals_video = (
        frames.groupby(['cluster', 'video_id'])
        .size()
        .reset_index(name='video_total_frames')
    )

    coverage_video = counts_video.merge(
        totals_video,
        on=['cluster', 'video_id'],
        how='left'
    )
    coverage_video['frame_coverage'] = (
        coverage_video['n_frames'] / coverage_video['video_total_frames']
    )

    complete_index = pd.MultiIndex.from_product(
        [cluster_order, sorted(frames['video_id'].unique()), group_order],
        names=['cluster', 'video_id', 'syllable_group']
    )

    coverage_video = (
        coverage_video.set_index(['cluster', 'video_id', 'syllable_group'])
        .reindex(complete_index, fill_value=0)
        .reset_index()
    )

    totals_video_lookup = totals_video.set_index(['cluster', 'video_id'])['video_total_frames']
    coverage_video['video_total_frames'] = [
        totals_video_lookup.get((c, v), 0)
        for c, v in zip(coverage_video['cluster'], coverage_video['video_id'])
    ]
    coverage_video['frame_coverage'] = np.where(
        coverage_video['video_total_frames'] > 0,
        coverage_video['n_frames'] / coverage_video['video_total_frames'],
        0
    )

    coverage_video.to_csv(
        os.path.join(output, 'grouped_syllable_frame_coverage_per_cluster_per_video.csv'),
        index=False
    )

    coverage_summary = (
        coverage_video.groupby(['cluster', 'syllable_group'], as_index=False)
        .agg(
            mean_frame_coverage=('frame_coverage', 'mean'),
            sd_frame_coverage=('frame_coverage', 'std'),
            n_videos=('video_id', 'nunique')
        )
    )
    coverage_summary.to_csv(
        os.path.join(output, 'grouped_syllable_frame_coverage_per_cluster_summary.csv'),
        index=False
    )

    # --------------------------------------------------
    # 6. plot: subplot = cluster, x = grouped syllable
    # --------------------------------------------------
    n = len(cluster_order)
    ncols = 4
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3.5), sharey=True)
    axes = np.atleast_1d(axes).flatten()

    for i, cl in enumerate(cluster_order):
        ax = axes[i]
        sub = coverage_video[coverage_video['cluster'] == cl].copy()

        sns.barplot(
            data=sub,
            x='syllable_group',
            y='frame_coverage',
            order=group_order,
            errorbar='sd',
            ax=ax,
            color='steelblue'
        )

        ax.set_title(f"Cluster {cl}")
        ax.set_xlabel("")
        ax.set_ylabel("Relative frame coverage")
        ax.tick_params(axis='x', rotation=45)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(
        os.path.join(output, "grouped_syllable_frame_coverage_clusters_as_subplots.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # --------------------------------------------------
    # 7. stats: grouped syllable association with cluster
    # --------------------------------------------------
    contingency = pd.crosstab(frames['cluster'], frames['syllable_group'])
    chi2, p, dof, expected = chi2_contingency(contingency)

    overall_stats = pd.DataFrame([{
        'chi2': chi2,
        'p_value': p,
        'dof': dof
    }])
    overall_stats.to_csv(
        os.path.join(output, 'grouped_syllable_frame_coverage_cluster_association_overall.csv'),
        index=False
    )

    per_group_rows = []
    for sg in group_order:
        tmp = frames.copy()
        tmp['is_this_group'] = tmp['syllable_group'] == sg
        cont = pd.crosstab(tmp['cluster'], tmp['is_this_group'])

        if cont.shape[0] < 2 or cont.shape[1] < 2:
            per_group_rows.append({
                'syllable_group': sg,
                'chi2': np.nan,
                'p_value': np.nan,
                'dof': np.nan
            })
            continue

        chi2_sg, p_sg, dof_sg, _ = chi2_contingency(cont)
        per_group_rows.append({
            'syllable_group': sg,
            'chi2': chi2_sg,
            'p_value': p_sg,
            'dof': dof_sg
        })

    stats_df = pd.DataFrame(per_group_rows)
    stats_df['p_fdr'] = np.nan
    stats_df['significant_fdr_0.05'] = False

    valid = stats_df['p_value'].notna()
    if valid.sum() > 0:
        reject, p_fdr, _, _ = multipletests(stats_df.loc[valid, 'p_value'], method='fdr_bh')
        stats_df.loc[valid, 'p_fdr'] = p_fdr
        stats_df.loc[valid, 'significant_fdr_0.05'] = reject

    stats_df.to_csv(
        os.path.join(output, 'grouped_syllable_frame_coverage_cluster_association_per_group.csv'),
        index=False
    )

    # --------------------------------------------------
    # 8. pre/post per VIDEO within CLUSTER
    # --------------------------------------------------
    counts_video_pp = (
        frames.groupby(['cluster', 'video_id', 'period', 'syllable_group'])
        .size()
        .reset_index(name='n_frames')
    )

    totals_video_pp = (
        frames.groupby(['cluster', 'video_id', 'period'])
        .size()
        .reset_index(name='video_period_total_frames')
    )

    coverage_video_pp = counts_video_pp.merge(
        totals_video_pp,
        on=['cluster', 'video_id', 'period'],
        how='left'
    )
    coverage_video_pp['frame_coverage'] = (
        coverage_video_pp['n_frames'] / coverage_video_pp['video_period_total_frames']
    )

    complete_index_pp = pd.MultiIndex.from_product(
        [cluster_order, sorted(frames['video_id'].unique()), ['pre', 'post'], group_order],
        names=['cluster', 'video_id', 'period', 'syllable_group']
    )

    coverage_video_pp = (
        coverage_video_pp.set_index(['cluster', 'video_id', 'period', 'syllable_group'])
        .reindex(complete_index_pp, fill_value=0)
        .reset_index()
    )

    totals_video_pp_lookup = totals_video_pp.set_index(['cluster', 'video_id', 'period'])['video_period_total_frames']
    coverage_video_pp['video_period_total_frames'] = [
        totals_video_pp_lookup.get((c, v, p), 0)
        for c, v, p in zip(
            coverage_video_pp['cluster'],
            coverage_video_pp['video_id'],
            coverage_video_pp['period']
        )
    ]
    coverage_video_pp['frame_coverage'] = np.where(
        coverage_video_pp['video_period_total_frames'] > 0,
        coverage_video_pp['n_frames'] / coverage_video_pp['video_period_total_frames'],
        0
    )

    coverage_video_pp.to_csv(
        os.path.join(output, 'grouped_syllable_frame_coverage_per_cluster_pre_post_per_video.csv'),
        index=False
    )

    # --------------------------------------------------
    # 9. plot pre/post: subplot = cluster
    # --------------------------------------------------
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3.5), sharey=True)
    axes = np.atleast_1d(axes).flatten()

    for i, cl in enumerate(cluster_order):
        ax = axes[i]
        sub = coverage_video_pp[coverage_video_pp['cluster'] == cl].copy()

        sns.barplot(
            data=sub,
            x='syllable_group',
            y='frame_coverage',
            hue='period',
            order=group_order,
            errorbar='sd',
            ax=ax
        )

        ax.set_title(f"Cluster {cl}")
        ax.set_xlabel("")
        ax.set_ylabel("Relative frame coverage")
        ax.tick_params(axis='x', rotation=45)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    for ax in axes[:len(cluster_order)]:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

    plt.tight_layout()
    plt.savefig(
        os.path.join(output, "grouped_syllable_frame_coverage_clusters_as_subplots_pre_post.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # --------------------------------------------------
    # 10. pre/post stats separately
    # --------------------------------------------------
    stats_rows_pp = []

    for period in ['pre', 'post']:
        sub_period = frames[frames['period'] == period].copy()
        if sub_period.empty:
            continue

        contingency_pp = pd.crosstab(sub_period['cluster'], sub_period['syllable_group'])
        if contingency_pp.shape[0] >= 2 and contingency_pp.shape[1] >= 2:
            chi2_pp, p_pp, dof_pp, _ = chi2_contingency(contingency_pp)
        else:
            chi2_pp, p_pp, dof_pp = np.nan, np.nan, np.nan

        stats_rows_pp.append({
            'period': period,
            'syllable_group': 'ALL',
            'chi2': chi2_pp,
            'p_value': p_pp,
            'dof': dof_pp
        })

        for sg in group_order:
            tmp = sub_period.copy()
            tmp['is_this_group'] = tmp['syllable_group'] == sg
            cont = pd.crosstab(tmp['cluster'], tmp['is_this_group'])

            if cont.shape[0] < 2 or cont.shape[1] < 2:
                stats_rows_pp.append({
                    'period': period,
                    'syllable_group': sg,
                    'chi2': np.nan,
                    'p_value': np.nan,
                    'dof': np.nan
                })
                continue

            chi2_sg, p_sg, dof_sg, _ = chi2_contingency(cont)
            stats_rows_pp.append({
                'period': period,
                'syllable_group': sg,
                'chi2': chi2_sg,
                'p_value': p_sg,
                'dof': dof_sg
            })

    stats_pp_df = pd.DataFrame(stats_rows_pp)
    stats_pp_df['p_fdr'] = np.nan
    stats_pp_df['significant_fdr_0.05'] = False

    for period in ['pre', 'post']:
        mask = (
            (stats_pp_df['period'] == period) &
            (stats_pp_df['syllable_group'] != 'ALL') &
            stats_pp_df['p_value'].notna()
        )
        if mask.sum() > 0:
            reject, p_fdr, _, _ = multipletests(
                stats_pp_df.loc[mask, 'p_value'],
                method='fdr_bh'
            )
            stats_pp_df.loc[mask, 'p_fdr'] = p_fdr
            stats_pp_df.loc[mask, 'significant_fdr_0.05'] = reject

    stats_pp_df.to_csv(
        os.path.join(output, 'grouped_syllable_frame_coverage_cluster_association_pre_post.csv'),
        index=False
    )

    return merged, coverage_video, stats_df, coverage_video_pp, stats_pp_df



def syllable_proximity(df, stats_df, output):
    """
    Quantify how close grouped syllables occur to other larvae.

    For each onset:
    - keep only good bouts (>= 2 frames)
    - keep only good syllables from stats_df
    - map raw syllable -> grouped syllable
    - for that onset frame, compute distance to every other larva in the same video/frame

    Saves:
    - grouped_syllable_proximity_analysis.csv
    - grouped_syllable_proximity_onset_summary.csv
    - grouped_syllable_mean_proximity.png
    - grouped_syllable_min_proximity.png
    - grouped_syllable_mean_proximity_per_group.png
    - grouped_syllable_min_proximity_per_group.png
    """

    output = os.path.join(output, 'syllable_proximity')
    if not os.path.exists(output):
        os.makedirs(output)

    df = df.sort_values(['name', 'frame_index']).copy()

    # --------------------------------------------------
    # 1. good bouts only
    # --------------------------------------------------
    df['bout_num'] = df.groupby('name')['onset'].cumsum()
    df['bout_id'] = df.groupby(['name', 'bout_num']).ngroup()

    bout_lengths = df.groupby('bout_id').size()
    good_bouts = bout_lengths[bout_lengths >= 2].index
    df = df[df['bout_id'].isin(good_bouts)].copy()

    # --------------------------------------------------
    # 2. good syllables only
    # --------------------------------------------------
    good_syllables = stats_df['syllable'].unique()
    df = df[df['syllable'].isin(good_syllables)].copy()

    # --------------------------------------------------
    # 3. grouped syllables only
    # --------------------------------------------------
    df['syllable_group'] = df['syllable'].map(syll_to_group)
    df = df.dropna(subset=['syllable_group']).copy()

    # --------------------------------------------------
    # 4. metadata
    # --------------------------------------------------
    df['group'] = df['name'].str.split('_').str[0]
    df = df[df['frame_index'] <= 3600].copy()

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
            return None

    df['video_id'] = df['name'].apply(extract_video_id)
    df['track'] = df['name'].apply(extract_track)

    # --------------------------------------------------
    # 5. proximity at onset frames
    # --------------------------------------------------
    proximity_list = []

    for video_id, df_v in df.groupby('video_id'):
        df_v = df_v.sort_values('frame_index')

        # only onset rows
        df_onsets = df_v[df_v['onset'] == True].copy()

        for onset_row in df_onsets.itertuples():
            group = onset_row.group
            frame = onset_row.frame_index
            track = onset_row.track
            syllable = onset_row.syllable
            syllable_group = onset_row.syllable_group
            x = onset_row.centroid_x
            y = onset_row.centroid_y

            others = df_v[
                (df_v['frame_index'] == frame) &
                (df_v['track'] != track)
            ]

            for other in others.itertuples():
                d = np.hypot(other.centroid_x - x, other.centroid_y - y)

                proximity_list.append({
                    'video_id': video_id,
                    'group': group,
                    'frame_index': frame,
                    'track': track,
                    'syllable': syllable,
                    'syllable_group': syllable_group,
                    'track_other': other.track,
                    'syllable_other': other.syllable,
                    'syllable_group_other': other.syllable_group,
                    'distance': d,
                })

    proximal_df = pd.DataFrame(proximity_list)
    proximal_df.to_csv(
        os.path.join(output, 'grouped_syllable_proximity_analysis.csv'),
        index=False
    )

    # --------------------------------------------------
    # 6. summarise per onset
    # --------------------------------------------------
    onset_summary = (
        proximal_df
        .groupby(['group', 'video_id', 'frame_index', 'track'], as_index=False)
        .agg(
            syllable=('syllable', 'first'),
            syllable_group=('syllable_group', 'first'),
            mean_distance=('distance', 'mean'),
            min_distance=('distance', 'min')
        )
    )

    onset_summary.to_csv(
        os.path.join(output, 'grouped_syllable_proximity_onset_summary.csv'),
        index=False
    )

    # order grouped syllables nicely
    group_order = list(syllable_groups.keys())
    onset_summary['syllable_group'] = pd.Categorical(
        onset_summary['syllable_group'],
        categories=group_order,
        ordered=True
    )

    # --------------------------------------------------
    # 7. grouped syllable proximity plots
    # --------------------------------------------------
    plt.figure(figsize=(10, 6))
    sns.pointplot(
        data=onset_summary,
        x='syllable_group',
        y='mean_distance',
        order=group_order,
        errorbar='sd'
    )
    plt.title('Mean Distance to Other Larvae at Grouped Syllable Onset')
    plt.xlabel('Grouped syllable')
    plt.ylabel('Mean Distance (pixels)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(
        os.path.join(output, 'grouped_syllable_mean_proximity.png'),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.pointplot(
        data=onset_summary,
        x='syllable_group',
        y='min_distance',
        order=group_order,
        errorbar='sd'
    )
    plt.title('Minimum Distance to Other Larvae at Grouped Syllable Onset')
    plt.xlabel('Grouped syllable')
    plt.ylabel('Minimum Distance (pixels)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(
        os.path.join(output, 'grouped_syllable_min_proximity.png'),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.pointplot(
        data=onset_summary,
        x='syllable_group',
        y='mean_distance',
        hue='group',
        order=group_order,
        errorbar='sd'
    )
    plt.title('Mean Distance to Other Larvae at Grouped Syllable Onset')
    plt.xlabel('Grouped syllable')
    plt.ylabel('Mean Distance (pixels)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(
        os.path.join(output, 'grouped_syllable_mean_proximity_per_group.png'),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.pointplot(
        data=onset_summary,
        x='syllable_group',
        y='min_distance',
        hue='group',
        order=group_order,
        errorbar='sd'
    )
    plt.title('Minimum Distance to Other Larvae at Grouped Syllable Onset')
    plt.xlabel('Grouped syllable')
    plt.ylabel('Minimum Distance (pixels)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(
        os.path.join(output, 'grouped_syllable_min_proximity_per_group.png'),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()

    return proximal_df, onset_summary



def grouped_syllable_quantifications(df, directory):

    df = df.copy()
    output = os.path.join(directory, 'grouped_syllable_quantifications')
    os.makedirs(output, exist_ok=True)

    # keep only syllables that are in your grouped mapping
    df['syllable_group'] = df['syllable'].map(syll_to_group)
    df = df.dropna(subset=['syllable_group']).copy()

    # make bouts exactly as in your code
    # df = df.sort_values(['name', 'frame_index']).reset_index(drop=True)
    # df['bout_id'] = df.groupby('name')['onset'].cumsum()

    df = df.sort_values(['name', 'frame_index']).reset_index(drop=True)
    df['bout_num'] = df.groupby('name')['onset'].cumsum()
    df['bout_id'] = df.groupby(['name', 'bout_num']).ngroup()

    # # translate + rotate exactly as before
    # dfs = Parallel(n_jobs=-1)(
    #     delayed(translate_rotate_syllables)(g.copy())
    #     for _, g in df.groupby('name')
    # )
    # df = pd.concat(dfs, ignore_index=True)
    # print("Translated and rotated grouped syllables")

    df = df.sort_values(['bout_id', 'frame_index'])
    df['relative_frame'] = df.groupby('bout_id').cumcount()

    # remove 1-frame bouts
    bout_lengths = df.groupby('bout_id').size()
    good_bouts = bout_lengths[bout_lengths >= 2].index
    df = df[df['bout_id'].isin(good_bouts)].copy()

    # normalised frame
    df['normalised_frame'] = df.groupby('bout_id')['relative_frame'].transform(
        lambda x: x / x.max()
    )

    df['angular_velocity_abs'] = df['angular_velocity'].abs()

    hmin, hmax = -3, 3
    avmin, avmax = 0, 20
    vmin, vmax = 0, 500

    # --------------------------------------------------
    # 1. PLOT OVER RAW TIME
    # --------------------------------------------------
    for group_name, grouped in df.groupby('syllable_group'):

        grouped = grouped.copy()

        # only keep frames supported by enough bouts
        total_bouts = grouped['bout_id'].nunique()
        threshold_bouts = 0.05 * total_bouts
        grouped['n'] = grouped.groupby('relative_frame')['bout_id'].transform('nunique')
        mean_trace = grouped[grouped['n'] >= threshold_bouts].copy()

        fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)

        sns.lineplot(
            data=mean_trace,
            x='relative_frame',
            y='velocity_px_s',
            ax=axes[0]
        )
        axes[0].set_title(f"{group_name} velocity")
        axes[0].set_ylabel("Velocity")
        axes[0].set_xlabel("")
        axes[0].set_ylim(vmin, vmax)

        sns.lineplot(
            data=mean_trace,
            x='relative_frame',
            y='heading',
            ax=axes[1]
        )
        axes[1].set_title(f"{group_name} heading")
        axes[1].set_ylabel("Heading")
        axes[1].set_xlabel("")
        axes[1].set_ylim(hmin, hmax)

        sns.lineplot(
            data=mean_trace,
            x='relative_frame',
            y='angular_velocity_abs',
            ax=axes[2]
        )
        axes[2].set_title(f"{group_name} angular velocity")
        axes[2].set_ylabel("Angular velocity")
        axes[2].set_xlabel("Relative frame")
        axes[2].set_ylim(avmin, avmax)

        plt.tight_layout()
        plt.savefig(
            os.path.join(output, f'grouped_quantification_{group_name}.png'),
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()

    # --------------------------------------------------
    # 2. PLOT OVER NORMALISED TIME
    # --------------------------------------------------
    for group_name, grouped in df.groupby('syllable_group'):

        grouped = grouped.copy()

        n_bins = 20
        grouped['norm_bin'] = (grouped['normalised_frame'] * (n_bins - 1)).round().astype(int)

        total_bouts = grouped['bout_id'].nunique()
        threshold_bouts = 0.05 * total_bouts

        grouped['n_norm'] = grouped.groupby('norm_bin')['bout_id'].transform('nunique')
        mean_trace_norm = grouped[grouped['n_norm'] >= threshold_bouts].copy()
        mean_trace_norm['norm_t'] = mean_trace_norm['norm_bin'] / (n_bins - 1)

        fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)

        sns.lineplot(
            data=mean_trace_norm,
            x='norm_t',
            y='velocity_px_s',
            ax=axes[0]
        )
        axes[0].set_title(f"{group_name} velocity")
        axes[0].set_ylabel("Velocity")
        axes[0].set_xlabel("")
        axes[0].set_ylim(vmin, vmax)

        sns.lineplot(
            data=mean_trace_norm,
            x='norm_t',
            y='heading',
            ax=axes[1]
        )
        axes[1].set_title(f"{group_name} heading")
        axes[1].set_ylabel("Heading")
        axes[1].set_xlabel("")
        axes[1].set_ylim(hmin, hmax)

        sns.lineplot(
            data=mean_trace_norm,
            x='norm_t',
            y='angular_velocity_abs',
            ax=axes[2]
        )
        axes[2].set_title(f"{group_name} angular velocity")
        axes[2].set_ylabel("Angular velocity")
        axes[2].set_xlabel("Normalised frame")
        axes[2].set_ylim(avmin, avmax)

        plt.tight_layout()
        plt.savefig(
            os.path.join(output, f'grouped_quantification_normalised_{group_name}.png'),
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()

    print("Finished grouped syllable quantifications")



def plot_raw_grouped(df, output_directory, n_examples=100, ncols=10, random_state=42):
    """
    For each grouped syllable, plot up to n_examples aligned raw bouts in a grid.
    Only bouts with duration >= 2 frames are included.

    Saves one PDF per grouped syllable into:
        <output_directory>/raw_grouped/<group_name>.pdf
    """

    os.makedirs(output_directory, exist_ok=True)
    raw_output = os.path.join(output_directory, "raw_grouped")
    os.makedirs(raw_output, exist_ok=True)

    df = df.sort_values(["name", "frame_index"]).copy()

    # map raw syllables -> grouped syllables
    df["syllable_group"] = df["syllable"].map(syll_to_group)
    df = df.dropna(subset=["syllable_group"]).copy()

    # bout ids within each recording
    df["bout_num"] = df.groupby("name")["onset"].cumsum()
    df["bout_id"] = df.groupby(["name", "bout_num"]).ngroup()

    # keep only bouts longer than 1 frame
    bout_lengths = df.groupby("bout_id").size()
    good_bouts = bout_lengths[bout_lengths >= 3].index
    df = df[df["bout_id"].isin(good_bouts)].copy()

    # align all bouts exactly like the other functions
    translated_df = translate_rotate_syllables(df)

    rng = np.random.default_rng(random_state)
    nrows = int(np.ceil(n_examples / ncols))

    group_order = list(syllable_groups.keys())

    for group_name in group_order:
        g_syl = translated_df[translated_df["syllable_group"] == group_name].copy()

        if g_syl.empty:
            continue

        bout_ids = g_syl["bout_id"].dropna().unique()
        if len(bout_ids) == 0:
            continue

        n_take = min(n_examples, len(bout_ids))
        sampled_bout_ids = rng.choice(bout_ids, size=n_take, replace=False)

        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
        axes = np.atleast_1d(axes).flatten()

        # consistent axis limits across all sampled bouts for this grouped syllable
        g_plot = g_syl[g_syl["bout_id"].isin(sampled_bout_ids)].copy()
        xmin = g_plot["rotated_centroid_x"].min()
        xmax = g_plot["rotated_centroid_x"].max()
        ymin = g_plot["rotated_centroid_y"].min()
        ymax = g_plot["rotated_centroid_y"].max()

        xpad = (xmax - xmin) * 0.05 if xmax > xmin else 1
        ypad = (ymax - ymin) * 0.05 if ymax > ymin else 1

        for ax, bout_id in zip(axes, sampled_bout_ids):
            g_bout = g_syl[g_syl["bout_id"] == bout_id].sort_values("frame_index")

            ax.plot(
                g_bout["rotated_centroid_x"],
                g_bout["rotated_centroid_y"],
                color="gray",
                linewidth=1.0
            )
            ax.scatter([0], [0], s=8, color="black", zorder=10)

            ax.set_aspect("equal", "box")
            ax.set_xlim(xmin - xpad, xmax + xpad)
            ax.set_ylim(ymin - ypad, ymax + ypad)
            ax.set_xticks([])
            ax.set_yticks([])

        for ax in axes[n_take:]:
            ax.axis("off")

        fig.suptitle(f"{group_name}", y=0.995)
        plt.tight_layout()

        out_pdf = os.path.join(raw_output, f"{group_name}.pdf")
        fig.savefig(out_pdf, bbox_inches="tight")
        plt.close(fig)




def syllable_nearest_neighbour_histogram(df, stats_df, output, bins=30):
    """
    For grouped syllable onsets only:
    - keep good bouts (>= 2 frames)
    - keep good syllables from stats_df
    - map raw syllables to grouped syllables
    - at each onset, compute distance to the nearest other larva in the same frame
    - plot histogram(s) of nearest-neighbour distance

    Saves:
    - grouped_syllable_nearest_neighbour_per_onset.csv
    - grouped_syllable_nearest_neighbour_histogram_all.png
    - grouped_syllable_nearest_neighbour_histogram_stacked.png
    - grouped_syllable_nearest_neighbour_histogram_per_group.png
    """

    output = os.path.join(output, 'syllable_nearest_neighbour')
    os.makedirs(output, exist_ok=True)

    df = df.sort_values(['name', 'frame_index']).copy()

    # --------------------------------------------------
    # 1. good bouts only
    # --------------------------------------------------
    df['bout_num'] = df.groupby('name')['onset'].cumsum()
    df['bout_id'] = df.groupby(['name', 'bout_num']).ngroup()

    bout_lengths = df.groupby('bout_id').size()
    good_bouts = bout_lengths[bout_lengths >= 2].index
    df = df[df['bout_id'].isin(good_bouts)].copy()

    # --------------------------------------------------
    # 2. good syllables only
    # --------------------------------------------------
    good_syllables = stats_df['syllable'].unique()
    df = df[df['syllable'].isin(good_syllables)].copy()

    # --------------------------------------------------
    # 3. grouped syllables only
    # --------------------------------------------------
    df['syllable_group'] = df['syllable'].map(syll_to_group)
    df = df.dropna(subset=['syllable_group']).copy()

    # --------------------------------------------------
    # 4. metadata
    # --------------------------------------------------
    df['group'] = df['name'].str.split('_').str[0]
    df = df[df['frame_index'] <= 3600].copy()

    def extract_track(name):
        match = re.search(r'track(\d+)', name)
        return int(match.group(1)) if match else 0

    def extract_video_id(name):
        match = re.search(r'_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_td\d+)', name)
        return match.group(1) if match else None

    df['video_id'] = df['name'].apply(extract_video_id)
    df['track'] = df['name'].apply(extract_track)

    # --------------------------------------------------
    # 5. nearest-neighbour distance at onset only
    # --------------------------------------------------
    nearest_rows = []

    for video_id, df_v in df.groupby('video_id'):
        df_v = df_v.sort_values('frame_index')

        df_onsets = df_v[df_v['onset'] == True].copy()

        for onset_row in df_onsets.itertuples():
            group = onset_row.group
            frame = onset_row.frame_index
            track = onset_row.track
            syllable = onset_row.syllable
            syllable_group = onset_row.syllable_group
            x = onset_row.centroid_x
            y = onset_row.centroid_y

            others = df_v[
                (df_v['frame_index'] == frame) &
                (df_v['track'] != track)
            ].copy()

            if others.empty:
                continue

            others['distance'] = np.hypot(others['centroid_x'] - x, others['centroid_y'] - y)

            nearest = others.loc[others['distance'].idxmin()]

            nearest_rows.append({
                'video_id': video_id,
                'group': group,
                'frame_index': frame,
                'track': track,
                'syllable': syllable,
                'syllable_group': syllable_group,
                'nearest_track': nearest['track'],
                'nearest_syllable': nearest['syllable'],
                'nearest_syllable_group': nearest['syllable_group'],
                'nearest_distance': nearest['distance'],
            })

    nearest_df = pd.DataFrame(nearest_rows)

    nearest_df.to_csv(
        os.path.join(output, 'grouped_syllable_nearest_neighbour_per_onset.csv'),
        index=False
    )

    if nearest_df.empty:
        print("No nearest-neighbour data found.")
        return nearest_df

    group_order = list(syllable_groups.keys())
    nearest_df['syllable_group'] = pd.Categorical(
        nearest_df['syllable_group'],
        categories=group_order,
        ordered=True
    )

    # --------------------------------------------------
    # 6. overall histogram of nearest-neighbour distance
    # --------------------------------------------------
    plt.figure(figsize=(8, 6))
    sns.histplot(
    data=nearest_df,
    x='nearest_distance',
    hue='syllable_group',
    hue_order=group_order,
    bins=100,
    stat='density',
    common_norm=False,
    element='step',
    fill=False
)
    plt.title('Nearest-neighbour distance at grouped syllable onset')
    plt.xlabel('Distance to nearest other larva (pixels)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(
        os.path.join(output, 'grouped_syllable_nearest_neighbour_histogram_all.png'),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()

    # --------------------------------------------------
    # 7. stacked histogram by grouped syllable
    # --------------------------------------------------
    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=nearest_df,
        x='nearest_distance',
        hue='syllable_group',
        bins=bins,
        multiple='stack',
        hue_order=group_order, stat="density",
    common_norm=False
    )
    plt.title('Nearest-neighbour distance at onset, stacked by grouped syllable')
    plt.xlabel('Distance to nearest other larva (pixels)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(
        os.path.join(output, 'grouped_syllable_nearest_neighbour_histogram_stacked.png'),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()

    # --------------------------------------------------
    # 8. one panel per grouped syllable
    # --------------------------------------------------
    n = len(group_order)
    ncols = 4
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).flatten()

    for i, sg in enumerate(group_order):
        ax = axes[i]
        sub = nearest_df[nearest_df['syllable_group'] == sg].copy()

        if not sub.empty:
            sns.histplot(
                data=sub,
                x='nearest_distance',
                bins=bins,
                stat="density",
                ax=ax
            )

        ax.set_title(str(sg), fontsize=10)
        ax.set_xlabel('Distance')
        ax.set_ylabel('Count')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(
        os.path.join(output, 'grouped_syllable_nearest_neighbour_histogram_per_group.png'),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()

    return nearest_df




###########################################
# 3. RUN PIPELINE FOR ANALYSIS OF SYLLABLES 
###########################################


""" PIPELINE TO RUN ANALYSIS ON STATS DF """

df_stats = pd.read_csv('/Users/cochral/Desktop/MOSEQ2/KEYPOINT-KAPPA1000/stats_df.csv')
output = '/Users/cochral/Desktop/MOSEQ2/KEYPOINT-KAPPA1000/plots'

# stats_df(df_stats, output)
# grouped_syllable_frequencies(df_stats, output)



""" PIPELINE TO RUN ANALYSIS ON MOSEQ DF """

df_moseq = pd.read_csv('/Users/cochral/Desktop/MOSEQ2/KEYPOINT-KAPPA1000/moseq_df.csv')
# output = '/Users/cochral/Desktop/MOSEQ2/KEYPOINT-KAPPA1000/plots'
# plot_raw(df_moseq, output, n_examples=100)

# output = '/Users/cochral/Desktop/MOSEQ2/KEYPOINT-KAPPA1000/plots'
# fraction_one_frame_duration(df_moseq, output)

# output = '/Users/cochral/Desktop/MOSEQ2/KEYPOINT-KAPPA1000/plots/traces_syllable'
# syllables_with_traces(df_moseq, output)

# # output = '/Users/cochral/Desktop/MOSEQ2/KEYPOINT-KAPPA1000/plots/mean_syllable'
# # syllables_just_mean(df_moseq, output)

# syllable_folder = '/Users/cochral/Desktop/MOSEQ2/KEYPOINT-KAPPA1000/plots/traces_syllable'
# syllable_order = [1, 25, 0, 20, 2, 23, 4, 19, 7, 12, 29, 15, 32, 8, 26, 28, 21, 31, 13, 9, 24, 17, 11, 14, 6, 16, 30, 5, 22, 18, 27, 3, 10]

# plot_syllable_row(syllable_folder, syllable_order)

# output = '/Users/cochral/Desktop/MOSEQ2/KEYPOINT-KAPPA1000/plots/grouped_traces'

# grouped_syllables_with_traces(df_moseq, output)


# output = '/Users/cochral/Desktop/MOSEQ2/KEYPOINT-KAPPA1000/plots/ethograms'
# grouped_ethogram_by_condition(df_moseq, output, max_frame=3600)




# interactions = pd.read_csv('/Users/cochral/Desktop/MOSEQ2/KEYPOINT-KAPPA1000/plots/interactions/cropped_interactions.csv')
# cluster = pd.read_csv('/Users/cochral/Desktop/MOSEQ2/KEYPOINT-KAPPA1000/plots/interactions/pca-data2-F18.csv')
# moseq = pd.read_csv('/Users/cochral/Desktop/MOSEQ2/KEYPOINT-KAPPA1000/moseq_df.csv')
# stat = pd.read_csv('/Users/cochral/Desktop/MOSEQ2/KEYPOINT-KAPPA1000/stats_df.csv')
# output = '/Users/cochral/Desktop/MOSEQ2/KEYPOINT-KAPPA1000/plots/interactions/grouped-ethograms'

# os.makedirs(output, exist_ok=True)

# interaction_grouped_ethogram(interactions, cluster, moseq, stat, output)


# interactions = pd.read_csv('/Users/cochral/Desktop/MOSEQ2/KEYPOINT-KAPPA1000/plots/interactions/cropped_interactions.csv')
# cluster = pd.read_csv('/Users/cochral/Desktop/MOSEQ2/KEYPOINT-KAPPA1000/plots/interactions/pca-data2-F18.csv')
# moseq = pd.read_csv('/Users/cochral/Desktop/MOSEQ2/KEYPOINT-KAPPA1000/moseq_df.csv')
# stat = pd.read_csv('/Users/cochral/Desktop/MOSEQ2/KEYPOINT-KAPPA1000/stats_df.csv')
# output = '/Users/cochral/Desktop/MOSEQ2/KEYPOINT-KAPPA1000/plots/interactions/partner_ethograms'

# partner_interaction_ethogram(interactions, cluster, moseq, stat, output)






# interactions = pd.read_csv('/Users/cochral/Desktop/MOSEQ2/KEYPOINT-KAPPA1000/plots/interactions/cropped_interactions.csv')
# cluster = pd.read_csv('/Users/cochral/Desktop/MOSEQ2/KEYPOINT-KAPPA1000/plots/interactions/pca-data2-F18.csv')
# moseq = pd.read_csv('/Users/cochral/Desktop/MOSEQ2/KEYPOINT-KAPPA1000/moseq_df.csv')
# stat = pd.read_csv('/Users/cochral/Desktop/MOSEQ2/KEYPOINT-KAPPA1000/stats_df.csv')
# output = '/Users/cochral/Desktop/MOSEQ2/KEYPOINT-KAPPA1000/plots/interactions/grouped_syllable_frequency'

# interaction_syllable_frequency(interactions, cluster, moseq, stat, output)



# interactions = pd.read_csv('/Users/cochral/Desktop/MOSEQ2/KEYPOINT-KAPPA1000/plots/interactions/cropped_interactions.csv')
# cluster = pd.read_csv('/Users/cochral/Desktop/MOSEQ2/KEYPOINT-KAPPA1000/plots/interactions/pca-data2-F18.csv')
# moseq = pd.read_csv('/Users/cochral/Desktop/MOSEQ2/KEYPOINT-KAPPA1000/moseq_df.csv')
# stat = pd.read_csv('/Users/cochral/Desktop/MOSEQ2/KEYPOINT-KAPPA1000/stats_df.csv')
# output = '/Users/cochral/Desktop/MOSEQ2/KEYPOINT-KAPPA1000/plots/interactions/grouped_syllable_frame_coverage'

# interaction_syllable_frame_coverage(interactions, cluster, moseq, stat, output)


# df = pd.read_csv('/Users/cochral/Desktop/MOSEQ2/KEYPOINT-KAPPA1000/moseq_df.csv')
# stats_df = pd.read_csv('/Users/cochral/Desktop/MOSEQ2/KEYPOINT-KAPPA1000/stats_df.csv')
# output = '/Users/cochral/Desktop/MOSEQ2/KEYPOINT-KAPPA1000/plots'

# proximal_df, onset_summary = syllable_proximity(df, stats_df, output)


# df = pd.read_csv('/Users/cochral/Desktop/MOSEQ2/KEYPOINT-KAPPA1000/moseq_df.csv')
# directory = '/Users/cochral/Desktop/MOSEQ2/KEYPOINT-KAPPA1000/plots'

# grouped_syllable_quantifications(df, directory)


# df_moseq = pd.read_csv('/Users/cochral/Desktop/MOSEQ2/KEYPOINT-KAPPA1000/moseq_df.csv')
# output = '/Users/cochral/Desktop/MOSEQ2/KEYPOINT-KAPPA1000/plots'
# plot_raw_grouped(df_moseq, output, n_examples=100)


df = pd.read_csv('/Users/cochral/Desktop/MOSEQ2/KEYPOINT-KAPPA1000/moseq_df.csv')
stats_df = pd.read_csv('/Users/cochral/Desktop/MOSEQ2/KEYPOINT-KAPPA1000/stats_df.csv')
output = '/Users/cochral/Desktop/MOSEQ2/KEYPOINT-KAPPA1000/plots'

nearest_df = syllable_nearest_neighbour_histogram(df, stats_df, output, bins=30)