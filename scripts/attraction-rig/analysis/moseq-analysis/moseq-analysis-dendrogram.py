import sys
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pyarrow.feather as feather
from matplotlib.collections import LineCollection
import matplotlib.image as mpimg
from PIL import Image

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

        out_png = os.path.join(output_directory, f"syllable_{syll}.png")
        fig.savefig(out_png, dpi=300, bbox_inches='tight')
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

        out_png = os.path.join(output_directory, f"syllable_{syll}.png")
        fig.savefig(out_png, dpi=900, bbox_inches='tight')
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
    fig.savefig(out_pdf, bbox_inches="tight")  # PDF
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




""" PLOT RAW SYLLABLES"""

df = pd.read_csv('/Users/cochral/Desktop/MOSEQ/KEYPOINT-KAPPA1500/moseq_df.csv')
output_directory = '/Users/cochral/Desktop/MOSEQ/KEYPOINT-KAPPA1500/plots/dendrogram/mean_syllable'
syllables_just_mean(df, output_directory)



""" PLOT SYLLABLES IN DESIRED ORDER """
syllable_folder = '/Users/cochral/Desktop/MOSEQ/KEYPOINT-KAPPA1500/plots/dendrogram/mean_syllable'
syllable_order = [2, 19, 33, 45, 43, 44, 28, 37, 6, 11, 1, 16, 39, 27, 40, 26, 25, 22, 30, 9, 23, 10, 24, 12, 17, 35, 29, 42, 5, 36, 13, 15, 4, 21, 3, 32, 0, 14, 38, 18, 20, 34, 8, 41, 7, 31]

plot_syllable_row(syllable_folder, syllable_order)


""" PLOT GIFS IN DESIRED ORDER """

# syllable_folder = '/Users/cochral/Desktop/MOSEQ/KEYPOINT-KAPPA1500/2025_08_20-12_22_20/trajectory_plots'
# syllable_order = [2, 19, 33, 45, 43, 44, 28, 37, 6, 11, 1, 16, 39, 27, 40, 26, 25, 22, 30, 9, 23, 10, 24, 12, 17, 35, 29, 42, 5, 36, 13, 15, 4, 21, 3, 32, 0, 14, 38, 18, 20, 34, 8, 41, 7, 31]

# plot_syllable_row_gif(syllable_folder, syllable_order)


[1, 25, 0, 20, 2, 23, 4, 19, 7, 12, 29, 15, 32, 8, 26, 28, 21, 31, 13, 9, 24, 17, 11, 14, 6, 16, 30, 5, 22, 18, 27, 3, 10]