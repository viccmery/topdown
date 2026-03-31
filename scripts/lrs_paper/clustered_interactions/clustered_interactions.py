import sys
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pyarrow.feather as feather
import cv2
import re
from scipy.spatial.distance import pdist
from shapely import wkt
import glob
from random import sample
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
import imageio.v2 as imageio
from scipy.stats import linregress
import matplotlib as mpl
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests
import ast
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.image as mpimg
import networkx as nx
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.stats import spearmanr
from scipy.stats import linregress



""" 
-------------------------------------
INTERACTION CLUSTER PIPELINE ANALYSIS 
-------------------------------------
"""

class ClusterPipeline:

    def __init__(self, directory, interactions, clusters, cluster_name, video_path):
        
        self.directory = directory
        self.interaction_path = interactions 
        self.cluster_path = clusters
        self.cluster_name = cluster_name
        self.video_path = video_path

        self.interactions = None
        self.clusters = None
        self.df = None
        

    ##########################################################################################################
    ## METHOD LOADING_DATA: LOAD AND MERGE DATAFRAMES
    ##########################################################################################################
    def loading_data(self):

        ## LOAD DATAFRAMES

        self.interactions = pd.read_csv(self.interaction_path)
        self.clusters = pd.read_csv(self.cluster_path)

        ## MISSING INTERACTIONS BETWEEN DATAFRAMES

        set1 = set(self.interactions['interaction_id'].unique())
        set2 = set(self.clusters['interaction_id'].unique())
        missing_from_cluster = sorted(set1 - set2)
        missing_from_cropped  = sorted(set2 - set1)
        print(f">>> {len(missing_from_cluster)} IDs in cropped not in cluster (e.g. {missing_from_cluster[:5]})")
        print(f">>> {len(missing_from_cropped)} IDs in cluster not in cropped (e.g. {missing_from_cropped[:5]})")

        ## MERGE DATAFRAMES

        self.df = pd.merge(
            self.interactions, 
            self.clusters[['interaction_id', self.cluster_name]], 
            on='interaction_id', 
            how='inner'
        )

    ##########################################################################################################
    ## METHOD ANCHOR_PARTNER: DEFINE ANCHOR AND PARTNER BASED ON LINEARITY
    ##########################################################################################################
    def anchor_partner(self):
        
        df = self.df

        ## CREATE ALIGNED AND PARTNER TRACKS FOR DRAWING TRAJECTORIES 

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
            idx = group.index[:len(A_al)]
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
        

        ## APPROACH ANGLE CALCULATION FLIPPED INITIALLY BY ACCIDENT
        df["track1_approach_angle"] = 180 - df["track1_approach_angle"]
        df["track2_approach_angle"] = 180 - df["track2_approach_angle"]

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

        
        self.df = df


    ##########################################################################################################
    ## METHOD SANITY_CHECK_ANCHOR_PARTNER: SANITY CHECK WHEREBY ANCHOR AND PARTNER REVERSED TO LINEARITY SCORE
    ##########################################################################################################
    def sanity_check_anchor_partner(self):
        
        df = self.df

        ## CREATE ALIGNED AND PARTNER TRACKS FOR DRAWING TRAJECTORIES 

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
                winner = 2
                anchor_pts, partner_pts, anchor_axis = coords2, coords1, axis2
            else:
                winner = 1
                anchor_pts, partner_pts, anchor_axis = coords1, coords2, axis1

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
            idx = group.index[:len(A_al)]
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

        
        self.df = df


    ##########################################################################################################
    ## METHOD BARPLOT_PROPORTION: PROPORTION OF INTERACTIONS IN EACH CLUSTER
    ##########################################################################################################
    def barplot_proportion(self):

        df = self.clusters
        df['condition'] = df['condition'].replace({'iso': 'SI', 'group': 'GH'})

        cluster_name = self.cluster_name

        mpl.rcParams['pdf.fonttype'] = 42
        mpl.rcParams['ps.fonttype'] = 42

        mpl.rcParams['font.family'] = 'sans-serif'
        mpl.rcParams['font.sans-serif'] = ['Arial']

        PALETTE = {
            "GH": 'steelblue',     
            "SI": 'darkorange',}

        HUE_ORDER = ["GH", "SI"]

        # Collapse to unique interactions (to avoid overweighting frames)
        inter_per_video = (
            df[['file', 'condition', 'interaction_id', cluster_name]]
            .drop_duplicates(subset=['file', 'interaction_id'])
        )

        # Count interactions per (video, condition, cluster)
        counts = (
            inter_per_video
            .groupby(['file', 'condition', cluster_name])
            .size()
            .reset_index(name='count')
        )

        # Compute total interactions per video (for normalization)
        totals = (
            counts.groupby('file')['count']
            .transform('sum')
        )

        # Add proportion column
        counts['proportion'] = counts['count'] / totals
        
        summary_df = (
            counts
            .set_index(['file', 'condition', cluster_name])
            .unstack(fill_value=0)
            .stack()
            .reset_index()
        )
        summary_df.rename(columns={0: 'proportion'}, inplace=True)

        # Now you can plot with seaborn
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(
            data=summary_df,
            x=cluster_name, y='proportion', hue='condition',ci='sd', alpha=0.8, palette=PALETTE, hue_order=HUE_ORDER, edgecolor='black', linewidth=2)

        plt.title("Proportion of Clusters")
        plt.xlabel("Cluster")
        plt.ylabel("Proportion")
        # plt.xticks(rotation=90)
        sns.despine()
        ax.legend(frameon=False, title=None, fontsize=11, loc="upper right")
        plt.tight_layout()

        output = os.path.join(self.directory, 'cluster_proportions.pdf')
        plt.savefig(output, format='pdf', bbox_inches='tight')
        plt.close()


    ##########################################################################################################
    ## METHOD BARPLOT_DEVIATION: DEVIATION FROM EXPECTED FOR EACH EACH CLUSTER
    ##########################################################################################################
    def barplot_deviation(self):

        df = self.clusters
        df['condition'] = df['condition'].replace({'iso': 'SI', 'group': 'GH'})
        df_interaction = self.df
        cluster_name = self.cluster_name

        mpl.rcParams['pdf.fonttype'] = 42
        mpl.rcParams['ps.fonttype'] = 42

        mpl.rcParams['font.family'] = 'sans-serif'
        mpl.rcParams['font.sans-serif'] = ['Arial']

        PALETTE = {
            "GH": 'steelblue',     
            "SI": 'darkorange',}

        HUE_ORDER = ["GH", "SI"]

        ### OBSERVED - EXPECTED DEVIATION 

        cluster_counts = (df.groupby([cluster_name, 'condition']).size().unstack(fill_value=0).reindex(columns=['GH', 'SI'], fill_value=0))  # count number per cluster per condition

        total_group = cluster_counts['GH'].sum()
        total_iso   = cluster_counts['SI'].sum()
        total_all   = total_group + total_iso

        expected_group = total_group / total_all   # e.g., ~0.56

        observed_group_frac = cluster_counts['GH'] / (cluster_counts['GH'] + cluster_counts['SI']).replace({0: np.nan}) ## observed fraction
        observed_group_frac = observed_group_frac.fillna(0.0)

        deviation = observed_group_frac - expected_group ## expected fraction

        deviation_sorted = deviation.sort_values()
        colors = ['C1' if val < 0 else 'C0' for val in deviation_sorted.values]

        ## Binomial test per cluster
        results = []
        for cluster_id, row in cluster_counts.iterrows():
            k = row['GH']
            n = row['GH'] + row['SI']
            if n > 0:
                p_exp = expected_group
                res = binomtest(k, n, p_exp, alternative='two-sided')
                results.append((cluster_id, res.pvalue))
            else:
                results.append((cluster_id, np.nan))

        pvals = pd.DataFrame(results, columns=['cluster_id', 'p_value'])

        ## correct for multple test - false positives
        pvals['p_adj'] = multipletests(pvals['p_value'], method='fdr_bh')[1]
        path = os.path.join(self.directory, 'deviation_pvals.csv')
        pvals.to_csv(path, index=False, float_format='%.10f')


         ### CONTACT FRAME SUMMARY

        interaction_contact_summary = []

        for cluster_id in sorted(df_interaction[cluster_name].unique()):
            cluster_df = df_interaction[df_interaction[cluster_name] == cluster_id]
            for inter_id in cluster_df["interaction_id"].unique():
                inter_df = cluster_df[cluster_df["interaction_id"] == inter_id]
                n_close = (inter_df["min_distance"] < 1).sum()
                interaction_contact_summary.append({
                    "cluster": cluster_id,
                    "interaction_id": inter_id,
                    "frames_below_1mm": n_close
                })

        # Convert to DataFrame
        df_interaction_contact = pd.DataFrame(interaction_contact_summary)



        ## === TOP: DEVIATION BARPLOT WITH SIGNIFICANCE STARS ==
        fig = plt.figure(figsize=(12, 8))
        gs = gridspec.GridSpec(2, 1, height_ratios=[6, 0.3], hspace=0.05)

        ax1 = fig.add_subplot(gs[0])  # top: bar chart
        ax2 = fig.add_subplot(gs[1])  # middle: heatmap box

        x_labels = deviation_sorted.index.astype(str)
        x_pos = np.arange(len(x_labels))

        # Plot with Matplotlib's bar (since sns.barplot expects a DataFrame)
        ax1.bar(
            deviation_sorted.index.astype(str),
            deviation_sorted.values,
            color=colors,            # keep your C0/C1 mapping
            alpha=0.7,              # <— makes the fill lighter
            edgecolor='black',       # <— black border
            linewidth=1.5            # <— thicker border
        )

        # Add reference line
        ax1.axhline(0, color='k', linestyle='--', linewidth=1)

        # --- Annotate significance stars ---

        # Map adjusted p-values to cluster ids
        p_map = pvals.set_index('cluster_id')['p_adj']

        def stars(p):
            if p < 0.001: return '***'
            if p < 0.01:  return '**'
            if p < 0.05:  return '*'
            return ''

        # Vertical offset for labels
        ymin, ymax = ax1.get_ylim()
        dy = 0.015 * (ymax - ymin)

        # Annotate bars
        for i, cid in enumerate(deviation_sorted.index):
            p = p_map.get(cid, np.nan)
            if pd.notna(p):
                s = stars(p)
                if s:
                    y = deviation_sorted.loc[cid]
                    ax1.text(
                        i,
                        y + (dy if y >= 0 else -dy),
                        s,
                        ha='center',
                        va='bottom' if y >= 0 else 'top',
                        fontsize=10,
                        fontweight='bold',
                        color='black')
        
        ax1.set_title("Cluster Deviation from Expected", fontsize=16, fontweight='bold', pad=15)
        ax1.set_ylabel("Deviation from Expected")
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels('')
        

        # === BOTTOM: 1×N heatmap strip of avg contact frames per cluster ===
        # 1) mean frames below 1mm per cluster
        mean_frames = (
            df_interaction_contact
            .groupby('cluster')['frames_below_1mm']
            .mean()
        )

        # 2) ensure index types match the deviation index (important for reindex)
        mean_frames.index = mean_frames.index.astype(deviation_sorted.index.dtype)

        # 3) align to the same x-order as the bar plot
        avg_contact_aligned = mean_frames.reindex(deviation_sorted.index, fill_value=0)

        vals = avg_contact_aligned.to_numpy()

        colors = ["aliceblue", "#56B19C", "darkgreen"]
        my_cmap = LinearSegmentedColormap.from_list("greenblue_custom", colors)

        norm = Normalize(vmin=0, vmax=np.nanmax(vals) if np.nanmax(vals) > 0 else 1)

        ax2.set_xlim(0, len(vals))
        ax2.set_ylim(0, 1)

        for i, v in enumerate(vals):
            ax2.add_patch(
                Rectangle((i, 0), 1, 1,
                        facecolor=my_cmap(norm(v)),
                        edgecolor='none')
            )

        ax2.set_yticks([0])
        ax2.set_yticklabels(['Average Contact\nFrames'])
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(x_labels, fontweight='bold', fontsize=14)
        ax2.tick_params(axis='x', pad=15)

        # optional: make the heat row compact and boxy
        for spine in ax2.spines.values():
            spine.set_visible(False)
    

        sm = ScalarMappable(norm=norm, cmap=my_cmap)
        sm.set_array([])

        cbar = fig.colorbar(
            sm,
            ax=[ax1, ax2],
            fraction=0.03,
            pad=0.02,
            location='right'
        )
        cbar.set_label('Average Contact Frames', rotation=270, labelpad=15)

        path = os.path.join(self.directory, 'deviations.pdf')  
        plt.savefig(path, format="pdf", bbox_inches="tight", dpi=300, transparent=True)
        plt.close()


    ##########################################################################################################
    ## SPEARMANS CORRELATION BETWEEN DEVIATION AND CONTACT FRAMES
    ##########################################################################################################
    def correlation_contact(self):

        df = self.clusters
        df['condition'] = df['condition'].replace({'iso': 'SI', 'group': 'GH'})
        df_interaction = self.df
        cluster_name = self.cluster_name

        mpl.rcParams['pdf.fonttype'] = 42
        mpl.rcParams['ps.fonttype'] = 42

        mpl.rcParams['font.family'] = 'sans-serif'
        mpl.rcParams['font.sans-serif'] = ['Arial']

        PALETTE = {
            "GH": 'steelblue',     
            "SI": 'darkorange',}

        HUE_ORDER = ["GH", "SI"]

        ### OBSERVED - EXPECTED DEVIATION 

        cluster_counts = (df.groupby([cluster_name, 'condition']).size().unstack(fill_value=0).reindex(columns=['GH', 'SI'], fill_value=0))  # count number per cluster per condition

        total_group = cluster_counts['GH'].sum()
        total_iso   = cluster_counts['SI'].sum()
        total_all   = total_group + total_iso

        expected_group = total_group / total_all   # e.g., ~0.56

        observed_group_frac = cluster_counts['GH'] / (cluster_counts['GH'] + cluster_counts['SI']).replace({0: np.nan}) ## observed fraction
        observed_group_frac = observed_group_frac.fillna(0.0)

        deviation = observed_group_frac - expected_group ## expected fraction

        deviation_sorted = deviation.sort_values()

        ## Binomial test per cluster
        results = []
        for cluster_id, row in cluster_counts.iterrows():
            k = row['GH']
            n = row['GH'] + row['SI']
            if n > 0:
                p_exp = expected_group
                res = binomtest(k, n, p_exp, alternative='two-sided')
                results.append((cluster_id, res.pvalue))
            else:
                results.append((cluster_id, np.nan))

        pvals = pd.DataFrame(results, columns=['cluster_id', 'p_value'])

        ## correct for multple test - false positives
        pvals['p_adj'] = multipletests(pvals['p_value'], method='fdr_bh')[1]


         ### CONTACT FRAME SUMMARY

        interaction_contact_summary = []

        for cluster_id in sorted(df_interaction[cluster_name].unique()):
            cluster_df = df_interaction[df_interaction[cluster_name] == cluster_id]
            for inter_id in cluster_df["interaction_id"].unique():
                inter_df = cluster_df[cluster_df["interaction_id"] == inter_id]
                n_close = (inter_df["min_distance"] < 1).sum()
                interaction_contact_summary.append({
                    "cluster": cluster_id,
                    "interaction_id": inter_id,
                    "frames_below_1mm": n_close
                })

        # Convert to DataFrame
        df_interaction_contact = pd.DataFrame(interaction_contact_summary)
        
        mean_frames = (
            df_interaction_contact
            .groupby('cluster')['frames_below_1mm']
            .mean()
        )

        mean_frames.index = mean_frames.index.astype(deviation_sorted.index.dtype)

        avg_contact_aligned = mean_frames.reindex(deviation_sorted.index)

        # Spearman correlation!
        correlation_df = pd.DataFrame({
            'deviation': deviation_sorted.values,
            'avg_contact_frames': avg_contact_aligned.values
        }).dropna()


        rho, p = spearmanr(
            correlation_df["deviation"],
            correlation_df["avg_contact_frames"]
        )

        n = len(correlation_df)
        print(f"Spearman ρ = {rho:.3f}, p = {p:.3g}, N = {n}")


        x = correlation_df["deviation"]
        y = correlation_df["avg_contact_frames"]

        slope, intercept, r_value, p_value, std_err = linregress(x, y)

        print(f"Linear regression: slope = {slope:.3f}, intercept = {intercept:.3f}, R² = {r_value**2:.3f}, p = {p_value:.3g}")

        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = intercept + slope * x_line

        fig, ax = plt.subplots(figsize=(4,4))
        ax = sns.scatterplot(
            x=correlation_df["deviation"],
            y=correlation_df["avg_contact_frames"],
            s=60,
            color='mediumseagreen',
            edgecolor="gray",
        )

        ax.plot(x_line, y_line, color="darkgray", linewidth=2)

        ax.set_xlim(-0.2, 0.2)
        ax.set_xlabel("Deviation from expected (GH bias)")
        ax.set_ylabel("Average contact frames")
        sns.despine()

        ax.set_title(f"Spearman ρ = {rho:.2f}, p = {p:.3g}")
        plt.tight_layout()
        plt.savefig(os.path.join(self.directory, 'correlation.pdf'), format='pdf', bbox_inches='tight')
        plt.close()






        










    ##########################################################################################################
    ## METHOD SUMMARY_ANCHOR_PARTNER: SUMMARY QUANTIFICATIONS ANCHOR/PARTNER
    ##########################################################################################################
    def summary_anchor_partner(self):

        df = self.df
        cluster_name = self.cluster_name 

        cluster_ids = sorted(df[cluster_name].unique())
        n_clusters = len(cluster_ids)
        n_rows = 13  # number of summary plots (trajectory, speed, accel, angle, etc.)

        # Create summary canvas
        # fig_ap, axes_ap = plt.subplots(n_rows, n_clusters, figsize=(n_clusters * 4, n_rows * 2))

        # width per column and height per "unit"
        width_per_col  = 2    # you already had n_clusters*4
        height_per_unit = 2.1

        # Row 0 gets 3 units, rows 1–5 get 1 each → total units = 3 + 5*1 = 8
        height_ratios = [1.5] + [1]*(n_rows-1) ## want mean trajectory to get 3 times the space as the other rows 
        total_units   = sum(height_ratios)          # = 8
        fig_w = n_clusters * width_per_col          # unchanged
        fig_h = total_units * height_per_unit       # 8 * 1.5 = 12"

        fig_ap, axes_ap = plt.subplots(
        n_rows,
        n_clusters,
        figsize=(fig_w, fig_h),
        gridspec_kw={'height_ratios': height_ratios},
        constrained_layout=True
    )

        if n_clusters == 1:
            axes_ap = axes_ap.reshape(n_rows, 1)

        # Mark all as invisible initially
        for ax in axes_ap.flatten():
            ax.set_visible(False)

        row_labels = [
            "Mean Trajectory",           # 0
            "Speed",                     # 1
            "Acceleration",              # 2
            "Heading Angle",             # 3
            "Heading Angle Change",      # 4
            "Approach Angle",            # 5
            "Approach Angle Change",     # 6
            "Distance Travelled",        # 7
            "Minimum Distance",          # 8
            "Interaction Type",          # 9
            "Initial Contact",       # 10  <-- new
            "Predominant Contact",   # 11  <-- new
            "Contact Frames <1mm"        # 12  (moved down from 10)
        ]


        for i, label in enumerate(row_labels):
            ax_label = axes_ap[i, 0]  # first column of each row
            ax_label.set_ylabel(label, fontsize=10, rotation=0, labelpad=40, va='center')


        df['anchor_distance'] = df.groupby('interaction_id').apply(
        lambda x: np.sqrt((x['anchor x_body'].diff()**2 + x['anchor y_body'].diff()**2))).reset_index(level=0, drop=True)

        df['partner_distance'] =  df.groupby('interaction_id').apply(
        lambda x: np.sqrt((x['partner x_body'].diff()**2 + x['partner y_body'].diff()**2))).reset_index(level=0, drop=True)
        

        for column, cluster_id in enumerate(cluster_ids):
            cluster_df = df[df[cluster_name] == cluster_id]

            ## 0. MEAN TRAJECTORIES

            ax0 = axes_ap[0, column]
            grouped = cluster_df.groupby("Normalized Frame")

            t1_x = grouped["anchor x_body"].mean()
            t1_y = grouped["anchor y_body"].mean()
            t2_x = grouped["partner x_body"].mean()
            t2_y = grouped["partner y_body"].mean()

            t1_x_std = grouped["anchor x_body"].std()
            t1_y_std = grouped["anchor y_body"].std()
            t2_x_std = grouped["partner x_body"].std()
            t2_y_std = grouped["partner y_body"].std()

                    # Combine into a DataFrame
            sd_summary = pd.DataFrame({
                "Normalized Frame": t1_y_std.index,
                "t1_y_std": t1_y_std.values,
                "t2_y_std": t2_y_std.values,
                "t1_x_std": t1_x_std.values,
                "t2_x_std": t2_x_std.values,
            })

            # Save to CSV
            # sd_summary.to_csv(os.path.join(output_dir, "std_trajectory_summary.csv"), index=False)


            ax0.plot(t1_x, t1_y, label="Anchor", color="blue")
            ax0.plot(t2_x, t2_y, label="Partner", color="orange")

            ax0.scatter(t1_x.iloc[0], t1_y.iloc[0], color="blue", marker="o", label="Anchor Start")
            ax0.scatter(t2_x.iloc[0], t2_y.iloc[0], color="orange", marker="o", label="Partner Start")

                # error bars in X and Y
            # ax0.errorbar(
            #     t1_x, t1_y,
            #     xerr=t1_x_std, yerr=t1_y_std,
            #     fmt="none", ecolor="blue", alpha=0.3, label="Anchor ±1 SD"
            # )
            # ax0.errorbar(
            #     t2_x, t2_y,
            #     xerr=t2_x_std, yerr=t2_y_std,
            #     fmt="none", ecolor="orange", alpha=0.3, label="Partner ±1 SD"
            # )

            ax0.errorbar(
                    t1_x.values, t1_y.values,
                    xerr=t1_x_std.values, yerr=t1_y_std.values,
                    fmt="none", ecolor="blue", alpha=0.3, label="Anchor ±1 SD"
                )
            
            ax0.errorbar(
                    t2_x.values, t2_y.values,
                    xerr=t2_x_std.values, yerr=t2_y_std.values,
                    fmt="none", ecolor="orange", alpha=0.3, label="Partner ±1 SD"
                )



            # ax0.set_xticks([])
            # ax_sum.set_yticks([])
            # ax0.set_aspect('equal', 'box')
            ax0.set_title(f"Cluster {cluster_id}", fontsize=8)
            ax0.set_visible(True)



            ## 1. SPEED
            ax1 = axes_ap[1, column]

            sns.lineplot(data=cluster_df, x='Normalized Frame', y='anchor_speed', label='Anchor', errorbar=('ci', 95), color='blue', ax=ax1)
            sns.lineplot(data=cluster_df, x='Normalized Frame', y='partner_speed', label='Partner', errorbar=('ci', 95), color='orange', ax=ax1)

            ax1.axvline(0, color="gray", ls="--", lw=0.5)
            ax1.set_ylim(0, 2)
            ax1.set_xticks([])
            # ax1.set_yticks([])
            ax1.set_visible(True)

            ## 2. ACCELERATION
            ax2 = axes_ap[2, column]

            sns.lineplot(data=cluster_df, x='Normalized Frame', y='anchor_acceleration', label='Anchor', errorbar=('ci', 95), color='blue', ax=ax2)
            sns.lineplot(data=cluster_df, x='Normalized Frame', y='partner_acceleration', label='Partner', errorbar=('ci', 95), color='orange', ax=ax2)

            ax2.axvline(0, color="gray", ls="--", lw=0.5)
            ax2.set_ylim(-1, 1)
            ax2.set_xticks([])
            # ax1.set_yticks([])
            ax2.set_visible(True)

            ## 3. HEADING ANGLE
            ax3 = axes_ap[3, column]

            sns.lineplot(data=cluster_df, x='Normalized Frame', y='anchor_angle', label='Anchor', errorbar=('ci', 95), color='blue', ax=ax3)
            sns.lineplot(data=cluster_df, x='Normalized Frame', y='partner_angle', label='Partner', errorbar=('ci', 95), color='orange', ax=ax3)

            ax3.axvline(0, color="gray", ls="--", lw=0.5)
            ax3.set_ylim(0, 180)
            ax3.set_xticks([])
            # ax1.set_yticks([])
            ax3.set_visible(True)

            ## 4. HEADING ANGLE CHANGE
            ax4 = axes_ap[4, column]

            sns.lineplot(data=cluster_df, x='Normalized Frame', y='anchor_heading_angle_change', label='Anchor', errorbar=('ci', 95), color='blue', ax=ax4)
            sns.lineplot(data=cluster_df, x='Normalized Frame', y='partner_heading_angle_change', label='Partner', errorbar=('ci', 95), color='orange', ax=ax4)

            ax4.axvline(0, color="gray", ls="--", lw=0.5)
            ax4.set_ylim(0, 60)
            ax4.set_xticks([])
            ax4.set_visible(True)

            ## 4. APPROACH ANGLE
            ax5 = axes_ap[5, column]

            sns.lineplot(data=cluster_df, x='Normalized Frame', y='anchor_approach_angle', label='Anchor', errorbar=('ci', 95), color='blue', ax=ax5)
            sns.lineplot(data=cluster_df, x='Normalized Frame', y='partner_approach_angle', label='Partner', errorbar=('ci', 95), color='orange', ax=ax5)

            ax5.axvline(0, color="gray", ls="--", lw=0.5)
            ax5.set_ylim(0, 180)
            ax5.set_xticks([])
            # ax1.set_yticks([])
            ax5.set_visible(True)

            ## 6. APPROACH ANGLE CHANGE
            ax6 = axes_ap[6, column]

            sns.lineplot(data=cluster_df, x='Normalized Frame', y='anchor_approach_angle_change', label='Anchor', errorbar=('ci', 95), color='blue', ax=ax6)
            sns.lineplot(data=cluster_df, x='Normalized Frame', y='partner_approach_angle_change', label='Partner', errorbar=('ci', 95), color='orange', ax=ax6)

            ax6.axvline(0, color="gray", ls="--", lw=0.5)
            ax6.set_ylim(0, 60)
            ax6.set_xticks([])
            ax6.set_visible(True)

            ## 7. DISTANCE TRAVELLED
            ax7 = axes_ap[7, column]

            sns.lineplot(data=cluster_df, x='Normalized Frame', y='anchor_distance', label='Anchor', errorbar=('ci', 95), color='blue', ax=ax7)
            sns.lineplot(data=cluster_df, x='Normalized Frame', y='partner_distance', label='Partner', errorbar=('ci', 95), color='orange', ax=ax7)

            ax7.axvline(0, color="gray", ls="--", lw=0.5)
            ax7.set_ylim(0, 14)
            # ax5.set_xticks([])
            # ax1.set_yticks([])
            ax7.set_visible(True)

            ## 8. MIN DISTANCE BETWEEN

            ax8 = axes_ap[8, column]
            grouped_min = cluster_df.groupby("Normalized Frame")["min_distance"]
            mean_min = grouped_min.mean()
            std_min  = grouped_min.std()

            mean_min = mean_min.sort_index()
            # Split windows: pre (<0), post (>0)
            pre = mean_min[mean_min.index < 0]
            post = mean_min[mean_min.index > 0]

            if len(pre) >= 2 and pre.index.nunique() >= 2:
                res_pre = linregress(pre.index.values.astype(float), pre.values.astype(float))
                slope_pre = res_pre.slope
            else:
                slope_pre = np.nan

            if len(post) >= 2 and post.index.nunique() >= 2:
                res_post = linregress(post.index.values.astype(float), post.values.astype(float))
                slope_post = res_post.slope
            else:
                slope_post = np.nan

            # ax8.plot(mean_min.index, mean_min.values, color='black')
            # ax8.fill_between(
            #     mean_min.index,
            #     mean_min - std_min,
            #     mean_min + std_min,
            #     color='gray',
            #     alpha=0.3
            # )
            sns.lineplot(
            data=cluster_df,
            x='Normalized Frame',
            y='min_distance',
            errorbar=('ci', 95),
            color='black',
            ax=ax8
        )
            ax8.axvline(0, color='red', linestyle='--', linewidth=0.5)
            ax8.set_ylim(0, 25)
            ax8.set_xticks([])
            ax8.set_visible(True)

            ax8.text(0.55, 0.92, f"pre slope:  {slope_pre:.2f}",  transform=ax8.transAxes, fontsize=8)
            ax8.text(0.55, 0.76, f"post slope: {slope_post:.2f}", transform=ax8.transAxes, fontsize=8)



            # ---- 9–12. CONTACT SUMMARY (match standalone) ----
            interaction_colors = {
                "head-head": "red",
                "head-body": "orange",
                "head-tail": "yellow",
                "body-body": "black",
                "tail-tail": "green",
                "tail-body": "purple"}


            interaction_merge_map = {
                "head-tail": "head-tail", "tail-head": "head-tail",
                "tail-body": "tail-body", "body-tail": "tail-body",
                "head-body": "head-body", "body-head": "head-body",
                "tail-tail": "tail-tail", "head-head": "head-head", "body-body": "body-body",
            }
            interaction_types = ["head-head", "head-body", "head-tail", "body-body", "tail-tail", "tail-body"]
            palette_list = [interaction_colors[t] for t in interaction_types]

            inter_ids = cluster_df["interaction_id"].unique()
            records = []
            init_labels = []
            pred_labels = []
            frames_close_list = []

            for inter_id in inter_ids:
                inter = cluster_df[cluster_df["interaction_id"] == inter_id].sort_values("Frame")

                # frames in contact (<1mm), merge symmetric labels
                close = inter[inter["min_distance"] < 1].copy()
                close["interaction_type_merged"] = close["interaction_type"].map(interaction_merge_map)

                # counts per interaction (merged types)
                counts = close["interaction_type_merged"].value_counts().to_dict()
                row = {"interaction_id": inter_id}
                for it in interaction_types:
                    row[it] = counts.get(it, 0)
                records.append(row)

                # initial & predominant labels, if any contact exists
                tm = close["interaction_type_merged"]
                if not tm.empty:
                    init_labels.append(tm.iloc[0])
                    pred_labels.append(tm.value_counts().idxmax())

                # total frames <1mm for this interaction
                frames_close_list.append((inter_id, (inter["min_distance"] < 1).sum()))

            # ---------- ROW 9: Interaction Type (mean ± sd frames per interaction) ----------
            ax9 = axes_ap[9, column]
            df_counts = pd.DataFrame(records)
            means = df_counts[interaction_types].mean()
            stds  = df_counts[interaction_types].std()

            # x = np.arange(len(interaction_types))
            # ax9.bar(x, means.values, yerr=stds.values, capsize=5,
            #         color=[interaction_colors[it] for it in interaction_types], alpha=0.8)
            # ax9.set_xticks(x)

            df_counts_long = df_counts.melt(value_vars=interaction_types,
                                var_name="interaction_type",
                                value_name="frames")

            sns.barplot(
                    data=df_counts_long,
                    x="interaction_type",
                    y="frames",
                    order=interaction_types,
                    palette=palette_list,
                    errorbar=('ci', 95),   # <-- key change for 95% CI
                    ax=ax9
                )
                            
            ax9.set_xticklabels(interaction_types, rotation=45, fontsize=6)
            ax9.set_ylim(0, (means + stds).max() * 1.1 if len(means) else 1)
            ax9.set_xticks([])
            ax9.set_visible(True)

            # ---------- ROW 10: Initial Contact (%) ----------
            ax10 = axes_ap[10, column]
            if len(init_labels):
                tmp_init = pd.DataFrame({
                    "contact_type": np.repeat(interaction_types, len(init_labels)),
                    "val": np.concatenate([(np.array(init_labels) == t).astype(int) for t in interaction_types])
                })
            else:
                tmp_init = pd.DataFrame({"contact_type": interaction_types, "val": np.zeros(len(interaction_types), dtype=int)})

            sns.barplot(
                data=tmp_init, x="contact_type", y="val",
                errorbar=('ci', 95),
                order=interaction_types, palette=palette_list, ax=ax10
            )
            ax10.set_ylim(0, 1)
            ax10.set_xticks([])
            ax10.set_visible(True)

            # ---------- ROW 11: Predominant Contact (%) ----------
            ax11 = axes_ap[11, column]
            if len(pred_labels):
                tmp_pred = pd.DataFrame({
                    "contact_type": np.repeat(interaction_types, len(pred_labels)),
                    "val": np.concatenate([(np.array(pred_labels) == t).astype(int) for t in interaction_types])
                })
            else:
                tmp_pred = pd.DataFrame({"contact_type": interaction_types, "val": np.zeros(len(interaction_types), dtype=int)})

            sns.barplot(
                data=tmp_pred, x="contact_type", y="val",
                errorbar=('ci', 95),
                order=interaction_types, palette=palette_list, ax=ax11
            )
            ax11.set_ylim(0, 1)
            ax11.set_xticks([])
            ax11.set_visible(True)


            # ---------- ROW 12: Contact Frames <1mm (mean ± sd) ----------
            ax12 = axes_ap[12, column]
            frames_vals = pd.Series([v for _, v in frames_close_list])
            mean_val = float(frames_vals.mean()) if len(frames_vals) else 0.0
            std_val  = float(frames_vals.std())  if len(frames_vals) else 0.0

            ax12.bar(0, mean_val, yerr=std_val, color='green', alpha=0.8, capsize=5)
            ax12.text(0, mean_val + 1, f"{mean_val:.1f}", ha='left', fontsize=12)
            ax12.set_ylim(0, 15)
            ax12.set_xticks([])
            ax12.set_visible(True)


        out_path = os.path.join(self.directory, "summary_anchor_partner.pdf")
        plt.savefig(out_path, format="pdf", bbox_inches='tight')
        plt.close(fig_ap)

    ##########################################################################################################
    ## METHOD HIERARCHAL_COMPARISONS: HIERARCHAL CLUSTER COMPARISONS
    ##########################################################################################################

    def hierarchal_comparisons(self):

        """Generate a hierarchal summary of the Clusters"""

        df = self.df
        cluster_name = self.cluster_name

        group_map = {
                    1: 'G1', 2: 'G1',
                    3: 'G2', 4: 'G2',
                    6: 'G3', 7: 'G3',
                    8: 'G4', 9: 'G4',
                    10: 'G5', 11: 'G5',
                    5: 'G6', 12: 'G6'
                }
        
        
        df['hierarchal_group'] = df[cluster_name].map(group_map)

        output = os.path.join(self.directory, 'hierarchal_summary')
        os.makedirs(output, exist_ok=True)

        ## 1. Distance Travelled 

        data = []

        for cluster_id in sorted(df[cluster_name].unique()):
            cluster_df = df[df[cluster_name] == cluster_id]

            for inter_id in cluster_df["interaction_id"].unique():
                inter_df = cluster_df[cluster_df["interaction_id"] == inter_id].sort_values("Normalized Frame").copy()

                inter_df["partner"] = np.hypot(
                        inter_df["partner x_body"].diff(),
                        inter_df["partner y_body"].diff()
                    ).fillna(0)
                
                inter_df["anchor"] = np.hypot(
                        inter_df["anchor x_body"].diff(),
                        inter_df["anchor y_body"].diff()
                    ).fillna(0)
                
                cols = ['interaction_id', cluster_name, 'hierarchal_group', 'Normalized Frame', 'partner', 'anchor']

                filtered = inter_df[cols]
                data.append(filtered)

        distance = pd.concat(data, ignore_index=True)

        distance_melted = distance.melt(
                id_vars=["interaction_id", cluster_name, "Normalized Frame", 'hierarchal_group'],
                value_vars=["partner", "anchor"],
                var_name="role",
                value_name="distance"
            )
        
        distance_melted['cluster_role'] = distance_melted[cluster_name].astype(str) + '_' + distance_melted['role']

        save_path = os.path.join(output, "partner_anchor_distance.csv")
        distance.to_csv(save_path, index=False)

        fig, axes = plt.subplots(2, 3, figsize=(10, 8), sharex=True, sharey=True, constrained_layout=True)
        axes = axes.flatten()

        ## pallette for cluster_role
        group_A = [1, 3, 5, 6, 8, 10]
        group_B = [2, 4, 7, 9, 11, 12]

        palette = {}

        for c in group_A:
            palette[f"{c}_anchor"]  = "#1f77b4"   # dark blue
            palette[f"{c}_partner"] = "#90c9e8"   # dark orange

        for c in group_B:
            palette[f"{c}_partner"]  = "#edc078"   # light blue
            palette[f"{c}_anchor"] = "#ff7f0e"   # light orange


        ### DISTANCE TRAVELLED 

        # Plotting the distance data
        for i, (role, group) in enumerate(distance_melted.groupby("hierarchal_group")):
            hue_order = sorted(group['cluster_role'].unique())
            ax = axes[i]
            sns.lineplot(data=group, x="Normalized Frame", y="distance", ax=ax, hue='cluster_role', palette=palette, errorbar=('ci', 95), hue_order=hue_order)
            ax.set_title(f"")
            ax.set_ylim(0, None)
            ax.set_xlim(-14, 15)

        plt.suptitle("Distance Travelled", fontsize=16, fontweight='bold')

        plt.tight_layout()        
        save_path = os.path.join(output, "distance.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        save_path = os.path.join(output, "distance.pdf")
        plt.savefig(save_path, format="pdf", bbox_inches='tight')
        plt.close(fig)



        #### SPEED

        speed_melted = df.melt(
            id_vars=["interaction_id", cluster_name, "Normalized Frame", "hierarchal_group"],
            value_vars=["anchor_speed", "partner_speed"],
            var_name="role",
            value_name="speed"
        )

        # normalize role names to match palette
        speed_melted["role"] = speed_melted["role"].str.replace("_speed", "", regex=False)
        speed_melted["cluster_role"] = speed_melted[cluster_name].astype(str) + "_" + speed_melted["role"]

        fig, axes = plt.subplots(2, 3, figsize=(10, 8), sharex=True, sharey=True, constrained_layout=True)
        axes = axes.flatten()

        for i, (hg, group) in enumerate(speed_melted.groupby("hierarchal_group")):
            ax = axes[i]
            hue_order = sorted(group['cluster_role'].unique())
            sns.lineplot(
                data=group,
                x="Normalized Frame",
                y="speed",
                ax=ax,
                hue="cluster_role",
                palette=palette,
                hue_order=hue_order,
                errorbar=("ci", 95),
            )

            ax.set_xlim(-14, 15)
            ax.set_ylim(0, None)

        fig.suptitle("Speed", fontsize=16, fontweight="bold", y=1.02)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        plt.savefig(os.path.join(output, "speed.png"), dpi=300, bbox_inches="tight")
        plt.savefig(os.path.join(output, "speed.pdf"), format="pdf", bbox_inches="tight")
        plt.close(fig)


        #### ACCELERATION

        speed_melted = df.melt(
            id_vars=["interaction_id", cluster_name, "Normalized Frame", "hierarchal_group"],
            value_vars=["anchor_acceleration", "partner_acceleration"],
            var_name="role",
            value_name="acceleration"
        )

        # normalize role names to match palette
        speed_melted["role"] = speed_melted["role"].str.replace("_acceleration", "", regex=False)
        speed_melted["cluster_role"] = speed_melted[cluster_name].astype(str) + "_" + speed_melted["role"]

        fig, axes = plt.subplots(2, 3, figsize=(10, 8), sharex=True, sharey=True, constrained_layout=True)
        axes = axes.flatten()

        for i, (hg, group) in enumerate(speed_melted.groupby("hierarchal_group")):
            ax = axes[i]
            hue_order = sorted(group['cluster_role'].unique())
            sns.lineplot(
                data=group,
                x="Normalized Frame",
                y="acceleration",
                ax=ax,
                hue="cluster_role",
                palette=palette,
                hue_order=hue_order,
                errorbar=("ci", 95),
            )

            ax.set_xlim(-14, 15)
            ax.set_ylim(-0.2, 0.2)

        fig.suptitle("Acceleration", fontsize=16, fontweight="bold", y=1.02)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        plt.savefig(os.path.join(output, "acceleration.png"), dpi=300, bbox_inches="tight")
        plt.savefig(os.path.join(output, "acceleration.pdf"), format="pdf", bbox_inches="tight")
        plt.close(fig)


        #### APPROACH ANGLE CHANGE

        speed_melted = df.melt(
            id_vars=["interaction_id", cluster_name, "Normalized Frame", "hierarchal_group"],
            value_vars=["anchor_approach_angle_change", "partner_approach_angle_change"],
            var_name="role",
            value_name="approach_angle_change"
        )

        # normalize role names to match palette
        speed_melted["role"] = speed_melted["role"].str.replace("_approach_angle_change", "", regex=False)
        speed_melted["cluster_role"] = speed_melted[cluster_name].astype(str) + "_" + speed_melted["role"]

        fig, axes = plt.subplots(2, 3, figsize=(10, 8), sharex=True, sharey=True, constrained_layout=True)
        axes = axes.flatten()

        for i, (hg, group) in enumerate(speed_melted.groupby("hierarchal_group")):
            ax = axes[i]
            hue_order = sorted(group['cluster_role'].unique())
            sns.lineplot(
                data=group,
                x="Normalized Frame",
                y="approach_angle_change",
                ax=ax,
                hue="cluster_role",
                palette=palette,
                hue_order=hue_order,
                errorbar=("ci", 95),
            )

            ax.set_xlim(-14, 15)
            ax.set_ylim(0, None)

        fig.suptitle("Approach Angle Change", fontsize=16, fontweight="bold", y=1.02)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        plt.savefig(os.path.join(output, "approach_angle_change.png"), dpi=300, bbox_inches="tight")
        plt.savefig(os.path.join(output, "approach_angle_change.pdf"), format="pdf", bbox_inches="tight")
        plt.close(fig)


        #### HEADING ANGLE CHANGE

        speed_melted = df.melt(
            id_vars=["interaction_id", cluster_name, "Normalized Frame", "hierarchal_group"],
            value_vars=["anchor_heading_angle_change", "partner_heading_angle_change"],
            var_name="role",
            value_name="heading_angle_change"
        )

        # normalize role names to match palette
        speed_melted["role"] = speed_melted["role"].str.replace("_heading_angle_change", "", regex=False)
        speed_melted["cluster_role"] = speed_melted[cluster_name].astype(str) + "_" + speed_melted["role"]

        fig, axes = plt.subplots(2, 3, figsize=(10, 8), sharex=True, sharey=True, constrained_layout=True)
        axes = axes.flatten()

        for i, (hg, group) in enumerate(speed_melted.groupby("hierarchal_group")):
            ax = axes[i]
            hue_order = sorted(group['cluster_role'].unique())
            sns.lineplot(
                data=group,
                x="Normalized Frame",
                y="heading_angle_change",
                ax=ax,
                hue="cluster_role",
                palette=palette,
                hue_order=hue_order,
                errorbar=("ci", 95),
            )

            ax.set_xlim(-14, 15)
            ax.set_ylim(0, None)

        fig.suptitle("Heading Angle Change", fontsize=16, fontweight="bold", y=1.02)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        plt.savefig(os.path.join(output, "heading_angle_change.png"), dpi=300, bbox_inches="tight")
        plt.savefig(os.path.join(output, "heading_angle_change.pdf"), format="pdf", bbox_inches="tight")
        plt.close(fig)



        ###### MIN DISTANCE

        fig, axes = plt.subplots(2, 3, figsize=(10, 8), sharex=True, sharey=True, constrained_layout=True)
        axes = axes.flatten()

        ## pallette for cluster_role
        group_A = [1, 3, 5, 6, 8, 10]
        group_B = [2, 4, 7, 9, 11, 12]

        palette = {}

        for c in group_A:
            palette[c]  = "#1f77b4"   # dark blue
        for c in group_B:
            palette[c] = "#ff7f0e"   # light orange

        # Plotting the distance data
        for i, (role, group) in enumerate(df.groupby("hierarchal_group")):
            ax = axes[i]
            sns.lineplot(data=group, x="Normalized Frame", y="min_distance", ax=ax, hue=cluster_name, palette=palette, errorbar=('ci', 95))
            ax.set_title(f"")
            ax.set_ylim(0, None)
            ax.set_xlim(-14, 15)
        
        plt.suptitle("Min Distance", fontsize=16, fontweight='bold')

        plt.tight_layout()        
        save_path = os.path.join(output, "min-distance.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        save_path = os.path.join(output, "min-distance.pdf")
        plt.savefig(save_path, format="pdf", bbox_inches='tight')

        plt.close(fig)





        


 #### METHOD RELATIVE_PARTNER_METRICS: COMPARES THE PROJECTED VECTOR BETWEEN THE PREVIOUS PARTNER AND THE ANCHOR AND THE PREVIOUS PARTNER TO THE CURRENT PARTNER
    def relative_partner_metrics(self):

        """https://en.wikipedia.org/wiki/Vector_projection#Scalar_projection_2"""

        df = self.df.copy()
        df = df.sort_values(['interaction_id', 'Normalized Frame'])

        cluster_name = self.cluster_name
        
        ## PARTNERS PREV X,Y COORD 
        df['prev partner x_body'] = df.groupby('interaction_id')['partner x_body'].shift(1)
        df['prev partner y_body'] = df.groupby('interaction_id')['partner y_body'].shift(1)

        ## transpose (.T) so each row represents one frame
        partner_position = np.array([df['partner x_body'], df['partner y_body']]).T
        partner_prev_position = np.array([df['prev partner x_body'], df['prev partner y_body']]).T
        anchor_position = np.array([df['anchor x_body'], df['anchor y_body']]).T

        projection = anchor_position - partner_prev_position
        partner_direction  = partner_position - partner_prev_position

        # axis=1 - accross each row / frame

        def calculate_angle_between_two_angles(a, b):
            # rowwise dot and norms
            cos_theta = (a * b).sum(axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
            cos_theta = np.clip(cos_theta, -1.0, 1.0)  # avoid rounding errors
            theta_radians = np.arccos(cos_theta)
            return theta_radians

        def get_scalar_rejection(a, theta):
            rejection_magnitude = np.linalg.norm(a, axis=1) * np.sin(theta)
            return rejection_magnitude
        
        def get_scalar_projection(a, theta):
            projection_magnitude = np.linalg.norm(a, axis=1) * np.cos(theta)
            return projection_magnitude
        
        def distance(x, y):
            diff = x - y
            distance = np.linalg.norm(diff, axis=1)
            return distance
        
        # deviation angle from projected path 
        df['deviation_angle'] = calculate_angle_between_two_angles(partner_direction, projection)
        # deviated movement / magnitude 
        df['sideways_movement'] = get_scalar_rejection(partner_direction, df['deviation_angle'])
        df['sideways_movement'] = df['sideways_movement'].fillna(np.nan)
        # forward movement 
        df['forward_movement'] = get_scalar_projection(partner_direction, df['deviation_angle'])

        df['deviation_angle'] = np.degrees(df['deviation_angle']) # radians to degrees

        ## distance between anchor + partner
        df['distance_between'] = distance(partner_position, anchor_position)

        ## distance between partner + previous partner
        df['partner_distance'] = distance(partner_position, partner_prev_position)

        ## represents the fraction of motion directed toward the anchor.
        # how aligned is the motion with the target direction?
        df['aligned_movement'] = df['forward_movement'] / df['partner_distance'] 


        columns_to_keep = ['interaction_id', 'Normalized Frame', cluster_name, 'deviation_angle', 'sideways_movement', 'forward_movement', 'distance_between', 'aligned_movement']
        data = df[columns_to_keep]

        outdir = os.path.join(self.directory, "relative_partner_metrics")
        os.makedirs(outdir, exist_ok=True)
        outpath = os.path.join(outdir, "projected_versus_actual_partner.csv")
        data.to_csv(outpath, index=False)


        ### PLOTTING
        plt.figure(figsize=(8,8))
        sns.lineplot(data=data, x='Normalized Frame', y='deviation_angle', hue=cluster_name, legend='full', ci=95)
        plt.xlabel('Normalised Time', fontsize=12, fontweight='bold')
        plt.ylabel('Angle', fontsize=12, fontweight='bold')
        plt.title('Deviation from Partner-Anchor Projection', fontsize=16, fontweight='bold')
    
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "deviated_angle.png"), dpi=300, bbox_inches="tight")
        plt.close()

        clusters = sorted(data[cluster_name].dropna().unique())
        cols = 4
        rows = int(np.ceil(len(clusters) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3.2*rows), sharex=True, sharey=True)
        axes = axes.flatten()

        for i, cluster in enumerate(clusters): # index n label
            ax = axes[i]  
            d = data[data[cluster_name] ==cluster]

            sns.lineplot(
                data=d,
                x='Normalized Frame',
                y='deviation_angle',
                ci=95,
                legend=False,
                ax=ax,
                color='purple'
            )

            ax.set_title(f"Cluster {cluster}", fontsize=12, pad=6)
            ax.axvline(0, linestyle='--', linewidth=0.8, color='0.5')
            ax.grid(alpha=0.2)
            ax.set_xlabel('Normalized Time')
            ax.set_ylabel('Angle Deviation')

        # hide any unused subplots
        for j in range(i + 1, rows * cols):
            fig.delaxes(axes[j])

        # common labels + overall title
        fig.suptitle('Angle Deviation from Expected Trajectory', fontsize=16, fontweight='bold')
        fig.supxlabel('')
        fig.supylabel('')

        plt.tight_layout(rect=[0, 0, 1, 0.97])  # leave room for suptitle
        plt.savefig(os.path.join(outdir, "deviated_angle_subplot.png"), dpi=300, bbox_inches="tight")
        plt.close()

        ## DISTANCE BETWEEN + ANGLE

        clusters = sorted(data[cluster_name].dropna().unique())
        cols = 4
        rows = int(np.ceil(len(clusters) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3.2*rows), sharex=True, sharey=True)
        axes = axes.flatten()

        for i, cluster in enumerate(clusters): # index n label
            ax = axes[i]  
            d = data[data[cluster_name] ==cluster]


            ## distance between 
            sns.lineplot(
                data=d,
                x='Normalized Frame',
                y='distance_between',
                ci=95,
                ax=ax,
                color='green',
                label='Distance'
            )

            # second y-axis for forward movement
            ax2 = ax.twinx()
            sns.lineplot(
                data=d,
                x='Normalized Frame',
                y='deviation_angle',
                ci=95,
                ax=ax2,
                color='purple',
                label='Angle Deviation'
            )


            ax.set_title(f"Cluster {cluster}", fontsize=12, pad=6)
            ax.axvline(0, linestyle='--', linewidth=0.8, color='0.5')
            ax.grid(alpha=0.2)
            ax.set_xlabel('Normalized Time')
            ax.set_ylabel('Distance Between Anchor and Partner')
            ax2.set_ylabel('Angle')

            ax2.spines['right'].set_color('purple')
            ax2.tick_params(axis='y', colors='purple')
            ax.spines['left'].set_color('green')
            ax.tick_params(axis='y', colors='green')
            ax2.set_ylim(0, 150)

        # hide any unused subplots
        for j in range(i + 1, rows * cols):
            fig.delaxes(axes[j])

        # common labels + overall title
        fig.suptitle('Distance Between Anchor and Partner', fontsize=16, fontweight='bold')
        fig.supxlabel('')
        fig.supylabel('')

        plt.tight_layout(rect=[0, 0, 1, 0.97])  # leave room for suptitle
        plt.savefig(os.path.join(outdir, "distance_between.png"), dpi=300, bbox_inches="tight")
        plt.close()



        ## DISTANCE COVERED ON THE SCALAR PROJECTION (DIRECTED MOVEMENT IN LINE WITH A-P TRAJECTORY) + ANGLE

        clusters = sorted(data[cluster_name].dropna().unique())
        cols = 4
        rows = int(np.ceil(len(clusters) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3.2*rows), sharex=True, sharey=True)
        axes = axes.flatten()

        for i, cluster in enumerate(clusters): # index n label
            ax = axes[i]  
            d = data[data[cluster_name] ==cluster]


            ## distance between 
            sns.lineplot(
                data=d,
                x='Normalized Frame',
                y='forward_movement',
                ci=95,
                ax=ax,
                color='green',
                label='Projected Distance'
            )

            # second y-axis for forward movement
            ax2 = ax.twinx()
            sns.lineplot(
                data=d,
                x='Normalized Frame',
                y='deviation_angle',
                ci=95,
                ax=ax2,
                color='purple',
                label='Angle Deviation'
            )


            ax.set_title(f"Cluster {cluster}", fontsize=12, pad=6)
            ax.axvline(0, linestyle='--', linewidth=0.8, color='0.5')
            ax.grid(alpha=0.2)
            ax.set_xlabel('Normalized Time')
            ax.set_ylabel('Scalar Projection')
            ax2.set_ylabel('Angle')

            ax2.spines['right'].set_color('purple')
            ax2.tick_params(axis='y', colors='purple')
            ax.spines['left'].set_color('green')
            ax.tick_params(axis='y', colors='green')
            ax2.set_ylim(0, 150)

        # hide any unused subplots
        for j in range(i + 1, rows * cols):
            fig.delaxes(axes[j])

        # common labels + overall title
        fig.suptitle('Projected Coverage and Angle', fontsize=16, fontweight='bold')
        fig.supxlabel('')
        fig.supylabel('')

        plt.tight_layout(rect=[0, 0, 1, 0.97])  # leave room for suptitle
        plt.savefig(os.path.join(outdir, "scalar_projection_angle.png"), dpi=300, bbox_inches="tight")
        plt.close()



        ##

        plt.figure(figsize=(8,8))
        sns.lineplot(data=data, x='Normalized Frame', y='sideways_movement', hue=cluster_name, legend='full', ci=95)
        plt.xlabel('Normalised Time', fontsize=12, fontweight='bold')
        plt.ylabel('Sideways Movement', fontsize=12, fontweight='bold')
        plt.title('Sideways Movement from Partner-Anchor Projection', fontsize=16, fontweight='bold')
    
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "sideways_movement.png"), dpi=300, bbox_inches="tight")
        plt.close()

        clusters = sorted(data[cluster_name].dropna().unique())
        cols = 4
        rows = int(np.ceil(len(clusters) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3.2*rows), sharex=True, sharey=True)
        axes = axes.flatten()

        for i, cluster in enumerate(clusters): # index n label
            ax = axes[i]  
            d = data[data[cluster_name] ==cluster]

            sns.lineplot(
                data=d,
                x='Normalized Frame',
                y='sideways_movement',
                ci=95,
                legend=False,
                ax=ax,
                color='purple'
            )

            ax.set_title(f"Cluster {cluster}", fontsize=12, pad=6)
            ax.axvline(0, linestyle='--', linewidth=0.8, color='0.5')
            ax.grid(alpha=0.2)
            ax.set_xlabel('Normalized Time')
            ax.set_ylabel('Sideways Deviation')

        # hide any unused subplots
        for j in range(i + 1, rows * cols):
            fig.delaxes(axes[j])

        # common labels + overall title
        fig.suptitle('Sideways Movement from Expected Trajectory', fontsize=16, fontweight='bold')
        fig.supxlabel('')
        fig.supylabel('')

        plt.tight_layout(rect=[0, 0, 1, 0.97])  # leave room for suptitle
        plt.savefig(os.path.join(outdir, "sideways_movement_subplot.png"), dpi=300, bbox_inches="tight")
        plt.close()






        plt.figure(figsize=(8,8))
        sns.lineplot(data=data, x='Normalized Frame', y='aligned_movement', hue=cluster_name, legend='full', ci=95, palette='summer')
        plt.xlabel('Normalized Time', fontsize=12, fontweight='bold')
        plt.ylabel('scalar projection movement / total movement', fontsize=12, fontweight='bold')
        plt.title('Fraction of Movement in Anchor Direction', fontsize=16, fontweight='bold')
    
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "aligned_movement.png"), dpi=300, bbox_inches="tight")
        plt.close()

        clusters = sorted(data[cluster_name].dropna().unique())
        cols = 4
        rows = int(np.ceil(len(clusters) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3.2*rows), sharex=True, sharey=True)
        axes = axes.flatten()

        for i, cluster in enumerate(clusters): # index n label
            ax = axes[i]  
            d = data[data[cluster_name] ==cluster]

            sns.lineplot(
                data=d,
                x='Normalized Frame',
                y='aligned_movement',
                ci=95,
                legend=False,
                ax=ax,
                color='mediumseagreen'
            )

            ax.set_title(f"Cluster {cluster}", fontsize=12, pad=6)
            ax.axvline(0, linestyle='--', linewidth=0.8, color='0.5')
            ax.grid(alpha=0.2)
            ax.set_xlabel('Normalized Time')
            ax.set_ylabel('scalar projection movement / total movement')

        # hide any unused subplots
        for j in range(i + 1, rows * cols):
            fig.delaxes(axes[j])

        # common labels + overall title
        fig.suptitle('Fraction of Movement in Anchor Trajectory', fontsize=16, fontweight='bold')
        fig.supxlabel('')
        fig.supylabel('')

        plt.tight_layout(rect=[0, 0, 1, 0.97])  # leave room for suptitle
        plt.savefig(os.path.join(outdir, "aligned_movement_subplot.png"), dpi=300, bbox_inches="tight")
        plt.close()


        plt.figure(figsize=(8,8))
        sns.lineplot(data=data, x='Normalized Frame', y='forward_movement', hue=cluster_name, legend='full', ci=95)
        plt.xlabel('Normalised Time', fontsize=12, fontweight='bold')
        plt.ylabel('Scalar Movement in Projected Axis', fontsize=12, fontweight='bold')
        plt.title('Scalar Movement in Projected Axis', fontsize=16, fontweight='bold')
    
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "forward_movement.png"), dpi=300, bbox_inches="tight")
        plt.close()



        clusters = sorted(data[cluster_name].dropna().unique())
        cols = 4
        rows = int(np.ceil(len(clusters) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3.2*rows), sharex=True, sharey=True)
        axes = axes.flatten()

        for i, cluster in enumerate(clusters): # index n label
            ax = axes[i]  
            d = data[data[cluster_name] ==cluster]

            sns.lineplot(
                data=d,
                x='Normalized Frame',
                y='forward_movement',
                ci=95,
                legend=False,
                ax=ax,
                color='purple'
            )

            ax.set_title(f"Cluster {cluster}", fontsize=12, pad=6)
            ax.axvline(0, linestyle='--', linewidth=0.8, color='0.5')
            ax.grid(alpha=0.2)
            ax.set_xlabel('Normalized Time')
            ax.set_ylabel('Movement Aligned to the Expected Trajectory')

        # hide any unused subplots
        for j in range(i + 1, rows * cols):
            fig.delaxes(axes[j])

        # common labels + overall title
        fig.suptitle('Movement in Expected Trajectory', fontsize=16, fontweight='bold')
        fig.supxlabel('')
        fig.supylabel('')

        plt.tight_layout(rect=[0, 0, 1, 0.97])  # leave room for suptitle
        plt.savefig(os.path.join(outdir, "forward_movement_subplpt.png"), dpi=300, bbox_inches="tight")
        plt.close()


        ####### HIERARAJCAL OF ABOVE #######

        group_map = {
                    1: 'G1', 2: 'G1',
                    3: 'G2', 4: 'G2',
                    6: 'G3', 7: 'G3',
                    8: 'G4', 9: 'G4',
                    10: 'G5', 11: 'G5',
                    5: 'G6', 12: 'G6'
                }
        
        

        df['hierarchal_group'] = df[cluster_name].map(group_map)

        #### ANGLE
        fig, axes = plt.subplots(2, 3, figsize=(10, 8), sharex=True, sharey=True, constrained_layout=True)
        axes = axes.flatten()

        ## pallette for cluster_role
        group_A = [1, 3, 5, 6, 8, 10]
        group_B = [2, 4, 7, 9, 11, 12]

        palette = {}

        for c in group_A:
            palette[c]  = "#1f77b4"   # dark blue
        for c in group_B:
            palette[c] = "#ff7f0e"   # light orange

        # Plotting the distance data
        for i, (role, group) in enumerate(df.groupby("hierarchal_group")):
            ax = axes[i]
            sns.lineplot(data=group, x="Normalized Frame", y="deviation_angle", ax=ax, hue=cluster_name, palette=palette, errorbar=('ci', 95))
            ax.set_title(f"")
            ax.set_ylim(0, 180)
            # ax.set_xlim(-14, 15)

        plt.suptitle("Deviation Angle", fontsize=16, fontweight='bold')

        plt.tight_layout()
        save_path = os.path.join(outdir, "hierarchical-deviation-angle.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        save_path = os.path.join(outdir, "hierarchical-deviation-angle.pdf")
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        plt.close(fig)




        #### SIDEWAYS MOVEMENT
        fig, axes = plt.subplots(2, 3, figsize=(10, 8), sharex=True, sharey=True, constrained_layout=True)
        axes = axes.flatten()

        # Plotting the distance data
        for i, (role, group) in enumerate(df.groupby("hierarchal_group")):
            ax = axes[i]
            sns.lineplot(data=group, x="Normalized Frame", y="sideways_movement", ax=ax, hue=cluster_name, palette=palette, errorbar=('ci', 95))
            ax.set_title(f"")
            # ax.set_ylim(0, 180)
            # ax.set_xlim(-14, 15)

        plt.suptitle("Sideways Movement", fontsize=16, fontweight='bold')

        plt.tight_layout()
        save_path = os.path.join(outdir, "hierarchical-sideways-movement.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        save_path = os.path.join(outdir, "hierarchical-sideways-movement.pdf")
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        plt.close(fig)

        #### FORWARD MOVEMENT
        fig, axes = plt.subplots(2, 3, figsize=(10, 8), sharex=True, sharey=True, constrained_layout=True)
        axes = axes.flatten()

        # Plotting the distance data
        for i, (role, group) in enumerate(df.groupby("hierarchal_group")):
            ax = axes[i]
            sns.lineplot(data=group, x="Normalized Frame", y="forward_movement", ax=ax, hue=cluster_name, palette=palette, errorbar=('ci', 95))
            ax.set_title(f"")
            # ax.set_ylim(0, 180)
            # ax.set_xlim(-14, 15)

        plt.suptitle("Forward Movement", fontsize=16, fontweight='bold')

        plt.tight_layout()
        save_path = os.path.join(outdir, "hierarchical-forward-movement.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        save_path = os.path.join(outdir, "hierarchical-forward-movement.pdf")
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        plt.close(fig)


        #### ALIGNED MOVEMENT
        fig, axes = plt.subplots(2, 3, figsize=(10, 8), sharex=True, sharey=True, constrained_layout=True)
        axes = axes.flatten()

        # Plotting the distance data
        for i, (role, group) in enumerate(df.groupby("hierarchal_group")):
            ax = axes[i]
            sns.lineplot(data=group, x="Normalized Frame", y="aligned_movement", ax=ax, hue=cluster_name, palette=palette, errorbar=('ci', 95))
            ax.set_title(f"")
            ax.set_ylim(-1, 1)
            ax.axhline(0, color='0.6', linestyle='--', linewidth=0.8)

            ax2 = ax.twinx()
            sns.lineplot(
                data=group,
                x="Normalized Frame",
                y="min_distance",
                ax=ax2,
                color='gray',
                errorbar=('ci', 95),
                linewidth=0.5
            )

            ax2.set_ylabel("")
            ax2.tick_params(axis='y', colors='gray')
            ax2.spines['right'].set_color('gray')
            ax2.set_ylim(0, 1)

            # ---- ADD THIS BLOCK (shading) ----



        plt.suptitle("Aligned Movement Toward Anchor", fontsize=16, fontweight='bold')

        plt.tight_layout()
        save_path = os.path.join(outdir, "hierarchical-aligned-movement.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        save_path = os.path.join(outdir, "hierarchical-aligned-movement.pdf")
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        plt.close(fig)



    def hierarchal_mean_trace_summary(self):
            
            df = self.df
            cluster_name = self.cluster_name
        
            hierarchal_tree = (( ((1,2), (3,4)), (5, (6,7)) ),( ((8,9), (10,11)), 12 ))

            ## LEVEL 0
            x1, y1 = 1, 0
            x2, y2 = 2, 0
            x3, y3 = 3, 0
            x4, y4 = 4, 0
            x5, y5 = 5, 0
            x6, y6 = 6, 0
            x7, y7 = 7, 0
            x8, y8 = 8, 0
            x9, y9 = 9, 0
            x10, y10 = 10, 0
            x11, y11 = 11, 0
            x12, y12 = 12, 0

            ## LEVEL 1
            x_12, y_12 = (x1 + x2) / 2, 1 # (midpoint of x1, x2) and (y=1) is coordinates 
            x34, y34 = (x3 + x4) / 2, 1
            x67, y67 = (x6 + x7) / 2, 1
            x89, y89 = (x8 + x9) / 2, 1
            x1011, y1011 = (x10 + x11) / 2, 1

            ## LEVEL 2
            x1234, y1234 = (x_12 + x34) / 2, 2
            x567, y567 = (x5 + x67) / 2, 2
            x891011, y891011 = (x89 + x1011) / 2, 2

            ## LEVEL 3
            x1234567, y1234567 = (x1234 + x567) / 2, 3
            x89101112, y89101112 = (x891011 + x12) / 2, 3   

            ## LEVEL 4
            x_123456789101112, y_123456789101112 = (x1234567 + x89101112) / 2, 4



            fig = plt.figure(figsize=(15, 4))
            gs = gridspec.GridSpec(3, 1, height_ratios=[0.3, 0.6, 0.05], hspace=0.3)

            ax1 = fig.add_subplot(gs[0])  # top: hierarchal tree
            # ax2 = fig.add_subplot(gs[1])
            sub_ax2 = gs[1].subgridspec(1, 12, wspace=0.1)
            axes_ax2 = [fig.add_subplot(sub_ax2[0, i]) for i in range(12)]

            #ax3 = fig.add_subplot(gs[2])  # bottom: average contact frames
            sub_ax3 = gs[2].subgridspec(1, 12, wspace=0.1)
            ax3 = fig.add_subplot(sub_ax3[0, :])  # one long axis spanning all 12 columns

            ## PLOTTING HIERARCHAL TREE - MANUALLY USING COORDINATES ABOVE 
            # ax.plot([x_start, x_end], [y_start, y_end])

            ax1.plot([x1, x1], [y1, y_12], color='black') #12
            ax1.plot([x2, x2], [y2, y_12], color='black') #12
            ax1.plot([x1, x2], [y_12, y_12], color='black') #12

            ax1.plot([x3, x3], [y3, y34], color='black') #34
            ax1.plot([x4, x4], [y4, y34], color='black') #34
            ax1.plot([x3, x4], [y34, y34], color='black') #34

            ax1.plot([x_12, x_12], [y_12, y1234], color='black') #1234
            ax1.plot([x34, x34], [y34, y1234], color='black') #1234
            ax1.plot([x_12, x34], [y1234, y1234], color='black') #1234

            ax1.plot([x6, x6], [y6, y67], color='black') #67
            ax1.plot([x7, x7], [y7, y67], color='black') #67
            ax1.plot([x6, x7], [y67, y67], color='black') #67

            ax1.plot([x67, x67], [y67, y567], color='black') #567
            ax1.plot([x5, x5], [y5, y567], color='black') #567
            ax1.plot([x5, x67], [y567, y567], color='black') #567

            ax1.plot([x8, x8], [y8, y89], color='black') #89
            ax1.plot([x9, x9], [y9, y89], color='black') #89
            ax1.plot([x8, x9], [y89, y89], color='black') #89

            ax1.plot([x10, x10], [y10, y1011], color='black') #1011
            ax1.plot([x11, x11], [y11, y1011], color='black') #1011
            ax1.plot([x10, x11], [y1011, y1011], color='black') #1011

            ax1.plot([x89, x89], [y89, y891011], color='black') #891011
            ax1.plot([x1011, x1011], [y1011, y891011], color='black') #891011
            ax1.plot([x89, x1011], [y891011, y891011], color='black') #891011

            ax1.plot([x1234, x1234], [y1234, y1234567], color='black') #1234567
            ax1.plot([x567, x567], [y567, y1234567], color='black') #1234567
            ax1.plot([x1234, x567], [y1234567, y1234567], color='black') #1234567

            ax1.plot([x12, x12], [y12, y89101112], color='black') #89101112
            ax1.plot([x891011, x891011], [y891011, y89101112], color='black') #89101112
            ax1.plot([x12, x891011], [y89101112, y89101112], color='black') #89101112

            ax1.plot([x1234567, x1234567], [y1234567, y_123456789101112], color='black') #123456789101112
            ax1.plot([x89101112, x89101112], [y89101112, y_123456789101112], color='black') #123456789101112
            ax1.plot([x1234567, x89101112], [y_123456789101112, y_123456789101112], color='black') #123456789101112

            for spine in ax1.spines.values():
                spine.set_visible(False)
            ax1.set_xticks([])
            ax1.set_yticks([])
            
            ## MEAN TRACES PLOTTING

            anchor_base = "#4F7942"   
            partner_base = "#916288"
            anchor_cmap = plt.cm.Greens
            partner_cmap = plt.cm.Purples

            ordered_frames = sorted(df["Normalized Frame"].dropna().unique())
            norm_frames = Normalize(vmin=ordered_frames[0], vmax=ordered_frames[-1]) #ltr for colour mapping

            clusters = sorted(df[cluster_name].unique())

            for cluster, ax in zip(clusters, axes_ax2):
                cluster_df = df[df[cluster_name] == cluster].copy()
                by_frame = cluster_df.sort_values("Normalized Frame").groupby("Normalized Frame")
                frames = [f for f in by_frame.groups.keys() if f == f] # list of frames (non-nan)
                
                # role+node, return mean X series and mean Y series (indexed by frame)
                def get_means(role, node):
                    mx = by_frame[f"{role} x_{node}"].mean()  # Series: index = frame → mean x
                    my = by_frame[f"{role} y_{node}"].mean()  # Series: index = frame → mean y
                    return mx, my

                ## DRAW TRAILS
                for node in ("head", "body", "tail"):
                    anchor_x, anchor_y = get_means("anchor",  node) # xy means per frame for anchor
                    partner_x, partner_y = get_means("partner", node) # xy means per frame for partner

                    for f0, f1 in zip(frames[:-1], frames[1:]): # consecutive frame pairs (frame 0, frame 1), (frame 1, frame 2), ...
                        # anchor
                        if f0 in anchor_x.index and f1 in anchor_x.index: # check that both frames have data
                            x0, y0 = anchor_x.loc[f0], anchor_y.loc[f0]
                            x1, y1 = anchor_x.loc[f1], anchor_y.loc[f1]
                            if np.isfinite(x0) and np.isfinite(y0) and np.isfinite(x1) and np.isfinite(y1):
                                ax.plot([x0, x1], [y0, y1],
                                        color=anchor_cmap(norm_frames(f1)), alpha=0.7, linewidth=1.2, zorder=1)
                        # partner
                        if f0 in partner_x.index and f1 in partner_x.index:
                            x0, y0 = partner_x.loc[f0], partner_y.loc[f0]
                            x1, y1 = partner_x.loc[f1], partner_y.loc[f1]
                            if np.isfinite(x0) and np.isfinite(y0) and np.isfinite(x1) and np.isfinite(y1):
                                ax.plot([x0, x1], [y0, y1],
                                        color=partner_cmap(norm_frames(f1)), alpha=0.7, linewidth=1.2, zorder=1) #zorder=1 trails goes underneath skeletons and markers
            
                ## DRAW SKELETONS: CONNECT HEAD→BODY→TAIL PER FRAME (TIME-COLORED)
                for f in frames:
                    # anchor
                    parts = {}
                    for node in ("head", "body", "tail"):
                        x_means = by_frame[f"anchor x_{node}"].mean() 
                        y_means = by_frame[f"anchor y_{node}"].mean()

                        if f in x_means.index:
                            x, y = x_means.loc[f], y_means.loc[f]
                            if np.isfinite(x) and np.isfinite(y):
                                parts[node] = (x, y)
                    
                    if len(parts) == 3:
                        ax.plot([parts["head"][0], parts["body"][0], parts["tail"][0]],
                                    [parts["head"][1], parts["body"][1], parts["tail"][1]],
                                    color=anchor_cmap(norm_frames(f)), alpha=0.75, linewidth=1.0, zorder=2)
                    # partner
                    parts = {}
                    for node in ("head", "body", "tail"):
                        x_means = by_frame[f"partner x_{node}"].mean()
                        y_means = by_frame[f"partner y_{node}"].mean()
                        if f in x_means.index:
                            x, y = x_means.loc[f], y_means.loc[f]
                            if np.isfinite(x) and np.isfinite(y):
                                parts[node] = (x, y)
                    if len(parts) == 3:
                        ax.plot([parts["head"][0], parts["body"][0], parts["tail"][0]],
                                [parts["head"][1], parts["body"][1], parts["tail"][1]],
                                color=partner_cmap(norm_frames(f)), alpha=0.75, linewidth=1.0, zorder=2)


                ## POINTS FOR NODES AT EACH FRAME (TIME COLORED; HEAD BIGGER)
                node_marker = {"head": "o", "body": "o", "tail": "o"}
                size_map  = {"head": 6, "body": 4, "tail": 2}

                for node in ("head", "body", "tail"):
                    anchor_x_means, anchor_y_means = get_means("anchor",  node)
                    partner_x_means, partner_y_means = get_means("partner", node)
                    for f in frames:
                        if f in anchor_x_means.index:
                            x, y = anchor_x_means.loc[f], anchor_y_means.loc[f]
                            if np.isfinite(x) and np.isfinite(y):
                                ax.scatter(x, y, s=size_map[node], marker=node_marker[node],
                                            color=anchor_cmap(norm_frames(f)), alpha=0.9, zorder=3)
                        
                        if f in partner_x_means.index:
                            x, y = partner_x_means.loc[f], partner_y_means.loc[f]
                            if np.isfinite(x) and np.isfinite(y):
                                ax.scatter(x, y, s=size_map[node], marker=node_marker[node],
                                            color=partner_cmap(norm_frames(f)), alpha=0.9, zorder=3)

                # 6) tidy the one big axis
                xlim = (-50, 100)
                ylim = (-10, 300)
                ax.set_xlim(*xlim)
                ax.set_ylim(*ylim)
                # ax.set_aspect("equal", adjustable="box")
                ax.set_title(cluster, fontsize=16, fontweight='bold')
                for ax in axes_ax2:
                    ax.axis('off')


            ## AVERAGE CONTACT FRAMES 

            interaction_contact_summary = []

            for cluster_id in sorted(df[cluster_name].unique()):
                cluster_df = df[df[cluster_name] == cluster_id]
                for inter_id in cluster_df["interaction_id"].unique():
                    inter_df = cluster_df[cluster_df["interaction_id"] == inter_id]
                    n_close = (inter_df["min_distance"] < 1).sum()
                    interaction_contact_summary.append({
                        "cluster": cluster_id,
                        "interaction_id": inter_id,
                        "frames_below_1mm": n_close})

            df_interaction_contact = pd.DataFrame(interaction_contact_summary)

            mean_frames = (
                df_interaction_contact
                .groupby('cluster')['frames_below_1mm']
                .mean()
            )

            # 2) ensure index types match the deviation index (important for reindex)
            mean_frames.index = mean_frames.index.astype(int)
            order = [1,2,3,4,5,6,7,8,9,10,11,12]   # matches your dendrogram order
            mean_frames = mean_frames.reindex(order, fill_value=0)

            
            # heat = mean_frames.to_numpy()[np.newaxis, :]


            # # colors = ["aliceblue", "mediumseagreen", "darkgreen"] #
            # colors = ["aliceblue", "#6AB5A3", "darkgreen"] #
            # colors = ["aliceblue", "#56B19C", "darkgreen"] #

            # # build a sequential colormap from those
            # my_cmap = LinearSegmentedColormap.from_list("greenblue_custom", colors)

            # # 5) draw the heat “boand sorry x” (single row)
            # im = ax3.imshow(
            #     heat,
            #     aspect='auto',
            #     interpolation='nearest',
            #     vmin=0,  # anchor scale at 0
            #     cmap=my_cmap,
            #     # cmap='PuBuGn' 
            # )

            # ax3.set_yticklabels([])
            # ax3.set_yticks([])
            # ax3.tick_params(left=False)
            # ax3.set_xlabel("")             # remove x-axis label
            # ax3.set_xticklabels([])        # remove tick labels
            # ax3.set_xticks([])             # remove ticks entirely
            # ax3.tick_params(bottom=False)  # remove the small tick lines


            # # optional: make the heat row compact and boxy
            # for spine in ax3.spines.values():
            #     spine.set_visible(False)
            
            # # === add colorbar OUTSIDE grid ===
            # cbar = fig.colorbar(
            #     im,
            #     ax=[ax1, *axes_ax2, ax3],        # anchors to both axes (so it aligns with total figure height)
            #     fraction=0.01,         # width of colorbar relative to figure
            #     pad=0.01,              # horizontal gap between plots and colorbar
            #     location='right'       # move to right side
            # )
            # cbar.set_label('Average Contact Frames', rotation=270, labelpad=15)

            # --- colormap (keep exactly as you had it) ---
            colors = ["aliceblue", "#56B19C", "darkgreen"]
            my_cmap = LinearSegmentedColormap.from_list("greenblue_custom", colors)

            # values in dendrogram order (already reindexed above)
            vals = mean_frames.to_numpy()

            vmin = 0
            vmax = float(np.nanmax(vals)) if np.nanmax(vals) > 0 else 1.0
            norm = Normalize(vmin=vmin, vmax=vmax)

            # --- draw vector heat strip (Acrobat-proof) ---
            ax3.set_xlim(0, len(vals))
            ax3.set_ylim(0, 1)

            for i, v in enumerate(vals):
                ax3.add_patch(
                    Rectangle((i, 0), 1, 1,
                            facecolor=my_cmap(norm(v)),
                            edgecolor='none')
                )

            # hide axes exactly like before
            ax3.set_yticks([])
            ax3.set_xticks([])
            for spine in ax3.spines.values():
                spine.set_visible(False)

            # --- colorbar (vector-safe) ---
            sm = ScalarMappable(norm=norm, cmap=my_cmap)
            sm.set_array([])

            cbar = fig.colorbar(
                sm,
                ax=[ax1, *axes_ax2, ax3],
                fraction=0.01,
                pad=0.01,
                location='right'
            )
            cbar.set_label('Average Contact Frames', rotation=270, labelpad=15)


  
            output = os.path.join(self.directory, "hierarchal_mean_trace_summary.pdf")
            plt.savefig(output, format='pdf', bbox_inches='tight')
            plt.close(fig)






    """NORMALSING FOR ORIGINAL CLUSTER"""
    def cluster_transitions(self):


        mpl.rcParams['pdf.fonttype'] = 42
        mpl.rcParams['ps.fonttype'] = 42

        mpl.rcParams['font.family'] = 'sans-serif'
        mpl.rcParams['font.sans-serif'] = ['Arial']

        df = self.df
        cluster_name = self.cluster_name
        df = df[df['Normalized Frame'] == 0].copy()

        df['track_ids'] = (
        df['Interaction Pair']
        .str.strip('()')
        .str.split(',')
        .apply(lambda x: [int(x[0]), int(x[1])]))

        df = df.explode('track_ids').rename(columns={'track_ids': 'track_id'}) # explode → two rows per interaction

        df = df[[
            'file',
            'interaction_id',
            'track_id',
            'Normalized Frame',
            'Frame',
            cluster_name, 'condition']]

        df  = df.rename(columns={cluster_name: 'cluster'})
        df = df.rename(columns={'Frame': 'time'})
        df = df.sort_values(by=['file', 'track_id', 'time', 'interaction_id']) #interaction_id
        df['cluster'] = df['cluster'].astype(int)

        df['next_cluster'] = (
            df
            .groupby(['file', 'track_id'])['cluster']
            .shift(-1))
        df = df.dropna(subset=['next_cluster']) # drop rows where next_cluster is NaN - last interaction for each track
        df['next_cluster'] = df['next_cluster'].astype(int)


        isolated_transitions = (
            df[df['condition'] == 'iso']
            .groupby(['cluster', 'next_cluster'])
            .size()
            .unstack(fill_value=0))

        isolated_transitions_normalised = isolated_transitions.div(isolated_transitions.sum(axis=1), axis=0)

        grouped_transitions = (
            df[df['condition'] == 'group']
            .groupby(['cluster', 'next_cluster'])
            .size()
            .unstack(fill_value=0))

        grouped_transitions_normalised = grouped_transitions.div(grouped_transitions.sum(axis=1), axis=0)

        all_clusters = sorted(
            set(isolated_transitions_normalised.index) |
            set(isolated_transitions_normalised.columns) |
            set(grouped_transitions_normalised.index) |
            set(grouped_transitions_normalised.columns)
        )

        P_iso = isolated_transitions_normalised.reindex(index=all_clusters, columns=all_clusters, fill_value=0)
        P_grp = grouped_transitions_normalised.reindex(index=all_clusters, columns=all_clusters, fill_value=0)

        # Difference: Group - Iso
        P_diff = P_grp - P_iso
        D = P_diff.copy()   # or D = P_diff.copy() if you want unweighted # P_diff_weighted_fair 

        # align to same cluster order
        all_clusters = sorted(set(D.index) | set(D.columns))
        D = D.reindex(index=all_clusters, columns=all_clusters, fill_value=0)

        # --- build graph from difference matrix ---
        def diff_matrix_to_digraph(D, thresh=0.02):
            """
            thresh = minimum absolute difference to draw an edge
            """
            G = nx.DiGraph()
            for c in D.index:
                G.add_node(int(c))
            for i in D.index:
                for j in D.columns:
                    w = float(D.loc[i, j])
                    if abs(w) >= thresh:
                        G.add_edge(int(i), int(j), weight=w)
            return G

        G_diff = diff_matrix_to_digraph(D, thresh=0.05)  # tune 0.01–0.03 usually

        # positions (same circular layout)
        pos = nx.circular_layout(all_clusters)

        # colour map for signed differences
        #cmap = plt.cm.RdBu  # red=negative, blue=positive (we'll set norm around 0)
        cmap = LinearSegmentedColormap.from_list(
            "steelblue_darkorange",
            ["darkorange", "honeydew", "steelblue"]
        )
        lim = float(np.nanmax(np.abs(D.to_numpy())))
        if lim == 0:
            lim = 1e-6
        norm = mpl.colors.TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)

        # helpers for width/alpha based on abs(weight)
        def w_to_lw(w, min_w=0.3, max_w=6.0):
            return min_w + (abs(w) / lim) * (max_w - min_w)

        def w_to_alpha(w, min_a=0.15, max_a=0.95):
            return min_a + (abs(w) / lim) * (max_a - min_a)

        # draw
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.set_title("Transition Likelihood Difference")
        ax.axis("off")

        nx.draw_networkx_nodes(G_diff, pos, ax=ax, node_size=1100, node_color='slategray')
        nx.draw_networkx_labels(G_diff, pos, ax=ax, font_size=12, font_color='black', font_weight='bold')

        # edges
        for u, v, d in G_diff.edges(data=True):
            w = float(d["weight"])
            color = cmap(norm(w))
            lw = w_to_lw(w)
            a = w_to_alpha(w)

            rad = 0.12 if u != v else 0.35
            patch = FancyArrowPatch(
                posA=pos[u], posB=pos[v],
                arrowstyle="-|>",
                mutation_scale=14,
                connectionstyle=f"arc3,rad={rad}",
                linewidth=lw,
                color=color,
                alpha=a,
                shrinkA=25,
                shrinkB=25
            )
            ax.add_patch(patch)

        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        cax = fig.add_axes([0.95, 0.25, 0.015, 0.5])  # smaller bar
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label("P(group) − P(iso)")
        plt.savefig(os.path.join(self.directory, "transition_diff_circlegraph.pdf"), format="pdf", bbox_inches="tight")
        plt.close()

    

    """NORMALSING FOR NEXT CLUSTER"""
    def cluster_transitions_different(self):


        mpl.rcParams['pdf.fonttype'] = 42
        mpl.rcParams['ps.fonttype'] = 42

        mpl.rcParams['font.family'] = 'sans-serif'
        mpl.rcParams['font.sans-serif'] = ['Arial']

        df = self.df
        cluster_name = self.cluster_name
        df = df[df['Normalized Frame'] == 0].copy()

        df['track_ids'] = (
        df['Interaction Pair']
        .str.strip('()')
        .str.split(',')
        .apply(lambda x: [int(x[0]), int(x[1])]))

        df = df.explode('track_ids').rename(columns={'track_ids': 'track_id'}) # explode → two rows per interaction

        df = df[[
            'file',
            'interaction_id',
            'track_id',
            'Normalized Frame',
            'Frame',
            cluster_name, 'condition']]

        df  = df.rename(columns={cluster_name: 'cluster'})
        df = df.rename(columns={'Frame': 'time'})
        df = df.sort_values(by=['file', 'track_id', 'time', 'interaction_id']) #interaction_id
        df['cluster'] = df['cluster'].astype(int)

        df['next_cluster'] = (
            df
            .groupby(['file', 'track_id'])['cluster']
            .shift(-1))
        df = df.dropna(subset=['next_cluster']) # drop rows where next_cluster is NaN - last interaction for each track
        df['next_cluster'] = df['next_cluster'].astype(int)


        isolated_transitions = (
            df[df['condition'] == 'iso']
            .groupby(['cluster', 'next_cluster'])
            .size()
            .unstack(fill_value=0))

        isolated_transitions_normalised = isolated_transitions.div(isolated_transitions.sum(axis=0), axis=1)

        grouped_transitions = (
            df[df['condition'] == 'group']
            .groupby(['cluster', 'next_cluster'])
            .size()
            .unstack(fill_value=0))

        grouped_transitions_normalised = grouped_transitions.div(grouped_transitions.sum(axis=0), axis=1)

        all_clusters = sorted(
            set(isolated_transitions_normalised.index) |
            set(isolated_transitions_normalised.columns) |
            set(grouped_transitions_normalised.index) |
            set(grouped_transitions_normalised.columns)
        )

        P_iso = isolated_transitions_normalised.reindex(index=all_clusters, columns=all_clusters, fill_value=0)
        P_grp = grouped_transitions_normalised.reindex(index=all_clusters, columns=all_clusters, fill_value=0)

        # Difference: Group - Iso
        P_diff = P_grp - P_iso
        D = P_diff.copy()   # or D = P_diff.copy() if you want unweighted # P_diff_weighted_fair 

        # align to same cluster order
        all_clusters = sorted(set(D.index) | set(D.columns))
        D = D.reindex(index=all_clusters, columns=all_clusters, fill_value=0)

        # --- build graph from difference matrix ---
        def diff_matrix_to_digraph(D, thresh=0.02):
            """
            thresh = minimum absolute difference to draw an edge
            """
            G = nx.DiGraph()
            for c in D.index:
                G.add_node(int(c))
            for i in D.index:
                for j in D.columns:
                    w = float(D.loc[i, j])
                    if abs(w) >= thresh:
                        G.add_edge(int(i), int(j), weight=w)
            return G

        G_diff = diff_matrix_to_digraph(D, thresh=0.05)  # tune 0.01–0.03 usually

        # positions (same circular layout)
        pos = nx.circular_layout(all_clusters)

        # colour map for signed differences
        #cmap = plt.cm.RdBu  # red=negative, blue=positive (we'll set norm around 0)
        cmap = LinearSegmentedColormap.from_list(
            "steelblue_darkorange",
            ["darkorange", "honeydew", "steelblue"]
        )
        lim = float(np.nanmax(np.abs(D.to_numpy())))
        if lim == 0:
            lim = 1e-6
        norm = mpl.colors.TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)

        # helpers for width/alpha based on abs(weight)
        def w_to_lw(w, min_w=0.3, max_w=6.0):
            return min_w + (abs(w) / lim) * (max_w - min_w)

        def w_to_alpha(w, min_a=0.15, max_a=0.95):
            return min_a + (abs(w) / lim) * (max_a - min_a)

        # draw
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.set_title("Transition Likelihood Difference")
        ax.axis("off")

        nx.draw_networkx_nodes(G_diff, pos, ax=ax, node_size=1100, node_color='slategray')
        nx.draw_networkx_labels(G_diff, pos, ax=ax, font_size=12, font_color='black', font_weight='bold')

        # edges
        for u, v, d in G_diff.edges(data=True):
            w = float(d["weight"])
            color = cmap(norm(w))
            lw = w_to_lw(w)
            a = w_to_alpha(w)

            rad = 0.12 if u != v else 0.35
            patch = FancyArrowPatch(
                posA=pos[u], posB=pos[v],
                arrowstyle="-|>",
                mutation_scale=14,
                connectionstyle=f"arc3,rad={rad}",
                linewidth=lw,
                color=color,
                alpha=a,
                shrinkA=25,
                shrinkB=25
            )
            ax.add_patch(patch)

        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        cax = fig.add_axes([0.95, 0.25, 0.015, 0.5])  # smaller bar
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label("P(group) − P(iso)")
        plt.savefig(os.path.join(self.directory, "transition_diff_circlegraph_natalie-norm.pdf"), format="pdf", bbox_inches="tight")
        plt.close()












    
    def cluster_duration_transitions(self):

        mpl.rcParams['pdf.fonttype'] = 42
        mpl.rcParams['ps.fonttype'] = 42

        mpl.rcParams['font.family'] = 'sans-serif'
        mpl.rcParams['font.sans-serif'] = ['Arial']

        df = self.df
        cluster_name = self.cluster_name
        df = df[df['Normalized Frame'] == 0].copy()

        df['track_ids'] = (
        df['Interaction Pair']
        .str.strip('()')
        .str.split(',')
        .apply(lambda x: [int(x[0]), int(x[1])]))

        df = df.explode('track_ids').rename(columns={'track_ids': 'track_id'}) # explode → two rows per interaction

        df = df[[
            'file',
            'interaction_id',
            'track_id',
            'Normalized Frame',
            'Frame',
            cluster_name, 'condition']]

        df  = df.rename(columns={cluster_name: 'cluster'})
        df = df.rename(columns={'Frame': 'time'})
        df = df.sort_values(by=['file', 'track_id', 'time', 'interaction_id']) #interaction_id
        df['cluster'] = df['cluster'].astype(int)

        df['next_cluster'] = (
            df
            .groupby(['file', 'track_id'])['cluster']
            .shift(-1))
        df = df.dropna(subset=['next_cluster']) # drop rows where next_cluster is NaN - last interaction for each track
        df['next_cluster'] = df['next_cluster'].astype(int)


        isolated_transitions = (
            df[df['condition'] == 'iso']
            .groupby(['cluster', 'next_cluster'])
            .size()
            .unstack(fill_value=0))

        isolated_transitions_normalised = isolated_transitions.div(isolated_transitions.sum(axis=1), axis=0)

        grouped_transitions = (
            df[df['condition'] == 'group']
            .groupby(['cluster', 'next_cluster'])
            .size()
            .unstack(fill_value=0))

        grouped_transitions_normalised = grouped_transitions.div(grouped_transitions.sum(axis=1), axis=0)

        all_clusters = sorted(
            set(isolated_transitions_normalised.index) |
            set(isolated_transitions_normalised.columns) |
            set(grouped_transitions_normalised.index) |
            set(grouped_transitions_normalised.columns)
        )

        P_iso = isolated_transitions_normalised.reindex(index=all_clusters, columns=all_clusters, fill_value=0)
        P_grp = grouped_transitions_normalised.reindex(index=all_clusters, columns=all_clusters, fill_value=0)

        # Difference: Group - Iso
        P_diff = P_grp - P_iso


        dur_map = {
            1: "medium",
            2: "short",
            3: "long",
            4: "long",
            5: "long",
            6: "short",
            7: "medium",
            8: "short",
            9: "medium",
            10: "short",
            11: "medium",
            12: "long",
        }

        dur_order = ["short", "medium", "long"]

        # 2) make duration labels for current + next
        df_dur = df.copy()
        df_dur["cluster_i"] = df_dur["cluster"].astype(int)
        df_dur["cluster_j"] = df_dur["next_cluster"].astype(int)

        df_dur["from_dur"] = df_dur["cluster_i"].map(dur_map)
        df_dur["to_dur"]   = df_dur["cluster_j"].map(dur_map)

        # if anything unmapped, drop it (shouldn't happen, but safe)
        df_dur = df_dur.dropna(subset=["from_dur", "to_dur"])

        # 3) build 3x3 transition *counts* per condition, then row-normalise to probabilities
        def dur_transition_matrix(dsub):
            C = (
                dsub.groupby(["from_dur", "to_dur"])
                    .size()
                    .unstack(fill_value=0)
                    .reindex(index=dur_order, columns=dur_order, fill_value=0)
            )
            P = C.div(C.sum(axis=1), axis=0).fillna(0)   # P(next_dur | current_dur)
            return C, P

        C_iso_dur, P_iso_dur = dur_transition_matrix(df_dur[df_dur["condition"] == "iso"])
        C_grp_dur, P_grp_dur = dur_transition_matrix(df_dur[df_dur["condition"] == "group"])

        # 4) difference matrix (optionally "fair" weighted like before)
        P_diff_dur = P_grp_dur - P_iso_dur


        D = P_diff_dur.copy()   # or P_diff_dur if you want unweighted # P_diff_dur_weighted

        nodes = dur_order
        pos = nx.circular_layout(nodes, scale=0.8)

        lim = float(np.nanmax(np.abs(D.to_numpy())))
        if lim == 0:
            lim = 1e-6

        # cmap = plt.cm.RdBu
        cmap = LinearSegmentedColormap.from_list(
            "steelblue_darkorange",
            ["darkorange", "honeydew", "steelblue"]
        )
        norm = mpl.colors.TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)

        def w_to_lw(w, min_w=1, max_w=10.0):
            return min_w + (abs(w) / lim) * (max_w - min_w)

        def w_to_alpha(w, min_a=0.2, max_a=0.95):
            return min_a + (abs(w) / lim) * (max_a - min_a)

        def diff_matrix_to_digraph_labels(D, thresh=0.02):
            G = nx.DiGraph()
            for c in D.index:
                G.add_node(c)
            for i in D.index:
                for j in D.columns:
                    w = float(D.loc[i, j])
                    if abs(w) >= thresh:
                        G.add_edge(i, j, weight=w)
            return G

        Gd = diff_matrix_to_digraph_labels(D, thresh=0.02)

        fig, ax = plt.subplots(1, 1, figsize=(9, 9))
        ax.set_title("", pad=30)
        ax.axis("off")

        nx.draw_networkx_nodes(Gd, pos, ax=ax, node_size=2000, node_color='slategray')
        nx.draw_networkx_labels(Gd, pos, ax=ax, font_size=12, font_weight="bold", font_color='black')

        # draw edges (including self loops)
        for u, v, dct in Gd.edges(data=True):
            w = float(dct["weight"])
            color = cmap(norm(w))
            lw = w_to_lw(w)
            a = w_to_alpha(w)

            rad = 0.18 if u != v else 0.45
            patch = FancyArrowPatch(
                posA=pos[u], posB=pos[v],
                arrowstyle="-|>",
                mutation_scale=22,
                connectionstyle=f"arc3,rad={rad}",
                linewidth=lw,
                color=color,
                alpha=a,
                shrinkA=35,
                shrinkB=35
            )
            ax.add_patch(patch)

        # smaller colorbar
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cax = fig.add_axes([0.99, 0.28, 0.02, 0.45])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label("P(group) - P(iso)")


        
        plt.savefig(os.path.join(self.directory, "duration_transition_duration_diff_circlegraph.pdf"), format="pdf", bbox_inches="tight", pad_inches=0.5)
        plt.close()





    

    def hierarchal_mean_trace_summary_gif(self):

        import os
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from matplotlib import gridspec
        from matplotlib.colors import Normalize, LinearSegmentedColormap
        from matplotlib.patches import Rectangle
        from matplotlib.cm import ScalarMappable
        from matplotlib.animation import FuncAnimation, PillowWriter

        df = self.df.copy()
        cluster_name = self.cluster_name

        hierarchal_tree = (((((1, 2), (3, 4)), (5, (6, 7))), (((8, 9), (10, 11)), 12)))

        ## LEVEL 0
        x1, y1 = 1, 0
        x2, y2 = 2, 0
        x3, y3 = 3, 0
        x4, y4 = 4, 0
        x5, y5 = 5, 0
        x6, y6 = 6, 0
        x7, y7 = 7, 0
        x8, y8 = 8, 0
        x9, y9 = 9, 0
        x10, y10 = 10, 0
        x11, y11 = 11, 0
        x12, y12 = 12, 0

        ## LEVEL 1
        x_12, y_12 = (x1 + x2) / 2, 1
        x34, y34 = (x3 + x4) / 2, 1
        x67, y67 = (x6 + x7) / 2, 1
        x89, y89 = (x8 + x9) / 2, 1
        x1011, y1011 = (x10 + x11) / 2, 1

        ## LEVEL 2
        x1234, y1234 = (x_12 + x34) / 2, 2
        x567, y567 = (x5 + x67) / 2, 2
        x891011, y891011 = (x89 + x1011) / 2, 2

        ## LEVEL 3
        x1234567, y1234567 = (x1234 + x567) / 2, 3
        x89101112, y89101112 = (x891011 + x12) / 2, 3

        ## LEVEL 4
        x_123456789101112, y_123456789101112 = (x1234567 + x89101112) / 2, 4

        fig = plt.figure(figsize=(15, 4))
        gs = gridspec.GridSpec(3, 1, height_ratios=[0.3, 0.6, 0.05], hspace=0.3)

        ax1 = fig.add_subplot(gs[0])  # top: hierarchal tree
        sub_ax2 = gs[1].subgridspec(1, 12, wspace=0.1)
        axes_ax2 = [fig.add_subplot(sub_ax2[0, i]) for i in range(12)]

        sub_ax3 = gs[2].subgridspec(1, 12, wspace=0.1)
        ax3 = fig.add_subplot(sub_ax3[0, :])  # one long axis spanning all 12 columns

        # =========================
        # STATIC: HIERARCHAL TREE
        # =========================
        ax1.plot([x1, x1], [y1, y_12], color="black")
        ax1.plot([x2, x2], [y2, y_12], color="black")
        ax1.plot([x1, x2], [y_12, y_12], color="black")

        ax1.plot([x3, x3], [y3, y34], color="black")
        ax1.plot([x4, x4], [y4, y34], color="black")
        ax1.plot([x3, x4], [y34, y34], color="black")

        ax1.plot([x_12, x_12], [y_12, y1234], color="black")
        ax1.plot([x34, x34], [y34, y1234], color="black")
        ax1.plot([x_12, x34], [y1234, y1234], color="black")

        ax1.plot([x6, x6], [y6, y67], color="black")
        ax1.plot([x7, x7], [y7, y67], color="black")
        ax1.plot([x6, x7], [y67, y67], color="black")

        ax1.plot([x67, x67], [y67, y567], color="black")
        ax1.plot([x5, x5], [y5, y567], color="black")
        ax1.plot([x5, x67], [y567, y567], color="black")

        ax1.plot([x8, x8], [y8, y89], color="black")
        ax1.plot([x9, x9], [y9, y89], color="black")
        ax1.plot([x8, x9], [y89, y89], color="black")

        ax1.plot([x10, x10], [y10, y1011], color="black")
        ax1.plot([x11, x11], [y11, y1011], color="black")
        ax1.plot([x10, x11], [y1011, y1011], color="black")

        ax1.plot([x89, x89], [y89, y891011], color="black")
        ax1.plot([x1011, x1011], [y1011, y891011], color="black")
        ax1.plot([x89, x1011], [y891011, y891011], color="black")

        ax1.plot([x1234, x1234], [y1234, y1234567], color="black")
        ax1.plot([x567, x567], [y567, y1234567], color="black")
        ax1.plot([x1234, x567], [y1234567, y1234567], color="black")

        ax1.plot([x12, x12], [y12, y89101112], color="black")
        ax1.plot([x891011, x891011], [y891011, y89101112], color="black")
        ax1.plot([x12, x891011], [y89101112, y89101112], color="black")

        ax1.plot([x1234567, x1234567], [y1234567, y_123456789101112], color="black")
        ax1.plot([x89101112, x89101112], [y89101112, y_123456789101112], color="black")
        ax1.plot([x1234567, x89101112], [y_123456789101112, y_123456789101112], color="black")

        for spine in ax1.spines.values():
            spine.set_visible(False)
        ax1.set_xticks([])
        ax1.set_yticks([])

        # =========================
        # STATIC: CONTACT STRIP + COLORBAR (vector-safe)
        # =========================
        interaction_contact_summary = []
        for cluster_id in sorted(df[cluster_name].unique()):
            cluster_df = df[df[cluster_name] == cluster_id]
            for inter_id in cluster_df["interaction_id"].unique():
                inter_df = cluster_df[cluster_df["interaction_id"] == inter_id]
                n_close = (inter_df["min_distance"] < 1).sum()
                interaction_contact_summary.append(
                    {
                        "cluster": cluster_id,
                        "interaction_id": inter_id,
                        "frames_below_1mm": n_close,
                    }
                )

        df_interaction_contact = pd.DataFrame(interaction_contact_summary)

        mean_frames = df_interaction_contact.groupby("cluster")["frames_below_1mm"].mean()
        mean_frames.index = mean_frames.index.astype(int)
        order = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        mean_frames = mean_frames.reindex(order, fill_value=0)

        colors = ["aliceblue", "#56B19C", "darkgreen"]
        my_cmap = LinearSegmentedColormap.from_list("greenblue_custom", colors)

        vals = mean_frames.to_numpy()
        vmin = 0
        vmax = float(np.nanmax(vals)) if np.nanmax(vals) > 0 else 1.0
        norm_contact = Normalize(vmin=vmin, vmax=vmax)

        ax3.set_xlim(0, len(vals))
        ax3.set_ylim(0, 1)

        for i, v in enumerate(vals):
            ax3.add_patch(
                Rectangle((i, 0), 1, 1, facecolor=my_cmap(norm_contact(v)), edgecolor="none")
            )

        ax3.set_yticks([])
        ax3.set_xticks([])
        for spine in ax3.spines.values():
            spine.set_visible(False)

        sm = ScalarMappable(norm=norm_contact, cmap=my_cmap)
        sm.set_array([])

        cbar = fig.colorbar(
            sm,
            ax=[ax1, *axes_ax2, ax3],
            fraction=0.01,
            pad=0.01,
            location="right",
        )
        cbar.set_label("Average Contact Frames", rotation=270, labelpad=15)

        # =========================
        # ANIMATION SETUP (temporal progression)
        # =========================
        anchor_base = "#4F7942"
        partner_base = "#916288"
        anchor_cmap = plt.cm.Greens
        partner_cmap = plt.cm.Purples

        ordered_frames = sorted(df["Normalized Frame"].dropna().unique())
        norm_frames = Normalize(vmin=ordered_frames[0], vmax=ordered_frames[-1])

        clusters = sorted(df[cluster_name].unique())

        # Precompute per-cluster, per-node mean series for speed + consistency
        cluster_cache = {}
        for cluster in clusters:
            cluster_df = df[df[cluster_name] == cluster].copy()
            by_frame = cluster_df.sort_values("Normalized Frame").groupby("Normalized Frame")
            frames = [f for f in by_frame.groups.keys() if f == f]  # non-nan

            means = {}
            for role in ("anchor", "partner"):
                means[role] = {}
                for node in ("head", "body", "tail"):
                    mx = by_frame[f"{role} x_{node}"].mean()
                    my = by_frame[f"{role} y_{node}"].mean()
                    means[role][node] = (mx, my)

            cluster_cache[cluster] = {"frames": sorted(frames), "means": means}

        xlim = (-50, 100)
        ylim = (-10, 300)

        # Make sure each axis is initialized the same way (kept identical to your final styling)
        for cluster, ax in zip(clusters, axes_ax2):
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_title(cluster, fontsize=16, fontweight="bold")
            ax.axis("off")

        def _draw_up_to(ax, cache, f_max):
            frames = cache["frames"]
            means = cache["means"]

            # frames up to the current timepoint (inclusive)
            frames_upto = [f for f in frames if f <= f_max]
            if len(frames_upto) == 0:
                return

            # 1) DRAW TRAILS (persisting, colored by time; previous stays)
            for node in ("head", "body", "tail"):
                anchor_x, anchor_y = means["anchor"][node]
                partner_x, partner_y = means["partner"][node]

                for f0, f1 in zip(frames_upto[:-1], frames_upto[1:]):
                    # anchor
                    if f0 in anchor_x.index and f1 in anchor_x.index:
                        x0, y0 = anchor_x.loc[f0], anchor_y.loc[f0]
                        x1, y1 = anchor_x.loc[f1], anchor_y.loc[f1]
                        if np.isfinite(x0) and np.isfinite(y0) and np.isfinite(x1) and np.isfinite(y1):
                            ax.plot(
                                [x0, x1],
                                [y0, y1],
                                color=anchor_cmap(norm_frames(f1)),
                                alpha=0.7,
                                linewidth=1.2,
                                zorder=1,
                            )

                    # partner
                    if f0 in partner_x.index and f1 in partner_x.index:
                        x0, y0 = partner_x.loc[f0], partner_y.loc[f0]
                        x1, y1 = partner_x.loc[f1], partner_y.loc[f1]
                        if np.isfinite(x0) and np.isfinite(y0) and np.isfinite(x1) and np.isfinite(y1):
                            ax.plot(
                                [x0, x1],
                                [y0, y1],
                                color=partner_cmap(norm_frames(f1)),
                                alpha=0.7,
                                linewidth=1.2,
                                zorder=1,
                            )

            # 2) DRAW SKELETONS PER FRAME (persisting; colored by that frame)
            for f in frames_upto:
                # anchor
                parts = {}
                for node in ("head", "body", "tail"):
                    x_means = means["anchor"][node][0]
                    y_means = means["anchor"][node][1]
                    if f in x_means.index:
                        x, y = x_means.loc[f], y_means.loc[f]
                        if np.isfinite(x) and np.isfinite(y):
                            parts[node] = (x, y)

                if len(parts) == 3:
                    ax.plot(
                        [parts["head"][0], parts["body"][0], parts["tail"][0]],
                        [parts["head"][1], parts["body"][1], parts["tail"][1]],
                        color=anchor_cmap(norm_frames(f)),
                        alpha=0.75,
                        linewidth=1.0,
                        zorder=2,
                    )

                # partner
                parts = {}
                for node in ("head", "body", "tail"):
                    x_means = means["partner"][node][0]
                    y_means = means["partner"][node][1]
                    if f in x_means.index:
                        x, y = x_means.loc[f], y_means.loc[f]
                        if np.isfinite(x) and np.isfinite(y):
                            parts[node] = (x, y)

                if len(parts) == 3:
                    ax.plot(
                        [parts["head"][0], parts["body"][0], parts["tail"][0]],
                        [parts["head"][1], parts["body"][1], parts["tail"][1]],
                        color=partner_cmap(norm_frames(f)),
                        alpha=0.75,
                        linewidth=1.0,
                        zorder=2,
                    )

            # 3) POINTS FOR NODES (persisting; colored by that frame)
            node_marker = {"head": "o", "body": "o", "tail": "o"}
            size_map = {"head": 6, "body": 4, "tail": 2}

            for node in ("head", "body", "tail"):
                anchor_x_means, anchor_y_means = means["anchor"][node]
                partner_x_means, partner_y_means = means["partner"][node]

                for f in frames_upto:
                    if f in anchor_x_means.index:
                        x, y = anchor_x_means.loc[f], anchor_y_means.loc[f]
                        if np.isfinite(x) and np.isfinite(y):
                            ax.scatter(
                                x,
                                y,
                                s=size_map[node],
                                marker=node_marker[node],
                                color=anchor_cmap(norm_frames(f)),
                                alpha=0.9,
                                zorder=3,
                            )

                    if f in partner_x_means.index:
                        x, y = partner_x_means.loc[f], partner_y_means.loc[f]
                        if np.isfinite(x) and np.isfinite(y):
                            ax.scatter(
                                x,
                                y,
                                s=size_map[node],
                                marker=node_marker[node],
                                color=partner_cmap(norm_frames(f)),
                                alpha=0.9,
                                zorder=3,
                            )

        # Draw frame k using the k-th global timepoint (Normalized Frame),
        # but only render data up to that timepoint (everything persists).
        def update(k):
            f_max = ordered_frames[k]

            for cluster, ax in zip(clusters, axes_ax2):
                ax.cla()  # clear ONLY the trace axes each frame (tree + strip stay unchanged)
                ax.set_xlim(*xlim)
                ax.set_ylim(*ylim)
                ax.set_title(cluster, fontsize=16, fontweight="bold")
                ax.axis("off")

                _draw_up_to(ax, cluster_cache[cluster], f_max)

            return axes_ax2

        # animation
        anim = FuncAnimation(fig, update, frames=len(ordered_frames), interval=50, blit=False)

        output = os.path.join(self.directory, "hierarchal_mean_trace_summary.gif")
        anim.save(output, writer=PillowWriter(fps=20))
        plt.close(fig)



                    





















""" RUNNING ANALYSIS PIPELINE """

if __name__ == "__main__":
    # Set your paths
    directory = "/Users/cochral/repos/behavioural-analysis/plots/lrs_paper/clustered_interactions"
    interactions = "/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/umap-pipeline/youngser/test4_F29/cropped_interactions.csv"
    clusters = "/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/umap-pipeline/youngser_2/idt1/pca-data2-F18.csv"
    cluster_name = "Yhat.idt.pca"   # or whatever your cluster column is
    video_path = "/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/umap-pipeline/youngser/videos_original"

    # Create instance
    pipeline = ClusterPipeline(directory, interactions, clusters, cluster_name, video_path)

    # Run methods
    pipeline.loading_data()
    pipeline.anchor_partner()
    # pipeline.barplot_proportion()
    # pipeline.barplot_deviation()
    pipeline.correlation_contact()
    # pipeline.summary_anchor_partner()
    # pipeline.hierarchal_mean_trace_summary()
    # pipeline.hierarchal_mean_trace_summary_gif()
    # pipeline.cluster_transitions()
    # pipeline.cluster_transitions_different()
    # pipeline.cluster_duration_transitions()