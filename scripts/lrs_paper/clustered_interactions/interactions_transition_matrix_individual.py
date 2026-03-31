import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import matplotlib.patches as mpatches
import os
import matplotlib as mpl
import networkx as nx
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import ListedColormap


"""
PURPOSE
------------


"""
# -----------------------------------------------------
# DEFINING CLUSTER TO CLUSTER TRANSITIONS PER INDIVIDUAL
# -----------------------------------------------------
output = '/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/umap-pipeline/youngser_2/idt1/transitions_between_clusters'

df_interaction = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/cropped_interactions.csv')
df_cluster = pd.read_csv("/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/umap-pipeline/youngser_2/idt1/pca-data2-F18.csv")
cluster_name = "Yhat.idt.pca"

df = pd.merge(
            df_interaction, 
            df_cluster[['interaction_id', cluster_name]], 
            on='interaction_id', 
            how='inner')


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



# -----------------------------------------------------
# HEATMAP: TRANSITION PROBABILITIES BETWEEN CLUSTER
# -----------------------------------------------------

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharex=False, sharey=False)
vmax = max(isolated_transitions_normalised.max().max(),grouped_transitions_normalised.max().max())

# --- ISO ---
hm1 = sns.heatmap(
    isolated_transitions_normalised,
    ax=ax1,
    vmin=0, vmax=vmax,
    cmap="viridis",
    square=True,
    cbar=False
)
ax1.set_title("Isolated")
ax1.set_xlabel("Next cluster")
ax1.set_ylabel("Current cluster")

# force tick label rotation THE reliable way:
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)
ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0)  # usually nicer; set 90 if you want

# --- GROUP ---
hm2 = sns.heatmap(
    grouped_transitions_normalised,
    ax=ax2,
    vmin=0, vmax=vmax,
    cmap="viridis",
    square=True,
    cbar=False
)
ax2.set_title("Group Housed")
ax2.set_xlabel("Next cluster")
ax2.set_ylabel("")  # optional: avoid repeating label

ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)
ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)

fig.subplots_adjust(right=0.88, wspace=0.25)

# add a new axis for the colorbar (this is "position it right of ax2")
cax = fig.add_axes([0.90, 0.15, 0.02, 0.70])  # [left, bottom, width, height]

cbar = fig.colorbar(hm1.collections[0], cax=cax)
cbar.set_label("Transition probability")

plt.savefig(os.path.join(output, "iso_vs_group_transition_heatmaps.png"), dpi=300, bbox_inches="tight")
plt.close()


# -------------------------------------------------------
# CIRCLE GRAPHS: TRANSITION PROBABILITIES BETWEEN CLUSTER
# -------------------------------------------------------

# --- pick your matrices ---
M_iso = isolated_transitions_normalised.copy()
M_grp = grouped_transitions_normalised.copy()

# 1) Make sure both have the same row/col order (same cluster set)
all_clusters = sorted(set(M_iso.index) | set(M_iso.columns) | set(M_grp.index) | set(M_grp.columns))

M_iso = M_iso.reindex(index=all_clusters, columns=all_clusters, fill_value=0)
M_grp = M_grp.reindex(index=all_clusters, columns=all_clusters, fill_value=0)

# 2) Global scaling so widths/colors comparable between iso and group
global_max = max(M_iso.to_numpy().max(), M_grp.to_numpy().max())
if global_max == 0:
    raise ValueError("No transitions found (all probabilities are 0).")

# 3) Build graphs from matrices (include ALL non-zero edges)
def matrix_to_digraph(M):
    G = nx.DiGraph()
    for c in M.index:
        G.add_node(int(c))
    for i in M.index:
        for j in M.columns:
            w = float(M.loc[i, j])
            if w > 0:
                G.add_edge(int(i), int(j), weight=w)
    return G

G_iso = matrix_to_digraph(M_iso)
G_grp = matrix_to_digraph(M_grp)

# 4) Same circular positions for both
pos = nx.circular_layout(all_clusters)

# --- colour/width scaling shared across both plots ---
cmap = plt.cm.viridis
norm = mpl.colors.Normalize(vmin=0, vmax=global_max)

def weight_to_lw(w, min_w=0.2, max_w=6.0):
    return min_w + (w / global_max) * (max_w - min_w)

def weight_to_alpha(w, min_a=0.05, max_a=0.95):
    return min_a + (w / global_max) * (max_a - min_a)

# ---- OUTSIDE self-loop drawer ----
def draw_self_loop(
    ax, xy, w,
    loop_offset=0.20,
    loop_size=0.11,
    rad=1.0,
    arrow_size=14,
    min_w=0.2, max_w=6.0,
    min_a=0.05, max_a=0.95,
):
    x, y = xy
    v = np.array([x, y], dtype=float)
    r = np.linalg.norm(v)
    if r == 0:
        return
    u = v / r                  # outward unit vector
    perp = np.array([-u[1], u[0]])

    # loop centre pushed outward
    c = v + loop_offset * u

    # start/end points around that centre (so it becomes a loop)
    start = c - loop_size * perp + 0.15 * loop_size * u
    end   = c + loop_size * perp + 0.15 * loop_size * u

    lw = min_w + (w / global_max) * (max_w - min_w)
    a  = min_a + (w / global_max) * (max_a - min_a)
    color = cmap(norm(w))

    patch = FancyArrowPatch(
        posA=start, posB=end,
        arrowstyle="-|>",
        mutation_scale=arrow_size,
        connectionstyle=f"arc3,rad={rad}",
        linewidth=lw,
        color=color,
        alpha=a,
        zorder=2
    )
    ax.add_patch(patch)

# ---- normal edge drawer (curved, no clutter into nodes) ----
def draw_edge(
    ax, xy_u, xy_v, w,
    rad=0.12,
    arrow_size=14,
    shrink=18,
    min_w=0.2, max_w=6.0,
    min_a=0.05, max_a=0.95,
):
    lw = min_w + (w / global_max) * (max_w - min_w)
    a  = min_a + (w / global_max) * (max_a - min_a)
    color = cmap(norm(w))

    patch = FancyArrowPatch(
        posA=xy_u, posB=xy_v,
        arrowstyle="-|>",
        mutation_scale=arrow_size,
        connectionstyle=f"arc3,rad={rad}",
        linewidth=lw,
        color=color,
        alpha=a,
        shrinkA=shrink,   # keeps arrows from entering node circles
        shrinkB=shrink,
        zorder=1
    )
    ax.add_patch(patch)

def draw_transition_circle(ax, G, title):
    ax.set_title(title)
    ax.axis("off")

    # nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=700)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=9)

    # edges (ALL non-zero)
    for u, v, d in G.edges(data=True):
        w = float(d["weight"])
        if w <= 0:
            continue

        if u == v:
            # self-loops outside the ring
            draw_self_loop(
                ax, pos[u], w,
                loop_offset=0.1,
                loop_size=0.08,
                rad=0.8,
                arrow_size=14
            )
        else:
            draw_edge(
                ax, pos[u], pos[v], w,
                rad=0.12,
                arrow_size=14,
                shrink=18
            )

# 6) Plot side-by-side with ONE shared colorbar outside
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

draw_transition_circle(ax1, G_iso, "Isolated")
draw_transition_circle(ax2, G_grp, "Group housed")

# shared colorbar (outside)
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

fig.subplots_adjust(right=0.88, wspace=0.25)
cax = fig.add_axes([0.90, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
cbar = fig.colorbar(sm, cax=cax)
cbar.set_label("Transition probability")

plt.savefig(os.path.join(output, "iso_vs_group_transition_circlegraphs.png"), dpi=300, bbox_inches="tight")
plt.close()



# ---------------------------------------------------------------------
# CIRCLE GRAPHS: TRANSITION PROBABILITIES BETWEEN CLUSTER (THRESHOLDED)
# ---------------------------------------------------------------------

M_iso = isolated_transitions_normalised.copy()
M_grp = grouped_transitions_normalised.copy()

# 1) Make sure both have the same row/col order (same cluster set)
all_clusters = sorted(set(M_iso.index) | set(M_iso.columns) | set(M_grp.index) | set(M_grp.columns))

M_iso = M_iso.reindex(index=all_clusters, columns=all_clusters, fill_value=0)
M_grp = M_grp.reindex(index=all_clusters, columns=all_clusters, fill_value=0)

# 2) Global scaling so widths/colors comparable between iso and group
global_max = max(M_iso.to_numpy().max(), M_grp.to_numpy().max())
if global_max == 0:
    raise ValueError("No transitions found (all probabilities are 0).")

# 3) Build graphs from matrices (include ALL non-zero edges)
def matrix_to_digraph(M):
    G = nx.DiGraph()
    for c in M.index:
        G.add_node(int(c))
    for i in M.index:
        for j in M.columns:
            w = float(M.loc[i, j])
            if w > 0.1: ## threshold changed 
                G.add_edge(int(i), int(j), weight=w)
    return G

G_iso = matrix_to_digraph(M_iso)
G_grp = matrix_to_digraph(M_grp)

# 4) Same circular positions for both
pos = nx.circular_layout(all_clusters)

# --- colour/width scaling shared across both plots ---
cmap = plt.cm.viridis
norm = mpl.colors.Normalize(vmin=0, vmax=global_max)

def weight_to_lw(w, min_w=0.2, max_w=6.0):
    return min_w + (w / global_max) * (max_w - min_w)

def weight_to_alpha(w, min_a=0.05, max_a=0.95):
    return min_a + (w / global_max) * (max_a - min_a)

# ---- OUTSIDE self-loop drawer ----
def draw_self_loop(
    ax, xy, w,
    loop_offset=0.20,
    loop_size=0.11,
    rad=1.0,
    arrow_size=14,
    min_w=0.2, max_w=6.0,
    min_a=0.05, max_a=0.95,
):
    x, y = xy
    v = np.array([x, y], dtype=float)
    r = np.linalg.norm(v)
    if r == 0:
        return
    u = v / r                  # outward unit vector
    perp = np.array([-u[1], u[0]])

    # loop centre pushed outward
    c = v + loop_offset * u

    # start/end points around that centre (so it becomes a loop)
    start = c - loop_size * perp + 0.15 * loop_size * u
    end   = c + loop_size * perp + 0.15 * loop_size * u

    lw = min_w + (w / global_max) * (max_w - min_w)
    a  = min_a + (w / global_max) * (max_a - min_a)
    color = cmap(norm(w))

    patch = FancyArrowPatch(
        posA=start, posB=end,
        arrowstyle="-|>",
        mutation_scale=arrow_size,
        connectionstyle=f"arc3,rad={rad}",
        linewidth=lw,
        color=color,
        alpha=a,
        zorder=2
    )
    ax.add_patch(patch)

# ---- normal edge drawer (curved, no clutter into nodes) ----
def draw_edge(
    ax, xy_u, xy_v, w,
    rad=0.12,
    arrow_size=14,
    shrink=18,
    min_w=0.2, max_w=6.0,
    min_a=0.05, max_a=0.95,
):
    lw = min_w + (w / global_max) * (max_w - min_w)
    a  = min_a + (w / global_max) * (max_a - min_a)
    color = cmap(norm(w))

    patch = FancyArrowPatch(
        posA=xy_u, posB=xy_v,
        arrowstyle="-|>",
        mutation_scale=arrow_size,
        connectionstyle=f"arc3,rad={rad}",
        linewidth=lw,
        color=color,
        alpha=a,
        shrinkA=shrink,   # keeps arrows from entering node circles
        shrinkB=shrink,
        zorder=1
    )
    ax.add_patch(patch)

def draw_transition_circle(ax, G, title):
    ax.set_title(title)
    ax.axis("off")

    # nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=700)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=9)

    # edges (ALL non-zero)
    for u, v, d in G.edges(data=True):
        w = float(d["weight"])
        if w <= 0:
            continue

        if u == v:
            # self-loops outside the ring
            draw_self_loop(
                ax, pos[u], w,
                loop_offset=0.1,
                loop_size=0.08,
                rad=0.8,
                arrow_size=14
            )
        else:
            draw_edge(
                ax, pos[u], pos[v], w,
                rad=0.12,
                arrow_size=14,
                shrink=18
            )

# 6) Plot side-by-side with ONE shared colorbar outside
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

draw_transition_circle(ax1, G_iso, "Isolated")
draw_transition_circle(ax2, G_grp, "Group housed")

# shared colorbar (outside)
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

fig.subplots_adjust(right=0.88, wspace=0.25)
cax = fig.add_axes([0.90, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
cbar = fig.colorbar(sm, cax=cax)
cbar.set_label("Transition probability")

plt.savefig(os.path.join(output, "iso_vs_group_transition_circlegraphs_10_percent.png"), dpi=300, bbox_inches="tight")
plt.close()



# ----------------------------------------------------------------
# HEATMAP: DIFFERENCE IN TRANSITION PROBABILITIES BETWEEN CLUSTERS 
# ----------------------------------------------------------------

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

# Plot unweighted difference
lim = float(np.nanmax(np.abs(P_diff.to_numpy())))
if lim == 0:
    lim = 1e-6  # avoid zero-range colormap if identical

plt.figure(figsize=(10, 8))
sns.heatmap(
    P_diff,
    cmap="RdBu",      # negative=red, positive=blue
    center=0,
    vmin=-lim, vmax=lim,
    square=True
)
plt.xlabel("To cluster (k+1)")
plt.ylabel("From cluster (k)")
plt.title("Transition difference: Group − Iso")
plt.tight_layout()
plt.savefig(os.path.join(output, "transition_diff_group_minus_iso.png"), dpi=300)
plt.savefig(os.path.join(output, "transition_diff_group_minus_iso.pdf"), format="pdf", bbox_inches="tight")
plt.close()


# --------------------------------------------------------------------------------------------------
# HEATMAP: DIFFERENCE IN TRANSITION PROBABILITIES BETWEEN CLUSTERS (DOWNWEIGHTS LESS COMMON CLUSTERS)
# --------------------------------------------------------------------------------------------------

# raw count matrices aligned (needed for support weighting)
C_iso = isolated_transitions.reindex(index=all_clusters, columns=all_clusters, fill_value=0)
C_grp = grouped_transitions.reindex(index=all_clusters, columns=all_clusters, fill_value=0)

# per-condition support per starting cluster (row sums)
support_iso = C_iso.sum(axis=1).astype(float)
support_grp = C_grp.sum(axis=1).astype(float)

# normalise each to 0..1 within that condition
W_iso = support_iso / support_iso.max() if support_iso.max() > 0 else support_iso * 0.0
W_grp = support_grp / support_grp.max() if support_grp.max() > 0 else support_grp * 0.0

# combine fairly (equal weight to iso and group)
W_fair = 0.5 * (W_iso + W_grp)

# apply row-wise weighting to the difference matrix
P_diff_weighted_fair = P_diff.mul(W_fair, axis=0)

# plot
lim = float(np.nanmax(np.abs(P_diff_weighted_fair.to_numpy())))
if lim == 0:
    lim = 1e-6

plt.figure(figsize=(10, 8))
sns.heatmap(
    P_diff_weighted_fair,
    cmap="RdBu",
    center=0,
    vmin=-lim, vmax=lim,
    square=True
)
plt.xlabel("To cluster (k+1)")
plt.ylabel("From cluster (k)")
plt.title("Fair weighted transition difference: (Group − Iso) × row-support")
plt.tight_layout()
plt.savefig(os.path.join(output, "transition_diff_weighted_group_minus_iso.png"), dpi=300)
plt.close()



# ---------------------------------------------------------------------
# CIRCLE GRAPH: DIFFERENCE IN TRANSITION PROBABILITIES BETWEEN CLUSTERS
# ----------------------------------------------------------------------

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
cmap = plt.cm.RdBu  # red=negative, blue=positive (we'll set norm around 0)
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

nx.draw_networkx_nodes(G_diff, pos, ax=ax, node_size=700)
nx.draw_networkx_labels(G_diff, pos, ax=ax, font_size=9)

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
        shrinkA=18,
        shrinkB=18
    )
    ax.add_patch(patch)

sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

cax = fig.add_axes([0.90, 0.25, 0.015, 0.5])  # smaller bar
cbar = fig.colorbar(sm, cax=cax)
cbar.set_label("P(group) − P(iso)")

plt.savefig(os.path.join(output, "transition_diff_circlegraph.png"), dpi=300, bbox_inches="tight")
plt.savefig(os.path.join(output, "transition_diff_circlegraph.pdf"), format="pdf", bbox_inches="tight")
plt.close()







# -------------------------------------------------------------
# COLLAPSE CLUSTERS INTO SHORT, MEDIUM OR LONG DURATION CLASSES
# -------------------------------------------------------------

# 1) map clusters -> duration class
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

support_iso = C_iso_dur.sum(axis=1).astype(float)
support_grp = C_grp_dur.sum(axis=1).astype(float)

W_iso = support_iso / support_iso.max() if support_iso.max() > 0 else support_iso * 0.0
W_grp = support_grp / support_grp.max() if support_grp.max() > 0 else support_grp * 0.0
W_fair = 0.5 * (W_iso + W_grp)

P_diff_dur_weighted = P_diff_dur.mul(W_fair, axis=0)


# --------------------------------------------------
# CIRCLE GRAPH: DURATION CLASS TRANSITION DIFFERENCE
# --------------------------------------------------

# 5) circlegraph for duration diff (Group - Iso)
D = P_diff_dur.copy()   # or P_diff_dur if you want unweighted # P_diff_dur_weighted

nodes = dur_order
pos = nx.circular_layout(nodes)

lim = float(np.nanmax(np.abs(D.to_numpy())))
if lim == 0:
    lim = 1e-6

cmap = plt.cm.RdBu
norm = mpl.colors.TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)

def w_to_lw(w, min_w=0.6, max_w=8.0):
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

fig, ax = plt.subplots(1, 1, figsize=(7, 7))
ax.set_title("Transition Likelihood")
ax.axis("off")

nx.draw_networkx_nodes(Gd, pos, ax=ax, node_size=1200)
nx.draw_networkx_labels(Gd, pos, ax=ax, font_size=10)

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
        mutation_scale=16,
        connectionstyle=f"arc3,rad={rad}",
        linewidth=lw,
        color=color,
        alpha=a,
        shrinkA=22,
        shrinkB=22
    )
    ax.add_patch(patch)

# smaller colorbar
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cax = fig.add_axes([0.95, 0.28, 0.02, 0.45])
cbar = fig.colorbar(sm, cax=cax)
cbar.set_label("P(group) - P(iso)")

plt.savefig(os.path.join(output, "duration_transition_duration_diff_circlegraph.png"), dpi=300, bbox_inches="tight")
plt.savefig(os.path.join(output, "duration_transition_duration_diff_circlegraph.pdf"), format="pdf", bbox_inches="tight")

plt.close()



# --------------------------------------
# RASTA PLOT: DURATION CLUSTER OVER TIME
# --------------------------------------


from matplotlib.colors import ListedColormap

# use the duration mapping you already defined
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
dur_code = {k: i for i, k in enumerate(dur_order)}

# build per-larva interaction index
df_raster = df.copy()
df_raster["duration"] = df_raster["cluster"].astype(int).map(dur_map)
df_raster["dur_code"] = df_raster["duration"].map(dur_code)

df_raster = df_raster.sort_values(["file","track_id","time","interaction_id"])

df_raster["interaction_idx"] = (
    df_raster
    .groupby(["file", "track_id"])
    .cumcount()
)

# def make_duration_raster(d):
#     # order larvae by total number of interactions (longest on top)
#     order = (
#         d.groupby(["file", "track_id"])["interaction_idx"]
#         .max()
#         .sort_values(ascending=False)
#     )
#     larvae = order.index.tolist()
#     max_len = int(order.max()) + 1

#     mat = np.full((len(larvae), max_len), np.nan)
#     row = {pid: i for i, pid in enumerate(larvae)}

#     for (file, tid, idx, code) in d[["file", "track_id", "interaction_idx", "dur_code"]].itertuples(index=False):
#         mat[row[(file, tid)], idx] = code

#     return mat

# mat_iso = make_duration_raster(df_raster[df_raster["condition"] == "iso"])
# mat_grp = make_duration_raster(df_raster[df_raster["condition"] == "group"])


# def make_duration_raster_sorted_by_first(d):
#     # for each larva (file,track_id): total interactions + first duration class
#     summary = (
#         d.sort_values("interaction_idx")
#          .groupby(["file", "track_id"])
#          .agg(
#              max_idx=("interaction_idx", "max"),
#              first_code=("dur_code", "first")   # first interaction type (0/1/2)
#          )
#     )

#     # sort rows: first by first interaction type, then by length (desc)
#     summary = summary.sort_values(["first_code", "max_idx"], ascending=[True, False])

#     larvae = summary.index.tolist()
#     max_len = int(summary["max_idx"].max()) + 1

#     mat = np.full((len(larvae), max_len), np.nan)
#     row = {pid: i for i, pid in enumerate(larvae)}

#     for file, tid, idx, code in d[["file", "track_id", "interaction_idx", "dur_code"]].itertuples(index=False):
#         mat[row[(file, tid)], idx] = code

#     return mat

def make_duration_raster_sorted(d):
    # ensure correct order first
    d = d.sort_values(["file", "track_id", "time", "interaction_id"])

    # per-larva summary
    summary = (
        d.groupby(["file", "track_id"])
         .agg(
             mean_code=("dur_code", "mean"),   # overall tendency
             n=("dur_code", "count")           # number of interactions
         )
    )

    # sort larvae:
    # 1) mostly short -> mostly long
    # 2) longer sequences on top
    summary = summary.sort_values(
        ["mean_code", "n"],
        ascending=[True, False]
    )

    larvae = summary.index.tolist()
    max_len = d["interaction_idx"].max() + 1

    mat = np.full((len(larvae), max_len), np.nan)
    row = {pid: i for i, pid in enumerate(larvae)}

    for file, tid, idx, code in d[["file", "track_id", "interaction_idx", "dur_code"]].itertuples(index=False):
        mat[row[(file, tid)], idx] = code

    return mat


# then use:
mat_iso = make_duration_raster_sorted(df_raster[df_raster["condition"] == "iso"])
mat_grp = make_duration_raster_sorted(df_raster[df_raster["condition"] == "group"])


# colors (bright, readable, black background)
cmap = ListedColormap([
    "aliceblue",  # short (purple)
    "#56B19C",  # medium (green)
    "darkgreen",  # long (orange)
])
cmap.set_bad("lightgray")  # for NaN (no interaction)

fig, ax = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

im0 = ax[0].imshow(mat_iso, aspect="auto", interpolation="nearest",
                   cmap=cmap, vmin=0, vmax=2)
ax[0].set_title("Isolated")
ax[0].set_xlabel("Interaction number")
ax[0].set_ylabel("Larva (file, track_id)")

im1 = ax[1].imshow(mat_grp, aspect="auto", interpolation="nearest",
                   cmap=cmap, vmin=0, vmax=2)
ax[1].set_title("Group housed")
ax[1].set_xlabel("Interaction number")
ax[1].set_ylabel("")

# shared legend
cbar = fig.colorbar(im0, ax=ax, fraction=0.03, pad=0.02)
cbar.set_ticks([0, 1, 2])
cbar.set_ticklabels(["short", "medium", "long"])
cbar.set_label("Interaction duration")

plt.savefig(
    os.path.join(output, "duration_raster_per_larva_interaction_index.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.savefig(
    os.path.join(output, "duration_raster_per_larva_interaction_index.pdf"),
    format="pdf",
    bbox_inches="tight"
)
plt.close()
