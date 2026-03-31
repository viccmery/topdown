import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import matplotlib.patches as mpatches
import os
import matplotlib as mpl

df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/group-housed/individual_approach_responses_4.csv')
df1['condition'] = 'GH'

df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/socially-isolated/individual_approach_responses_4.csv')
df2['condition'] = 'SI'

# df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/pseudo-n10/group-housed/individual_approach_responses.csv')
# df2['condition'] = 'PSEUDO'

df = pd.concat([df1, df2], ignore_index=True)


per_video = (
    df
    .groupby(['condition', 'filename']) #, 'filename'])
    .agg(
        n_encounters=('touch', 'size'),   # total encounters
        n_touch=('touch', 'sum'),          # encounters with touch
        touch_rate=('touch', 'mean')       # fraction that touched
    )
    .reset_index()
)



plt.figure(figsize=(2,6))

sns.stripplot(
    data=per_video,
    x='condition',
    y='touch_rate',
    jitter=True,
    alpha=0.7
)


sns.pointplot(
    data=per_video,
    x='condition',
    y='touch_rate',
    errorbar=('ci', 95),
    color='black',
    markers='_',
    linestyle='none'
)

plt.title('Potential Interactions\n(Threshold = 4 cm)')
plt.ylabel('P(touch)')
plt.xlabel('')
# plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/potential_interactions/potential_interactions_individual_4.png', dpi=300, bbox_inches='tight')
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/potential_interactions/potential_interactions_individual_4.pdf', format='pdf', bbox_inches='tight')
plt.show()








########################## OLD INDIIVDUAL POTENTIAL INTERACTIONS ANALYSIS ##########################

# bins = np.arange(0, 42, 2)   # 2–4, 4–6, …, 18–20
# labels = [f"{bins[i]}–{bins[i+1]}" for i in range(len(bins)-1)]

# df['distance_bin'] = pd.cut(
#     df['distance'],
#     bins=bins,
#     labels=labels,
#     include_lowest=True,
#     right=False
# )


# ## APPROACH AVOID NEUTRAL COUNTS

# prob = (
#     df.groupby(['condition','filename','distance_bin'])['outcome']
#       .value_counts(normalize=True)
#       .rename('probability')
#       .reset_index()
# )


# plt.figure(figsize=(6,6))

# sns.lineplot(data=prob, x='distance_bin', y='probability', hue='outcome', errorbar=('ci', 95))

# plt.ylabel('P(approach)')
# plt.xlabel('Distance Bin (mm)')
# plt.ylim(0, 1)
# plt.title('Approach Probability by Distance')
# plt.legend(title='Condition')
# plt.tight_layout()
# plt.show()




# fig, axes = plt.subplots(1, 2, figsize=(15, 4), sharey=True)

# for ax, condition in zip(axes, prob['condition'].unique()):
#     sub = prob[prob['condition'] == condition]

#     # mean curve per condition across videos + 95% CI
#     sns.lineplot(
#         data=sub,
#         x='distance_bin',
#         y='probability',
#         hue='outcome',
#         errorbar=('ci', 95),
#         ax=ax
#     )

#     ax.set_title(f'{condition}')
#     ax.set_xlabel('Distance (mm)')
#     ax.set_ylabel('' if ax is axes[0] else '')
#     ax.set_ylim(0, 1)

#     # rotate x tick labels so bins are readable
#     ax.tick_params(axis='x', rotation=90)

#     # keep legend only on last plot (or first, your choice)
#     if ax is not axes[-1]:
#         ax.get_legend().remove()

# axes[-1].legend(title='Condition', loc='upper right')
# plt.tight_layout()
# plt.show()



# fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

# for ax, outcome in zip(axes, prob['outcome'].unique()):
#     sub = prob[prob['outcome'] == outcome]

#     # mean curve per condition across videos + 95% CI
#     sns.lineplot(
#         data=sub,
#         x='distance_bin',
#         y='probability',
#         hue='condition',
#         errorbar=('ci', 95),
#         ax=ax
#     )

#     ax.set_title(f'{outcome}')
#     ax.set_xlabel('Distance (mm)')
#     ax.set_ylabel('' if ax is axes[0] else '')
#     ax.set_ylim(0, 1)

#     # rotate x tick labels so bins are readable
#     ax.tick_params(axis='x', rotation=90)

#     # keep legend only on last plot (or first, your choice)
#     if ax is not axes[-1]:
#         ax.get_legend().remove()

# axes[-1].legend(title='Condition', loc='upper right')
# plt.tight_layout()
# plt.show()




# ## APPROACH PROBABILITY PER DISTANCE BIN

# per_video = (
#     df
#     .groupby(['condition', 'filename', 'distance_bin'])
#     .apply(lambda g: pd.Series({
#         'n_events': len(g),
#         'p_approach': (g['outcome'] == 'approach').mean()
#     }))
#     .reset_index()
# )


# ## APPROACH PROBABILITY PER DISTANCE BIN AND ANGLE NODE
# per_video_angle = (
#     df
#     .groupby(['condition', 'filename', 'distance_bin', 'angle_node'])
#     .apply(lambda g: pd.Series({
#         'n_events': len(g),
#         'p_approach': (g['outcome'] == 'approach').mean()
#     }))
#     .reset_index()
# )

# ## APPROACH PROBABILITY PER DISTANCE BIN AND CLOSEST NODE
# per_video_distnode = (
#     df
#     .groupby(['condition', 'filename', 'distance_bin', 'closest_node'])
#     .apply(lambda g: pd.Series({
#         'n_events': len(g),
#         'p_approach': (g['outcome'] == 'approach').mean()
#     }))
#     .reset_index()
# )


# ## APPROACH PROBABILITY PER DISTANCE BIN

# plt.figure(figsize=(6,6))

# sns.lineplot(data=per_video, x='distance_bin', y='p_approach', hue='condition', errorbar=('ci', 95))

# plt.ylabel('P(approach)')
# plt.xlabel('Distance Bin (mm)')
# plt.ylim(0, 1)
# plt.title('Approach Probability by Distance')
# plt.legend(title='Condition')
# plt.tight_layout()
# plt.close()



# ## APPROACH PROBABILITY PER DISTANCE BIN AND ANGLE NODE


# plt.figure(figsize=(6,6))

# sns.lineplot(data=per_video_angle, x='distance_bin', y='p_approach', hue='angle_node', errorbar=('ci', 95))

# plt.ylabel('P(approach)')
# plt.xlabel('Distance Bin (mm)')
# plt.ylim(0, 1)
# plt.title('Approach Probability by Distance')
# plt.legend(title='Condition')
# plt.tight_layout()
# plt.close()




# nodes = ['head', 'body', 'tail']

# fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

# for ax, node in zip(axes, nodes):
#     sub = per_video_angle[per_video_angle['angle_node'] == node]

#     # mean curve per condition across videos + 95% CI
#     sns.lineplot(
#         data=sub,
#         x='distance_bin',
#         y='p_approach',
#         hue='condition',
#         errorbar=('ci', 95),
#         ax=ax
#     )

#     ax.set_title(f'Closest node: {node}')
#     ax.set_xlabel('Distance (mm)')
#     ax.set_ylabel('P(approach)' if ax is axes[0] else '')
#     ax.set_ylim(0, 1)

#     # rotate x tick labels so bins are readable
#     ax.tick_params(axis='x', rotation=90)

#     # keep legend only on last plot (or first, your choice)
#     if ax is not axes[-1]:
#         ax.get_legend().remove()

# axes[-1].legend(title='Condition', loc='upper right')
# plt.tight_layout()
# plt.close()




# plt.figure(figsize=(6,6))

# sns.lineplot(data=per_video_distnode, x='distance_bin', y='p_approach', hue='closest_node', errorbar=('ci', 95))

# plt.ylabel('P(approach)')
# plt.xlabel('Distance Bin (mm)')
# plt.ylim(0, 1)
# plt.title('Approach Probability by Distance')
# plt.legend(title='Condition')
# plt.tight_layout()
# plt.show()



# nodes = ['head', 'body', 'tail']

# fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

# for ax, node in zip(axes, nodes):
#     sub = per_video_distnode[per_video_distnode['closest_node'] == node]

#     # mean curve per condition across videos + 95% CI
#     sns.lineplot(
#         data=sub,
#         x='distance_bin',
#         y='p_approach',
#         hue='condition',
#         errorbar=('ci', 95),
#         ax=ax
#     )

#     ax.set_title(f'Closest node: {node}')
#     ax.set_xlabel('Distance (mm)')
#     ax.set_ylabel('P(approach)' if ax is axes[0] else '')
#     ax.set_ylim(0, 1)

#     # rotate x tick labels so bins are readable
#     ax.tick_params(axis='x', rotation=90)

#     # keep legend only on last plot (or first, your choice)
#     if ax is not axes[-1]:
#         ax.get_legend().remove()

# axes[-1].legend(title='Condition', loc='upper right')
# plt.tight_layout()
# plt.show()
