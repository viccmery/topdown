import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import matplotlib.patches as mpatches
import os
import matplotlib as mpl

##### GROUPHOUSE 
df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/group-housed/individual_approach_responses_2.csv')
df2['condition'] = 'GH'
df2['response_distance'] = '2'

df3 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/group-housed/individual_approach_responses_3.csv')
df3['condition'] = 'GH'
df3['response_distance'] = '3'

df4 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/group-housed/individual_approach_responses_4.csv')
df4['condition'] = 'GH'
df4['response_distance'] = '4'

df5 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/group-housed/individual_approach_responses_5.csv')
df5['condition'] = 'GH'
df5['response_distance'] = '5'

df6 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/group-housed/individual_approach_responses_6.csv')
df6['condition'] = 'GH'
df6['response_distance'] = '6'

df7 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/group-housed/individual_approach_responses_7.csv')
df7['condition'] = 'GH'
df7['response_distance'] = '7'

df8 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/group-housed/individual_approach_responses_8.csv')
df8['condition'] = 'GH'
df8['response_distance'] = '8'

df9 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/group-housed/individual_approach_responses_9.csv')
df9['condition'] = 'GH'
df9['response_distance'] = '9'

df10 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/group-housed/individual_approach_responses_10.csv')
df10['condition'] = 'GH'
df10['response_distance'] = '10'

##### SOCIALLY ISOLATED  
si2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/socially-isolated/individual_approach_responses_2.csv')
si2['condition'] = 'SI'
si2['response_distance'] = '2'

si3 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/socially-isolated/individual_approach_responses_3.csv')
si3['condition'] = 'SI'
si3['response_distance'] = '3'

si4 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/socially-isolated/individual_approach_responses_4.csv')
si4['condition'] = 'SI'
si4['response_distance'] = '4'

si5 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/socially-isolated/individual_approach_responses_5.csv')
si5['condition'] = 'SI'
si5['response_distance'] = '5'

si6 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/socially-isolated/individual_approach_responses_6.csv')
si6['condition'] = 'SI'
si6['response_distance'] = '6'

si7 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/socially-isolated/individual_approach_responses_7.csv')
si7['condition'] = 'SI'
si7['response_distance'] = '7'

si8 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/socially-isolated/individual_approach_responses_8.csv')
si8['condition'] = 'SI'
si8['response_distance'] = '8'

si9 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/socially-isolated/individual_approach_responses_9.csv')
si9['condition'] = 'SI'
si9['response_distance'] = '9'

si10 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/socially-isolated/individual_approach_responses_10.csv')
si10['condition'] = 'SI'
si10['response_distance'] = '10'

df_all = pd.concat([df2, df3, df4, df5, df6, df7, df8, df9, df10, si2, si3, si4, si5, si6, si7, si8, si9, si10], ignore_index=True)
df_all['response_distance'] = pd.to_numeric(df_all['response_distance'])



per_video = (
    df_all
    .groupby(['condition', 'filename', 'response_distance']) #, 'filename'])
    .agg(
        n_encounters=('touch', 'size'),   # total encounters
        n_touch=('touch', 'sum'),          # encounters with touch
        touch_rate=('touch', 'mean') ,      # fraction that touched
        no_touch_rate=('touch', lambda s: 1 - s.mean()),
    )
    .reset_index()
)



plt.figure(figsize=(4,6))

sns.lineplot(data=per_video, x='response_distance', y='touch_rate', hue='condition', errorbar=('ci', 95), palette={'GH': 'steelblue', 'SI': 'darkorange'})

plt.title('Potential Interactions\n', fontsize=14, fontweight='bold')
plt.ylabel('P(touch)', fontsize=12, fontweight='bold')
plt.xlabel('Distance (mm)', fontsize=12, fontweight='bold')
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/potential_interactions/potential_interactions_individual_combined.png', dpi=300, bbox_inches='tight')
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/potential_interactions/potential_interactions_individual_combined.pdf', format='pdf', bbox_inches='tight')
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
