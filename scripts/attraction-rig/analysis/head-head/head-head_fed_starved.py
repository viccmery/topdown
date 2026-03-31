import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import matplotlib.patches as mpatches
import matplotlib as mpl

# ---- Adobe-friendly fonts (must be set BEFORE plotting) ----
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


default_pal = sns.color_palette("deep", 2)

COND_PALETTE = {
    "fed": default_pal[0],         # blue
    "starved": default_pal[1],     # orange
}

# (optional) enforce condition order everywhere
COND_ORDER = ["fed", "starved"]


df = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/2/socially-isolated/fed-starved/nearest_neighbour.csv')
df['role'] = df['track_id'].map({0: 'fed', 1: 'starved'})


bins = np.arange(0, 91, 1)   # edges: 0,1,2,...,90 (1mm bins)
df['bin'] = pd.cut(df['body-body'], bins=bins, include_lowest=True)
df['bin_center'] = df['bin'].apply(lambda x: x.mid)

grouped_speed = (
    df
    .groupby(['file', 'role', 'bin_center'])['speed']
    .mean()
    .reset_index()
)

sns.lineplot(data=grouped_speed, x='bin_center', y='speed', hue='role', errorbar='sd')

plt.xlabel('Nearest Neighbour Distance (mm)', fontsize=12, fontweight='bold')
plt.ylabel('Speed', fontsize=12, fontweight='bold')

plt.title('Speed vs Nearest Neighbour Distance', fontsize=16, fontweight='bold')

plt.tight_layout(rect=[1, 1, 1, 1])


plt.xticks(rotation=45)

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/head-head/fed_starved/nearest_neighbour_distance_speed.png', dpi=300, bbox_inches='tight')
plt.close()


grouped_angle = (
    df
    .groupby(['file', 'role', 'bin_center'])['angle']
    .mean()
    .reset_index()
)

sns.lineplot(data=grouped_angle, x='bin_center', y='angle', hue='role', errorbar='sd')

plt.xlabel('Nearest Neighbour Distance (mm)', fontsize=12, fontweight='bold')
plt.ylabel('Angle', fontsize=12, fontweight='bold')

plt.title('Angle vs Nearest Neighbour Distance', fontsize=16, fontweight='bold')

plt.tight_layout(rect=[1, 1, 1, 1])


plt.xticks(rotation=45)

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/head-head/fed_starved/nearest_neighbour_distance_angle.png', dpi=300, bbox_inches='tight')
plt.close()




plt.figure(figsize=(8,8))
sns.histplot(
    data=df,
    x='speed',              # <-- your speed column
    hue='role',
    bins=50,
    stat='density',         # compare shapes not raw counts
    common_norm=False,      # don't force same total area
    element='step',
    fill=False, palette=COND_PALETTE
)


plt.xlabel('Speed (mm/s)', fontsize=12, fontweight='bold')
plt.ylabel('Density', fontsize=12, fontweight='bold')

plt.title('Speed Distribution', fontsize=16, fontweight='bold')

plt.tight_layout(rect=[1, 1, 1, 1])

plt.xticks(rotation=45)

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/head-head/fed_starved/speed.png', dpi=300, bbox_inches='tight')
plt.savefig(
    '/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/head-head/fed_starved/speed.pdf',
    format='pdf',
    bbox_inches='tight'
)
plt.close()



plt.figure(figsize=(8,8))
sns.histplot(
    data=df,
    x='angle',              # <-- your speed column
    hue='role',
    bins=50,
    stat='density',         # compare shapes not raw counts
    common_norm=False,      # don't force same total area
    element='step',
    fill=False
)


plt.xlabel('Angle (degrees)', fontsize=12, fontweight='bold')
plt.ylabel('Density', fontsize=12, fontweight='bold')

plt.title('Angle Distribution', fontsize=16, fontweight='bold')

plt.tight_layout(rect=[1, 1, 1, 1])

plt.xticks(rotation=45)

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/head-head/fed_starved/angle.png', dpi=300, bbox_inches='tight')
plt.close()



#### WITHIN 10MM ONLY
df_within_10mm = df[df['body-body'] <= 10]

plt.figure(figsize=(8,8))
sns.histplot(
    data=df_within_10mm,
    x='speed',              # <-- your speed column
    hue='role',
    bins=50,
    stat='density',         # compare shapes not raw counts
    common_norm=False,      # don't force same total area
    element='step',
    fill=False
)


plt.xlabel('Speed (mm/s)', fontsize=12, fontweight='bold')
plt.ylabel('Density', fontsize=12, fontweight='bold')

plt.title('Speed Distribution', fontsize=16, fontweight='bold')

plt.tight_layout(rect=[1, 1, 1, 1])

plt.xticks(rotation=45)

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/head-head/fed_starved/speed_within_10mm.png', dpi=300, bbox_inches='tight')
plt.close()


plt.figure(figsize=(8,8))
sns.histplot(
    data=df_within_10mm,
    x='angle',              # <-- your speed column
    hue='role',
    bins=50,
    stat='density',         # compare shapes not raw counts
    common_norm=False,      # don't force same total area
    element='step',
    fill=False
)


plt.xlabel('Angle (degrees)', fontsize=12, fontweight='bold')
plt.ylabel('Density', fontsize=12, fontweight='bold')

plt.title('Angle Distribution', fontsize=16, fontweight='bold')

plt.tight_layout(rect=[1, 1, 1, 1])

plt.xticks(rotation=45)

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/head-head/fed_starved/angle_within_10mm.png', dpi=300, bbox_inches='tight')
plt.close()


#### WITHIN 5MM ONLY
df_within_5mm = df[df['body-body'] <= 5]

plt.figure(figsize=(8,8))
sns.histplot(
    data=df_within_5mm,
    x='speed',              # <-- your speed column
    hue='role',
    bins=50,
    stat='density',         # compare shapes not raw counts
    common_norm=False,      # don't force same total area
    element='step',
    fill=False
)


plt.xlabel('Speed (mm/s)', fontsize=12, fontweight='bold')
plt.ylabel('Density', fontsize=12, fontweight='bold')

plt.title('Speed Distribution', fontsize=16, fontweight='bold')

plt.tight_layout(rect=[1, 1, 1, 1])

plt.xticks(rotation=45)

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/head-head/fed_starved/speed_within_5mm.png', dpi=300, bbox_inches='tight')
plt.close()


plt.figure(figsize=(8,8))
sns.histplot(
    data=df_within_5mm,
    x='angle',              # <-- your speed column
    hue='role',
    bins=50,
    stat='density',         # compare shapes not raw counts
    common_norm=False,      # don't force same total area
    element='step',
    fill=False
)


plt.xlabel('Angle (degrees)', fontsize=12, fontweight='bold')
plt.ylabel('Density', fontsize=12, fontweight='bold')

plt.title('Angle Distribution', fontsize=16, fontweight='bold')

plt.tight_layout(rect=[1, 1, 1, 1])

plt.xticks(rotation=45)

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/head-head/fed_starved/angle_within_5mm.png', dpi=300, bbox_inches='tight')
plt.close()





df = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/2/socially-isolated/fed-starved/head_head_approach_angles.csv')
df['role'] = df['track_id'].map({0: 'fed', 1: 'starved'})

# df = df[df['body_body_distance'] < 5]
# df = df[df['closest_other_node'] == 'tail']

bins = np.arange(0, 91, 1)   # edges: 0,1,2,...,90 (1mm bins)
df['bin'] = pd.cut(df['body_body_distance'], bins=bins, include_lowest=True)
df['bin_center'] = df['bin'].apply(lambda x: x.mid)


plt.figure(figsize=(8,8))
sns.lineplot(
    data=df,
    x='bin_center',
    y='approach_angle',
    hue='role',
    errorbar=('sd'),   # standard error bar, very clean
    marker='o'
)

plt.xlabel("Body–Body Distance (mm)")
plt.ylabel("Approach Angle (degrees)")
plt.title("Approach Angle vs Distance (Fed vs Starved)")
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/head-head/fed_starved/head_head_approach_angle.png', dpi=300, bbox_inches='tight')
plt.close()





df = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/2/socially-isolated/fed-starved/head_head_approach_angles.csv')
df['role'] = df['track_id'].map({0: 'fed', 1: 'starved'})

## identify tailing events

tailing = df[
    (df['closest_other_node'] == 'tail') &
    (df['closest_other_node_distance'] < 2) &  
    (df['approach_angle'] < 30)
]


print(tailing['role'].value_counts())

total_frames_per_file = tailing.groupby(['file', 'role']).size().reset_index(name='total_frames')

sns.barplot(data=total_frames_per_file, x='role', y='total_frames', ci='sd')

plt.xlabel('Role', fontsize=12, fontweight='bold')
plt.ylabel('Number of Tailing Frames', fontsize=12, fontweight='bold')  
plt.ylim(0, None)

plt.title('Tailing Events within 2mm and <30° Approach Angle', fontsize=16, fontweight='bold')
plt.tight_layout(rect=[1, 1, 1, 1])
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/head-head/fed_starved/head_head_tailing_events.png', dpi=300, bbox_inches='tight')
plt.close()



df = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/2/socially-isolated/fed-starved/head_head_approach_angles.csv')
df['role'] = df['track_id'].map({0: 'fed', 1: 'starved'})

## identify tailing events

tailing = df[
    (df['closest_other_node'] == 'tail') & 
    (df['approach_angle'] < 30)
]


print(tailing['role'].value_counts())

total_frames_per_file = tailing.groupby(['file', 'role']).size().reset_index(name='total_frames')

sns.barplot(data=total_frames_per_file, x='role', y='total_frames', ci='sd')

plt.xlabel('Role', fontsize=12, fontweight='bold')
plt.ylabel('Number of Tailing Frames', fontsize=12, fontweight='bold')  
plt.ylim(0, None)

plt.title('Tailing Events <30° Approach Angle (any distance)', fontsize=16, fontweight='bold')
plt.tight_layout(rect=[1, 1, 1, 1])
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/head-head/fed_starved/head_head_tailing_events_any distance.png', dpi=300, bbox_inches='tight')
plt.close()




df = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/2/socially-isolated/fed-starved/head_head_approach_angles.csv')
df['role'] = df['track_id'].map({0: 'fed', 1: 'starved'})

## identify tailing events

tailing = df[
    (df['closest_other_node'] == 'tail') &
    (df['closest_other_node_distance'] < 10) &  
    (df['approach_angle'] < 30)
]


print(tailing['role'].value_counts())

total_frames_per_file = tailing.groupby(['file', 'role']).size().reset_index(name='total_frames')

sns.barplot(data=total_frames_per_file, x='role', y='total_frames', ci='sd')

plt.xlabel('Role', fontsize=12, fontweight='bold')
plt.ylabel('Number of Tailing Frames', fontsize=12, fontweight='bold')  
plt.ylim(0, None)

plt.title('Tailing Events within 10mm and <30° Approach Angle', fontsize=16, fontweight='bold')
plt.tight_layout(rect=[1, 1, 1, 1])
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/head-head/fed_starved/head_head_tailing_events_10mm.png', dpi=300, bbox_inches='tight')
plt.close()



df = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/2/socially-isolated/fed-starved/head_head_approach_angles.csv')
df['role'] = df['track_id'].map({0: 'fed', 1: 'starved'})

## identify tailing events

tailing = df[
    (df['closest_other_node'] == 'tail') &
    (df['closest_other_node_distance'] < 5) &  
    (df['approach_angle'] < 30)
]


print(tailing['role'].value_counts())

total_frames_per_file = tailing.groupby(['file', 'role']).size().reset_index(name='total_frames')

sns.barplot(data=total_frames_per_file, x='role', y='total_frames', ci='sd')

plt.xlabel('Role', fontsize=12, fontweight='bold')
plt.ylabel('Number of Tailing Frames', fontsize=12, fontweight='bold')  
plt.ylim(0, None)

plt.title('Tailing Events within 5mm and <30° Approach Angle', fontsize=16, fontweight='bold')
plt.tight_layout(rect=[1, 1, 1, 1])
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/head-head/fed_starved/head_head_tailing_events_5mm.png', dpi=300, bbox_inches='tight')
plt.close()





tailing = df[
    (df['closest_other_node'] == 'tail') &
    (df['closest_other_node_distance'] < 1) &  
    (df['approach_angle'] < 30)
]


print(tailing['role'].value_counts())

total_frames_per_file = tailing.groupby(['file', 'role']).size().reset_index(name='total_frames')

sns.barplot(data=total_frames_per_file, x='role', y='total_frames', ci='sd', palette=COND_PALETTE)

plt.xlabel('Role', fontsize=12, fontweight='bold')
plt.ylabel('Number of Tailing Frames', fontsize=12, fontweight='bold')  
plt.ylim(0, None)

plt.title('Tailing Events within 1mm and <30° Approach Angle', fontsize=16, fontweight='bold')
plt.tight_layout(rect=[1, 1, 1, 1])
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/head-head/fed_starved/head_head_tailing_events_1mm.png', dpi=300, bbox_inches='tight')
plt.savefig(
    '/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/head-head/fed_starved/head_head_tailing_events_1mm.pdf',
    format='pdf',
    bbox_inches='tight'
)
plt.show()



df = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/2/socially-isolated/fed-starved/closest_contacts_1mm.csv')

df = df[df['Closest Interaction Type'] == 'head_tail']

long = df.melt(
    id_vars=['file'],
    value_vars=['track_0_node', 'track_1_node'],
    var_name='track',
    value_name='node'
)

counts = (
    long.groupby(['file', 'track', 'node'])
        .size()
        .reset_index(name='n_frames')
)
counts['track'] = counts['track'].map({
    'track_0_node': 'Fed',
    'track_1_node': 'Starved'
})


plt.figure(figsize=(8,8))
sns.barplot(
    data=counts,
    x='track',
    y='n_frames',
    hue='node', legend={'0: fed', '1: starved'}, alpha=0.8, edgecolor='black', linewidth=2, ci='sd'
)


plt.xlabel('')
plt.ylim(0, None)
plt.ylabel('Number of frames', fontsize=12, fontweight='bold')
plt.title('1mm Contact Frames: Head vs Tail', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/head-head/fed_starved/1mm_contact_headvtail.png', dpi=300, bbox_inches='tight')
plt.savefig(
    '/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/head-head/fed_starved/1mm_contact_headvtail.pdf',
    format='pdf',
    bbox_inches='tight'
)   
plt.show()