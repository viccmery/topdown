
######################

# %% PRINT UNIQUE TRACKS AND CHECK GAPS IN EACH TRACK 

import sys
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re

USER = "meryv"

home_folder = Path("/Users") / USER
desktop_folder = home_folder / "Desktop"
sleap_folder = desktop_folder / "sleap"
csv_folder = sleap_folder / "clean-csv"
feather_folder = sleap_folder / "csv-tofeather"
feather_folder.mkdir(parents=True, exist_ok=True)

#change everytime
csv_file = csv_folder / "2026-06-26_td4_anosmic.tracks.2026-06-26_td4_anosmic.analysis.csv"
feather_file = feather_folder / "2026-06-26_td4_anosmic.tracks.feather"


print(f"Reading CSVs from: {csv_folder}")
print(f"Saving feathers to: {feather_folder}")

df = pd.read_csv(csv_file)

print(df['track'].unique())

track_gaps = {}

for track_id in df['track'].unique():
    track_df = df[df['track'] == track_id].sort_values(by='frame_idx')
    frames = track_df['frame_idx'].tolist()

    missing_frames = [f for f in range(min(frames), max(frames) + 1) if f not in frames]
    
    if missing_frames:
        track_gaps[track_id] = missing_frames

if track_gaps:
    print("Tracks with missing frames:")
    for track, gaps in track_gaps.items():
        print(f"Track {track} has missing frames: {gaps}")
else:
    print("No missing frames detected in any track.")



############################
# %% MISSING FRAMES IN GENERAL

frames = df['frame_idx'].unique()  # Get unique frames in your DataFrame
all_frames = set(range(0, 3601))  # Set of all expected frame numbers (from 0 to 3600)

# Find missing frames by checking the difference between the full set and the frames in the DataFrame
missing_frames = list(all_frames - set(frames))

# If there are missing frames, print them
if missing_frames:
    print(f"Missing frames: {missing_frames}")
else:
    print("No missing frames.")

################################################################################
# %% HIGHLIGHT TRACK JUMPS

df['displacement'] = df.groupby('track').apply(
    lambda x: np.sqrt(x['body.x'].diff()**2 + x['body.y'].diff()**2)
).reset_index(level=0, drop=True)

jumped_tracks = df[df['displacement'] > 30][['frame_idx', 'track', 'displacement']]
print(jumped_tracks)


# %%

# %% PRINT FRAMES OF CERTAIN TRACK

print(df[df['track'] == 'track_15']['frame_idx'])


######################
# %% CHECK FRAMES IN WHICH ALL ANIMALS WERE NOT PREDICTED

# frames with a mix of NaN and values in instance.score indicate missing predictions on those frames

frames_with_mixed_scores = df.groupby('frame_idx')['instance.score'].apply(
    lambda x: x.isna().any() and x.notna().any()
)

mixed_frames = frames_with_mixed_scores[frames_with_mixed_scores].index

print(f"Frames:{mixed_frames}")




######################
# %% RENAME COLUMNS 

df.rename(columns={'frame_idx': 'frame', 'track': 'track_id', 'head.x': 'x_head', 'head.y': 'y_head', 'body.x': 'x_body', 'body.y': 'y_body', 'tail.x': 'x_tail', 'tail.y': 'y_tail', 'instance.score': 'instance_score', 'head.score': 'score_head', 'body.score': 'score_body', 'tail.score': 'score_tail'}, inplace=True)

df['track_id'] = df['track_id'].str.replace('track_', '', regex=True).astype(int)

df = df.drop(columns=['displacement'])


print(df.columns)

print(df.head())


# %% CSV -> FEATHER AND SLP
df.to_feather(feather_file)
print(f"Saved: {feather_file.name}")

#############################################################

# %%

# ---------------------------------------------------------------------
# LOOP THROUGH ALL CSV FILES
# ---------------------------------------------------------------------

