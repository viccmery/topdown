import sys
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


#### -------------------------------------------------------------------------- ####
#### TRYING TO CORRECT H-T SWAPS BEFORE ANYTHING TO SEE IF I CAN                ####
#### -------------------------------------------------------------------------- ####

# df = pd.read_csv("/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/behavioural-detection-testing/2026-02-18_10-44-42_td17-track_0.csv")
# df = df.sort_values("frame").reset_index(drop=True)

# ## vector from tail to head 
# df["vector_tail_head_x"] = df["x_head"] - df["x_tail"]
# df["vector_tail_head_y"] = df["y_head"] - df["y_tail"]

# ## vector from head to tail 
# df["vector_head_tail_x"] = -df["vector_tail_head_x"]
# df["vector_head_tail_y"] = -df["vector_tail_head_y"]

# ## vector from previous frame tail to head
# df["prev_vector_x"] = df["vector_tail_head_x"].shift(1)
# df["prev_vector_y"] = df["vector_tail_head_y"].shift(1)

# ## use dot product to see how much the vectors point in the same direction (positive) or opposite direction (negative)
# ## compare the tail-head and head-tail vectors to see which one is more similar to the previous frame tail-head vector

# df["align_tail_head"] = (
#     df["vector_tail_head_x"] * df["prev_vector_x"] +
#     df["vector_tail_head_y"] * df["prev_vector_y"]
# )

# df["align_head_tail"] = (
#     df["vector_head_tail_x"] * df["prev_vector_x"] +
#     df["vector_head_tail_y"] * df["prev_vector_y"]
# )

# df["flipped"] = df["align_head_tail"] > df["align_tail_head"]
# print(df['flipped'].sum())

# ## second pass = actual corrected coordinates
# df["x_head_corrected"] = np.nan
# df["y_head_corrected"] = np.nan
# df["x_tail_corrected"] = np.nan
# df["y_tail_corrected"] = np.nan
# df["flipped_corrected"] = False

# ## first frame keep as is
# df.loc[0, "x_head_corrected"] = df.loc[0, "x_head"]
# df.loc[0, "y_head_corrected"] = df.loc[0, "y_head"]
# df.loc[0, "x_tail_corrected"] = df.loc[0, "x_tail"]
# df.loc[0, "y_tail_corrected"] = df.loc[0, "y_tail"]

# for i in range(1, len(df)):

#     ## previous corrected frame
#     prev_head_x = df.loc[i - 1, "x_head_corrected"]
#     prev_head_y = df.loc[i - 1, "y_head_corrected"]
#     prev_tail_x = df.loc[i - 1, "x_tail_corrected"]
#     prev_tail_y = df.loc[i - 1, "y_tail_corrected"]

#     ## if first pass says FALSE, keep raw coords
#     if df.loc[i, "flipped"] == False:
#         df.loc[i, "x_head_corrected"] = df.loc[i, "x_head"]
#         df.loc[i, "y_head_corrected"] = df.loc[i, "y_head"]
#         df.loc[i, "x_tail_corrected"] = df.loc[i, "x_tail"]
#         df.loc[i, "y_tail_corrected"] = df.loc[i, "y_tail"]

#     ## if first pass says TRUE, test no-flip vs flip properly
#     else:
#         noflip_dist = (
#             np.hypot(df.loc[i, "x_head"] - prev_head_x, df.loc[i, "y_head"] - prev_head_y) +
#             np.hypot(df.loc[i, "x_tail"] - prev_tail_x, df.loc[i, "y_tail"] - prev_tail_y)
#         )

#         flip_dist = (
#             np.hypot(df.loc[i, "x_tail"] - prev_head_x, df.loc[i, "y_tail"] - prev_head_y) +
#             np.hypot(df.loc[i, "x_head"] - prev_tail_x, df.loc[i, "y_head"] - prev_tail_y)
#         )

#         if flip_dist < noflip_dist:
#             df.loc[i, "flipped_corrected"] = True
#             df.loc[i, "x_head_corrected"] = df.loc[i, "x_tail"]
#             df.loc[i, "y_head_corrected"] = df.loc[i, "y_tail"]
#             df.loc[i, "x_tail_corrected"] = df.loc[i, "x_head"]
#             df.loc[i, "y_tail_corrected"] = df.loc[i, "y_head"]
#         else:
#             df.loc[i, "x_head_corrected"] = df.loc[i, "x_head"]
#             df.loc[i, "y_head_corrected"] = df.loc[i, "y_head"]
#             df.loc[i, "x_tail_corrected"] = df.loc[i, "x_tail"]
#             df.loc[i, "y_tail_corrected"] = df.loc[i, "y_tail"]

# print(df["flipped_corrected"].sum())

# df.to_csv(
#     "/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/behavioural-detection-testing/2026-02-18_10-44-42_td17-track_0-flipped.csv",
#     index=False
# )
















# import pandas as pd
# import numpy as np

# df = pd.read_csv("/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/behavioural-detection-testing/2026-02-18_10-44-42_td17-track_3.csv")
# df = df.sort_values("frame").reset_index(drop=True)

# # ------------------------------------------------------------------
# # raw first-pass vector check, only used to TRIGGER a 6-frame window
# # ------------------------------------------------------------------

# df["vector_tail_head_x"] = df["x_head"] - df["x_tail"]
# df["vector_tail_head_y"] = df["y_head"] - df["y_tail"]

# df["vector_head_tail_x"] = -df["vector_tail_head_x"]
# df["vector_head_tail_y"] = -df["vector_tail_head_y"]

# df["prev_vector_x"] = df["vector_tail_head_x"].shift(1)
# df["prev_vector_y"] = df["vector_tail_head_y"].shift(1)

# df["align_tail_head"] = (
#     df["vector_tail_head_x"] * df["prev_vector_x"] +
#     df["vector_tail_head_y"] * df["prev_vector_y"]
# )

# df["align_head_tail"] = (
#     df["vector_head_tail_x"] * df["prev_vector_x"] +
#     df["vector_head_tail_y"] * df["prev_vector_y"]
# )

# # this only triggers checking mode
# # df["flipped"] = df["align_head_tail"] > df["align_tail_head"]

# df["flipped"] = (
#     (df["align_head_tail"] > df["align_tail_head"]) &
#     (df["instance_score"].notna())
# )

# # ------------------------------------------------------------------
# # corrected columns
# # ------------------------------------------------------------------

# df["x_head_corrected"] = np.nan
# df["y_head_corrected"] = np.nan
# df["x_tail_corrected"] = np.nan
# df["y_tail_corrected"] = np.nan
# df["flipped_corrected"] = False
# df["checked_window"] = False

# # first frame stays as-is
# df.loc[0, "x_head_corrected"] = df.loc[0, "x_head"]
# df.loc[0, "y_head_corrected"] = df.loc[0, "y_head"]
# df.loc[0, "x_tail_corrected"] = df.loc[0, "x_tail"]
# df.loc[0, "y_tail_corrected"] = df.loc[0, "y_tail"]

# i = 1
# while i < len(df):

#     # outside a trigger window: just copy raw frame unless this row triggers one
#     if df.loc[i, "flipped"] == False:
#         df.loc[i, "x_head_corrected"] = df.loc[i, "x_head"]
#         df.loc[i, "y_head_corrected"] = df.loc[i, "y_head"]
#         df.loc[i, "x_tail_corrected"] = df.loc[i, "x_tail"]
#         df.loc[i, "y_tail_corrected"] = df.loc[i, "y_tail"]
#         i += 1
#         continue

#     # --------------------------------------------------------------
#     # raw True at row i -> now CHECK row i and next 5 rows properly
#     # using previous CORRECTED vector
#     # --------------------------------------------------------------
#     end_idx = min(i + 5, len(df) - 1)

#     for j in range(i, end_idx + 1):

#         prev_head_x = df.loc[j - 1, "x_head_corrected"]
#         prev_head_y = df.loc[j - 1, "y_head_corrected"]
#         prev_tail_x = df.loc[j - 1, "x_tail_corrected"]
#         prev_tail_y = df.loc[j - 1, "y_tail_corrected"]

#         # previous corrected vector: tail -> head
#         prev_vector_x = prev_head_x - prev_tail_x
#         prev_vector_y = prev_head_y - prev_tail_y

#         # current frame as raw orientation
#         tail_head_x = df.loc[j, "x_head"] - df.loc[j, "x_tail"]
#         tail_head_y = df.loc[j, "y_head"] - df.loc[j, "y_tail"]

#         # current frame as swapped orientation
#         head_tail_x = -tail_head_x
#         head_tail_y = -tail_head_y

#         align_tail_head = (
#             tail_head_x * prev_vector_x +
#             tail_head_y * prev_vector_y
#         )

#         align_head_tail = (
#             head_tail_x * prev_vector_x +
#             head_tail_y * prev_vector_y
#         )

#         df.loc[j, "checked_window"] = True

#         # ONLY NOW decide whether to swap
#         if align_head_tail > align_tail_head:
#             df.loc[j, "flipped_corrected"] = True
#             df.loc[j, "x_head_corrected"] = df.loc[j, "x_tail"]
#             df.loc[j, "y_head_corrected"] = df.loc[j, "y_tail"]
#             df.loc[j, "x_tail_corrected"] = df.loc[j, "x_head"]
#             df.loc[j, "y_tail_corrected"] = df.loc[j, "y_head"]
#         else:
#             df.loc[j, "x_head_corrected"] = df.loc[j, "x_head"]
#             df.loc[j, "y_head_corrected"] = df.loc[j, "y_head"]
#             df.loc[j, "x_tail_corrected"] = df.loc[j, "x_tail"]
#             df.loc[j, "y_tail_corrected"] = df.loc[j, "y_tail"]

#     # after checking this 6-frame block, continue from the frame after it
#     i = end_idx + 1

# print("Raw trigger Trues:", df["flipped"].sum())
# print("Actually flipped after checking:", df["flipped_corrected"].sum())

# df.to_csv(
#     "/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/behavioural-detection-testing/2026-02-18_10-44-42_td17-track_3-flipped.csv",
#     index=False
# )











import sys
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/behavioural-detection-testing/2026-02-18_10-44-42_td17-track_0.csv")
df = df.sort_values("frame").reset_index(drop=True)

# ------------------------------------------------------------------
# raw first-pass vector check, only used to TRIGGER a 6-frame window
# ------------------------------------------------------------------

df["vector_tail_head_x"] = df["x_head"] - df["x_tail"]
df["vector_tail_head_y"] = df["y_head"] - df["y_tail"]

df["vector_head_tail_x"] = -df["vector_tail_head_x"]
df["vector_head_tail_y"] = -df["vector_tail_head_y"]

df["prev_vector_x"] = df["vector_tail_head_x"].shift(1)
df["prev_vector_y"] = df["vector_tail_head_y"].shift(1)

df["align_tail_head"] = (
    df["vector_tail_head_x"] * df["prev_vector_x"] +
    df["vector_tail_head_y"] * df["prev_vector_y"]
)

df["align_head_tail"] = (
    df["vector_head_tail_x"] * df["prev_vector_x"] +
    df["vector_head_tail_y"] * df["prev_vector_y"]
)

# raw trigger only
df["flipped"] = (
    (df["align_head_tail"] > df["align_tail_head"]) &
    (df["instance_score"].notna())
)

# ------------------------------------------------------------------
# corrected columns
# ------------------------------------------------------------------

df["x_head_corrected"] = np.nan
df["y_head_corrected"] = np.nan
df["x_tail_corrected"] = np.nan
df["y_tail_corrected"] = np.nan
df["flipped_corrected"] = False
df["checked_window"] = False

# optional debug columns
df["align_tail_head_corrected"] = np.nan
df["align_head_tail_corrected"] = np.nan
df["tail_dist_noflip"] = np.nan
df["tail_dist_flip"] = np.nan

# first frame stays as-is
df.loc[0, "x_head_corrected"] = df.loc[0, "x_head"]
df.loc[0, "y_head_corrected"] = df.loc[0, "y_head"]
df.loc[0, "x_tail_corrected"] = df.loc[0, "x_tail"]
df.loc[0, "y_tail_corrected"] = df.loc[0, "y_tail"]

i = 1
while i < len(df):

    # if this row does not trigger a check window, just copy raw values
    if df.loc[i, "flipped"] == False:
        df.loc[i, "x_head_corrected"] = df.loc[i, "x_head"]
        df.loc[i, "y_head_corrected"] = df.loc[i, "y_head"]
        df.loc[i, "x_tail_corrected"] = df.loc[i, "x_tail"]
        df.loc[i, "y_tail_corrected"] = df.loc[i, "y_tail"]
        i += 1
        continue

    # raw True at row i -> check this row and next 5 rows
    end_idx = min(i + 5, len(df) - 1)

    for j in range(i, end_idx + 1):

        # if this frame has no instance_score, keep raw and move on
        if pd.isna(df.loc[j, "instance_score"]):
            df.loc[j, "x_head_corrected"] = df.loc[j, "x_head"]
            df.loc[j, "y_head_corrected"] = df.loc[j, "y_head"]
            df.loc[j, "x_tail_corrected"] = df.loc[j, "x_tail"]
            df.loc[j, "y_tail_corrected"] = df.loc[j, "y_tail"]
            df.loc[j, "checked_window"] = True
            continue

        # previous corrected frame
        prev_head_x = df.loc[j - 1, "x_head_corrected"]
        prev_head_y = df.loc[j - 1, "y_head_corrected"]
        prev_tail_x = df.loc[j - 1, "x_tail_corrected"]
        prev_tail_y = df.loc[j - 1, "y_tail_corrected"]

        # previous corrected vector: tail -> head
        prev_vector_x = prev_head_x - prev_tail_x
        prev_vector_y = prev_head_y - prev_tail_y

        # current raw coordinates
        curr_head_x = df.loc[j, "x_head"]
        curr_head_y = df.loc[j, "y_head"]
        curr_tail_x = df.loc[j, "x_tail"]
        curr_tail_y = df.loc[j, "y_tail"]

        # no-flip orientation
        tail_head_x = curr_head_x - curr_tail_x
        tail_head_y = curr_head_y - curr_tail_y

        # flipped orientation
        head_tail_x = -tail_head_x
        head_tail_y = -tail_head_y

        # vector alignment scores
        align_tail_head = (
            tail_head_x * prev_vector_x +
            tail_head_y * prev_vector_y
        )

        align_head_tail = (
            head_tail_x * prev_vector_x +
            head_tail_y * prev_vector_y
        )

        # tail continuity scores
        tail_dist_noflip = np.hypot(curr_tail_x - prev_tail_x, curr_tail_y - prev_tail_y)
        tail_dist_flip = np.hypot(curr_head_x - prev_tail_x, curr_head_y - prev_tail_y)

        df.loc[j, "checked_window"] = True
        df.loc[j, "align_tail_head_corrected"] = align_tail_head
        df.loc[j, "align_head_tail_corrected"] = align_head_tail
        df.loc[j, "tail_dist_noflip"] = tail_dist_noflip
        df.loc[j, "tail_dist_flip"] = tail_dist_flip

        # final decision:
        # only flip if BOTH:
        # 1) flipped orientation aligns better with previous corrected vector
        # 2) flipped orientation keeps the tail closer to previous corrected tail
        if (align_head_tail > align_tail_head) and (tail_dist_flip < tail_dist_noflip):
            df.loc[j, "flipped_corrected"] = True
            df.loc[j, "x_head_corrected"] = curr_tail_x
            df.loc[j, "y_head_corrected"] = curr_tail_y
            df.loc[j, "x_tail_corrected"] = curr_head_x
            df.loc[j, "y_tail_corrected"] = curr_head_y
        else:
            df.loc[j, "x_head_corrected"] = curr_head_x
            df.loc[j, "y_head_corrected"] = curr_head_y
            df.loc[j, "x_tail_corrected"] = curr_tail_x
            df.loc[j, "y_tail_corrected"] = curr_tail_y

    # move to frame after the checked window
    i = end_idx + 1

print("Raw trigger Trues:", df["flipped"].sum())
print("Actually flipped after checking:", df["flipped_corrected"].sum())

df.to_csv(
    "/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/behavioural-detection-testing/2026-02-18_10-44-42_td17-track_0-flipped.csv",
    index=False
)





import pandas as pd
import numpy as np


def correct_head_tail_swaps(track_df):
    df = track_df.copy()
    df = df.sort_values("frame").reset_index(drop=True)

    # raw first-pass vector check, only used to trigger a 6-frame window
    df["vector_tail_head_x"] = df["x_head"] - df["x_tail"]
    df["vector_tail_head_y"] = df["y_head"] - df["y_tail"]

    df["vector_head_tail_x"] = -df["vector_tail_head_x"]
    df["vector_head_tail_y"] = -df["vector_tail_head_y"]

    df["prev_vector_x"] = df["vector_tail_head_x"].shift(1)
    df["prev_vector_y"] = df["vector_tail_head_y"].shift(1)

    df["align_tail_head"] = (
        df["vector_tail_head_x"] * df["prev_vector_x"] +
        df["vector_tail_head_y"] * df["prev_vector_y"]
    )

    df["align_head_tail"] = (
        df["vector_head_tail_x"] * df["prev_vector_x"] +
        df["vector_head_tail_y"] * df["prev_vector_y"]
    )

    df["flipped"] = (
        (df["align_head_tail"] > df["align_tail_head"]) &
        (df["instance_score"].notna())
    )

    # corrected columns
    df["x_head_corrected"] = np.nan
    df["y_head_corrected"] = np.nan
    df["x_tail_corrected"] = np.nan
    df["y_tail_corrected"] = np.nan
    df["flipped_corrected"] = False
    df["checked_window"] = False

    # debug columns
    df["align_tail_head_corrected"] = np.nan
    df["align_head_tail_corrected"] = np.nan
    df["tail_dist_noflip"] = np.nan
    df["tail_dist_flip"] = np.nan

    # first frame stays as-is
    df.loc[0, "x_head_corrected"] = df.loc[0, "x_head"]
    df.loc[0, "y_head_corrected"] = df.loc[0, "y_head"]
    df.loc[0, "x_tail_corrected"] = df.loc[0, "x_tail"]
    df.loc[0, "y_tail_corrected"] = df.loc[0, "y_tail"]

    i = 1
    while i < len(df):

        # if this row does not trigger a check window, just copy raw values
        if df.loc[i, "flipped"] == False:
            df.loc[i, "x_head_corrected"] = df.loc[i, "x_head"]
            df.loc[i, "y_head_corrected"] = df.loc[i, "y_head"]
            df.loc[i, "x_tail_corrected"] = df.loc[i, "x_tail"]
            df.loc[i, "y_tail_corrected"] = df.loc[i, "y_tail"]
            i += 1
            continue

        # raw True at row i -> check this row and next 5 rows
        end_idx = min(i + 5, len(df) - 1)

        for j in range(i, end_idx + 1):

            # if this frame has no instance_score, keep raw and move on
            if pd.isna(df.loc[j, "instance_score"]):
                df.loc[j, "x_head_corrected"] = df.loc[j, "x_head"]
                df.loc[j, "y_head_corrected"] = df.loc[j, "y_head"]
                df.loc[j, "x_tail_corrected"] = df.loc[j, "x_tail"]
                df.loc[j, "y_tail_corrected"] = df.loc[j, "y_tail"]
                df.loc[j, "checked_window"] = True
                continue

            # previous corrected frame
            prev_head_x = df.loc[j - 1, "x_head_corrected"]
            prev_head_y = df.loc[j - 1, "y_head_corrected"]
            prev_tail_x = df.loc[j - 1, "x_tail_corrected"]
            prev_tail_y = df.loc[j - 1, "y_tail_corrected"]

            # previous corrected vector: tail -> head
            prev_vector_x = prev_head_x - prev_tail_x
            prev_vector_y = prev_head_y - prev_tail_y

            # current raw coordinates
            curr_head_x = df.loc[j, "x_head"]
            curr_head_y = df.loc[j, "y_head"]
            curr_tail_x = df.loc[j, "x_tail"]
            curr_tail_y = df.loc[j, "y_tail"]

            # no-flip orientation
            tail_head_x = curr_head_x - curr_tail_x
            tail_head_y = curr_head_y - curr_tail_y

            # flipped orientation
            head_tail_x = -tail_head_x
            head_tail_y = -tail_head_y

            # vector alignment scores
            align_tail_head = (
                tail_head_x * prev_vector_x +
                tail_head_y * prev_vector_y
            )

            align_head_tail = (
                head_tail_x * prev_vector_x +
                head_tail_y * prev_vector_y
            )

            # tail continuity scores
            tail_dist_noflip = np.hypot(curr_tail_x - prev_tail_x, curr_tail_y - prev_tail_y)
            tail_dist_flip = np.hypot(curr_head_x - prev_tail_x, curr_head_y - prev_tail_y)

            df.loc[j, "checked_window"] = True
            df.loc[j, "align_tail_head_corrected"] = align_tail_head
            df.loc[j, "align_head_tail_corrected"] = align_head_tail
            df.loc[j, "tail_dist_noflip"] = tail_dist_noflip
            df.loc[j, "tail_dist_flip"] = tail_dist_flip

            # final decision
            if (align_head_tail > align_tail_head) and (tail_dist_flip < tail_dist_noflip):
                df.loc[j, "flipped_corrected"] = True
                df.loc[j, "x_head_corrected"] = curr_tail_x
                df.loc[j, "y_head_corrected"] = curr_tail_y
                df.loc[j, "x_tail_corrected"] = curr_head_x
                df.loc[j, "y_tail_corrected"] = curr_head_y
            else:
                df.loc[j, "x_head_corrected"] = curr_head_x
                df.loc[j, "y_head_corrected"] = curr_head_y
                df.loc[j, "x_tail_corrected"] = curr_tail_x
                df.loc[j, "y_tail_corrected"] = curr_tail_y

        i = end_idx + 1

    return df



df = pd.read_csv("/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/behavioural-detection-testing/2026-02-18_10-44-42_td17-track_0.csv")

df_corrected = (
    df
    .groupby("track_id", group_keys=False)
    .apply(correct_head_tail_swaps)
    .reset_index(drop=True)
)

print("Raw trigger Trues:", df_corrected["flipped"].sum())
print("Actually flipped after checking:", df_corrected["flipped_corrected"].sum())

df_corrected.to_csv(
    "/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/behavioural-detection-testing/2026-02-18_10-44-42_td17-track_0-flipped.csv",
    index=False
)   