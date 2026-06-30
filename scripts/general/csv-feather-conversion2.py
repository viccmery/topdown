from pathlib import Path
import numpy as np
import pandas as pd


INPUT_CSV = Path(
    "/Users/meryv/Desktop/sleap/clean/"
    "2026-03-31_12-08-02_td11_9047.tracks.000_"
    "2026-03-31_12-08-02_td11_9047.analysis.csv"
)

OUTPUT_FEATHER = Path(
    "/Users/meryv/Desktop/sleap/"
    "2026-03-31_12-08-02_td11_9047.static-filled.tracks.feather"
)

OUTPUT_CHANGE_LOG = OUTPUT_FEATHER.with_suffix(".change_log.txt")


# For 3600 zero-indexed frames, use 0-3599.
EXPECTED_START_FRAME = 0
EXPECTED_END_FRAME = 3599

EXPECTED_TRACKS = ["track_0", "track_1", "track_2", "track_3", "track_4"]

POSITION_COLUMNS = [
    "head.x",
    "head.y",
    "body.x",
    "body.y",
    "tail.x",
    "tail.y",
]

SCORE_COLUMNS = [
    "instance.score",
    "head.score",
    "body.score",
    "tail.score",
]

#change according to notes
DIGGING_INTERVALS = {
    "track_0": [(2630, 3108), (3361, 3599)],
    "track_1": [(2158, 2258), (3029, 3079)],
    "track_2": [
        (1995, 2200),
        (2253, 2258),
        (2283, 2344)
        ],
    "track_3": [
        (1760, 2194),
        (2244, 2591),
        (2621, 2791),
        (2872, 3094),
        (3162, 3536),
    ],
    "track_4": [
        (1987, 2342)
        ],
}

COLUMN_RENAMES = {
    "frame_idx": "frame",
    "track": "track_id",
    "head.x": "x_head",
    "head.y": "y_head",
    "body.x": "x_body",
    "body.y": "y_body",
    "tail.x": "x_tail",
    "tail.y": "y_tail",
    "instance.score": "instance_score",
    "head.score": "score_head",
    "body.score": "score_body",
    "tail.score": "score_tail",
}


def make_complete_grid():
    frames = range(EXPECTED_START_FRAME, EXPECTED_END_FRAME + 1)

    grid = pd.MultiIndex.from_product(
        [frames, EXPECTED_TRACKS],
        names=["frame_idx", "track"],
    ).to_frame(index=False)

    return grid


def static_fill_from_digging_notes(df):
    df = df.copy()

    df["row_was_present_in_original_csv"] = df["row_was_present_in_original_csv"].fillna(False)
    df["imputed"] = False
    df["imputation_reason"] = ""

    change_log = []

    for track_id, intervals in DIGGING_INTERVALS.items():
        for start_frame, end_frame in intervals:

            interval_mask = (
                (df["track"] == track_id)
                & (df["frame_idx"] >= start_frame)
                & (df["frame_idx"] <= end_frame)
            )

            interval_df = df.loc[interval_mask]

            # Use only originally observed rows to estimate static position.
            observed_interval_df = interval_df[
                interval_df["row_was_present_in_original_csv"]
            ]

            if observed_interval_df.empty:
                change_log.append(
                    f"{track_id} frames {start_frame}-{end_frame}: skipped; "
                    "no originally observed rows available in this interval."
                )
                continue

            fill_values = observed_interval_df[POSITION_COLUMNS].median()
            fill_values = fill_values.dropna()

            if fill_values.empty:
                change_log.append(
                    f"{track_id} frames {start_frame}-{end_frame}: skipped; "
                    "no usable coordinate values available."
                )
                continue

            # Fill only rows with missing coordinate values inside the static interval.
            missing_coord_mask = interval_mask & df[POSITION_COLUMNS].isna().any(axis=1)

            n_rows_to_fill = int(missing_coord_mask.sum())

            for col, value in fill_values.items():
                df.loc[missing_coord_mask, col] = df.loc[
                    missing_coord_mask, col
                ].fillna(value)

            if n_rows_to_fill > 0:
                df.loc[missing_coord_mask, "imputed"] = True
                df.loc[missing_coord_mask, "imputation_reason"] = "static_digging_fill"

                # Scores should not pretend to be real SLEAP confidence scores.
                existing_score_cols = [c for c in SCORE_COLUMNS if c in df.columns]
                df.loc[missing_coord_mask, existing_score_cols] = np.nan

            change_log.append(
                f"{track_id} frames {start_frame}-{end_frame}: filled {n_rows_to_fill} "
                "rows with missing coordinates using the median observed static position "
                "from originally present rows in that interval."
            )

    return df, change_log


def validate_output(df):
    expected_n_frames = EXPECTED_END_FRAME - EXPECTED_START_FRAME + 1
    expected_n_rows = expected_n_frames * len(EXPECTED_TRACKS)

    problems = []

    if len(df) != expected_n_rows:
        problems.append(
            f"Expected {expected_n_rows} rows, but found {len(df)} rows."
        )

    counts = df.groupby("track")["frame_idx"].nunique()

    for track_id in EXPECTED_TRACKS:
        if track_id not in counts.index:
            problems.append(f"{track_id} is missing entirely.")
        elif counts.loc[track_id] != expected_n_frames:
            problems.append(
                f"{track_id} has {counts.loc[track_id]} frames instead of "
                f"{expected_n_frames}."
            )

    unexpected_tracks = sorted(set(df["track"].dropna()) - set(EXPECTED_TRACKS))
    if unexpected_tracks:
        problems.append(f"Unexpected tracks found: {unexpected_tracks}")

    duplicated = df.duplicated(subset=["frame_idx", "track"]).sum()
    if duplicated > 0:
        problems.append(f"Found {duplicated} duplicated frame-track rows.")

    if problems:
        raise ValueError(
            "Validation failed:\n" + "\n".join(f"- {p}" for p in problems)
        )

    print("Validation passed.")
    print(f"Final rows: {len(df)}")
    print(f"Expected rows: {expected_n_rows}")
    print(f"Frames per track: {expected_n_frames}")
    print(f"Tracks: {EXPECTED_TRACKS}")


def main():
    df_original = pd.read_csv(INPUT_CSV)

    change_log = []

    change_log.append(f"Input CSV: {INPUT_CSV}")
    change_log.append(f"Original shape: {df_original.shape[0]} rows x {df_original.shape[1]} columns")
    change_log.append("Original CSV was not overwritten.")
    change_log.append("")

    # Restrict to the five expected tracks.
    unexpected_tracks = sorted(set(df_original["track"].dropna()) - set(EXPECTED_TRACKS))

    if unexpected_tracks:
        change_log.append(f"Removed unexpected tracks: {unexpected_tracks}")
        df_original = df_original[df_original["track"].isin(EXPECTED_TRACKS)].copy()
    else:
        change_log.append("No unexpected tracks found.")

    # Mark original rows before merging onto complete grid.
    df_original["row_was_present_in_original_csv"] = True

    # Create complete 3600-frame x 5-track structure.
    grid = make_complete_grid()

    df = grid.merge(
        df_original,
        on=["frame_idx", "track"],
        how="left",
    )

    change_log.append("")
    change_log.append(
        "Created a complete frame-track grid containing every expected frame "
        f"from {EXPECTED_START_FRAME} to {EXPECTED_END_FRAME} for tracks "
        f"{EXPECTED_TRACKS}."
    )

    missing_after_grid = df["row_was_present_in_original_csv"].isna().sum()
    change_log.append(
        f"Added {missing_after_grid} frame-track rows that were absent from the original CSV."
    )

    # Fill only annotated static digging gaps.
    df, fill_log = static_fill_from_digging_notes(df)

    change_log.append("")
    change_log.append("Static digging fill:")
    change_log.extend(fill_log)

    # Validate before renaming.
    validate_output(df)

    # Rename columns for downstream analysis.
    df = df.rename(columns=COLUMN_RENAMES)

    df["track_id"] = (
        df["track_id"]
        .str.replace("track_", "", regex=False)
        .astype(int)
    )

    df = df.sort_values(["frame", "track_id"]).reset_index(drop=True)

    change_log.append("")
    change_log.append("Columns renamed:")
    for old, new in COLUMN_RENAMES.items():
        if new in df.columns:
            change_log.append(f"{old} -> {new}")

    change_log.append("")
    change_log.append(
        "Converted track_id from strings such as 'track_0' to integers such as 0."
    )

    change_log.append("")
    change_log.append("Added QC columns:")
    change_log.append("- row_was_present_in_original_csv")
    change_log.append("- imputed")
    change_log.append("- imputation_reason")

    change_log.append("")
    change_log.append(f"Final shape: {df.shape[0]} rows x {df.shape[1]} columns")
    change_log.append(f"Output Feather: {OUTPUT_FEATHER}")

    OUTPUT_FEATHER.parent.mkdir(parents=True, exist_ok=True)

    df.to_feather(OUTPUT_FEATHER)

    with open(OUTPUT_CHANGE_LOG, "w") as f:
        f.write("\n".join(change_log))

    print(f"Saved Feather file: {OUTPUT_FEATHER}")
    print(f"Saved change log: {OUTPUT_CHANGE_LOG}")


if __name__ == "__main__":
    main()