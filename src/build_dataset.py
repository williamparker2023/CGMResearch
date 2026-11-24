from pathlib import Path
import pyarrow.parquet as pq
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "metabonet_public.parquet"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "iobp2_segments.parquet"

SOURCE = "IOBP2"
MIN_SEGMENT_LENGTH = 288  # 24 hours at 5-minute intervals

COLUMNS = [
    "CGM",
    "date",
    "id",
    "source_file",
]


def main():
    pf = pq.ParquetFile(DATA_PATH)

    chunks = []

    for i in range(pf.num_row_groups):
        print(f"Processing row group {i + 1}/{pf.num_row_groups}")

        df = pf.read_row_group(i, columns=COLUMNS).to_pandas()

        df = df[
            (df["source_file"] == SOURCE)
            & (df["CGM"].notna())
            & (df["date"].notna())
        ]

        if not df.empty:
            chunks.append(df[["id", "date", "CGM"]])

    data = pd.concat(chunks, ignore_index=True)

    data = data.drop_duplicates(subset=["id", "date"])
    data = data.sort_values(["id", "date"]).reset_index(drop=True)

    output_segments = []

    for subject_id, subject_df in data.groupby("id"):
        subject_df = subject_df.sort_values("date").copy()

        # Start a new segment whenever the time difference is not exactly 5 minutes
        diffs = subject_df["date"].diff()

        subject_df["local_segment"] = (
            (diffs != pd.Timedelta(minutes=5))
            .cumsum()
        )

        for local_segment, segment_df in subject_df.groupby("local_segment"):
            if len(segment_df) < MIN_SEGMENT_LENGTH:
                continue

            segment_df = segment_df.copy()

            output_segments.append(segment_df[["id", "date", "CGM"]])

    processed = []

    for segment_id, segment_df in enumerate(output_segments):
        segment_df = segment_df.copy()
        segment_df["segment_id"] = segment_id
        processed.append(segment_df)

    final_df = pd.concat(processed, ignore_index=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    final_df = final_df[
        ["id", "segment_id", "date", "CGM"]
    ]

    final_df.to_parquet(OUTPUT_PATH, index=False)

    print("\n--- PROCESSED DATASET SUMMARY ---")
    print("Rows:", len(final_df))
    print("Subjects:", final_df["id"].nunique())
    print("Segments:", final_df["segment_id"].nunique())

    segment_lengths = (
        final_df.groupby("segment_id")
        .size()
    )

    print("\nSegment length statistics:")
    print(segment_lengths.describe())

    print("\nSaved to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()