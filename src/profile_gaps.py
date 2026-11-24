from pathlib import Path
import pyarrow.parquet as pq
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "metabonet_public.parquet"

SOURCE = "IOBP2"

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

    gap_counts = {}
    total_gaps = 0

    for subject_id, subject_df in data.groupby("id"):
        dates = (
            subject_df["date"]
            .drop_duplicates()
            .sort_values()
        )

        diffs = dates.diff().dropna()

        # Only actual gaps larger than the expected 5 minutes
        gaps = diffs[diffs > pd.Timedelta(minutes=5)]

        total_gaps += len(gaps)

        for gap in gaps:
            minutes = int(gap.total_seconds() / 60)
            gap_counts[minutes] = gap_counts.get(minutes, 0) + 1

    print("\n--- GAP SUMMARY ---")
    print("Total gaps:", total_gaps)

    print("\nMost common gap lengths:")
    for minutes, count in sorted(
        gap_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )[:20]:
        print(f"{minutes:4d} min: {count}")

    # Useful buckets
    buckets = {
        "10 min": 0,
        "15-30 min": 0,
        "35-60 min": 0,
        "1-6 hours": 0,
        ">6 hours": 0,
    }

    for minutes, count in gap_counts.items():
        if minutes == 10:
            buckets["10 min"] += count
        elif 15 <= minutes <= 30:
            buckets["15-30 min"] += count
        elif 35 <= minutes <= 60:
            buckets["35-60 min"] += count
        elif 60 < minutes <= 360:
            buckets["1-6 hours"] += count
        elif minutes > 360:
            buckets[">6 hours"] += count

    print("\nGap buckets:")
    for name, count in buckets.items():
        print(f"{name}: {count}")


if __name__ == "__main__":
    main()