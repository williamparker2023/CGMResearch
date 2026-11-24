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


def longest_five_minute_run(dates):
    dates = dates.sort_values().drop_duplicates()

    if len(dates) == 0:
        return 0

    diffs = dates.diff()

    longest = 1
    current = 1

    for diff in diffs.iloc[1:]:
        if diff == pd.Timedelta(minutes=5):
            current += 1
            longest = max(longest, current)
        else:
            current = 1

    return longest


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

    print(f"\nRows for {SOURCE}: {len(data)}")
    print(f"Subjects: {data['id'].nunique()}")

    results = []

    for subject_id, subject_df in data.groupby("id"):
        subject_df = subject_df.sort_values("date")

        dates = subject_df["date"]
        diffs = dates.diff()

        five_minute_fraction = (
            (diffs == pd.Timedelta(minutes=5)).sum()
            / max(len(diffs) - 1, 1)
        )

        longest_run = longest_five_minute_run(dates)

        results.append({
            "id": subject_id,
            "rows": len(subject_df),
            "five_minute_fraction": five_minute_fraction,
            "longest_run": longest_run,
            "longest_run_days": longest_run * 5 / 60 / 24,
        })

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        "longest_run",
        ascending=False
    )

    print("\n--- TOP SUBJECTS BY CONTINUOUS 5-MINUTE RUN ---")
    print(results_df.head(30).to_string(index=False))

    print("\nSubjects with >= 7 continuous days:")
    print((results_df["longest_run"] >= 2016).sum())


if __name__ == "__main__":
    main()