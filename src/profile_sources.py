from pathlib import Path
import pyarrow.parquet as pq
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "metabonet_public.parquet"

COLUMNS = [
    "CGM",
    "date",
    "id",
    "source_file",
    "cgm_device",
]


def main():
    pf = pq.ParquetFile(DATA_PATH)

    chunks = []

    for i in range(pf.num_row_groups):
        print(f"Processing row group {i + 1}/{pf.num_row_groups}")

        df = pf.read_row_group(i, columns=COLUMNS).to_pandas()
        df = df[df["CGM"].notna()]

        # We only need these for the summary
        chunks.append(
            df[["id", "source_file", "cgm_device"]]
            .drop_duplicates()
        )

    subjects = pd.concat(chunks, ignore_index=True).drop_duplicates()

    print("\n--- SUBJECTS BY SOURCE ---")
    source_summary = (
        subjects.groupby("source_file")
        .agg(
            subjects=("id", "nunique"),
            devices=("cgm_device", lambda x: sorted(x.dropna().unique()))
        )
        .sort_values("subjects", ascending=False)
    )

    print(source_summary.to_string())


if __name__ == "__main__":
    main()