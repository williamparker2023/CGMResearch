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
    "is_test",
]


def main():
    pf = pq.ParquetFile(DATA_PATH)

    total_cgm_rows = 0
    subjects = set()
    source_counts = {}
    device_counts = {}

    for i in range(pf.num_row_groups):
        print(f"Processing row group {i + 1}/{pf.num_row_groups}")

        df = pf.read_row_group(i, columns=COLUMNS).to_pandas()

        # Keep only actual CGM observations
        df = df[df["CGM"].notna()]

        total_cgm_rows += len(df)

        subjects.update(df["id"].dropna().unique())

        for source, count in df["source_file"].value_counts().items():
            source_counts[source] = source_counts.get(source, 0) + count

        for device, count in df["cgm_device"].value_counts().items():
            device_counts[device] = device_counts.get(device, 0) + count

        del df

    print("\n--- CGM SUMMARY ---")
    print("Total CGM rows:", total_cgm_rows)
    print("Unique subjects:", len(subjects))

    print("\nCGM rows by source:")
    for source, count in sorted(
        source_counts.items(),
        key=lambda x: x[1],
        reverse=True
    ):
        print(f"{source}: {count}")

    print("\nCGM rows by device:")
    for device, count in sorted(
        device_counts.items(),
        key=lambda x: x[1],
        reverse=True
    ):
        print(f"{device}: {count}")


if __name__ == "__main__":
    main()