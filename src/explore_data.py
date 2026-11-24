from pathlib import Path
import pyarrow.parquet as pq

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "metabonet_public.parquet"


def main():
    pf = pq.ParquetFile(DATA_PATH)

    print("Rows:", pf.metadata.num_rows)
    print("Columns:", pf.metadata.num_columns)
    print("Row groups:", pf.num_row_groups)

    print("\nSchema:")
    print(pf.schema)

    print("\nFirst row group:")
    df = pf.read_row_group(0).to_pandas()

    print(df.head(10))

    print("\nData types:")
    print(df.dtypes)

    print("\nShape of first row group:")
    print(df.shape)


if __name__ == "__main__":
    main()