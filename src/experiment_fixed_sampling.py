from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "iobp2_segments.parquet"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "fixed_sampling_results.csv"

SAMPLING_INTERVALS = [5, 10, 15, 20]


def reconstruct_segment(segment_df, interval_minutes):
    """
    Simulate fixed-rate sampling on one continuous 5-minute CGM segment.

    Returns:
        mae
        rmse
        max_error
        samples_retained
        sample_fraction
    """

    segment_df = segment_df.sort_values("date").copy()

    # 5-minute baseline: every point is sampled
    if interval_minutes == 5:
        actual = segment_df["CGM"].to_numpy()

        return {
            "mae": 0.0,
            "rmse": 0.0,
            "max_error": 0.0,
            "samples_retained": len(segment_df),
            "sample_fraction": 1.0,
        }

    # Since the ground-truth data is spaced every 5 minutes:
    # 10 min -> every 2nd point
    # 15 min -> every 3rd point
    # 20 min -> every 4th point
    step = interval_minutes // 5

    sampled_indices = np.arange(0, len(segment_df), step)

    # Make sure the final point is included so interpolation does not
    # extrapolate past the last sampled measurement.
    if sampled_indices[-1] != len(segment_df) - 1:
        sampled_indices = np.append(sampled_indices, len(segment_df) - 1)

    sampled = segment_df.iloc[sampled_indices].copy()

    # Use timestamp as the index so interpolation respects time.
    reconstructed = pd.Series(
        index=segment_df["date"],
        dtype=float
    )

    reconstructed.loc[sampled["date"]] = sampled["CGM"].to_numpy()

    # Fill skipped values using linear interpolation.
    reconstructed = reconstructed.interpolate(method="time")

    actual = segment_df["CGM"].to_numpy()
    predicted = reconstructed.to_numpy()

    errors = np.abs(actual - predicted)

    mae = np.mean(errors)
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    max_error = np.max(errors)

    return {
        "mae": mae,
        "rmse": rmse,
        "max_error": max_error,
        "samples_retained": len(sampled),
        "sample_fraction": len(sampled) / len(segment_df),
    }


def main():
    df = pd.read_parquet(DATA_PATH)

    print("Loaded processed dataset")
    print("Rows:", len(df))
    print("Subjects:", df["id"].nunique())
    print("Segments:", df["segment_id"].nunique())

    results = []

    grouped = df.groupby("segment_id")

    total_segments = df["segment_id"].nunique()

    for count, (segment_id, segment_df) in enumerate(grouped, start=1):
        if count % 100 == 0:
            print(f"Processing segment {count}/{total_segments}")

        subject_id = segment_df["id"].iloc[0]

        for interval in SAMPLING_INTERVALS:
            metrics = reconstruct_segment(segment_df, interval)

            results.append({
                "subject_id": subject_id,
                "segment_id": segment_id,
                "interval_minutes": interval,
                "segment_length": len(segment_df),
                **metrics,
            })

    results_df = pd.DataFrame(results)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_PATH, index=False)

    print("\n--- OVERALL RESULTS ---")

    summary = (
        results_df
        .groupby("interval_minutes")
        .agg(
            mean_mae=("mae", "mean"),
            median_mae=("mae", "median"),
            mean_rmse=("rmse", "mean"),
            mean_max_error=("max_error", "mean"),
            worst_max_error=("max_error", "max"),
            mean_sample_fraction=("sample_fraction", "mean"),
        )
    )

    summary["sample_reduction_percent"] = (
        1 - summary["mean_sample_fraction"]
    ) * 100

    print(summary.to_string())

    print("\nSaved detailed results to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()