from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "delivery"
OUT_PATH = PROJECT_ROOT / "reports" / "city_screening.csv"
CITY_CODES = ["jl", "yt", "cq", "sh", "hz"]


def missingShare(series: pd.Series) -> float:
    """Share of values that are NaN or exactly 0 (possible missing sentinel)."""
    return float((series.isna() | (series == 0)).mean())


def screenCity(cityCode: str) -> dict:
    df = pd.read_parquet(RAW_DIR / f"delivery_{cityCode}.parquet")

    dayCount = df["ds"].nunique()
    regionCount = df["region_id"].nunique()

    # Each groupby below is O(N)
    ordersPerDay = df.groupby("ds").size()
    couriersPerDay = df.groupby("ds")["courier_id"].nunique()
    ordersPerCourierDay = df.groupby(["ds", "courier_id"]).size()
    ordersPerRegionDay = df.groupby(["region_id", "ds"]).size()
    stopsPerRegionDay = (
        df.drop_duplicates(["region_id", "ds", "lng", "lat"])
        .groupby(["region_id", "ds"])
        .size()
    )

    observedCells = len(ordersPerRegionDay)
    zeroCellShare = 1 - observedCells / (regionCount * dayCount)

    return {
        "city": cityCode,
        "rows": len(df),
        "couriers": df["courier_id"].nunique(),
        "regions": regionCount,
        "aois": df["aoi_id"].nunique(),
        "days": dayCount,
        "dsMin": df["ds"].min(),
        "dsMax": df["ds"].max(),
        "ordersPerDayMean": ordersPerDay.mean(),
        "couriersPerDayMean": couriersPerDay.mean(),
        "ordersPerCourierDayMean": ordersPerCourierDay.mean(),
        "ordersPerCourierDayP90": ordersPerCourierDay.quantile(0.9),
        "ordersPerRegionDayMean": ordersPerRegionDay.mean(),
        "ordersPerRegionDayMedian": ordersPerRegionDay.median(),
        "zeroCellShare": zeroCellShare,
        "stopsPerRegionDayMedian": stopsPerRegionDay.median(),
        "stopsPerRegionDayP90": stopsPerRegionDay.quantile(0.9),
        "stopsPerRegionDayMax": stopsPerRegionDay.max(),
        "acceptGpsMissing": missingShare(df["accept_gps_lng"]),
        "deliveryGpsMissing": missingShare(df["delivery_gps_lng"]),
        "dupOrderIds": int(df["order_id"].duplicated().sum()),
    }


def printFormatSamples(cityCode: str) -> None:
    df = pd.read_parquet(RAW_DIR / f"delivery_{cityCode}.parquet")
    print(f"\n--- format samples ({cityCode}) ---")
    print("ds unique (first 5):", sorted(df["ds"].unique())[:5])
    print("ds unique (last 5): ", sorted(df["ds"].unique())[-5:])
    cols = ["ds", "accept_time", "accept_gps_time", "delivery_time", "delivery_gps_time"]
    print(df[cols].head(3).to_string())


def main() -> None:
    rows = []
    for cityCode in CITY_CODES:
        print(f"screening {cityCode} ...")
        rows.append(screenCity(cityCode))

    summary = pd.DataFrame(rows).set_index("city").round(3)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_PATH)

    pd.set_option("display.width", 200)
    print("\n=== CITY SCREENING (transposed) ===")
    print(summary.T.to_string())

    printFormatSamples("jl")


if __name__ == "__main__":
    main()
