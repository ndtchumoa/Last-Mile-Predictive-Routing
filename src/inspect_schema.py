from pathlib import Path

import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "delivery"

# Row counts declared in the README dataset_info block
EXPECTED_ROWS = {
    "jl": 31415, "cq": 931351, "yt": 206431, "sh": 1483864, "hz": 1861600,
}

# Columns declared in the README dataset_info block
EXPECTED_COLUMNS = [
    "order_id", "region_id", "city", "courier_id", "lng", "lat",
    "aoi_id", "aoi_type", "accept_time", "accept_gps_time",
    "accept_gps_lng", "accept_gps_lat", "delivery_time",
    "delivery_gps_time", "delivery_gps_lng", "delivery_gps_lat", "ds",
]


def inspectCity(cityCode: str, expectedRows: int) -> None:
    path = RAW_DIR / f"delivery_{cityCode}.parquet"
    # Reads footer metadata only, no row data is loaded
    parquetFile = pq.ParquetFile(path)
    numRows = parquetFile.metadata.num_rows
    schema = parquetFile.schema_arrow

    actualColumns = set(schema.names)
    missingColumns = sorted(set(EXPECTED_COLUMNS) - actualColumns)
    extraColumns = sorted(actualColumns - set(EXPECTED_COLUMNS))

    print(f"\n=== {cityCode.upper()} ===")
    print(f"rows: {numRows} (README: {expectedRows}, match={numRows == expectedRows})")
    print(f"missing vs README: {missingColumns}")
    print(f"extra vs README:   {extraColumns}")
    for field in schema:
        print(f"  {field.name:<20} {field.type}")


def main() -> None:
    for cityCode, expectedRows in EXPECTED_ROWS.items():
        inspectCity(cityCode, expectedRows)


if __name__ == "__main__":
    main()
