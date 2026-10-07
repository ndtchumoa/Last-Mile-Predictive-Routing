# %% [markdown]
# # 03 - YT diagnostics
# Blocks: demand label, GPS robustness, allocation binding with K_t, spikes, depot and Q.
# Complexity: O(N) per block (N = rows of one city); region-day grid O(R*T).

# %%
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path.cwd()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "delivery"

CITY = "yt"  # switch to "hz" for the backup city
# ds encodes MMDD; the real year is unknown, so a dummy year is used for the calendar only
ORIGIN = pd.Timestamp("2001-05-01")
NUM_DAYS = (pd.Timestamp("2001-10-31") - ORIGIN).days + 1  # 184
EARTH_RADIUS_KM = 6371.0088
TIME_COLS = ["accept_time", "accept_gps_time", "delivery_time", "delivery_gps_time"]


def toOffset(ts):
    # Calendar-day offset from May 1; NaT becomes NaN
    return (ts.dt.normalize() - ORIGIN).dt.days


def toMMDD(dayOffset):
    return (ORIGIN + pd.Timedelta(days=int(dayOffset))).strftime("%m-%d")


def loadCity(cityCode):
    df = pd.read_parquet(RAW_DIR / f"delivery_{cityCode}.parquet")
    dsDate = pd.to_datetime("2001" + df["ds"].astype(str).str.zfill(4), format="%Y%m%d")
    df["dayOffset"] = (dsDate - ORIGIN).dt.days
    for col in TIME_COLS:
        df[col + "Dt"] = pd.to_datetime(
            "2001-" + df[col], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    df["acceptOffset"] = toOffset(df["accept_timeDt"])
    df["deliveryOffset"] = toOffset(df["delivery_timeDt"])
    return df


def haversineKm(lat1, lng1, lat2, lng2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dLat = p2 - p1
    dLng = np.radians(lng2) - np.radians(lng1)
    a = np.sin(dLat / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dLng / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def buildGrid(df, col):
    # Region x day order counts for a chosen day label; days outside the calendar are dropped
    valid = df[col].dropna()
    valid = valid[(valid >= 0) & (valid < NUM_DAYS)].astype(int)
    frame = pd.DataFrame({"region_id": df.loc[valid.index, "region_id"], "day": valid})
    grid = frame.groupby(["region_id", "day"]).size().unstack(fill_value=0)
    return grid.reindex(columns=range(NUM_DAYS), fill_value=0)


df = loadCity(CITY)
print(CITY.upper(), "rows:", len(df))

# %% [markdown]
# ## Block 1 - which date should define demand? (ds vs accept vs delivery)

# %%
def reportDemandLabel(df):
    ds = df["dayOffset"]
    sameAccept = df["acceptOffset"] == ds
    sameDelivery = df["deliveryOffset"] == ds
    both = sameAccept & sameDelivery
    neither = ~sameAccept & ~sameDelivery
    print("ds == accept date:", round(float(sameAccept.mean()), 4),
          "| ds == delivery date:", round(float(sameDelivery.mean()), 4),
          "| both:", round(float(both.mean()), 4),
          "| neither:", round(float(neither.mean()), 4))
    gapDays = df["deliveryOffset"] - df["acceptOffset"]
    print("delivery - accept (days):",
          gapDays.value_counts(normalize=True).sort_index().round(4).head(6).to_dict())
    print("accept hour share:",
          df["accept_timeDt"].dt.hour.value_counts(normalize=True).sort_index().round(3).to_dict())
    print("examples where ds matches neither date:")
    print(df.loc[neither, ["ds", "accept_time", "delivery_time"]].head(5).to_string())

    gridDs = buildGrid(df, "dayOffset")
    dailyDs = gridDs.sum(axis=0)
    for label, col in [("accept", "acceptOffset"), ("delivery", "deliveryOffset")]:
        other = buildGrid(df, col).reindex(gridDs.index, fill_value=0)
        wape = np.abs(gridDs.to_numpy() - other.to_numpy()).sum() / gridDs.to_numpy().sum()
        outside = int(((df[col] < 0) | (df[col] >= NUM_DAYS)).sum())
        print(f"label={label}: WAPE of region-day grid vs ds grid:", round(float(wape), 4),
              "| daily-total corr with ds:", round(float(dailyDs.corr(other.sum(axis=0))), 4),
              "| orders outside May1-Oct31:", outside)


reportDemandLabel(df)

# %% [markdown]
# ## Block 2 - robust GPS diagnostics (no point is removed)

# %%
def reportGpsRobustness(df):
    qs = [0, 0.001, 0.01, 0.5, 0.99, 0.999, 1]
    print(df[["lng", "lat"]].quantile(qs).round(4).to_string())
    distCity = haversineKm(df["lat"], df["lng"], df["lat"].median(), df["lng"].median())
    print("km to city median center q50/q90/q99/q999/max:",
          distCity.quantile([0.5, 0.9, 0.99, 0.999, 1]).round(1).tolist())
    for thr in [25, 50, 100, 200]:  # arbitrary diagnostic thresholds
        print(f"points farther than {thr} km from the city center:", int((distCity > thr).sum()))

    stopGap = haversineKm(df["lat"], df["lng"], df["delivery_gps_lat"], df["delivery_gps_lng"])
    print("stop vs delivery-GPS km q50/q90/q99/q999/max:",
          stopGap.quantile([0.5, 0.9, 0.99, 0.999, 1]).round(2).tolist())
    print("share gap > 1 km:", round(float((stopGap > 1).mean()), 4),
          "| > 10 km:", round(float((stopGap > 10).mean()), 4))

    medLat = df.groupby("region_id")["lat"].transform("median")
    medLng = df.groupby("region_id")["lng"].transform("median")
    distRegion = haversineKm(df["lat"], df["lng"], medLat, medLng)
    byRegion = distRegion.groupby(df["region_id"])
    perRegion = pd.DataFrame({"n": df.groupby("region_id").size(),
                              "distP50": byRegion.median(),
                              "distP99": byRegion.quantile(0.99),
                              "distMax": byRegion.max()})
    perRegion["maxOverP99"] = perRegion["distMax"] / perRegion["distP99"]
    print("regions with the largest max distance to their own median center (km):")
    print(perRegion.sort_values("distMax", ascending=False).round(1).head(8).to_string())


reportGpsRobustness(df)

# %% [markdown]
# ## Block 3 - Oracle allocation with daily K_t: how often does k_r = 1 bind?

# %%
def reportAllocationBinding(df):
    grid = buildGrid(df, "dayOffset")
    kDaily = df.groupby("dayOffset")["courier_id"].nunique().reindex(range(NUM_DAYS))
    totalDaily = grid.sum(axis=0)
    raw = grid.mul(kDaily, axis=1).div(totalDaily, axis=1)  # K_t * y_r / sum(y)
    rawArr = raw.to_numpy()
    print("regions:", len(grid), "| cells:", rawArr.size)
    print("share of cells with raw < 1:", round(float((rawArr < 1).mean()), 3),
          "| raw < 0.5:", round(float((rawArr < 0.5).mean()), 3))
    forcedTotal = np.maximum(np.round(rawArr), 1).sum(axis=0)
    print("share of days where sum(max(1, round(raw))) > K_t:",
          round(float((forcedTotal > kDaily.to_numpy()).mean()), 3))
    print("corr(K_t, total daily orders):", round(float(kDaily.corr(totalDaily)), 3),
          "| orders per active courier per day q10/q50/q90:",
          (totalDaily / kDaily).quantile([0.1, 0.5, 0.9]).round(1).tolist())
    table = pd.DataFrame({"meanOrders": grid.mean(axis=1),
                          "meanRawK": raw.mean(axis=1),
                          "shareRawBelow1": (raw < 1).mean(axis=1)})
    print(table.sort_values("meanOrders", ascending=False).round(2).to_string())


reportAllocationBinding(df)

# %% [markdown]
# ## Block 4 - demand spikes: uniform or region-specific?

# %%
def reportSpikes(df):
    grid = buildGrid(df, "dayOffset")
    daily = grid.sum(axis=0)
    baseline = daily.rolling(15, center=True, min_periods=8).median()
    isSpike = (daily > 1.25 * baseline).to_numpy()  # 1.25 is an arbitrary threshold
    windows, start = [], None
    for d in range(NUM_DAYS):
        if isSpike[d] and start is None:
            start = d
        if (not isSpike[d]) and start is not None:
            windows.append((start, d - 1))
            start = None
    if start is not None:
        windows.append((start, NUM_DAYS - 1))

    print("test window (last 30 days):", toMMDD(NUM_DAYS - 30), "to", toMMDD(NUM_DAYS - 1))
    for a, b in windows:
        if b - a + 1 < 2:
            continue
        lo = max(0, a - 14)
        preCols = ~isSpike[lo:a]
        pre = grid.iloc[:, lo:a].loc[:, preCols]
        label = f"{toMMDD(a)}..{toMMDD(b)} (dayOffset {a}-{b})"
        if pre.shape[1] < 3:
            print(label, "-> too few clean pre-days, skipped")
            continue
        inWindow = grid.iloc[:, a:b + 1].mean(axis=1)
        preMean = pre.mean(axis=1)
        uplift = inWindow / preMean.replace(0, np.nan)
        excess = inWindow - preMean
        topShare = (excess / excess.sum()).nlargest(3).round(3).to_dict()
        print(label, "| uplift ratio q10/q50/q90:", uplift.quantile([0.1, 0.5, 0.9]).round(2).tolist(),
              "| corr(uplift, pre-mean):", round(float(uplift.corr(preMean)), 2),
              "| top-3 region share of excess:", topShare)


reportSpikes(df)

# %% [markdown]
# ## Block 5 - depot candidates and capacity Q

# %%
def reportDepotAndQ(df):
    byRegion = df.groupby("region_id")
    centroid = byRegion[["lat", "lng"]].mean()
    medianCenter = byRegion[["lat", "lng"]].median()
    shiftKm = haversineKm(centroid["lat"].to_numpy(), centroid["lng"].to_numpy(),
                          medianCenter["lat"].to_numpy(), medianCenter["lng"].to_numpy())
    print("mean-centroid vs median-center shift km q50/q90/max:",
          np.round(np.quantile(shiftKm, [0.5, 0.9, 1]), 2).tolist())

    acc = df.dropna(subset=["accept_gps_lat", "accept_gps_lng"])
    cells = pd.DataFrame({"region_id": acc["region_id"],
                          "latCell": acc["accept_gps_lat"].round(3),
                          "lngCell": acc["accept_gps_lng"].round(3)})
    counts = cells.groupby(["region_id", "latCell", "lngCell"]).size().rename("n").reset_index()
    top = counts.loc[counts.groupby("region_id")["n"].idxmax()].set_index("region_id")
    topShare = top["n"] / cells.groupby("region_id").size()
    distToCentroid = haversineKm(top["latCell"].to_numpy(), top["lngCell"].to_numpy(),
                                 centroid.loc[top.index, "lat"].to_numpy(),
                                 centroid.loc[top.index, "lng"].to_numpy())
    print("densest 0.001-degree accept-GPS cell: share of region's accept points q50/max:",
          np.round(np.quantile(topShare, [0.5, 1]), 3).tolist())
    print("distance densest cell -> centroid km q50/q90/max:",
          np.round(np.quantile(distToCentroid, [0.5, 0.9, 1]), 2).tolist())

    courierDay = df.groupby(["dayOffset", "courier_id"]).agg(
        orders=("order_id", "size"), region=("region_id", "first"))
    qs = [0.5, 0.75, 0.9, 0.95, 0.99]
    print("orders per courier-day quantiles", qs, ":", courierDay["orders"].quantile(qs).tolist())
    perRegionP90 = courierDay.groupby("region")["orders"].quantile(0.9)
    print("per-region p90 of orders per courier-day min/q50/max:",
          perRegionP90.quantile([0, 0.5, 1]).tolist())


reportDepotAndQ(df)
