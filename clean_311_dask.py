#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NYC 311 Cleaning (Parquet → monthly CSV) — Project 3

Usage:
  python clean_311_dask.py --input-dir nyc_311_data --output-dir clean_311_csv
  Optional:
    --start-date 2025-05-01 --end-date 2025-11-01
    --time-features
    --flag-outliers
"""

# === 0) SETUP & CLI ===
# - Parse CLI args, init output dir.
# - Configure deterministic run (no randomness) and progress bar.

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask.diagnostics import ProgressBar


# -----------------------------
# Helpers: column normalization
# -----------------------------
def to_snake(name: str) -> str:
    # Normalize names to snake_case; collapse duplicate underscores; trim.
    name = re.sub(r"[\s/]+", "_", str(name).strip())
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    name = re.sub(r"__+", "_", name)
    return name.lower()


ALIASES: Dict[str, List[str]] = {
    "unique_key": ["unique_key", "uniquekey", "id", "uniqueid", "unique_id"],
    "created_date": ["createddate", "created_date", "created", "createdtime", "created_ts"],
    "closed_date": ["closeddate", "closed_date", "closed", "closedtime", "closedts"],
    "resolution_action_updated_date": [
        "resolutionactionupdateddate",
        "resolution_action_updated_date",
        "resolutionupdateddate",
        "ra_updated",
    ],
    "due_date": ["duedate", "due_date", "sla_due", "sla_duedate"],
    "agency": ["agency"],
    "agency_name": ["agencyname", "agency_name"],
    "complaint_type": ["complainttype", "complait_type", "complaint", "type"],
    "descriptor": ["descriptor", "description_detail"],
    "borough": ["borough", "boro"],
    "city": ["city", "locality"],
    "incident_zip": ["incidentzip", "incident_zip", "zipcode", "zip", "zip_code"],
    "latitude": ["latitude", "lat"],
    "longitude": ["longitude", "lon", "lng"],
    "x_coordinate_state_plane": ["xcoordinatestateplane", "x_state_plane", "xcoord", "x_coordinate_state_plane"],
    "y_coordinate_state_plane": ["ycoordinatestateplane", "y_state_plane", "ycoord", "y_coordinate_state_plane"],
    "resolution_description": ["resolutiondescription", "resolution_description", "resolution", "resolution_text"],
    "location": ["location", "loc"],
    "status": ["status"],
}


def _alias_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def map_aliases(columns: List[str]) -> Dict[str, str]:
    existing_keys = {c: _alias_key(c) for c in columns}
    have = set(columns)
    already_has_complaint = any(_alias_key(c) in set(ALIASES["complaint_type"]) for c in have)

    rename: Dict[str, str] = {}
    for canon, alias_list in ALIASES.items():
        for col, key in existing_keys.items():
            if col in rename:
                continue
            if key in alias_list:
                if canon == "complaint_type" and already_has_complaint and _alias_key(col) == "type":
                    continue
                rename[col] = canon
    return rename


# -----------------------------
# Helpers: datetime parsing
# -----------------------------
def parse_nyc_datetime(series: pd.Series) -> pd.Series:
    """
    Parse timestamps assuming America/New_York if tz-naive, then convert to UTC.
    On failure → NaT.
    """
    s = pd.to_datetime(series, errors="coerce", utc=False)
    if getattr(s.dt, "tz", None) is None:
        try:
            s = s.dt.tz_localize("America/New_York", nonexistent="NaT", ambiguous="NaT")
        except Exception:
            s = s.dt.tz_localize("UTC", nonexistent="NaT", ambiguous="NaT")
    s = s.dt.tz_convert("UTC")
    return s


def to_numeric_coerce(series: dd.Series):
    return dd.to_numeric(series, errors="coerce")


def str_trim(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


EMPTY_LIKE = {"", ".", "-", "n/a", "na", "null", "unknown"}


def normalize_empty_tokens(series: pd.Series) -> pd.Series:
    s = series.astype("string")
    mask = s.str.lower().isin(EMPTY_LIKE)
    return s.mask(mask)


def ts_to_ns_int(s: pd.Series) -> pd.Series:
    """
    Convert tz-aware datetime64[ns, UTC] to int64 nanoseconds safely; NaT → -1.
    Avoids deprecated Series.view by using DatetimeArray.asi8.
    """
    v = s.array.asi8.copy()
    iNAT = np.iinfo(np.int64).min
    v = np.where(v == iNAT, -1, v)
    return pd.Series(v, index=s.index, dtype="int64")


# -----------------------------
# 1) LOAD & COLUMN NORMALIZATION
# - Read all Parquet files recursively (pyarrow). If none found, treat input dir as Parquet dataset.
# - Normalize column names to snake_case.
# - Apply alias mapping to canonical names.
# -----------------------------
def load_parquet_dataset(input_dir: Path) -> dd.DataFrame:
    files = sorted(input_dir.rglob("*.parquet"))
    if files:
        df = dd.read_parquet([str(p) for p in files], engine="pyarrow")
    else:
        df = dd.read_parquet(str(input_dir), engine="pyarrow")
    df = df.rename(columns={c: to_snake(c) for c in df.columns})
    rename_map = map_aliases(list(df.columns))
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


# -----------------------------
# 2) STRICT SCHEMA & DTYPES
# - Enforce required columns: unique_key (string), created_date (datetime UTC).
# - Optional timestamps left absent if not present.
# - Strings to string dtype; incident_zip to string; lat/long/x/y to numeric.
# - Drop rows missing unique_key.
# -----------------------------
CANON_COLS = [
    "unique_key",
    "created_date",
    "closed_date",
    "resolution_action_updated_date",
    "due_date",
    "agency",
    "agency_name",
    "complaint_type",
    "descriptor",
    "borough",
    "city",
    "incident_zip",
    "latitude",
    "longitude",
    "x_coordinate_state_plane",
    "y_coordinate_state_plane",
    "resolution_description",
    "location",
]


def enforce_schema(df: dd.DataFrame) -> dd.DataFrame:
    if "unique_key" not in df.columns:
        id_like = [c for c in df.columns if _alias_key(c) in {"id", "uniqueid", "uniquekey"}]
        if id_like:
            df = df.rename(columns={id_like[0]: "unique_key"})
    if "unique_key" not in df.columns:
        raise ValueError("Missing required column: unique_key (after alias normalization).")

    df["unique_key"] = df["unique_key"].astype("string").map_partitions(str_trim, meta=("unique_key", "string"))
    df["unique_key"] = df["unique_key"].map_partitions(normalize_empty_tokens, meta=("unique_key", "string"))
    df = df.dropna(subset=["unique_key"])

    # Timestamps (only parse those present)
    for c in ["created_date", "closed_date", "resolution_action_updated_date", "due_date"]:
        if c in df.columns:
            df[c] = df[c].map_partitions(parse_nyc_datetime, meta=(c, "datetime64[ns, UTC]"))

    if "created_date" not in df.columns:
        raise ValueError("Missing required column: created_date (after alias normalization).")

    for c in ["agency", "agency_name", "complaint_type", "descriptor", "borough", "city", "resolution_description"]:
        if c in df.columns:
            df[c] = df[c].astype("object")

    if "incident_zip" in df.columns:
        df["incident_zip"] = df["incident_zip"].astype("string")

    for c in ["latitude", "longitude", "x_coordinate_state_plane", "y_coordinate_state_plane"]:
        if c in df.columns:
            df[c] = to_numeric_coerce(df[c])

    return df


# -----------------------------
# 3) STRING HYGIENE
# - Trim all object/string columns.
# - Convert empty-like tokens to null.
# - Uppercase normalization for borough and city.
# - ZIP extract 5-digits only.
# -----------------------------
def apply_string_hygiene(df: dd.DataFrame) -> dd.DataFrame:
    obj_cols = [c for c, t in zip(df.columns, df.dtypes) if t == "object" or str(t).startswith("string")]
    for c in obj_cols:
        df[c] = df[c].map_partitions(str_trim, meta=(c, "string"))
        if c != "resolution_description":
            df[c] = df[c].map_partitions(normalize_empty_tokens, meta=(c, "string"))

    for c in ["borough", "city"]:
        if c in df.columns:
            df[c] = df[c].map_partitions(lambda s: s.str.upper(), meta=(c, "string"))

    if "incident_zip" in df.columns:
        df["incident_zip"] = df["incident_zip"].map_partitions(
            lambda s: s.astype("string").str.extract(r"(\d{5})", expand=False), meta=("incident_zip", "string")
        )
    return df


# -----------------------------
# 4) KEY & DUPLICATES
# - Ensure uniqueness of unique_key using priority:
#   (1) CLOSED over OPEN (closed_date not null),
#   (2) later resolution_action_updated_date,
#   (3) later closed_date,
#   (4) longer resolution_description length.
# - Deterministic: set index to unique_key, sort by tie-breakers, keep last per key.
# -----------------------------
def dedupe_by_priority(df: dd.DataFrame) -> dd.DataFrame:
    if "resolution_description" not in df.columns:
        df["resolution_description"] = ""

    df["closed_flag"] = df["closed_date"].notnull().astype("int64") if "closed_date" in df.columns else 0

    df["desc_len"] = df["resolution_description"].map_partitions(
        lambda s: s.fillna("").astype(str).str.len().astype("int64"), meta=("desc_len", "int64")
    )

    for c in ["resolution_action_updated_date", "closed_date"]:
        if c in df.columns:
            df[f"{c}_ns"] = df[c].map_partitions(ts_to_ns_int, meta=(f"{c}_ns", "int64"))
        else:
            df[f"{c}_ns"] = -1

    df = df.dropna(subset=["unique_key"]).set_index("unique_key", shuffle="tasks")

    order_cols = ["closed_flag", "resolution_action_updated_date_ns", "closed_date_ns", "desc_len"]

    def _keep_best(pdf: pd.DataFrame) -> pd.DataFrame:
        if not len(pdf):
            return pdf
        pdf_sorted = pdf.sort_values(order_cols, kind="mergesort")
        return pdf_sorted[~pdf_sorted.index.duplicated(keep="last")]

    meta = df._meta
    df = df.map_partitions(_keep_best, meta=meta)

    df = df.drop(columns=["closed_flag", "desc_len", "resolution_action_updated_date_ns", "closed_date_ns"])
    df = df.reset_index()
    return df


# -----------------------------
# 5) DATE PARSING & LOGICAL SANITY
# - Drop null created_date.
# - If created_date in future → drop.
# - If closed_date < created_date → set closed_date = NaT.
# - Optional --start-date/--end-date filter (UTC).
# -----------------------------
def apply_date_sanity(
    df: dd.DataFrame, start_date: Optional[str] = None, end_date: Optional[str] = None
) -> Tuple[dd.DataFrame, int]:
    null_created = int(df["created_date"].isna().sum().compute())
    df = df.dropna(subset=["created_date"])

    now_utc = pd.Timestamp.now(tz="UTC")
    df = df[df["created_date"] <= now_utc]

    if "closed_date" in df.columns:
        df["closed_date"] = df["closed_date"].where(df["closed_date"] >= df["created_date"])

    if start_date:
        start_ts = pd.Timestamp(start_date).tz_localize("UTC")
        df = df[df["created_date"] >= start_ts]
    if end_date:
        end_ts = pd.Timestamp(end_date).tz_localize("UTC")
        df = df[df["created_date"] < end_ts]

    return df, null_created


# -----------------------------
# 6) STATUS NORMALIZATION
# - Binary status: CLOSED iff closed_date not null; else OPEN.
# -----------------------------
def derive_status(df: dd.DataFrame) -> dd.DataFrame:
    if "closed_date" in df.columns:
        df["status"] = df["closed_date"].notnull().map_partitions(
            lambda s: s.replace({True: "CLOSED", False: "OPEN"}), meta=("status", "object")
        )
    else:
        df["status"] = "OPEN"
    return df


# -----------------------------
# 7) FEATURE ENGINEERING (MINIMAL)
# - response_hours = (closed_date - created_date) in hours; NaN if closed_date null/missing.
# - Optional --time-features: created_year, created_month, created_dow, created_hour, is_weekend.
# -----------------------------
def add_features(df: dd.DataFrame, time_features: bool = False) -> dd.DataFrame:
    if "closed_date" in df.columns:
        df["response_hours"] = ((df["closed_date"] - df["created_date"]).dt.total_seconds() / 3600.0).astype("float64")
    else:
        df["response_hours"] = np.nan

    if time_features:
        df["created_year"] = df["created_date"].dt.year
        df["created_month"] = df["created_date"].dt.month
        df["created_dow"] = df["created_date"].dt.dayofweek
        df["created_hour"] = df["created_date"].dt.hour
        # ensure pure bool without NA
        df["is_weekend"] = df["created_dow"].isin([5, 6]).fillna(False).astype("bool")

    return df


# -----------------------------
# 8) GEO SANITY (LIGHTWEIGHT)
# - has_geo True iff lat/lon numeric and within coarse NYC bounds.
# - Drop free-text 'location' field if present.
# -----------------------------
def apply_geo_checks(df: dd.DataFrame) -> dd.DataFrame:
    if "latitude" in df.columns and "longitude" in df.columns:
        lat = dd.to_numeric(df["latitude"], errors="coerce")
        lon = dd.to_numeric(df["longitude"], errors="coerce")
        lat_ok = lat.between(40.3, 41.2).fillna(False)
        lon_ok = lon.between(-74.5, -73.4).fillna(False)
        df["has_geo"] = (lat_ok & lon_ok).astype("bool")
    else:
        df["has_geo"] = False

    if "location" in df.columns:
        df = df.drop(columns=["location"])
    return df


# -----------------------------
# 9) ZIP CLEANING
# - incident_zip: extract 5-digit ZIP only; non-matching → null.
# -----------------------------
def clean_zip(df: dd.DataFrame) -> dd.DataFrame:
    if "incident_zip" in df.columns:
        df["incident_zip"] = df["incident_zip"].map_partitions(
            lambda s: s.astype("string").str.extract(r"(\d{5})", expand=False), meta=("incident_zip", "string")
        )
    return df


# -----------------------------
# 10) MISSING DATA POLICY
# - No arbitrary imputation.
# - Optionally derive agency_name from agency when 1:1 mapping exists in data.
# -----------------------------
def backfill_agency_name(df: dd.DataFrame) -> dd.DataFrame:
    if "agency" in df.columns and "agency_name" in df.columns:
        try:
            pairs = df[["agency", "agency_name"]].dropna().drop_duplicates().compute()
            one_to_one_agencies = pairs.groupby("agency")["agency_name"].nunique()
            ok_keys = set(one_to_one_agencies[one_to_one_agencies == 1].index)
            if ok_keys:
                unique_pairs = pairs[pairs["agency"].isin(ok_keys)].drop_duplicates("agency")
                amap = dict(zip(unique_pairs["agency"], unique_pairs["agency_name"]))
                mapped = df["agency"].map(amap, meta=("agency_name", "object"))
                df["agency_name"] = df["agency_name"].fillna(mapped)
        except Exception:
            pass
    return df


# -----------------------------
# 11) OPTIONAL OUTLIER FLAG
# - If --flag-outliers: is_duration_outlier by complaint_type using 1st/99th pct.
# -----------------------------
def add_outlier_flag(df: dd.DataFrame, enable: bool) -> dd.DataFrame:
    if not enable or "complaint_type" not in df.columns or "response_hours" not in df.columns:
        return df
    try:
        q = df.groupby("complaint_type")["response_hours"].quantile([0.01, 0.99]).unstack()
        q = q.rename(columns={0.01: "q01", 0.99: "q99"}).compute()
        q01 = q["q01"].to_dict()
        q99 = q["q99"].to_dict()
        df["__q01"] = df["complaint_type"].map(q01, meta=("__q01", "float64"))
        df["__q99"] = df["complaint_type"].map(q99, meta=("__q99", "float64"))
        df["is_duration_outlier"] = (df["response_hours"] < df["__q01"]) | (df["response_hours"] > df["__q99"])
        df = df.drop(columns=["__q01", "__q99"])
    except Exception:
        q1, q99 = dd.compute(df["response_hours"].quantile(0.01), df["response_hours"].quantile(0.99))
        df["is_duration_outlier"] = (df["response_hours"] < q1) | (df["response_hours"] > q99)
    return df


# -----------------------------
# 12) FINAL COLUMN SET & ORDER
# - Keep only canonical columns in the specified order (+optional extras via flags).
# -----------------------------
BASE_ORDER = [
    "unique_key",
    "created_date",
    "closed_date",
    "response_hours",
    "status",
    "resolution_action_updated_date",
    "due_date",
    "agency",
    "agency_name",
    "complaint_type",
    "descriptor",
    "resolution_description",
    "borough",
    "city",
    "incident_zip",
    "latitude",
    "longitude",
    "x_coordinate_state_plane",
    "y_coordinate_state_plane",
    "has_geo",
]


def reorder_columns(df: dd.DataFrame, time_features: bool, outliers: bool) -> dd.DataFrame:
    cols = [c for c in BASE_ORDER if c in df.columns]
    if time_features:
        cols += [c for c in ["created_year", "created_month", "created_dow", "created_hour", "is_weekend"] if c in df.columns]
    if outliers and "is_duration_outlier" in df.columns:
        cols.append("is_duration_outlier")
    keep = [c for c in cols if c in df.columns]
    return df[keep]


# -----------------------------
# 13) OUTPUT & QA LOGGING
# - Write multi-part CSVs by month (created_date) to --output-dir (e.g., data-YYYY-MM-*.csv), header in each part.
# - Emit etl_run_log.json with essential QA metrics; print same summary to stdout.
# -----------------------------
def write_monthly_csvs(df: dd.DataFrame, out_dir: Path) -> List[str]:
    """
    Write ONE CSV per local NYC month:
      clean_311_csv/data-YYYY-MM.csv
    """
    # If for some reason created_date is missing, write a single file
    if "created_date" not in df.columns:
        path = out_dir / "data-all.csv"
        df.to_csv(str(path), index=False, header=True, single_file=True)
        return [str(path)]

    # Determine months using NYC local time to avoid UTC month drift
    months = (
        df["created_date"]
        .dt.tz_convert("America/New_York")
        .dt.strftime("%Y-%m")
        .dropna()
        .unique()
        .compute()
    )

    files: List[str] = []
    for ym in sorted(months):
        year, month = ym.split("-")
        # Local month window [start_local, end_local)
        start_local = pd.Timestamp(f"{year}-{month}-01 00:00:00", tz="America/New_York")
        end_local = start_local + pd.offsets.MonthBegin(1)  # first day of next month (local)

        # Convert boundaries to UTC to compare with UTC 'created_date'
        start_utc = start_local.tz_convert("UTC")
        end_utc = end_local.tz_convert("UTC")

        dff = df[(df["created_date"] >= start_utc) & (df["created_date"] < end_utc)]

        # Skip empty months just in case
        if int(dff.shape[0].compute()) == 0:
            continue

        out_path = out_dir / f"data-{year}-{month}.csv"
        dff.to_csv(str(out_path), index=False, header=True, single_file=True)
        files.append(str(out_path))

    return files


def run_summary(df: dd.DataFrame, in_dir: Path, out_dir: Path, null_created_drop_count: int) -> Dict:
    rows_after = int(df.shape[0].compute())
    unique_keys = int(df["unique_key"].nunique().compute())
    has_geo_rate = float(df["has_geo"].mean().compute()) if "has_geo" in df.columns else 0.0
    status_counts = df["status"].value_counts().compute().to_dict() if "status" in df.columns else {}

    resp_metrics = {}
    if "response_hours" in df.columns:
        null_share = float((df["response_hours"].isna().mean()).compute())
        q1, q50, q99 = dd.compute(
            df["response_hours"].quantile(0.01), df["response_hours"].quantile(0.50), df["response_hours"].quantile(0.99)
        )

        def _asf(x):
            return None if (x is None or (isinstance(x, float) and math.isnan(x))) else float(x)

        resp_metrics = {"null_share": null_share, "p01": _asf(q1), "p50": _asf(q50), "p99": _asf(q99)}

    summary = {
        "run_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "input_dir": str(in_dir),
        "output_dir": str(out_dir),
        "rows_after_clean": rows_after,
        "unique_keys_after_clean": unique_keys,
        "null_created_date_dropped": null_created_drop_count,
        "status_counts": status_counts,
        "has_geo_rate": has_geo_rate,
        "response_hours": resp_metrics,
    }
    return summary


# -----------------------------
# MAIN
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="NYC 311 cleaning (Parquet in → CSV out)")
    parser.add_argument("--input-dir", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--start-date", type=str, default=None, help="UTC date (YYYY-MM-DD) inclusive")
    parser.add_argument("--end-date", type=str, default=None, help="UTC date (YYYY-MM-DD) exclusive")
    parser.add_argument("--time-features", action="store_true", help="Add created_* and weekend flags")
    parser.add_argument("--flag-outliers", action="store_true", help="Add is_duration_outlier by complaint_type")
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dask.config.set({"dataframe.shuffle.algorithm": "tasks"})  # deterministic task-based shuffle

    with ProgressBar():
        # 1) Load & normalize columns
        df = load_parquet_dataset(in_dir)

        # 2) Strict schema & dtype enforcement
        df = enforce_schema(df)

        # 3) String hygiene
        df = apply_string_hygiene(df)

        # 4) Deduplicate by priority
        df = dedupe_by_priority(df)

        # 5) Date parsing & logical sanity (+ window filter)
        df, null_created_dropped = apply_date_sanity(df, args.start_date, args.end_date)

        # 6) Status normalization
        df = derive_status(df)

        # 7) Minimal features
        df = add_features(df, time_features=args.time_features)

        # 8) Geo sanity checks
        df = apply_geo_checks(df)

        # 9) ZIP cleaning
        df = clean_zip(df)

        # 10) Missing data policy (deterministic backfill for agency_name if 1:1)
        df = backfill_agency_name(df)

        # 11) Optional outlier flag
        df = add_outlier_flag(df, enable=args.flag_outliers)

        # 12) Final column order
        df = reorder_columns(df, time_features=args.time_features, outliers=args.flag_outliers)

        # 13) Output CSV by month + QA logging
        write_monthly_csvs(df, out_dir)
        summary = run_summary(df, in_dir, out_dir, null_created_dropped)

        print("\n" + "=" * 30 + "\nRUN SUMMARY\n" + "=" * 30)
        print(json.dumps(summary, indent=2))
        (out_dir / "etl_run_log.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
