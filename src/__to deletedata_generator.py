"""
Load the Phosphorus Homeostasis dataset (Mendeley, DOI 10.17632/6m32bh6rjt.1)
and produce a schema-compliant training CSV for the MBDP model.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

RAW_DIR = Path("data/raw/mendeley_phosphorus")
OUT_PATH = Path("data/processed/train.csv")

# --- COLUMN MAPPINGS ---
# Replace keys with the ACTUAL headers in your Excel files.
# Keys = names in your Excel; values = canonical names used below.
COLUMN_MAP = {
    # <-- VERIFY these against the real headers
    "Gestational_Age": "ga",
    "Birth_Weight": "bw",
    "Sex": "sex",
    "ALP": "alp",
    "Phosphorus": "phosphate",
    # "Infant_ID": "infant_id",   # <-- VERIFY: the grouping key
    # "MBDP": "mbdp_label",       # <-- VERIFY: if a diagnosis column exists
    # "SGA": "sga",               # <-- VERIFY: if present
    # "PN_Days": "cum_pn_days",   # <-- VERIFY: if present
    # "Weight": "weight",         # <-- VERIFY: if serial weights exist
    # "Age_Days": "age_days",     # <-- VERIFY: if a time column exists
}

# --- MBDP PROXY LABEL THRESHOLDS ---
# Used ONLY if no diagnosis column exists in the raw data.
# Values follow common NICU screening cutoffs; document your choice.
PROXY_ALP_THRESHOLD = 500.0       # U/L
PROXY_PHOS_THRESHOLD = 4.0        # mg/dL


def _load_raw() -> pd.DataFrame:
    files = sorted(RAW_DIR.glob("*.xls*"))
    if not files:
        raise FileNotFoundError(f"No Excel files found in {RAW_DIR}")
    log.info("Loading %d Excel file(s) from %s", len(files), RAW_DIR)
    frames = [pd.read_excel(f) for f in files]
    raw = pd.concat(frames, ignore_index=True)
    log.info("Combined raw shape: %s", raw.shape)
    return raw


def _slope(y: pd.Series) -> float:
    """OLS slope of y against its row index. Returns 0.0 if <2 points."""
    y = y.dropna()
    if len(y) < 2:
        return 0.0
    x = np.arange(len(y), dtype=float)
    return float(np.polyfit(x, y.values, 1)[0])


def _derive_target(df: pd.DataFrame, raw: pd.DataFrame) -> pd.Series:
    """Use an existing diagnosis column if present, else proxy-label."""
    if "mbdp_label" in df.columns:
        log.info("Using existing diagnosis column for target.")
        return df["mbdp_label"].astype(int)

    log.warning(
        "No diagnosis column found. Deriving proxy label: "
        "ALP>%.0f OR phosphate<%.1f. Document this in your methods.",
        PROXY_ALP_THRESHOLD, PROXY_PHOS_THRESHOLD,
    )
    return (
        (df["last_alp"] > PROXY_ALP_THRESHOLD)
        | (df["last_phosphate"] < PROXY_PHOS_THRESHOLD)
    ).astype(int)


def load_and_engineer() -> pd.DataFrame:
    raw = _load_raw().rename(columns=COLUMN_MAP)

    if "infant_id" not in raw.columns:
        raise KeyError(
            "No 'infant_id' column after mapping. Confirm the grouping key "
            "in COLUMN_MAP before proceeding."
        )

    g = raw.groupby("infant_id")

    engineered = pd.DataFrame({
        "ga":       g["ga"].first(),
        "bw":       g["bw"].first(),
        "sex":      g["sex"].first(),
        "last_alp": g["alp"].last(),
        "mean_alp": g["alp"].mean(),
        "slope_alp": g["alp"].apply(_slope),
        "last_phosphate": g["phosphate"].last(),
    })

    # Optional features — included only if the source data provides them.
    for opt in ["sga", "cum_pn_days", "weight_gain_velocity"]:
        if opt in raw.columns:
            engineered[opt] = g[opt].first()
        elif opt == "weight_gain_velocity" and {"weight", "age_days"} <= set(raw.columns):
            # derive g/day from serial weight & age
            sub = raw.dropna(subset=["weight", "age_days"])
            v = sub.groupby("infant_id").apply(
                lambda d: np.polyfit(d["age_days"], d["weight"], 1)[0]
                if len(d) >= 2 else np.nan
            )
            engineered["weight_gain_velocity"] = v
        else:
            log.warning("Optional feature '%s' not derivable — will be dropped.", opt)

    engineered = engineered.reset_index(drop=True)
    engineered["mbdp"] = _derive_target(engineered, raw)

    log.info("Engineered shape: %s", engineered.shape)
    log.info("Target balance: %s", engineered["mbdp"].value_counts().to_dict())
    return engineered


if __name__ == "__main__":
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = load_and_engineer()
    df.to_csv(OUT_PATH, index=False)
    log.info("Wrote %s (%d rows, %d cols)", OUT_PATH, *df.shape)