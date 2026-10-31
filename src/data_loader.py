"""
Load the MBDP-labelled cohort from 1.xlsx (sheet 'total') and prepare a
training CSV that matches the model's (reduced) schema.

Data comes from the Phosphorus Homeostasis dataset (Mendeley, DOI 10.17632/6m32bh6rjt.1)
"""
from pathlib import Path
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

RAW_PATH = Path("data/raw/Mendeley/1.xlsx")
OUT_PATH = Path("data/processed/train.csv")

# Source column -> model name
COLUMN_MAP = {
    "no":       "infant_id",
    "gender":   "sex_raw",       # 1=boy, 2=girl
    "GA":       "ga",
    "bw":       "bw",
    "SGA":      "sga_raw",       # 1=no, 2=yes
    "MBDP":     "mbdp",          # target, already 0/1
    "allday":   "cum_pn_days",
    "ALPmax":   "last_alp",      # proxy for last ALP
    "sp":       "last_phosphate",
    "day":      "los_days",
}

KEEP = ["infant_id", "ga", "bw", "sga", "sex",
        "last_alp", "last_phosphate", "cum_pn_days", "mbdp"]


def load_and_engineer() -> pd.DataFrame:
    raw = pd.read_excel(RAW_PATH, sheet_name="total")
    log.info("Raw shape: %s", raw.shape)

    df = raw.rename(columns=COLUMN_MAP)

    # Encode per 'Data conversion rule.doc'
    df["sex"] = df["sex_raw"].map({1: 0, 2: 1})      # boy=0, girl=1
    df["sga"] = df["sga_raw"].map({1: 0, 2: 1})      # no=0, yes=1

    df = df[KEEP].copy()

    # Drop any row missing the target
    before = len(df)
    df = df.dropna(subset=["mbdp"])
    df["mbdp"] = df["mbdp"].astype(int)
    log.info("Dropped %d rows missing MBDP; %d remain", before - len(df), len(df))

    # Report missingness on features
    for c in ["ga", "bw", "sex", "sga", "last_alp", "last_phosphate", "cum_pn_days"]:
        n_miss = df[c].isna().sum()
        if n_miss:
            log.warning("Feature '%s' has %d missing value(s) (%.1f%%)",
                        c, n_miss, 100 * n_miss / len(df))

    log.info("Class balance: %s", df["mbdp"].value_counts().to_dict())
    log.info("Final shape: %s", df.shape)
    return df


if __name__ == "__main__":
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = load_and_engineer()
    df.to_csv(OUT_PATH, index=False)
    log.info("Wrote %s", OUT_PATH)