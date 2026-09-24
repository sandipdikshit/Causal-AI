"""Free daily WTI observations from FRED/EIA; no API key or intraday claim."""

import hashlib
import io
import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DCOILWTICO"
SOURCE = "https://fred.stlouisfed.org/series/DCOILWTICO"


def parse_csv(raw: bytes) -> pd.DataFrame:
    frame = pd.read_csv(io.BytesIO(raw))
    if len(frame.columns) != 2 or "DCOILWTICO" not in frame:
        raise ValueError("Unexpected FRED CSV schema")
    frame.columns = ["date", "price"]
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
    frame = frame.dropna().sort_values("date").drop_duplicates("date")
    if frame.empty:
        raise ValueError("No usable WTI observations returned")
    # Negative WTI prices are valid historical observations, not missing values.
    return frame.reset_index(drop=True)


def fetch(directory: Path) -> dict:
    request = Request(URL, headers={"User-Agent": "Causal-AI-research-demo/0.1"})
    with urlopen(request, timeout=45) as response:
        raw = response.read()
    frame = parse_csv(raw)
    latest = frame.date.max()
    frame = frame[frame.date >= latest - pd.DateOffset(years=10)]
    directory.mkdir(parents=True, exist_ok=True)
    csv = frame.to_csv(index=False).encode()
    (directory / "wti_daily.csv").write_bytes(csv)
    metadata = {
        "series": "DCOILWTICO",
        "source": SOURCE,
        "download_url": URL,
        "provider": "U.S. Energy Information Administration via FRED",
        "units": "USD per barrel",
        "frequency": "daily published observations; not intraday",
        "retrieved_at_utc": datetime.now(UTC).isoformat(),
        "first_observation": frame.date.min().date().isoformat(),
        "last_observation": latest.date().isoformat(),
        "observations": len(frame),
        "sha256": hashlib.sha256(csv).hexdigest(),
        "role": "Market context only; not used to estimate liquidation effects or train PPO",
    }
    (directory / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata
