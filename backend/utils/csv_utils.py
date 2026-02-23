from __future__ import annotations

import io
from typing import Optional

import pandas as pd


def read_csv_bytes(content: bytes) -> pd.DataFrame:
    """Read CSV bytes into a DataFrame with reasonable defaults."""
    return pd.read_csv(io.BytesIO(content))


def read_optional_csv_bytes(content: Optional[bytes]) -> Optional[pd.DataFrame]:
    if content is None:
        return None
    return read_csv_bytes(content)
