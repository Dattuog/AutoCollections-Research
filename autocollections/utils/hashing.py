import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return sha256_bytes(encoded)


def sha256_dataframe(frame: pd.DataFrame) -> str:
    csv = frame.to_csv(index=False, lineterminator="\n", float_format="%.17g", na_rep="<NA>")
    return sha256_bytes(csv.encode())
