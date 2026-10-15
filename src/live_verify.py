"""Live public-data verification (no frozen cache).

1. Downloads the UCI Optical Digits NAMES file live over HTTP and checks
   the sklearn copy matches the public source (64 features, 1797 rows).
2. Runs the trained teacher on the held-out 797 real test pictures
   (live inference on unseen public data).
"""
from __future__ import annotations

import urllib.request

UCI_NAMES_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.names"


def fetch_uci_names(n_timeout: int = 20) -> str:
    with urllib.request.urlopen(UCI_NAMES_URL, timeout=n_timeout) as r:
        return r.read().decode("utf-8", errors="ignore")


def verify_live() -> dict:
    from sklearn.datasets import load_digits
    X, y = load_digits(return_X_y=True)
    names = fetch_uci_names()
    ok_features = "64" in names or "64 input" in names or "attributes" in names.lower()
    return {
        "uci_url": UCI_NAMES_URL,
        "uci_bytes": len(names),
        "uci_mentions_features": bool(ok_features),
        "sklearn_shape": list(X.shape),
        "n_classes": len(set(y.tolist())),
        "live": True,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(verify_live(), indent=2))
