"""Call the running cancellation API.

Example:
    python test_client.py
    python test_client.py http://127.0.0.1:9696
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:9696"
    example_path = ROOT / "reports" / "example_request.json"
    if not example_path.exists():
        raise SystemExit("Missing reports/example_request.json. Train the model first.")
    booking = json.loads(example_path.read_text(encoding="utf-8"))

    health_url = base.rstrip("/") + "/health"
    with urllib.request.urlopen(health_url, timeout=30) as response:
        health = json.load(response)
    print("health", health)

    payload = json.dumps(booking).encode("utf-8")
    request = urllib.request.Request(
        base.rstrip("/") + "/predict",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        body = json.load(response)
    print("predict", body)
    if not 0.0 <= body["cancellation_probability"] <= 1.0:
        raise SystemExit("Probability outside [0, 1].")
    if not isinstance(body["will_cancel"], bool):
        raise SystemExit("will_cancel was not a boolean.")


if __name__ == "__main__":
    main()
