"""Download the public Hotel Booking Demand file used by this project.

The CSV is the TidyTuesday (2020-02-11) redistribution of the dataset published by
Antonio, Almeida, and Nunes (2019), Data in Brief, DOI 10.1016/j.dib.2018.11.126.
The article is the open-access source. This script checks a SHA-256 recorded
from the file downloaded for this project.
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

URL = (
    "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/"
    "data/2020/2020-02-11/hotels.csv"
)
EXPECTED_SHA256 = "7c2ae42a7353905ea136e5c2287f17c92c5435826598bfbb8491c6f0c7b1fc06"
DESTINATION = Path(__file__).resolve().parent / "data" / "hotel_bookings.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    if DESTINATION.exists() and sha256(DESTINATION) == EXPECTED_SHA256:
        print(f"Already present and verified: {DESTINATION}")
        return
    print(f"Downloading {URL}")
    urllib.request.urlretrieve(URL, DESTINATION)
    actual = sha256(DESTINATION)
    if actual != EXPECTED_SHA256:
        raise SystemExit(
            f"SHA-256 mismatch.\nExpected: {EXPECTED_SHA256}\nActual:   {actual}"
        )
    print(f"Saved {DESTINATION} ({DESTINATION.stat().st_size} bytes)")
    print(f"SHA-256 {actual}")


if __name__ == "__main__":
    main()
