"""Fetch public-domain images from the Metropolitan Museum of Art collection API.

    python tools/met_fetch.py "hokusai great wave"          # list candidates, download nothing
    python tools/met_fetch.py --id 45434 --name great_wave_met   # download one object
    python tools/met_fetch.py "seurat" --list 20 --department 11

API: https://metmuseum.github.io (no key). Only objects flagged isPublicDomain (the Met's
Open Access, CC0) with an image are listed or downloaded; anything else is refused.
A download saves assets/paintings/<name>.jpg (the "web-large" image by default, --full for
the original scan) and appends its source to assets/paintings/SOURCES.md. An existing file is
never overwritten.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = "https://collectionapi.metmuseum.org/public/collection/v1"
SEARCH = f"{API}.1/search"  # v1 search was retired on 2026-10-01 (objects stay on v1)
AGENT = "mosaic-prototype/0.1 (research script)"


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": AGENT})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def get_object(oid: int) -> dict:
    return json.loads(get(f"{API}/objects/{oid}"))


def search(query: str, department: int | None) -> list[int]:
    par = {"q": query, "hasImages": "true", "limit": "100"}
    if department is not None:
        par["departmentId"] = str(department)
    d = json.loads(get(f"{SEARCH}?{urllib.parse.urlencode(par)}"))
    return d.get("objectIDs") or []


def usable(o: dict) -> bool:
    return bool(o.get("isPublicDomain")) and bool(
        o.get("primaryImageSmall") or o.get("primaryImage")
    )


def describe(o: dict) -> str:
    return (
        f"{o['objectID']:>7}  {o.get('artistDisplayName') or 'unknown'}, "
        f"{o.get('title', '')} ({o.get('objectDate', '')}), {o.get('medium', '')[:50]}"
    )


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:40] or "painting"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("query", nargs="?", help="search words (lists candidates)")
    ap.add_argument("--id", type=int, default=None, help="object to download")
    ap.add_argument("--name", default=None, help="file name without extension")
    ap.add_argument("--list", type=int, default=10, help="candidates to show")
    ap.add_argument(
        "--department",
        type=int,
        default=None,
        help="Met department id (11 = European Paintings, 6 = Asian Art)",
    )
    ap.add_argument("--full", action="store_true", help="original scan instead of web-large")
    ap.add_argument("--out", type=Path, default=ROOT / "assets" / "paintings")
    a = ap.parse_args()
    if a.id is None:
        if not a.query:
            ap.error("give search words or --id")
        ids = search(a.query, a.department)
        print(f"{len(ids)} objects with images for {a.query!r}; public-domain ones:")
        shown = 0
        for oid in ids:
            if shown >= a.list:
                break
            try:
                o = get_object(oid)
            except OSError:
                continue
            if usable(o):
                print(describe(o))
                shown += 1
        print("download one with: python tools/met_fetch.py --id <number> --name <name>")
        return
    o = get_object(a.id)
    if not usable(o):
        raise SystemExit(f"object {a.id} is not public domain or has no image: not downloaded")
    url = o["primaryImage"] if a.full and o.get("primaryImage") else o["primaryImageSmall"]
    name = a.name or slug(o.get("title", str(a.id)))
    ext = Path(urllib.parse.urlparse(url).path).suffix.lower() or ".jpg"
    path = a.out / f"{name}{ext}"
    if path.exists():
        raise SystemExit(f"{path} exists: choose another --name")
    a.out.mkdir(parents=True, exist_ok=True)
    path.write_bytes(get(url))
    src = a.out / "SOURCES.md"
    if not src.exists():
        src.write_text("# Painting sources\n")
    with open(src, "a") as f:
        f.write(
            f"\n- {path.name}: {o.get('artistDisplayName') or 'unknown'}, *{o.get('title', '')}*, "
            f"{o.get('objectDate', '')}, {o.get('medium', '')}.\n"
            f"  Metropolitan Museum of Art, Open Access (CC0), object {o['objectID']}, "
            f"{o.get('objectURL', '')}\n"
            f"  ({'original scan' if url == o.get('primaryImage') else 'web-large image'}). "
            f"Downloaded {dt.date.today().isoformat()}.\n"
        )
    print(describe(o))
    print(f"saved {path} ({path.stat().st_size // 1024} kB); source added to {src}")


if __name__ == "__main__":
    main()
