"""Import the paintings picked on the pick page (tools/pick_page.py) into the toy.

    python tools/import_picks.py PICKS_DIR --images Met=FOLDER NGA=FOLDER [--skip Met_1 ...]

PICKS_DIR holds one JSON file per pick (the page's "picks" collection: source, id, version,
share, artist, title, url). Each FOLDER is the candidate folder the page was built from
(candidates.json and img/<id>.jpg). For every pick the image is copied to
assets/paintings/<name>.jpg (never overwritten), its source is appended to
assets/paintings/SOURCES.md, and a [[pictures]] entry is inserted in toy/toy.toml before the
generated pictures. Version "present" -> palette "present"; "contrast 2 %" -> "contrast";
"contrast 0.4 %" -> "accent"; any other share -> palette "contrast" with share = value.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import shutil
import tomllib
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MUSEUM = {
    "Met": "Metropolitan Museum of Art, Open Access (CC0)",
    "NGA": "National Gallery of Art, Washington, Open Access (CC0)",
}


def ascii_(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


def short(s: str, n: int = 48) -> str:
    """Title or artist without parentheses, brackets and anything after a semicolon."""
    s = re.sub(r"\s*[\(\[].*?[\)\]]", "", s).split(";")[0].strip(" ,")
    s = s.replace('"', "'")  # safe inside a TOML basic string
    return ascii_(s if len(s) <= n else s[:n].rsplit(" ", 1)[0])


def slug(artist: str, title: str) -> str:
    last = short(artist).split()[-1] if short(artist) else "unknown"
    words = re.sub(r"[^a-z0-9]+", " ", short(title).lower()).split()[:4]
    return "_".join([re.sub(r"[^a-z0-9]", "", last.lower()), *words])[:44].strip("_")


def palette_of(p: dict) -> str:
    if p["version"] == "present":
        return 'palette = "present"\n'
    if p["share"] == 0.02:
        return 'palette = "contrast"\n'
    if p["share"] == 0.004:
        return 'palette = "accent"\n'
    return f'palette = "contrast"\nshare = {p["share"]}\n'


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("picks", type=Path)
    ap.add_argument("--images", nargs="+", required=True, metavar="SOURCE=FOLDER")
    ap.add_argument("--skip", nargs="*", default=[], help="pick keys to leave out")
    a = ap.parse_args()
    folders = {k: Path(v) for k, v in (s.split("=", 1) for s in a.images)}
    dates = {
        (src, o["objectID"]): o.get("objectDate") or ""
        for src, f in folders.items()
        for o in json.loads((f / "candidates.json").read_text())
    }
    toml_path = ROOT / "toy" / "toy.toml"
    text = toml_path.read_text()
    have = {p["key"] for p in tomllib.loads(text)["pictures"]}
    out = ROOT / "assets" / "paintings"
    entries, sources = [], []
    for f in sorted(a.picks.glob("*.json")):
        p = json.loads(f.read_text())
        if f.stem in a.skip:
            continue
        key = slug(p["artist"], p["title"])
        if key in have:
            print(f"skipped {f.stem}: {key} is already in toy.toml")
            continue
        have.add(key)
        dest = out / f"{key}.jpg"
        if not dest.exists():
            shutil.copyfile(folders[p["source"]] / "img" / f"{p['id']}.jpg", dest)
        date = dates.get((p["source"], p["id"]), "")
        sources.append(
            f"\n- {dest.name}: {p['artist']}, *{p['title']}*, {date}.\n"
            f"  {MUSEUM[p['source']]}, object {p['id']}, {p['url']}\n"
            f"  (small image). Downloaded {dt.date.today().isoformat()}.\n"
        )
        entries.append(
            f'[[pictures]]\nkey = "{key}"\nfile = "assets/paintings/{key}.jpg"\n'
            f'artist = "{short(p["artist"], 40)}"\ntitle = "{short(p["title"])}"\n'
            f"{palette_of(p)}\n"
        )
        print(f"{f.stem} -> {key}")
    marker = '[[pictures]]\nkey = "destijl"'
    assert marker in text
    toml_path.write_text(text.replace(marker, "".join(entries) + marker, 1))
    with open(out / "SOURCES.md", "a") as fh:
        fh.writelines(sources)
    print(f"{len(entries)} paintings added to {toml_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
