"""Fetch the icons of the game from Ionicons (npm package, via jsDelivr).

    python tools/fetch_icons.py

Reads the latest version and the licence from the package.json, downloads the SVG files of
that exact version (next, reset, undo, redo, credits, settings, lock), normalises them
(24 x 24, white instead of currentColor) and writes game/data/icons.json ({set: {package,
version, licence, icons: {name: svg}}}) and game/data/ICONS.md (credit). The game draws them
with game/ui/icons.gd; the licence text is in game/data/licences/ionicons_LICENSE.txt.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CDN = "https://cdn.jsdelivr.net/npm"
AGENT = "icon-fetch/0.1 (script)"

# set name -> (npm package, {icon of the game: path of the file in the package}); only the
# set that ships (31 others were compared on 2026-11-21 and dropped before the freeze)
SETS = {
    "ionicons": (
        "ionicons",
        {
            "next": "dist/svg/play-skip-forward.svg",
            "reset": "dist/svg/play-skip-back.svg",
            "undo": "dist/svg/arrow-undo.svg",
            "redo": "dist/svg/arrow-redo.svg",
            "credits": "dist/svg/information-circle.svg",
            "settings": "dist/svg/settings.svg",
            "lock": "dist/svg/lock-closed.svg",
        },
    ),
}


def get(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": AGENT})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode()


def normalise(svg: str) -> str:
    """One line, 24 x 24, drawn in white; a root tag without fill gets fill='white'."""
    svg = re.sub(r"<!--.*?-->", "", svg, flags=re.S)
    svg = re.sub(r"<\?xml.*?\?>", "", svg)
    svg = " ".join(svg.split()).replace("currentColor", "white")
    root = re.search(r"<svg\b[^>]*>", svg).group(0)
    new = re.sub(r'\s(width|height|class)="[^"]*"', "", root)
    if "fill=" not in new:
        new = new[:-1] + ' fill="white">'
    new = new[:-1] + ' width="24" height="24">'
    return svg.replace(root, new, 1).replace('"', "'")


def main() -> None:
    out, credits, versions = {}, [], {}
    for name, (package, files) in SETS.items():
        try:
            if package not in versions:
                meta = json.loads(get(f"{CDN}/{package}/package.json"))
                versions[package] = (meta["version"], str(meta.get("license", "?")))
            version, licence = versions[package]
            icons = {k: normalise(get(f"{CDN}/{package}@{version}/{f}")) for k, f in files.items()}
        except (urllib.error.URLError, AttributeError, KeyError, ValueError) as err:
            print(f"{name}: left out ({err})")
            continue
        out[name] = {"package": package, "version": version, "licence": licence, "icons": icons}
        used = ", ".join(files.values())
        credits.append(f"- {name}: npm package {package} {version}, licence {licence} ({used})")
        print(f"{name}: {package} {version}, {licence}")
    data = ROOT / "game" / "data"
    (data / "icons.json").write_text(json.dumps(out, indent=1) + "\n")
    (data / "ICONS.md").write_text(
        "# Icons fetched by tools/fetch_icons.py\n\n"
        "Licence as stated in the package's package.json; its LICENSE file was read and copied\n"
        "to licences/ionicons_LICENSE.txt, and the credit is on the game's credits panel.\n\n"
        + "\n".join(credits)
        + "\n"
    )


if __name__ == "__main__":
    main()
