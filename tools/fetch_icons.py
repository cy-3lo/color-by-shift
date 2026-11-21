"""Fetch the four icons of the bar from free icon libraries (npm packages, via jsDelivr).

    python tools/fetch_icons.py

For every library below: reads the latest version and the licence from its package.json,
downloads the four SVG files of that exact version, normalises them (24 x 24, white instead
of currentColor) and writes game/data/icons.json ({set: {package, version, licence, icons:
{next, reset, undo, redo}}}) and game/data/ICONS.md (credits). A set with a missing file is
left out and reported. The game draws them with game/ui/icons.gd.
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

# set name -> (npm package, {icon of the bar: path of the file in the package})
SETS = {
    "lucide": (
        "lucide-static",
        {
            "next": "icons/skip-forward.svg",
            "reset": "icons/skip-back.svg",
            "undo": "icons/undo-2.svg",
            "redo": "icons/redo-2.svg",
        },
    ),
    "tabler": (
        "@tabler/icons",
        {
            "next": "icons/outline/player-skip-forward.svg",
            "reset": "icons/outline/player-skip-back.svg",
            "undo": "icons/outline/arrow-back-up.svg",
            "redo": "icons/outline/arrow-forward-up.svg",
        },
    ),
    "tabler-filled": (
        "@tabler/icons",
        {
            "next": "icons/filled/player-skip-forward.svg",
            "reset": "icons/filled/player-skip-back.svg",
            "undo": "icons/outline/arrow-back-up.svg",
            "redo": "icons/outline/arrow-forward-up.svg",
        },
    ),
    "phosphor-bold": (
        "@phosphor-icons/core",
        {
            "next": "assets/bold/skip-forward-bold.svg",
            "reset": "assets/bold/skip-back-bold.svg",
            "undo": "assets/bold/arrow-u-up-left-bold.svg",
            "redo": "assets/bold/arrow-u-up-right-bold.svg",
        },
    ),
    "phosphor-fill": (
        "@phosphor-icons/core",
        {
            "next": "assets/fill/skip-forward-fill.svg",
            "reset": "assets/fill/skip-back-fill.svg",
            "undo": "assets/fill/arrow-u-up-left-fill.svg",
            "redo": "assets/fill/arrow-u-up-right-fill.svg",
        },
    ),
    "heroicons": (
        "heroicons",
        {
            "next": "24/solid/forward.svg",
            "reset": "24/solid/backward.svg",
            "undo": "24/solid/arrow-uturn-left.svg",
            "redo": "24/solid/arrow-uturn-right.svg",
        },
    ),
    "bootstrap": (
        "bootstrap-icons",
        {
            "next": "icons/skip-end-fill.svg",
            "reset": "icons/skip-start-fill.svg",
            "undo": "icons/arrow-counterclockwise.svg",
            "redo": "icons/arrow-clockwise.svg",
        },
    ),
    "remix": (
        "remixicon",
        {
            "next": "icons/Media/skip-forward-fill.svg",
            "reset": "icons/Media/skip-back-fill.svg",
            "undo": "icons/Arrows/arrow-go-back-fill.svg",
            "redo": "icons/Arrows/arrow-go-forward-fill.svg",
        },
    ),
    "material": (
        "@material-symbols/svg-600",
        {
            "next": "rounded/skip_next-fill.svg",
            "reset": "rounded/skip_previous-fill.svg",
            "undo": "rounded/undo.svg",
            "redo": "rounded/redo.svg",
        },
    ),
    "iconoir": (
        "iconoir",
        {
            "next": "icons/regular/skip-next.svg",
            "reset": "icons/regular/skip-prev.svg",
            "undo": "icons/regular/undo.svg",
            "redo": "icons/regular/redo.svg",
        },
    ),
    # curvier undo and redo, with filled skip triangles
    "lucide-curve": (
        "lucide-static",
        {
            "next": "icons/skip-forward.svg",
            "reset": "icons/skip-back.svg",
            "undo": "icons/undo.svg",
            "redo": "icons/redo.svg",
        },
    ),
    "lucide-rotate": (
        "lucide-static",
        {
            "next": "icons/skip-forward.svg",
            "reset": "icons/skip-back.svg",
            "undo": "icons/rotate-ccw.svg",
            "redo": "icons/rotate-cw.svg",
        },
    ),
    "tabler-curve": (
        "@tabler/icons",
        {
            "next": "icons/filled/player-skip-forward.svg",
            "reset": "icons/filled/player-skip-back.svg",
            "undo": "icons/outline/arrow-back.svg",
            "redo": "icons/outline/arrow-forward.svg",
        },
    ),
    "phosphor-arc": (
        "@phosphor-icons/core",
        {
            "next": "assets/fill/skip-forward-fill.svg",
            "reset": "assets/fill/skip-back-fill.svg",
            "undo": "assets/bold/arrow-arc-left-bold.svg",
            "redo": "assets/bold/arrow-arc-right-bold.svg",
        },
    ),
    "phosphor-turn": (
        "@phosphor-icons/core",
        {
            "next": "assets/fill/skip-forward-fill.svg",
            "reset": "assets/fill/skip-back-fill.svg",
            "undo": "assets/bold/arrow-counter-clockwise-bold.svg",
            "redo": "assets/bold/arrow-clockwise-bold.svg",
        },
    ),
    "phosphor-bend": (
        "@phosphor-icons/core",
        {
            "next": "assets/fill/skip-forward-fill.svg",
            "reset": "assets/fill/skip-back-fill.svg",
            "undo": "assets/fill/arrow-bend-up-left-fill.svg",
            "redo": "assets/fill/arrow-bend-up-right-fill.svg",
        },
    ),
    "ionicons": (
        "ionicons",
        {
            "next": "dist/svg/play-skip-forward.svg",
            "reset": "dist/svg/play-skip-back.svg",
            "undo": "dist/svg/arrow-undo.svg",
            "redo": "dist/svg/arrow-redo.svg",
        },
    ),
    "ionicons-outline": (
        "ionicons",
        {
            "next": "dist/svg/play-skip-forward-outline.svg",
            "reset": "dist/svg/play-skip-back-outline.svg",
            "undo": "dist/svg/arrow-undo-outline.svg",
            "redo": "dist/svg/arrow-redo-outline.svg",
        },
    ),
    "fluent": (
        "@fluentui/svg-icons",
        {
            "next": "icons/next_24_filled.svg",
            "reset": "icons/previous_24_filled.svg",
            "undo": "icons/arrow_undo_24_filled.svg",
            "redo": "icons/arrow_redo_24_filled.svg",
        },
    ),
    "fluent-regular": (
        "@fluentui/svg-icons",
        {
            "next": "icons/next_24_regular.svg",
            "reset": "icons/previous_24_regular.svg",
            "undo": "icons/arrow_undo_24_regular.svg",
            "redo": "icons/arrow_redo_24_regular.svg",
        },
    ),
    "fontawesome": (
        "@fortawesome/fontawesome-free",
        {
            "next": "svgs/solid/forward-step.svg",
            "reset": "svgs/solid/backward-step.svg",
            "undo": "svgs/solid/rotate-left.svg",
            "redo": "svgs/solid/rotate-right.svg",
        },
    ),
    "fontawesome-reply": (
        "@fortawesome/fontawesome-free",
        {
            "next": "svgs/solid/forward-step.svg",
            "reset": "svgs/solid/backward-step.svg",
            "undo": "svgs/solid/reply.svg",
            "redo": "svgs/solid/share.svg",
        },
    ),
    "carbon": (
        "@carbon/icons",
        {
            "next": "svg/32/skip--forward--filled.svg",
            "reset": "svg/32/skip--back--filled.svg",
            "undo": "svg/32/undo.svg",
            "redo": "svg/32/redo.svg",
        },
    ),
    "boxicons": (
        "boxicons",
        {
            "next": "svg/regular/bx-skip-next.svg",
            "reset": "svg/regular/bx-skip-previous.svg",
            "undo": "svg/regular/bx-undo.svg",
            "redo": "svg/regular/bx-redo.svg",
        },
    ),
    "eva": (
        "eva-icons",
        {
            "next": "fill/svg/skip-forward.svg",
            "reset": "fill/svg/skip-back.svg",
            "undo": "fill/svg/undo.svg",
            "redo": "outline/svg/corner-up-right-outline.svg",
        },
    ),
    "feather": (
        "feather-icons",
        {
            "next": "dist/icons/skip-forward.svg",
            "reset": "dist/icons/skip-back.svg",
            "undo": "dist/icons/rotate-ccw.svg",
            "redo": "dist/icons/rotate-cw.svg",
        },
    ),
    "material-sharp": (
        "@material-symbols/svg-600",
        {
            "next": "sharp/skip_next-fill.svg",
            "reset": "sharp/skip_previous-fill.svg",
            "undo": "sharp/undo.svg",
            "redo": "sharp/redo.svg",
        },
    ),
    "remix-line": (
        "remixicon",
        {
            "next": "icons/Media/skip-forward-line.svg",
            "reset": "icons/Media/skip-back-line.svg",
            "undo": "icons/Arrows/arrow-go-back-line.svg",
            "redo": "icons/Arrows/arrow-go-forward-line.svg",
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
        "# Icon sets fetched by tools/fetch_icons.py\n\n"
        "Licence as stated in each package's package.json; check the package's LICENSE file\n"
        "before release and keep the credit for the set that ships.\n\n" + "\n".join(credits) + "\n"
    )


if __name__ == "__main__":
    main()
