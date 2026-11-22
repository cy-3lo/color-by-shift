#!/bin/sh
# Builds the size-reduced web export template of Color by Shift.
#   sh tools/godot_template/build.sh
# Needs (in build/, git-ignored): the Godot source at tag 4.7.2-stable (build/godot), emsdk
# with Emscripten 4.0.20 activated (build/emsdk), SCons in a virtual environment (build/venv):
#   git clone --depth 1 --branch 4.7.2-stable https://github.com/godotengine/godot.git build/godot
#   git clone --depth 1 https://github.com/emscripten-core/emsdk.git build/emsdk
#   build/emsdk/emsdk install 4.0.20 && build/emsdk/emsdk activate 4.0.20
#   python3 -m venv build/venv && build/venv/bin/pip install scons
# Writes build/web_template/godot.web.template_release.wasm32.nothreads.zip, which the "Web"
# export preset of game/ uses as its custom release template.
set -e
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
. "$ROOT/build/emsdk/emsdk_env.sh" > /dev/null 2>&1
cd "$ROOT/build/godot"
"$ROOT/build/venv/bin/scons" platform=web target=template_release threads=no profile="$ROOT/tools/godot_template/custom.py" -j"$(sysctl -n hw.ncpu)"
mkdir -p "$ROOT/build/web_template"
cp bin/godot.web.template_release.wasm32.nothreads.zip "$ROOT/build/web_template/"
ls -l "$ROOT/build/web_template"
