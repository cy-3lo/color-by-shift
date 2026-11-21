"""Screenshots of the web build in headless Chrome (no window), with the browser console.

    python tools/web_shot.py --out runs/web_shot [--level P01] [--wait 25]
        [--do move:640,360 click:640,360 wait:1.5 key:z shot:name ...]

Serves game/export/web on a local port, opens index.html?level=ID in Chrome started with
--headless=new, waits for the game, then runs the actions in order (pointer positions in
page pixels; "shot:name" writes <out>/<name>.png; a final shot "end" is always taken).
Console messages and exceptions are printed. Talks to Chrome over its DevTools websocket
with the standard library only.
"""

from __future__ import annotations

import argparse
import base64
import functools
import http.server
import json
import os
import socket
import struct
import subprocess
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


class DevTools:
    """A minimal client of the Chrome DevTools protocol (one websocket, text frames)."""

    def __init__(self, ws_url: str):
        host, rest = ws_url.removeprefix("ws://").split("/", 1)
        name, port = host.split(":")
        self.sock = socket.create_connection((name, int(port)))
        key = base64.b64encode(os.urandom(16)).decode()
        self.sock.sendall(
            (
                f"GET /{rest} HTTP/1.1\r\nHost: {host}\r\nUpgrade: websocket\r\n"
                f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\n"
                "Sec-WebSocket-Version: 13\r\n\r\n"
            ).encode()
        )
        head = b""
        while b"\r\n\r\n" not in head:
            head += self.sock.recv(1)
        self.next_id = 0
        self.events: list[dict] = []

    def _read(self, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            part = self.sock.recv(n - len(buf))
            if not part:
                raise ConnectionError("Chrome closed the connection")
            buf += part
        return buf

    def _message(self) -> dict:
        data = b""
        while True:
            b0, b1 = self._read(2)
            n = b1 & 0x7F
            if n == 126:
                (n,) = struct.unpack(">H", self._read(2))
            elif n == 127:
                (n,) = struct.unpack(">Q", self._read(8))
            data += self._read(n)
            if b0 & 0x80:
                return json.loads(data)

    def call(self, method: str, **params) -> dict:
        self.next_id += 1
        payload = json.dumps({"id": self.next_id, "method": method, "params": params}).encode()
        mask = os.urandom(4)
        n = len(payload)
        head = bytes([0x81]) + (
            bytes([0x80 | n])
            if n < 126
            else bytes([0x80 | 126]) + struct.pack(">H", n)
            if n < 65536
            else bytes([0x80 | 127]) + struct.pack(">Q", n)
        )
        self.sock.sendall(head + mask + bytes(c ^ mask[i % 4] for i, c in enumerate(payload)))
        while True:
            msg = self._message()
            if msg.get("id") == self.next_id:
                return msg.get("result", msg)
            self.events.append(msg)

    def wait(self, seconds: float) -> None:
        """Lets time pass while collecting the events Chrome sends."""
        end = time.time() + seconds
        while time.time() < end:
            self.sock.settimeout(max(0.01, end - time.time()))
            try:
                self.events.append(self._message())
            except (TimeoutError, socket.timeout):
                break
        self.sock.settimeout(None)


def console(events: list[dict]) -> list[str]:
    out = []
    for e in events:
        p = e.get("params", {})
        if e.get("method") == "Runtime.consoleAPICalled":
            out.append(f"{p['type']}: " + " ".join(str(a.get("value", "")) for a in p["args"]))
        elif e.get("method") == "Runtime.exceptionThrown":
            out.append("exception: " + p["exceptionDetails"].get("text", ""))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="folder for the screenshots")
    ap.add_argument("--level", default="")
    ap.add_argument("--wait", type=float, default=25.0, help="seconds before the first action")
    ap.add_argument("--size", default="1280x720")
    ap.add_argument("--do", nargs="*", default=[], help="move:x,y click:x,y key:c wait:s shot:name")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    w, h = (int(v) for v in args.size.split("x"))

    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler, directory=str(ROOT / "game" / "export" / "web")
    )
    handler.log_message = lambda *a, **k: None
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/index.html"
    if args.level:
        url += f"?level={args.level}"

    with tempfile.TemporaryDirectory() as profile:
        chrome = subprocess.Popen(
            [
                CHROME, "--headless=new", "--no-first-run", "--disable-extensions",
                "--enable-unsafe-swiftshader", f"--window-size={w},{h}", "--hide-scrollbars",
                f"--user-data-dir={profile}", "--remote-debugging-port=0", "about:blank",
            ],
            stderr=subprocess.PIPE, text=True,
        )  # fmt: skip
        try:
            port = None
            for line in chrome.stderr:
                if "DevTools listening on" in line:
                    port = line.split("ws://")[1].split("/")[0].split(":")[1]
                    break
            tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json"))
            page = next(t for t in tabs if t["type"] == "page")
            dev = DevTools(page["webSocketDebuggerUrl"])
            dev.call("Runtime.enable")
            dev.call("Page.enable")
            dev.call("Page.navigate", url=url)
            dev.wait(args.wait)

            def shot(name: str) -> None:
                png = dev.call("Page.captureScreenshot", format="png")["data"]
                (out / f"{name}.png").write_bytes(base64.b64decode(png))
                print("wrote", out / f"{name}.png")

            def mouse(kind: str, x: float, y: float, **more) -> None:
                dev.call("Input.dispatchMouseEvent", type=kind, x=x, y=y, **more)

            for action in args.do:
                verb, _, arg = action.partition(":")
                if verb == "move":
                    x, y = (float(v) for v in arg.split(","))
                    mouse("mouseMoved", x, y)
                elif verb == "click":
                    x, y = (float(v) for v in arg.split(","))
                    mouse("mouseMoved", x, y)
                    mouse("mousePressed", x, y, button="left", clickCount=1)
                    mouse("mouseReleased", x, y, button="left", clickCount=1)
                elif verb == "key":
                    code = ord(arg.upper())
                    for kind in ("keyDown", "keyUp"):
                        dev.call(
                            "Input.dispatchKeyEvent", type=kind, key=arg,
                            code=f"Key{arg.upper()}", windowsVirtualKeyCode=code,
                        )  # fmt: skip
                elif verb == "wait":
                    dev.wait(float(arg))
                elif verb == "shot":
                    shot(arg)
                else:
                    raise SystemExit(f"unknown action {action}")
                dev.wait(0.15)
            shot("end")
            for line in console(dev.events):
                print("console", line)
        finally:
            chrome.terminate()
            server.shutdown()


if __name__ == "__main__":
    main()
