"""
satviz_web — serve the CesiumJS satellite globe and hand back a link.

    from satviz_web import launch
    launch(groups=["stations", "gps-ops", "starlink"])
    # → prints (and in Colab, displays) a URL; open it in a browser tab.

Works locally (http://localhost:PORT), in Google Colab (proxied link), and on any
box you can reach over the network (`host="0.0.0.0"`). The browser does the SGP4
propagation with satellite.js; this server only ships the page and fresh TLEs.

    python satviz_web.py --groups stations gps-ops starlink --port 8765
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import threading
import time
import webbrowser
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import requests

HERE = Path(__file__).resolve().parent
WEB_DIR = HERE / "web"
CELESTRAK = "https://celestrak.org/NORAD/elements/gp.php?GROUP={group}&FORMAT=tle"
CESIUM_VERSION = "1.120.0"
CESIUM_CDN = f"https://cdn.jsdelivr.net/npm/cesium@{CESIUM_VERSION}/Build/Cesium/"
SATELLITE_JS_CDN = "https://cdn.jsdelivr.net/npm/satellite.js@5.0.0/dist/satellite.min.js"

GROUP_COLORS = {
    "stations": "#ff4d4d", "gps-ops": "#ffd700", "starlink": "#7fdbff", "oneweb": "#ff9f1c",
    "galileo": "#b10dc9", "glo-ops": "#2ecc40", "beidou": "#ff69b4", "weather": "#39cccc",
    "noaa": "#39cccc", "iridium-NEXT": "#01ff70", "geo": "#dddddd", "active": "#aaaaaa",
}


def parse_tle_text(text: str) -> list[list[str]]:
    lines = [l.rstrip() for l in text.splitlines() if l.strip()]
    out, i = [], 0
    while i < len(lines) - 1:
        if lines[i].startswith("1 ") and lines[i + 1].startswith("2 "):
            out.append([f"NORAD {lines[i][2:7].strip()}", lines[i], lines[i + 1]]); i += 2
        elif i + 2 < len(lines) and lines[i + 1].startswith("1 ") and lines[i + 2].startswith("2 "):
            out.append([lines[i].strip(), lines[i + 1], lines[i + 2]]); i += 3
        else:
            i += 1
    return out


class TLEStore:
    def __init__(self, groups, refetch_hours=2.0, tle_text=None):
        self.groups = list(groups)
        self.refetch_hours = refetch_hours
        self.lock = threading.Lock()
        self.data = {"fetched": None, "groups": {}}
        self.error = None
        if tle_text:
            self.data = {"fetched": datetime.now(timezone.utc).isoformat(), "groups": {"custom": parse_tle_text(tle_text)}}
        else:
            self.fetch()

    def fetch(self):
        groups = {}
        try:
            for g in self.groups:
                r = requests.get(CELESTRAK.format(group=g), timeout=30, headers={"User-Agent": "satviz/0.3"})
                r.raise_for_status()
                groups[g] = parse_tle_text(r.text)
            with self.lock:
                self.data = {"fetched": datetime.now(timezone.utc).isoformat(), "groups": groups}
                self.error = None
        except Exception as e:  # keep serving the previous set
            self.error = repr(e)
        return self.data

    def json(self) -> bytes:
        with self.lock:
            return json.dumps(self.data).encode()

    def stale(self) -> bool:
        if not self.data["fetched"]:
            return True
        age = datetime.now(timezone.utc) - datetime.fromisoformat(self.data["fetched"])
        return age.total_seconds() > self.refetch_hours * 3600


def make_handler(store: TLEStore, config: dict, cesium_dir: Path | None):
    index_html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    page = index_html.replace("/*CONFIG*/{}/*ENDCONFIG*/", json.dumps(config)).encode()

    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _send(self, body: bytes, ctype="text/html; charset=utf-8", status=200):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path in ("/", "/index.html"):
                return self._send(page)
            if u.path == "/tles.json":
                if "refresh=1" in u.query or store.stale():
                    store.fetch()
                return self._send(store.json(), "application/json")
            if u.path == "/health":
                return self._send(json.dumps({"ok": True, "error": store.error}).encode(), "application/json")
            if cesium_dir and u.path.startswith("/cesium/"):
                f = (cesium_dir / u.path[len("/cesium/"):]).resolve()
                if cesium_dir in f.parents and f.is_file():
                    ctype = mimetypes.guess_type(str(f))[0] or "application/octet-stream"
                    return self._send(f.read_bytes(), ctype)
            if u.path.startswith("/satellite.js") and config.get("_satjs_local"):
                return self._send(Path(config["_satjs_local"]).read_bytes(), "application/javascript")
            self._send(b"not found", "text/plain", 404)

    return Handler


class SatvizServer:
    def __init__(self, groups=("stations", "gps-ops", "starlink"), *, port=8765, host="127.0.0.1",
                 trail_minutes=30, max_trails=400, point_size=5, refetch_hours=2.0, ion_token=None,
                 cesium_dir=None, satellite_js=None, tle_text=None):
        self.store = TLEStore(groups, refetch_hours, tle_text)
        cesium_dir = Path(cesium_dir).resolve() if cesium_dir else None
        config = {
            "tle_url": "/tles.json", "trail_minutes": trail_minutes, "max_trails": max_trails,
            "point_size": point_size, "refetch_hours": refetch_hours, "colors": GROUP_COLORS,
            "ion_token": ion_token or os.environ.get("CESIUM_ION_TOKEN") or "",
            "cesium_base": "/cesium/" if cesium_dir else CESIUM_CDN,
            "satellite_js": "/satellite.js" if satellite_js else SATELLITE_JS_CDN,
            "_satjs_local": str(satellite_js) if satellite_js else "",
        }
        self.httpd = ThreadingHTTPServer((host, port), make_handler(self.store, config, cesium_dir))
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self.host = host
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    @property
    def local_url(self):
        return f"http://localhost:{self.port}/"

    def public_url(self):
        """Best URL for the current environment (Colab proxy, else localhost)."""
        try:
            from google.colab.output import eval_js  # type: ignore
            return eval_js(f"google.colab.kernel.proxyPort({self.port})")
        except Exception:
            return self.local_url

    def stop(self):
        self.httpd.shutdown()


def launch(groups=("stations", "gps-ops", "starlink"), *, open_browser=True, **kw) -> SatvizServer:
    srv = SatvizServer(groups, **kw)
    url = srv.public_url()
    if srv.store.error:
        print("TLE fetch problem:", srv.store.error, file=sys.stderr)
    try:
        from IPython.display import display, HTML
        display(HTML(f'<p>satviz is running → <a href="{url}" target="_blank" style="font-size:1.1em">{url}</a></p>'))
    except Exception:
        pass
    print("satviz running at", url)
    if open_browser and url.startswith("http://localhost"):
        try:
            webbrowser.open(url)
        except Exception:
            pass
    return srv


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--groups", nargs="+", default=["stations", "gps-ops", "starlink"])
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to share on your LAN")
    ap.add_argument("--trail", type=int, default=30)
    ap.add_argument("--max-trails", type=int, default=400)
    ap.add_argument("--ion-token", default=None, help="Cesium ion token for Bing/ion imagery (optional)")
    a = ap.parse_args()
    srv = launch(a.groups, port=a.port, host=a.host, trail_minutes=a.trail, max_trails=a.max_trails, ion_token=a.ion_token)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        srv.stop()


if __name__ == "__main__":
    main()
