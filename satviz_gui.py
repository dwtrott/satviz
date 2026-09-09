"""
satviz_gui — standalone 3D satellite viewer window (PyVista / VTK + Qt).

From a notebook (kernel stays free, window updates in the background):

    %gui qt
    from satviz_gui import launch
    app = launch(groups=["stations", "gps-ops", "starlink"])

From a terminal:

    python satviz_gui.py --groups stations gps-ops starlink

Mouse: left-drag orbit · scroll zoom · middle/shift-drag pan · click a satellite to label it.
Control panel (right side): play/pause, time scrub & speed, trail length, per-group
visibility, ECI/Earth-fixed frame, search & focus, refresh TLEs.

Textures: tries NASA Blue Marble (cached in ~/.satviz/), falls back to PyVista's bundled globe.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pyvista as pv

from satviz import Catalog, R_EARTH, GROUP_COLORS, _FALLBACK, gmst_rad, eci_to_ecef

CACHE = Path(os.environ.get("SATVIZ_CACHE", Path.home() / ".satviz"))
EARTH_URLS = [
    # NASA Blue Marble Next Generation, topography + bathymetry, 5400x2700
    "https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73909/world.topo.bathy.200412.3x5400x2700.jpg",
    "https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73776/world.topo.bathy.200408.3x5400x2700.jpg",
]
NIGHT_URL = "https://eoimages.gsfc.nasa.gov/images/imagerecords/55000/55167/earth_lights_lrg.jpg"


# --------------------------------------------------------------------------- #
# Assets
# --------------------------------------------------------------------------- #
def _download(url: str, dest: Path, timeout=60) -> bool:
    try:
        import requests
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "satviz/0.2"})
        r.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(r.content)
        return True
    except Exception:
        return False


def earth_texture(path: str | None = None) -> tuple[pv.Texture, str]:
    """Return (texture, source description)."""
    if path and Path(path).exists():
        return pv.read_texture(path), f"custom: {path}"
    cached = CACHE / "earth_bluemarble.jpg"
    if not cached.exists():
        for u in EARTH_URLS:
            if _download(u, cached):
                break
    if cached.exists():
        try:
            return pv.read_texture(str(cached)), "NASA Blue Marble (cached)"
        except Exception:
            pass
    from pyvista import examples
    return examples.load_globe_texture(), "PyVista bundled globe (2k)"


def earth_mesh(radius=R_EARTH, n_lon=360, n_lat=180) -> pv.PolyData:
    """Sphere with equirectangular texture coords; +X axis = 0° lon, +Z = north."""
    lon = np.linspace(-np.pi, np.pi, n_lon + 1)
    lat = np.linspace(-np.pi / 2, np.pi / 2, n_lat + 1)
    LON, LAT = np.meshgrid(lon, lat)
    x = radius * np.cos(LAT) * np.cos(LON)
    y = radius * np.cos(LAT) * np.sin(LON)
    z = radius * np.sin(LAT)
    grid = pv.StructuredGrid(x, y, z)
    # texture coords: u along longitude (west→east), v along latitude (south→north).
    # Assign on the grid BEFORE extracting — extract_surface permutes point order.
    u = (LON.ravel(order="F") + np.pi) / (2 * np.pi)
    v = (LAT.ravel(order="F") + np.pi / 2) / np.pi
    grid.active_texture_coordinates = np.column_stack([u, v]).astype(np.float32)
    surf = grid.extract_surface(algorithm="dataset_surface")
    surf.compute_normals(inplace=True, auto_orient_normals=True)
    return surf


def star_points(n=4000, radius=8e5, seed=0) -> pv.PolyData:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3)); v /= np.linalg.norm(v, axis=1)[:, None]
    pts = pv.PolyData(v * radius)
    pts["mag"] = rng.uniform(0.3, 1.0, n)
    return pts


def sun_direction_eci(t: datetime) -> np.ndarray:
    """Approximate unit vector Earth→Sun in ECI (good to ~1°)."""
    from sgp4.api import jday
    jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute, t.second)
    n = jd + fr - 2451545.0
    L = math.radians((280.460 + 0.9856474 * n) % 360)
    g = math.radians((357.528 + 0.9856003 * n) % 360)
    lam = L + math.radians(1.915) * math.sin(g) + math.radians(0.020) * math.sin(2 * g)
    eps = math.radians(23.439 - 0.0000004 * n)
    return np.array([math.cos(lam), math.cos(eps) * math.sin(lam), math.sin(eps) * math.sin(lam)])


def _hex_to_rgb(h: str):
    h = h.lstrip("#"); return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


# --------------------------------------------------------------------------- #
# Scene (Qt-independent so it can run off-screen too)
# --------------------------------------------------------------------------- #
class SatelliteScene:
    def __init__(self, plotter: pv.Plotter, catalog: Catalog, *, trail_minutes=45.0,
                 trail_points=60, max_trails=600, point_size=7.0, earth_fixed=False,
                 texture_path=None, show_grid=False):
        self.p = plotter
        self.catalog = catalog
        self.trail_minutes = trail_minutes
        self.trail_points = trail_points
        self.max_trails = max_trails
        self.point_size = point_size
        self.earth_fixed = earth_fixed
        self.time_offset = timedelta(0)
        self.speed = 1.0                 # simulated seconds per real second
        self.paused = False
        self.selected: int | None = None
        self._last_tick = None
        self._lock = threading.Lock()
        self.group_actors: dict[str, dict] = {}
        self._build(texture_path, show_grid)

    # ---- time -------------------------------------------------------------- #
    def now(self) -> datetime:
        return datetime.now(timezone.utc) + self.time_offset

    # ---- build ------------------------------------------------------------- #
    def _build(self, texture_path, show_grid):
        p = self.p
        p.set_background("#02030a")
        p.enable_anti_aliasing("msaa") if hasattr(p, "enable_anti_aliasing") else None

        # stars
        stars = star_points()
        p.add_mesh(stars, scalars="mag", cmap="gray", clim=[0, 1], point_size=1.5,
                   render_points_as_spheres=False, show_scalar_bar=False, name="stars",
                   lighting=False, pickable=False)

        # earth
        tex, self.texture_source = earth_texture(texture_path)
        self.earth = earth_mesh()
        self.earth_actor = p.add_mesh(self.earth, texture=tex, name="earth", smooth_shading=True,
                                      ambient=0.22, diffuse=1.0, specular=0.2, specular_power=25,
                                      pickable=False)
        # thin atmosphere glow
        atmo = earth_mesh(radius=R_EARTH * 1.012, n_lon=120, n_lat=60)
        p.add_mesh(atmo, color="#5aa9ff", opacity=0.10, smooth_shading=True, name="atmo",
                   pickable=False, specular=0.0)
        if show_grid:
            p.add_mesh(pv.Sphere(radius=R_EARTH * 1.002), style="wireframe", color="white",
                       opacity=0.1, name="grid", pickable=False)

        # lighting: single sun + faint ambient
        p.remove_all_lights()
        self.sun = pv.Light(light_type="scene light", intensity=1.9, color="white")
        p.add_light(self.sun)
        p.add_light(pv.Light(light_type="headlight", intensity=0.08))

        # satellites, one actor pair per group
        t = self.now()
        heads, trails = self._positions(t)
        self.valid = ~np.isnan(heads).any(axis=1)
        groups = np.array(self.catalog.groups)
        for gi, g in enumerate(sorted(set(groups))):
            idx = np.where((groups == g) & self.valid)[0]
            if len(idx) == 0:
                continue
            col = GROUP_COLORS.get(g) or _FALLBACK[gi % len(_FALLBACK)]
            pts = pv.PolyData(heads[idx])
            pa = p.add_mesh(pts, color=col, point_size=self.point_size, render_points_as_spheres=True,
                            name=f"pts_{g}", lighting=False)
            tidx = idx[: self.max_trails]
            k = self.trail_points
            tp = trails[tidx].reshape(-1, 3)
            lines = np.hstack([np.concatenate([[k], np.arange(k) + i * k]) for i in range(len(tidx))])
            tmesh = pv.PolyData(tp, lines=lines)
            # fade along trail: scalar 0..1 → opacity via color ramp toward background
            fade = np.tile(np.linspace(0.05, 1.0, k), len(tidx))
            tmesh["fade"] = fade
            from matplotlib.colors import LinearSegmentedColormap
            cmap = LinearSegmentedColormap.from_list(f"fade_{g}", ["#02030a", col])
            ta = p.add_mesh(tmesh, scalars="fade", cmap=cmap, clim=[0, 1],
                            line_width=1.0, show_scalar_bar=False, name=f"trl_{g}",
                            lighting=False, pickable=False, opacity=0.7)
            self.group_actors[g] = dict(idx=idx, tidx=tidx, pts=pts, trails=tmesh,
                                        pts_actor=pa, trl_actor=ta, color=col, visible=True)

        # selection marker + label
        self.sel_mesh = pv.PolyData(np.zeros((1, 3)))
        self.sel_actor = p.add_mesh(self.sel_mesh, color="white", point_size=self.point_size * 2.4,
                                    render_points_as_spheres=True, name="sel", pickable=False)
        self.sel_actor.SetVisibility(False)
        self.label = p.add_text("", position="upper_left", font_size=11, color="white", name="label")
        self.clock = p.add_text("", position="lower_left", font_size=9, color="#9fb3c8", name="clock")

        p.enable_point_picking(callback=self._on_pick, show_message=False, use_picker=True,
                               pickable_window=False, show_point=False, tolerance=0.02)
        p.enable_trackball_style()
        self.reset_view()
        self._apply_frame(t)
        self._update_clock(t)

    # ---- geometry ---------------------------------------------------------- #
    def _positions(self, t: datetime):
        k = self.trail_points
        span = timedelta(minutes=max(self.trail_minutes, 0.01))
        times = [t - span * (1 - j / (k - 1)) for j in range(k)]
        r = self.catalog.propagate(times)
        if self.earth_fixed:
            for j, tj in enumerate(times):
                r[:, j, :] = eci_to_ecef(r[:, j, :], tj)
        return r[:, -1, :], r

    def _apply_frame(self, t: datetime):
        """Rotate Earth to sidereal angle (ECI) or hold it fixed (ECEF); place the Sun."""
        g = math.degrees(gmst_rad(t))
        s = sun_direction_eci(t)
        if self.earth_fixed:
            self.earth_actor.SetOrientation(0, 0, 0)
            s = eci_to_ecef(s, t)
        else:
            self.earth_actor.SetOrientation(0, 0, g)
        self.sun.position = tuple(s * 1.5e8)
        self.sun.focal_point = (0, 0, 0)

    def _update_clock(self, t):
        frame = "Earth-fixed" if self.earth_fixed else "ECI"
        off = self.time_offset.total_seconds() / 60
        offs = f"  ({off:+.0f} min)" if abs(off) >= 0.5 else ""
        fetched = self.catalog.fetched_at.strftime("%H:%M") if self.catalog.fetched_at else "n/a"
        self.clock.SetText(0, f"{t:%Y-%m-%d %H:%M:%S} UTC{offs}   ×{self.speed:g}   {frame}   "
                              f"{len(self.catalog)} objects   TLEs {fetched} UTC   {self.texture_source}")

    # ---- update -------------------------------------------------------------- #
    def tick(self, dt_real: float):
        if not self.paused and self.speed != 1.0:
            self.time_offset += timedelta(seconds=(self.speed - 1.0) * dt_real)
        if self.paused:
            self.time_offset -= timedelta(seconds=dt_real)   # freeze the displayed time
        self.update()

    def update(self):
        t = self.now()
        with self._lock:
            heads, trails = self._positions(t)
        heads = np.nan_to_num(heads, nan=1e9)
        trails = np.nan_to_num(trails, nan=1e9)
        for g, a in self.group_actors.items():
            a["pts"].points = heads[a["idx"]]
            a["trails"].points = trails[a["tidx"]].reshape(-1, 3)
        if self.selected is not None:
            self.sel_mesh.points = heads[[self.selected]]
            alt = np.linalg.norm(heads[self.selected]) - R_EARTH
            self.label.SetText(2, f"{self.catalog.names[self.selected]}\n"
                                  f"NORAD {self.catalog.norad[self.selected]}   "
                                  f"{self.catalog.groups[self.selected]}   alt {alt:,.0f} km")
        self._apply_frame(t)
        self._update_clock(t)
        self.p.render()

    def set_trail_minutes(self, minutes: float):
        self.trail_minutes = minutes
        self.update()

    def set_frame(self, earth_fixed: bool):
        self.earth_fixed = earth_fixed
        self.update()

    def set_group_visible(self, g: str, visible: bool):
        a = self.group_actors.get(g)
        if a:
            a["pts_actor"].SetVisibility(visible); a["trl_actor"].SetVisibility(visible)
            a["visible"] = visible; self.p.render()

    def select(self, i: int | None):
        self.selected = i
        self.sel_actor.SetVisibility(i is not None)
        if i is None:
            self.label.SetText(2, "")
        self.update()

    def find(self, text: str) -> list[int]:
        t = text.lower().strip()
        if not t:
            return []
        if t.isdigit():
            return [i for i, n in enumerate(self.catalog.norad) if str(n) == t]
        return [i for i, n in enumerate(self.catalog.names) if t in n.lower()]

    def focus(self, i: int, distance_km: float = 3500.0):
        self.select(i)
        pos = np.asarray(self.sel_mesh.points[0])
        d = pos / np.linalg.norm(pos)
        eye = pos + d * distance_km + np.array([0, 0, distance_km * 0.3])
        self.p.camera_position = [tuple(eye), tuple(pos), (0, 0, 1)]
        self.p.camera.clipping_range = (10, 3e6)
        self.p.render()

    def reset_view(self):
        self.p.camera_position = [(R_EARTH * 3.6, R_EARTH * 2.6, R_EARTH * 1.8), (0, 0, 0), (0, 0, 1)]
        self.p.camera.clipping_range = (100, 3e6)
        self.p.render()

    def _on_pick(self, point, *args):
        # nearest visible satellite to the picked point
        best, bd = None, np.inf
        for g, a in self.group_actors.items():
            if not a["visible"]:
                continue
            d = np.linalg.norm(a["pts"].points - point, axis=1)
            j = int(np.argmin(d))
            if d[j] < bd:
                best, bd = int(a["idx"][j]), d[j]
        if best is not None and bd < R_EARTH * 0.15:
            self.select(best)

    def refetch(self):
        with self._lock:
            self.catalog.fetch(sorted(set(self.catalog.groups)))
        self.update()


# --------------------------------------------------------------------------- #
# Qt window
# --------------------------------------------------------------------------- #
class SatelliteApp:
    def __init__(self, groups=("stations", "gps-ops", "starlink"), *, update_ms=1000,
                 refetch_hours=2.0, window_size=(1500, 950), tle_text=None, **scene_kw):
        from pyvistaqt import BackgroundPlotter
        from qtpy import QtWidgets, QtCore

        self.catalog = Catalog()
        if tle_text:
            self.catalog.add_tle_text(tle_text, "custom")
        else:
            self.catalog.fetch(groups)

        self.plotter = BackgroundPlotter(title="satviz — live satellite viewer", window_size=window_size,
                                         lighting="none", toolbar=False, menu_bar=False)
        self.scene = SatelliteScene(self.plotter, self.catalog, **scene_kw)
        self._last = datetime.now(timezone.utc)
        self._last_fetch = self._last
        self.refetch_hours = refetch_hours
        self._build_panel(QtWidgets, QtCore)
        self.plotter.add_callback(self._tick, interval=update_ms)

    # ---- controls -------------------------------------------------------- #
    def _build_panel(self, W, C):
        win = self.plotter.app_window
        dock = W.QDockWidget("Controls", win)
        dock.setFeatures(W.QDockWidget.DockWidgetMovable | W.QDockWidget.DockWidgetFloatable)
        panel = W.QWidget(); lay = W.QVBoxLayout(panel); lay.setSpacing(6)
        s = self.scene

        # playback
        row = W.QHBoxLayout()
        self.play_btn = W.QPushButton("⏸ Pause"); self.play_btn.clicked.connect(self._toggle_pause)
        now_btn = W.QPushButton("Now"); now_btn.clicked.connect(self._go_now)
        row.addWidget(self.play_btn); row.addWidget(now_btn); lay.addLayout(row)

        lay.addWidget(W.QLabel("Speed"))
        self.speed_box = W.QComboBox()
        for v in (1, 10, 60, 300, 1000, 3600):
            self.speed_box.addItem(f"×{v}", v)
        self.speed_box.currentIndexChanged.connect(lambda _: self._set_speed(self.speed_box.currentData()))
        lay.addWidget(self.speed_box)

        self.offset_lbl = W.QLabel("Time offset: 0 min"); lay.addWidget(self.offset_lbl)
        self.offset = W.QSlider(C.Qt.Horizontal); self.offset.setRange(-1440, 1440); self.offset.setValue(0)
        self.offset.valueChanged.connect(self._set_offset); lay.addWidget(self.offset)

        self.trail_lbl = W.QLabel(f"Trail: {s.trail_minutes:.0f} min"); lay.addWidget(self.trail_lbl)
        trail = W.QSlider(C.Qt.Horizontal); trail.setRange(0, 240); trail.setValue(int(s.trail_minutes))
        trail.valueChanged.connect(self._set_trail); lay.addWidget(trail)

        lay.addWidget(W.QLabel("Frame"))
        fr = W.QHBoxLayout()
        self.r_eci = W.QRadioButton("ECI (inertial)"); self.r_ecef = W.QRadioButton("Earth-fixed")
        (self.r_ecef if s.earth_fixed else self.r_eci).setChecked(True)
        self.r_ecef.toggled.connect(lambda on: s.set_frame(on))
        fr.addWidget(self.r_eci); fr.addWidget(self.r_ecef); lay.addLayout(fr)

        lay.addWidget(W.QLabel("Groups"))
        for g, a in s.group_actors.items():
            cb = W.QCheckBox(f"{g}  ({len(a['idx'])})"); cb.setChecked(True)
            cb.setStyleSheet(f"color: {a['color']}; font-weight: bold;")
            cb.toggled.connect(lambda on, g=g: s.set_group_visible(g, on))
            lay.addWidget(cb)

        lay.addWidget(W.QLabel("Find (name or NORAD id)"))
        fl = W.QHBoxLayout()
        self.search = W.QLineEdit(); self.search.setPlaceholderText("ISS, 25544, NAVSTAR…")
        self.search.returnPressed.connect(self._search)
        go = W.QPushButton("Focus"); go.clicked.connect(self._search)
        fl.addWidget(self.search); fl.addWidget(go); lay.addLayout(fl)
        self.results = W.QListWidget(); self.results.setMaximumHeight(140)
        self.results.itemClicked.connect(lambda it: s.focus(it.data(C.Qt.UserRole)))
        lay.addWidget(self.results)

        row2 = W.QHBoxLayout()
        clr = W.QPushButton("Clear selection"); clr.clicked.connect(lambda: s.select(None))
        rv = W.QPushButton("Reset view"); rv.clicked.connect(s.reset_view)
        row2.addWidget(clr); row2.addWidget(rv); lay.addLayout(row2)
        rf = W.QPushButton("Refresh TLEs now"); rf.clicked.connect(self._refetch); lay.addWidget(rf)

        self.status = W.QLabel(""); self.status.setWordWrap(True); lay.addWidget(self.status)
        lay.addStretch(1)
        panel.setMinimumWidth(260)
        dock.setWidget(panel)
        win.addDockWidget(C.Qt.RightDockWidgetArea, dock)

    def _toggle_pause(self):
        self.scene.paused = not self.scene.paused
        self.play_btn.setText("▶ Play" if self.scene.paused else "⏸ Pause")

    def _go_now(self):
        self.scene.time_offset = timedelta(0); self.offset.blockSignals(True)
        self.offset.setValue(0); self.offset.blockSignals(False)
        self.offset_lbl.setText("Time offset: 0 min"); self.scene.update()

    def _set_speed(self, v):
        self.scene.speed = float(v)

    def _set_offset(self, minutes):
        self.scene.time_offset = timedelta(minutes=minutes)
        self.offset_lbl.setText(f"Time offset: {minutes:+d} min")
        self.scene.update()

    def _set_trail(self, minutes):
        self.trail_lbl.setText(f"Trail: {minutes} min"); self.scene.set_trail_minutes(minutes)

    def _search(self):
        from qtpy import QtCore, QtWidgets
        hits = self.scene.find(self.search.text())
        self.results.clear()
        for i in hits[:200]:
            it = QtWidgets.QListWidgetItem(f"{self.catalog.names[i]}  [{self.catalog.norad[i]}]")
            it.setData(QtCore.Qt.UserRole, i); self.results.addItem(it)
        if hits:
            self.scene.focus(hits[0])
        self.status.setText(f"{len(hits)} match(es)")

    def _refetch(self):
        try:
            self.scene.refetch(); self._last_fetch = datetime.now(timezone.utc)
            self.status.setText(f"TLEs refreshed: {len(self.catalog)} objects")
        except Exception as e:
            self.status.setText(f"Refresh failed: {e}")

    def _tick(self):
        now = datetime.now(timezone.utc)
        dt = (now - self._last).total_seconds(); self._last = now
        try:
            if (now - self._last_fetch).total_seconds() > self.refetch_hours * 3600:
                self._refetch()
            self.scene.tick(dt)
            off = self.scene.time_offset.total_seconds() / 60
            if abs(off - self.offset.value()) > 1 and abs(off) <= 1440:
                self.offset.blockSignals(True); self.offset.setValue(int(off)); self.offset.blockSignals(False)
                self.offset_lbl.setText(f"Time offset: {off:+.0f} min")
        except Exception as e:
            self.status.setText(f"update error: {e}")

    def close(self):
        self.plotter.close()


def launch(groups=("stations", "gps-ops", "starlink"), **kw) -> SatelliteApp:
    """Open the viewer window (non-blocking). In Jupyter the Qt event loop is enabled automatically."""
    try:  # equivalent of `%gui qt` so the window stays live while the kernel is idle
        from IPython import get_ipython
        ip = get_ipython()
        if ip is not None:
            ip.run_line_magic("gui", "qt")
    except Exception:
        pass
    return SatelliteApp(groups, **kw)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--groups", nargs="+", default=["stations", "gps-ops", "starlink"])
    ap.add_argument("--trail", type=float, default=45.0, help="trail length, minutes")
    ap.add_argument("--max-trails", type=int, default=600)
    ap.add_argument("--texture", default=None, help="path to an equirectangular Earth image")
    ap.add_argument("--earth-fixed", action="store_true")
    a = ap.parse_args()
    from qtpy import QtWidgets
    qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app = launch(a.groups, trail_minutes=a.trail, max_trails=a.max_trails,
                 texture_path=a.texture, earth_fixed=a.earth_fixed)
    qapp.exec_()


if __name__ == "__main__":
    main()
