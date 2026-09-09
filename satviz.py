"""
satviz — live 3D satellite viewer for Jupyter.

    from satviz import SatelliteViewer
    v = SatelliteViewer(groups=["stations", "gps-ops", "starlink"])
    v.show()           # interactive Plotly globe; orbit/zoom/pan with the mouse
    v.start()          # background thread: re-propagate every few seconds,
                       # re-fetch TLEs from CelesTrak every couple of hours
    v.stop()

Data: CelesTrak GP sets (https://celestrak.org). Propagation: SGP4.
Frame: ECI (TEME) — Earth is drawn fixed, so satellites move through inertial
space. Toggle `earth_fixed=True` to rotate into an Earth-fixed (ECEF) view
instead, so ground tracks make sense.
"""
from __future__ import annotations

import math
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Iterable

import numpy as np
import plotly.graph_objects as go
import requests
from sgp4.api import Satrec, SatrecArray, jday

R_EARTH = 6378.137  # km
CELESTRAK = "https://celestrak.org/NORAD/elements/gp.php?GROUP={group}&FORMAT=tle"

GROUP_COLORS = {
    "stations": "#ff4d4d",
    "gps-ops": "#ffd700",
    "starlink": "#7fdbff",
    "oneweb": "#ff9f1c",
    "galileo": "#b10dc9",
    "glo-ops": "#2ecc40",
    "beidou": "#ff69b4",
    "weather": "#39cccc",
    "noaa": "#39cccc",
    "iridium-NEXT": "#01ff70",
    "geo": "#dddddd",
    "active": "#aaaaaa",
}
_FALLBACK = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
             "#46f0f0", "#f032e6", "#bcf60c", "#fabebe", "#008080"]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
class Catalog:
    """A set of TLEs, tagged by group, with vectorised SGP4 propagation."""

    def __init__(self):
        self.names: list[str] = []
        self.groups: list[str] = []
        self.norad: list[int] = []
        self._sats: list[Satrec] = []
        self._arr: SatrecArray | None = None
        self.fetched_at: datetime | None = None

    # ---- loading --------------------------------------------------------- #
    def add_tle_text(self, text: str, group: str = "custom") -> int:
        lines = [l.rstrip() for l in text.splitlines() if l.strip()]
        n = 0
        i = 0
        while i < len(lines) - 1:
            if lines[i].startswith("1 ") and lines[i + 1].startswith("2 "):
                name, l1, l2 = f"NORAD {lines[i][2:7].strip()}", lines[i], lines[i + 1]
                i += 2
            elif i + 2 < len(lines) and lines[i + 1].startswith("1 ") and lines[i + 2].startswith("2 "):
                name, l1, l2 = lines[i].strip(), lines[i + 1], lines[i + 2]
                i += 3
            else:
                i += 1
                continue
            try:
                s = Satrec.twoline2rv(l1, l2)
            except Exception:
                continue
            self._sats.append(s)
            self.names.append(name)
            self.groups.append(group)
            self.norad.append(s.satnum)
            n += 1
        self._arr = SatrecArray(self._sats) if self._sats else None
        return n

    def fetch(self, groups: Iterable[str], timeout: float = 30) -> dict[str, int]:
        """Replace catalog contents with fresh TLEs from CelesTrak."""
        new = Catalog()
        counts = {}
        for g in groups:
            r = requests.get(CELESTRAK.format(group=g), timeout=timeout,
                             headers={"User-Agent": "satviz/0.1"})
            r.raise_for_status()
            counts[g] = new.add_tle_text(r.text, g)
        new.fetched_at = datetime.now(timezone.utc)
        self.__dict__.update(new.__dict__)
        return counts

    def __len__(self):
        return len(self._sats)

    # ---- propagation ----------------------------------------------------- #
    def propagate(self, times: np.ndarray | list[datetime]) -> np.ndarray:
        """Return ECI (TEME) positions, shape (n_sats, n_times, 3), in km. NaN on error."""
        if self._arr is None:
            return np.empty((0, len(times), 3))
        jd = np.empty(len(times)); fr = np.empty(len(times))
        for k, t in enumerate(times):
            jd[k], fr[k] = jday(t.year, t.month, t.day, t.hour, t.minute,
                                t.second + t.microsecond * 1e-6)
        err, r, _v = self._arr.sgp4(jd, fr)
        r = r.astype(float)
        r[err != 0] = np.nan
        return r


def gmst_rad(t: datetime) -> float:
    """Greenwich mean sidereal time (radians) — enough for a visualisation."""
    jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute, t.second + t.microsecond * 1e-6)
    T = (jd + fr - 2451545.0) / 36525.0
    g = 280.46061837 + 360.98564736629 * (jd + fr - 2451545.0) + 0.000387933 * T**2
    return math.radians(g % 360.0)


def eci_to_ecef(r: np.ndarray, t: datetime) -> np.ndarray:
    th = gmst_rad(t)
    c, s = math.cos(th), math.sin(th)
    x, y, z = r[..., 0], r[..., 1], r[..., 2]
    return np.stack([c * x + s * y, -s * x + c * y, z], axis=-1)


# --------------------------------------------------------------------------- #
# Viewer
# --------------------------------------------------------------------------- #
class SatelliteViewer:
    def __init__(
        self,
        groups: Iterable[str] = ("stations", "gps-ops", "starlink"),
        trail_minutes: float = 45.0,
        trail_points: int = 40,
        update_seconds: float = 5.0,
        refetch_hours: float = 2.0,
        earth_fixed: bool = False,
        max_trails: int = 400,
        marker_size: float = 2.5,
        height: int = 750,
        auto_fetch: bool = True,
    ):
        """
        groups        CelesTrak GROUP names (stations, gps-ops, starlink, oneweb,
                      galileo, glo-ops, beidou, weather, geo, active, ...).
        trail_minutes length of the trajectory drawn behind each satellite.
        max_trails    only draw trails for this many satellites (perf); markers
                      are always drawn for everything.
        earth_fixed   rotate into ECEF so Earth's surface features stay put and
                      ground tracks are meaningful (LEO trails curve west).
        """
        self.groups = list(groups)
        self.trail_minutes = trail_minutes
        self.trail_points = trail_points
        self.update_seconds = update_seconds
        self.refetch_hours = refetch_hours
        self.earth_fixed = earth_fixed
        self.max_trails = max_trails
        self.marker_size = marker_size
        self.height = height

        self.catalog = Catalog()
        self.fig: go.FigureWidget | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.time_offset = timedelta(0)   # set to look into the past/future
        self.last_error: str | None = None

        if auto_fetch:
            self.refresh_tles()

    # ---- data ------------------------------------------------------------ #
    def refresh_tles(self) -> dict[str, int]:
        with self._lock:
            counts = self.catalog.fetch(self.groups)
        return counts

    def load_tle_text(self, text: str, group: str = "custom") -> int:
        with self._lock:
            return self.catalog.add_tle_text(text, group)

    def now(self) -> datetime:
        return datetime.now(timezone.utc) + self.time_offset

    # ---- geometry -------------------------------------------------------- #
    def _positions(self, t: datetime):
        """Return (heads (n,3), trails (n, k, 3)) in the display frame."""
        n = len(self.catalog)
        k = self.trail_points
        times = [t - timedelta(minutes=self.trail_minutes) * (1 - j / (k - 1)) for j in range(k)]
        r = self.catalog.propagate(times)  # (n, k, 3)
        if self.earth_fixed:
            for j, tj in enumerate(times):
                r[:, j, :] = eci_to_ecef(r[:, j, :], tj)
        return r[:, -1, :], r

    @staticmethod
    def _earth_mesh(res: int = 48):
        u = np.linspace(0, 2 * np.pi, res)
        v = np.linspace(0, np.pi, res // 2)
        x = R_EARTH * np.outer(np.cos(u), np.sin(v))
        y = R_EARTH * np.outer(np.sin(u), np.sin(v))
        z = R_EARTH * np.outer(np.ones_like(u), np.cos(v))
        # simple shading: brighter toward +x so the sphere reads as 3D
        shade = (x / R_EARTH + 1) / 2
        return x, y, z, shade

    def _color(self, group: str, i: int) -> str:
        return GROUP_COLORS.get(group) or _FALLBACK[i % len(_FALLBACK)]

    # ---- figure ---------------------------------------------------------- #
    def build_figure(self) -> go.FigureWidget:
        t = self.now()
        heads, trails = self._positions(t)
        names, groups = self.catalog.names, self.catalog.groups
        uniq = sorted(set(groups))

        data = []
        x, y, z, shade = self._earth_mesh()
        data.append(go.Surface(
            x=x, y=y, z=z, surfacecolor=shade, showscale=False,
            colorscale=[[0, "#0b2d5c"], [0.5, "#1f5fa8"], [1, "#3b8fd9"]],
            opacity=1.0, hoverinfo="skip", name="Earth",
            contours=dict(x=dict(show=False), y=dict(show=False), z=dict(show=False)),
        ))
        # lat/long grid so rotation is visible
        gx, gy, gz = [], [], []
        for lat in range(-60, 61, 30):
            lon = np.linspace(0, 2 * np.pi, 73)
            la = math.radians(lat)
            gx += list(R_EARTH * 1.001 * np.cos(la) * np.cos(lon)) + [None]
            gy += list(R_EARTH * 1.001 * np.cos(la) * np.sin(lon)) + [None]
            gz += list(R_EARTH * 1.001 * np.sin(la) * np.ones_like(lon)) + [None]
        for lon in range(0, 360, 30):
            lat = np.linspace(-np.pi / 2, np.pi / 2, 37)
            lo = math.radians(lon)
            gx += list(R_EARTH * 1.001 * np.cos(lat) * np.cos(lo)) + [None]
            gy += list(R_EARTH * 1.001 * np.cos(lat) * np.sin(lo)) + [None]
            gz += list(R_EARTH * 1.001 * np.sin(lat)) + [None]
        data.append(go.Scatter3d(x=gx, y=gy, z=gz, mode="lines", hoverinfo="skip",
                                 line=dict(color="rgba(255,255,255,0.15)", width=1),
                                 name="grid", showlegend=False))

        self._trace_index = {}
        for gi, g in enumerate(uniq):
            idx = np.array([i for i, gg in enumerate(groups) if gg == g])
            col = self._color(g, gi)
            # trails: one trace per group, None-separated segments
            tidx = idx[: self.max_trails]
            tx, ty, tz = self._trail_arrays(trails, tidx)
            data.append(go.Scatter3d(
                x=tx, y=ty, z=tz, mode="lines", hoverinfo="skip",
                line=dict(color=col, width=1.5), opacity=0.5,
                name=f"{g} trails", legendgroup=g, showlegend=False,
            ))
            h = heads[idx]
            data.append(go.Scatter3d(
                x=h[:, 0], y=h[:, 1], z=h[:, 2], mode="markers",
                marker=dict(size=self.marker_size, color=col),
                text=[names[i] for i in idx],
                customdata=[self.catalog.norad[i] for i in idx],
                hovertemplate="<b>%{text}</b><br>NORAD %{customdata}<br>"
                              "alt≈%{meta:.0f} km<extra>" + g + "</extra>",
                meta=0,  # replaced per-update below
                name=f"{g} ({len(idx)})", legendgroup=g,
            ))
            self._trace_index[g] = (len(data) - 2, len(data) - 1, idx, tidx)

        rmax = np.nanmax(np.linalg.norm(heads, axis=1)) if len(heads) else R_EARTH
        lim = float(max(1.15 * rmax, 1.6 * R_EARTH))
        fig = go.FigureWidget(data=data)
        fig.update_layout(
            height=self.height, margin=dict(l=0, r=0, t=30, b=0),
            paper_bgcolor="#05070d", font=dict(color="#dddddd"),
            title=self._title(t),
            legend=dict(bgcolor="rgba(0,0,0,0.3)", itemsizing="constant"),
            scene=dict(
                bgcolor="#05070d", aspectmode="data",
                xaxis=dict(visible=False, range=[-lim, lim]),
                yaxis=dict(visible=False, range=[-lim, lim]),
                zaxis=dict(visible=False, range=[-lim, lim]),
                camera=dict(eye=dict(x=1.1, y=1.1, z=0.6)),
                dragmode="orbit",
            ),
            uirevision="keep",   # keeps your camera position across updates
        )
        self._fix_altitude_hover(fig, heads)
        self.fig = fig
        return fig

    @staticmethod
    def _trail_arrays(trails, idx):
        if len(idx) == 0:
            return [], [], []
        seg = trails[idx]                                  # (m, k, 3)
        pad = np.full((seg.shape[0], 1, 3), np.nan)
        seg = np.concatenate([seg, pad], axis=1).reshape(-1, 3)
        return seg[:, 0], seg[:, 1], seg[:, 2]

    def _fix_altitude_hover(self, fig, heads):
        alt = np.linalg.norm(heads, axis=1) - R_EARTH
        for g, (_ti, hi, idx, _t) in self._trace_index.items():
            a = alt[idx]
            fig.data[hi].hovertemplate = ("<b>%{text}</b><br>NORAD %{customdata[0]}"
                                          "<br>alt %{customdata[1]:.0f} km<extra>" + g + "</extra>")
            fig.data[hi].customdata = np.column_stack(
                [[self.catalog.norad[i] for i in idx], a])

    def _title(self, t: datetime) -> str:
        frame = "ECEF (Earth-fixed)" if self.earth_fixed else "ECI (inertial)"
        fetched = self.catalog.fetched_at.strftime("%H:%M UTC") if self.catalog.fetched_at else "n/a"
        return (f"{len(self.catalog)} objects — {t.strftime('%Y-%m-%d %H:%M:%S')} UTC — "
                f"{frame} — TLEs fetched {fetched}")

    # ---- live update ----------------------------------------------------- #
    def update(self, t: datetime | None = None):
        """Recompute positions and push them into the existing figure."""
        if self.fig is None:
            return
        t = t or self.now()
        with self._lock:
            heads, trails = self._positions(t)
        alt = np.linalg.norm(heads, axis=1) - R_EARTH
        with self.fig.batch_update():
            for g, (ti, hi, idx, tidx) in self._trace_index.items():
                tx, ty, tz = self._trail_arrays(trails, tidx)
                self.fig.data[ti].x, self.fig.data[ti].y, self.fig.data[ti].z = tx, ty, tz
                h = heads[idx]
                self.fig.data[hi].x, self.fig.data[hi].y, self.fig.data[hi].z = h[:, 0], h[:, 1], h[:, 2]
                self.fig.data[hi].customdata = np.column_stack(
                    [[self.catalog.norad[i] for i in idx], alt[idx]])
            self.fig.layout.title = self._title(t)

    def _loop(self):
        last_fetch = time.time()
        while not self._stop.is_set():
            try:
                if time.time() - last_fetch > self.refetch_hours * 3600:
                    self.refresh_tles()
                    last_fetch = time.time()
                self.update()
                self.last_error = None
            except Exception as e:  # keep the loop alive on transient errors
                self.last_error = repr(e)
            self._stop.wait(self.update_seconds)

    def show(self) -> go.FigureWidget:
        """Build (if needed) and return the widget; Jupyter renders it."""
        if self.fig is None:
            self.build_figure()
        return self.fig

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        if self.fig is None:
            self.build_figure()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    # ---- conveniences ---------------------------------------------------- #
    def set_time_offset(self, **kwargs):
        """e.g. v.set_time_offset(hours=3) to see where everything is in 3 h."""
        self.time_offset = timedelta(**kwargs)
        self.update()

    def focus(self, name_substring: str):
        """Move the camera to look toward the first satellite whose name matches."""
        if self.fig is None:
            return
        for i, n in enumerate(self.catalog.names):
            if name_substring.lower() in n.lower():
                p = self.catalog.propagate([self.now()])[i, 0]
                if self.earth_fixed:
                    p = eci_to_ecef(p, self.now())
                p = p / np.linalg.norm(p)
                self.fig.layout.scene.camera.eye = dict(x=p[0] * 1.8, y=p[1] * 1.8, z=p[2] * 1.8)
                return n
        return None
