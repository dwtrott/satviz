# satviz

Live 3D satellite viewer: CelesTrak TLEs → SGP4 → textured Earth in a standalone
PyVista/VTK window with a Qt control panel. Launch it from a notebook or a terminal.

```
pip install sgp4 numpy requests matplotlib pyvista pyvistaqt PyQt5
```

## Standalone window (satviz_gui.py)

```python
from satviz_gui import launch
app = launch(groups=["stations", "gps-ops", "starlink"])   # window opens; kernel stays free
```
or `python satviz_gui.py --groups stations gps-ops starlink oneweb`.

- Mouse: left-drag orbit, scroll zoom, middle/shift-drag pan, click a satellite to label it
- Panel: pause/play, speed ×1–×3600, time scrub ±24 h, trail length, per-group toggles,
  ECI / Earth-fixed frame, find by name or NORAD id → focus, refresh TLEs
- Earth rotates at sidereal rate, lit by the Sun at the current time (day/night terminator)
- First run downloads NASA Blue Marble (5400×2700) to `~/.satviz/`; falls back to PyVista's
  bundled 2k globe if offline. Use `launch(..., texture_path="my_earth.jpg")` for your own
  equirectangular image (e.g. an 8k/16k Blue Marble or a night-lights composite).
- Positions re-propagate every second; TLEs re-fetched every 2 h (CelesTrak's guidance).

Programmatic: `app.scene.focus(i)`, `app.scene.speed`, `app.scene.set_frame(True)`,
`app.scene.set_trail_minutes(90)`, `app.catalog.propagate(times)` → ECI km array `(n, t, 3)`.

## Inline Plotly version (satviz.py)
`SatelliteViewer(...).show()` renders inside the notebook instead; same data layer.

CelesTrak GROUP names: stations, gps-ops, galileo, glo-ops, beidou, starlink, oneweb,
iridium-NEXT, weather, noaa, geo, active (everything), cosmos-2251-debris, …
