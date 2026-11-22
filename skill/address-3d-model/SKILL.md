---
name: address-3d-model
description: Build an interactive 3D model of a US street address (the house, one neighbor on each side, real terrain, street and trees) from USGS LiDAR, hands-off from just the address. Use when someone gives a US address and asks for a 3D model, 3D view or neighborhood model of it.
---

# Address to 3D neighborhood model

**Input:** one US street address.
**Output:** a page you can orbit, showing the house, the nearest neighbor on each side, the real terrain, the street and the trees, followed by a short summary.

Run the whole thing without asking questions. Ask only if there is no address, or if it is clearly outside the US. All geometry comes from public data: USGS 3DEP LiDAR and terrain, the U.S. DOT National Address Database, OpenStreetMap, and the Census geocoder. Never look up or show who lives at or owns the address.

## 1. Get the pipeline (code: https://github.com/lagostimgomes/address-3d-model)

```bash
A="$HOME/.address-3d-model"
if [ -d "$A/.git" ]; then git -C "$A" pull -q; else git clone -q --depth 1 https://github.com/lagostimgomes/address-3d-model.git "$A"; fi
PYTHON="$(command -v python3 || command -v python)"
"$PYTHON" -m venv "$A/.venv" 2>/dev/null
PY="$A/.venv/bin/python"; [ -x "$PY" ] || PY="$A/.venv/Scripts/python.exe"; [ -x "$PY" ] || PY="$PYTHON"
"$PY" -m pip install -q -r "$A/requirements.txt" || "$PY" -m pip install -q --break-system-packages -r "$A/requirements.txt"
```

## 2. Run it

Run in the working directory, with a long timeout (up to 10 minutes; the LiDAR download is 5–40 MB):

```bash
"$PY" "$A/run.py" "<address exactly as given>" --work sites 2>&1 | tail -60
```

The last line is `RESULT: {...}` (JSON). The exit code tells you what happened:

| Exit | Meaning | What to do |
| --- | --- | --- |
| 0 | Done. `RESULT.html` is the page. | Go to step 3. |
| 3 | Network blocked. `RESULT.blocked_hosts` lists the hosts. | If a `device_bash` tool for the user's computer is available, rerun steps 1–2 there, in a connected folder. Otherwise stop. Tell the user, kindly and briefly: on Pro and Max plans, allow these hosts under network access in Settings → Capabilities; on Team or Enterprise, ask an org owner to add them to the code execution network allowlist; the Free plan can't reach them. List the blocked hosts and link https://lagostimgomes.github.io/address-3d-model/#network. |
| 4 | Fetch failed. | Read the last 30 lines of `sites/<slug>/fetch.log`. If geocoding failed, retry once with a cleaned-up address (full street name, city, state, ZIP). If no LiDAR dataset covers the point, tell the user that the area has no public USGS point cloud yet. |
| 5 | Analysis failed, for example no buildings found. | Report it plainly. It usually means very sparse or old LiDAR, or the geocode landed off the parcel. |
| 6 | Build failed. | Report the error. |

## 3. Check the result

- `RESULT.houses` should hold the featured house plus up to 2 neighbors. Fewer neighbors is fine at corner lots or rural sites; mention it.
- Large or complex buildings (campus, apartment, commercial) come back with roof type `massing`: their measured outline and heights in 3 ft columns. Describe them that way, with no pitch, and don't apply the house limits below.
- Sanity limits for a house:
  - wall height (eave): 8–32 ft
  - ridge height: 12–45 ft
  - footprint sides: 16–120 ft
  - roof pitch: 0–50°

  If the featured house breaks them, say the LiDAR fit is unreliable rather than presenting it as measured.

## 4. Deliver

- **If an artifact or page-publishing tool is available** (claude.ai): publish `RESULT.html` as a new artifact from the file.
  - title: the street address, for example "1600 Pennsylvania Ave NW"
  - icon: `home`

  Artifact pages allow three.js loaded from cdnjs and jsdelivr.
- **Otherwise** (for example Claude Code): give the user the path to `RESULT.html` and open it in the default browser: `open` on macOS, `start ""` on Windows, `xdg-open` on Linux. The page needs internet to load three.js.

## 5. Reply

Write three or four sentences, without pasting the HTML:
- the featured building's measured ridge height, roof type and pitch, and footprint (for `massing`: tallest point, typical roof height, roof area)
- which neighbors were found
- the LiDAR survey years (`RESULT.datasets`)
- if `RESULT.location_precision` is `street`, say the address wasn't in the National Address Database, so the highlighted house is the nearest one to a street-level point and may be a neighbor; offer to move it
- caveats: windows, doors, garages and colors are generated, not observed; the street side was inferred if `RESULT.street_source` is `inferred`; complex roofs are simplified to a main roof plus cross gables

## Notes

- Model frame: feet, +x east, +z south, origin at the geocode, heights relative to the terrain at the origin.
- If the user says the wrong house is highlighted, rerun with `--lat/--lon` of the right house (taken from scene.json's neighbor `x`/`z`, or from the user) in a new `--work` folder.
- To refine a model on request (colors, house numbers, a porch), edit `sites/<slug>/scene.json` and rerun `"$PY" "$A/pipeline/build/build_model.py" sites/<slug>`. Don't refetch.
- Sources to credit if asked: USGS 3DEP (public domain), U.S. DOT National Address Database (public domain), © OpenStreetMap contributors (ODbL), the U.S. Census Bureau geocoder.
