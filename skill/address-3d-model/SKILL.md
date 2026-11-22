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

## 3. Name and enrich (automatic, still no questions)

Outside material makes notable buildings look right: names, floor counts, wall and roof materials, and style. Use this general procedure for every address. Never tailor it to one place.

1. Run `"$PY" "$A/pipeline/enrich/identify.py" sites/<slug>`. It lists institutions or sites around the building, each with a name, website and Wikidata ID.
2. **Privacy:** research a building only if it's non-residential or a public landmark. That means a named institution appears in step 1 within about 1,000 ft, or the featured building's roof type is `grid`. For ordinary homes, skip the web research; OpenStreetMap tags from step 1 are fine. Never look up owners or occupants.
3. **Candidate names:** search the web for the address, then for "<institution> campus buildings" or "<institution> campus map". Collect the building names you find.
4. **Verify:** run `identify.py sites/<slug> --names "Name 1" "Name 2" ...`. A name is accepted only if OpenStreetMap places it on one of the measured buildings. Use only the accepted names.
5. **Facts:** for each accepted building, and for the institution as a whole, search for its architecture. Prefer authoritative sources:
   - National Register nominations (npgallery.nps.gov)
   - state historic inventories
   - the institution's facilities pages
   - the architect's project page
   - Wikipedia

   Record only what a source states: wall material or color, roof material, style, number of floors, year, and a one-line summary. Mark each building's `basis` as `building` for facts about that building, or `site` for campus or district facts applied to it.
6. Write `sites/<slug>/facts.json` (the format is in the `build_model.py` docstring), with at least one source URL per building. Then run `"$PY" "$A/pipeline/build/build_model.py" sites/<slug>`.

Leave out anything you can't source. If nothing is found, skip this step: the model is complete without it.

## 4. Check the result

- The model shows the building at the address plus every building, tree and paved area within 100 ft of its footprint. Touching buildings are shown whole, up to 40,000 sq ft of roof.
- Buildings other than simple houses come back with roof type `grid`: one continuous roof rebuilt from their measured planes, with walls at the outline and at height steps. Describe them by tallest point, typical eave height, roof area and typical pitch, and don't apply the house limits below.
- Sanity limits for a house:
  - wall height (eave): 8–32 ft
  - ridge height: 12–45 ft
  - footprint sides: 16–120 ft
  - roof pitch: 0–50°

  If the featured house breaks them, say the LiDAR fit is unreliable rather than presenting it as measured.

## 5. Deliver

- **If an artifact or page-publishing tool is available** (claude.ai): publish `RESULT.html` as a new artifact from the file.
  - title: the street address, for example "1600 Pennsylvania Ave NW"
  - icon: `home`

  Artifact pages allow three.js loaded from cdnjs and jsdelivr.
- **Otherwise** (for example Claude Code): give the user the path to `RESULT.html` and open it in the default browser: `open` on macOS, `start ""` on Windows, `xdg-open` on Linux. The page needs internet to load three.js.

## 6. Reply

Write three or four sentences, without pasting the HTML:
- the featured building's measured ridge height, roof type and pitch, and footprint (for `grid`: tallest point, typical eave height, roof area, typical pitch)
- the building's name and the sourced facts, if step 3 found any, saying whether they're about the building or the campus/site
- how many buildings and trees the 100 ft site includes
- the LiDAR survey years (`RESULT.datasets`)
- if `RESULT.location_precision` is `street`, say the address wasn't in the National Address Database, so the highlighted house is the nearest one to a street-level point and may be a neighbor; offer to move it
- caveats: windows, doors, garages and colors are generated, not observed; streets and parking are drawn from LiDAR-measured pavement (plus OpenStreetMap centerlines when reachable); complex roofs are simplified to a main roof plus cross gables

## Notes

- Model frame: feet, +x east, +z south, origin at the geocode, heights relative to the terrain at the origin.
- If the user says the wrong house is highlighted, rerun with `--lat/--lon` of the right house (taken from scene.json's neighbor `x`/`z`, or from the user) in a new `--work` folder.
- To refine a model on request (colors, house numbers, a porch), edit `sites/<slug>/scene.json` and rerun `"$PY" "$A/pipeline/build/build_model.py" sites/<slug>`. Don't refetch.
- Sources to credit if asked: USGS 3DEP (public domain), U.S. DOT National Address Database (public domain), © OpenStreetMap contributors (ODbL), the U.S. Census Bureau geocoder.
