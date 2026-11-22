# address-3d-model

**No coding? Start here: https://lagostimgomes.github.io/address-3d-model/**
It walks you through 4 steps: download the skill, add it to Claude, allow the map sites, and ask for a model.

Address in, 3D neighborhood model out. For a US street address this builds a single-file
Three.js page of the house, the nearest neighbor on each side, the real terrain, the street,
and the surrounding trees. All geometry is measured from public USGS 3DEP LiDAR.

```bash
pip install -r requirements.txt
python run.py "1600 Pennsylvania Ave NW, Washington, DC 20500"
# -> sites/<slug>/index.html, plus RESULT: {...} on the last line
```

## What is measured vs generated

| Measured (LiDAR, DEM, OSM) | Generated |
| --- | --- |
| Terrain (USGS 3DEP or state DEM, 120x120 cells of 5.29 ft) | Window, door and garage layout |
| House position, rotation, footprint | Siding colours |
| Wall (eave) height, ridge height, roof type and pitch, cross gables | Shrubs, deck, driveway shape |
| Tree positions and heights | |
| Streets (OpenStreetMap) and house numbers where OSM has them | |

## Use it as a Claude skill

- **claude.ai or the Claude app:** upload `docs/address-3d-model.zip` in Customize → Skills, and allow the
  network hosts listed below. Then ask: *"Make a 3D model of <address>"*.
- **Claude Code:** `cp -r skill/address-3d-model ~/.claude/skills/`. It runs on your own machine,
  so no network setting is needed.

The skill (`skill/address-3d-model/SKILL.md`) clones this repo, runs `run.py`, checks the result
and publishes the page.

## Pipeline

1. `pipeline/fetch/fetch_site.py`: geocode (Census, then Nominatim), terrain grid, and LiDAR
   tiles via `fetch_lidar.py` (USGS 3DEP Entwine Point Tiles on public S3).
2. `pipeline/osm/fetch_osm.py`: streets and building footprints from the Overpass API
   (optional; on failure the builder guesses the street side).
3. `pipeline/build/analyze_site.py`: finds roofs by clustering single-return points, fits roof
   planes with RANSAC, and picks the featured house and one neighbor on each side along the
   street. Writes `scene.json`.
4. `pipeline/build/build_model.py`: writes `index.html` from `template.html` and the scene.

## Network

Needs HTTPS access to: `geocoding.geo.census.gov`, `nominatim.openstreetmap.org`,
`raw.githubusercontent.com`, `s3-us-west-2.amazonaws.com`, `elevation.nationalmap.gov`,
`overpass-api.de` (or `overpass.kumi.systems`, `overpass.private.coffee`) and, in Maryland,
`mdgeodata.md.gov`. `run.py` checks them first and exits with code 3, listing the blocked hosts.

## Limits

- 3DEP coverage is nationwide but uneven in age and density. Very old or sparse surveys give
  rough roofs.
- Complex roofs are reduced to a main gable, hip or flat roof plus cross gables. Dormers,
  porches and turrets are not modeled.
- Attached houses (townhouses) are treated as one building.

Data: USGS 3DEP (public domain), OpenStreetMap contributors (ODbL), U.S. Census geocoder.
Built by Claude and Codex.

## License

Copyright 2026 lagostimgomes. Licensed under the [Apache License, Version 2.0](LICENSE).
The data the pipeline downloads keeps its own terms: USGS 3DEP is public domain, and OpenStreetMap is ODbL.
