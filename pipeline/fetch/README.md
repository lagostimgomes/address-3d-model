# Data acquisition v2

The lead owns `run.py` and the builder. This layer produces mapped evidence and
matching DEM/LiDAR extents without modifying the original `fetch_lidar.py`.

```powershell
python pipeline/fetch/fetch_site.py "<US address>" --lat <NAD-lat> --lon <NAD-lon> --out-root sites
python pipeline/fetch/fetch_site.py "<same address>" --bounds-ft "-400,600,-500,350" --expand-into sites/<slug>
```

The runner resolves the structure point using `pipeline/locate` first. Direct
address-only invocation of this fetcher still uses Census/Nominatim street
geocoding; it is not a substitute for the NAD locator. Explicit coordinates fix
the model origin for every expansion. An existing output directory is refused
except for the explicit `--expand-into` operation.

## Sequence and CLI

1. Query OpenStreetMap interpreters for footprints, multipolygon areas, paths,
   road tags, and entrance nodes. There is no `/api/status` gate. Parse complete
   rings only, retaining holes. Partial Overpass responses are rejected.
2. Pick the mapped polygon containing the origin (otherwise the nearest within
   120 ft). Select every footprint within 100 ft of that original polygon. Do not
   recursively include neighbors of neighbors. Include their full bounds plus
   30 ft margin. Context buildings over 40,000 sq ft may be section cuts; record
   that separately from missing data. Bound each raster side to 1,500 ft.
3. Export a DEM and download EPT nodes intersecting the same rectangular bounds.
   The EPT helper checks dataset polygons against the entire requested rectangle.
   Extra points in intersecting tiles are expected; the builder crops them.
4. Publish `site.json.status=complete` only if both required acquisitions pass.
   Coverage is a separate mapped preselection result, not a claim that LiDAR
   has verified every building or cell. Sparse surveys still need builder checks.

`--bounds-ft` accepts four space-separated numbers or one comma-separated value,
ordered xmin,xmax,zmin,zmax. It is a minimum request combined with footprint needs.
`--expansions 0|1` records the pass. `--no-features` preserves the original 120x120
default terrain mode and root-downloader path (unless explicit bounds are supplied).
`--half` controls only that legacy LiDAR path, in Web Mercator units.
`--osm-raw-json FILE` replays a captured Overpass response and is explicitly labeled
`cached-input` in provenance. It does not pretend to be a fresh map acquisition.

## Expansion safety

`--expand-into` uses a sibling staging folder, verifies the same address/origin and
at most one expansion, and checks that the new extent contains the old extent.
Cached EPT tiles with recorded byte sizes are reused; no original input is edited
during network requests. On success, existing generated fetch inputs are moved to
`fetch-history/<id>/`, then staged files are promoted, `site.json` last. During
promotion the site status is `expanding`, so an interrupted update cannot be read
as a complete site. A local
promotion error rolls back the inputs. Interrupted promotions retain the staging
folder, history and lock for recovery; multi-file replacement cannot be atomic
across abrupt power loss. `facts.json`, `locate.json`, `scene.json` and HTML are
not replaced by this command; the runner re-analyzes and rebuilds after success.
An explicit lock prevents two expansions. Failed refetches preserve the old site
and write `expansion-failure.json`. No old dataset or model is deleted.

The runner should attempt the expansion once if its measured outlines touch the
data boundary. It must retain `incomplete` if coverage is still missing. Do not
gate map acquisition on a status probe. A missing map can be useful with LiDAR,
but it cannot establish mapped coverage completeness.

## Raster frame (additive dem.json fields)

`bounds_ft=[xmin,xmax,zmin,zmax]` describes outer CELL EDGES relative to
`site.json.mercator_center`. It can be asymmetric. Grid samples are cell centers:

```
x = xmin + (col + 0.5) * cell_size_ft
z = zmin + (row + 0.5) * cell_size_ft
elevation = elevations_ft[row * cols + col]
```

Rows go north to south; columns west to east; +x east and +z south. `rows`, `cols`
and `cell_size_ft` are authoritative. Values are absolute NAVD88 feet, with no PAD
subtraction or quantization. Legacy grids without bounds are centered at the origin.
Mercator bboxes use [west,south,east,north], which is a different ordering from
model bounds. One ground foot is `.3048/cos(latitude)` Mercator units.

Before using Maryland services, the fetcher resolves the point's Census state and
county. DC is state 11, Maryland state 24, Frederick county 021. If jurisdiction
lookup fails, only the national source is attempted. It does not use a Maryland
bounding box as proof of membership. State rasters crossing the boundary can still
return NoData; the national fallback remains available.

Transient national export errors get four attempts, with 5/15/30 second waits.
If a full export fails, four tiles at exact pixel edges are fetched and stitched
without overlap, flip or resampling. Odd row/column counts are supported. Raw TIFFs,
metadata, URLs, tile offsets and request parameters are retained. Missing/NaN values,
wrong image sizes, rendered rasters and unsupported vertical datums fail explicitly.

## features.json schema 1

`buildings`: stable `id`, `polygon`, `holes`, `tags`; `building_parts` is an additive
array that is not automatically selected as another whole building.

`surfaces`: polygons/holes and `kind` (`parking`, `sidewalk`, `driveway`, `plaza`,
`road_area`, `lawn`, `water`, `other_paved`, `other`). Landuse=residential, school
or park does not mean the entire polygon is lawn. Only explicit grass mapping
produces lawn. Unmapped areas remain unknown.

`ways`: `kind` (highway value), `centerline`, `width_ft`, `width_source`, `lanes`,
`sidewalk`, `layer`, `bridge`, `tunnel`, `name`, `tags`. Explicit widths accept metres
and feet; ambiguous values stay null. No lane-derived width is claimed as measured.

`entrances`: `x`, `z`, `building_id` or null, `tags`. A mapped node is associated with
the smallest containing/nearby footprint within 6 ft. No doorway is invented here.

Every record retains `source`, `source_id`, `source_date`, `edited_utc`, `confidence`
and `confidence_components`. `source_date` means imagery/acquisition date, which
OSM usually does not supply, so it stays null. `edited_utc` is the map edit date,
not the acquisition date. A default evidence score of 0.65 means mapped geometry
with unknown acquisition age and no point-cloud agreement measurement; it is not
a calibrated probability. Components separately state geometry, temporal knowledge,
completeness and method. The builder should add LiDAR agreement/temporal compatibility
and preserve conflicts rather than blindly prefer either an older map or newer scan.

`coverage` includes actual `lidar_bounds_ft`, `dem_bounds_ft`, requested bounds,
`status`, `expansions`, `edge_touching`, `selected_building_ids`, `target_building_id`,
`section_cut_ids`, and notes. Status is `complete` only for supported mapped
preselection with no unresolved edge; `expanded` is the same after expansion;
`incomplete` wins over either when any uncertainty about target coverage remains.
Post-analysis scene coverage is the lead's final authority.

Legacy roads.json and osm_buildings.json are still written. Map failure writes
explicit unavailable evidence, never fabricated grass or roads. A failed refresh
retains existing mapped evidence as `status=stale` with the error; coverage remains
incomplete until validated. OSM relations are assembled from unordered/reversed
member segments, with inner rings retained and member duplicates suppressed.

## Current limits and sources

The implemented map adapter is OSM. Overture and Microsoft native metadata/index
access and NAIP's four-band metadata were independently verified; they are not
yet alternative footprint adapters or an aerial-classification implementation.
Public source availability does not guarantee local coverage or geometry quality.
Raw intensity alone is not a reliable pavement classifier. RGB cannot provide
NDVI; raw NIR and acquisition metadata must be checked first. There are no paid
imagery credentials or owner/occupant lookups in this code.

- [Census geographies API](https://geocoding.geo.census.gov/geocoder/Geocoding_Services_API.html/)
- [USGS NAIP raw imagery service](https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPImagery/ImageServer)
- [Overture official data access](https://docs.overturemaps.org/getting-data/cloud-sources/)
- [Microsoft footprint release](https://github.com/microsoft/GlobalMLBuildingFootprints)
- OpenStreetMap data: ODbL, attribution to OpenStreetMap contributors retained.

Tests: `python -m unittest discover -s pipeline/fetch -p 'test_*.py' -v` and
`python -m unittest discover -s pipeline/osm -p 'test_*.py' -v`.
