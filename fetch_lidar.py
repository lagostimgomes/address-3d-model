#!/usr/bin/env python3
"""
fetch_lidar.py - download USGS 3DEP LiDAR point-cloud tiles around a US street address.

Usage (Windows / macOS / Linux, Python 3.8+, no extra packages needed):

    python fetch_lidar.py "1600 Pennsylvania Ave NW, Washington, DC 20500"

Optional flags:
    --lat 38.89768 --lon -77.03655   skip geocoding and use these coordinates
    --half 125                       half-width of the box, in Web-Mercator units (~0.78 m each at 39 deg N)
    --out lidar_download             output folder name

Output: a folder plus a .zip of it, containing for each covering dataset:
    ept.json, the ept-hierarchy/*.json files that were read, ept-data/*.laz tiles,
and a manifest.json describing the address, coordinates and what was downloaded.
"""
import argparse, json, math, os, sys, time, urllib.parse, urllib.request, zipfile

INDEX_URL = "https://raw.githubusercontent.com/hobuinc/usgs-lidar/master/boundaries/resources.geojson"
GEOCODER = ("https://geocoding.geo.census.gov/geocoder/locations/onelineaddress"
            "?benchmark=Public_AR_Current&format=json&address=")
UA = {"User-Agent": "address-modeling-lidar-fetch/1.0"}


def get(url, retries=3):
    for i in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return r.read()
        except Exception as e:
            if i == retries - 1:
                raise
            print(f"  retrying ({e})")
            time.sleep(2 * (i + 1))


def get_json(url):
    return json.loads(get(url).decode("utf-8"))


def geocode(address):
    d = get_json(GEOCODER + urllib.parse.quote(address))
    m = d.get("result", {}).get("addressMatches", [])
    if not m:
        sys.exit("Geocoder found no match. Re-run with --lat and --lon (copy them from Google Maps: "
                 "right-click the house > the first line is lat, lon).")
    c = m[0]["coordinates"]
    return float(c["y"]), float(c["x"]), m[0].get("matchedAddress", address)


def to_mercator(lat, lon):
    r = 6378137.0
    return math.radians(lon) * r, r * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))


def in_ring(x, y, ring):
    inside, j = False, len(ring) - 1
    for i in range(len(ring)):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def in_geom(lon, lat, g):
    polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
    for p in polys:
        if in_ring(lon, lat, p[0]) and not any(in_ring(lon, lat, h) for h in p[1:]):
            return True
    return False


def node_box(key, bounds):
    d, x, y, _ = (int(v) for v in key.split("-"))
    sx = (bounds[3] - bounds[0]) / 2 ** d
    sy = (bounds[4] - bounds[1]) / 2 ** d
    x0, y0 = bounds[0] + x * sx, bounds[1] + y * sy
    return x0, y0, x0 + sx, y0 + sy


def overlaps(nb, box):
    return nb[0] < box[2] and nb[2] > box[0] and nb[1] < box[3] and nb[3] > box[1]


def fetch_dataset(ept_url, cx, cy, half, outdir):
    base = ept_url.rsplit("/", 1)[0]
    os.makedirs(os.path.join(outdir, "ept-hierarchy"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "ept-data"), exist_ok=True)
    raw = get(ept_url)
    open(os.path.join(outdir, "ept.json"), "wb").write(raw)
    ept = json.loads(raw)
    srs = ept.get("srs", {})
    if str(srs.get("horizontal", "3857")) != "3857":
        print(f"  WARNING: dataset SRS is {srs}; box assumes EPSG:3857.")
    bounds = ept["bounds"]
    box = (cx - half, cy - half, cx + half, cy + half)
    wanted, queue = [], ["0-0-0-0"]
    while queue:
        hk = queue.pop()
        hraw = get(f"{base}/ept-hierarchy/{hk}.json")
        open(os.path.join(outdir, "ept-hierarchy", f"{hk}.json"), "wb").write(hraw)
        for key, count in json.loads(hraw).items():
            if not overlaps(node_box(key, bounds), box):
                continue
            if count == -1:
                queue.append(key)
            elif count > 0:
                wanted.append((key, count))
    total_pts, total_bytes = 0, 0
    for n, (key, count) in enumerate(sorted(wanted), 1):
        data = get(f"{base}/ept-data/{key}.laz")
        open(os.path.join(outdir, "ept-data", f"{key}.laz"), "wb").write(data)
        total_pts += count
        total_bytes += len(data)
        print(f"  [{n}/{len(wanted)}] {key}.laz  {count:,} pts  {len(data)/1e6:.2f} MB")
    return {"ept_url": ept_url, "nodes": [k for k, _ in sorted(wanted)],
            "points": total_pts, "bytes": total_bytes}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("address", nargs="?", default="")
    ap.add_argument("--lat", type=float)
    ap.add_argument("--lon", type=float)
    ap.add_argument("--half", type=float, default=125.0)
    ap.add_argument("--out", default="lidar_download")
    ap.add_argument("--index", default=INDEX_URL, help=argparse.SUPPRESS)
    a = ap.parse_args()

    if a.lat is not None and a.lon is not None:
        lat, lon, matched = a.lat, a.lon, a.address or "(coordinates given)"
    elif a.address:
        print("Geocoding address ...")
        lat, lon, matched = geocode(a.address)
    else:
        sys.exit('Give an address in quotes, e.g.  python fetch_lidar.py "123 Main St, Town, ST 12345"')
    cx, cy = to_mercator(lat, lon)
    print(f"  {matched}\n  lat {lat:.7f}, lon {lon:.7f}  ->  Web Mercator ({cx:.2f}, {cy:.2f})")

    print("Finding USGS 3DEP datasets that cover this point ...")
    idx = get_json(a.index)
    hits = [f["properties"] for f in idx["features"] if in_geom(lon, lat, f["geometry"])]
    if not hits:
        sys.exit("No USGS 3DEP point-cloud dataset covers this location. Tell the requester.")
    for h in hits:
        print(f"  {h['name']}  ({h.get('count', 0):,} points total)")

    os.makedirs(a.out, exist_ok=True)
    manifest = {"address_input": a.address, "matched_address": matched, "lat": lat, "lon": lon,
                "mercator_center": [cx, cy], "half_width_mercator": a.half,
                "downloaded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "datasets": {}}
    for h in hits:
        print(f"Downloading tiles from {h['name']} ...")
        try:
            manifest["datasets"][h["name"]] = fetch_dataset(h["url"], cx, cy, a.half, os.path.join(a.out, h["name"]))
        except Exception as e:
            print(f"  FAILED: {e}")
            manifest["datasets"][h["name"]] = {"ept_url": h["url"], "error": str(e)}
    json.dump(manifest, open(os.path.join(a.out, "manifest.json"), "w"), indent=2)

    zpath = a.out.rstrip("/\\") + ".zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:
        for root, _, files in os.walk(a.out):
            for f in files:
                p = os.path.join(root, f)
                z.write(p, os.path.relpath(p, os.path.dirname(os.path.abspath(a.out)) or "."))
    ok = [d for d in manifest["datasets"].values() if "error" not in d]
    print(f"\nDone. {sum(d['points'] for d in ok):,} points, "
          f"{sum(d['bytes'] for d in ok)/1e6:.1f} MB of LAZ in {len(ok)} dataset(s).")
    print(f"Send this file back: {os.path.abspath(zpath)}")


if __name__ == "__main__":
    main()
