"""Ground cover from an aerial orthomosaic (generic; no per-site tuning).

Input: a georeferenced RGB mosaic in the model frame (world file: feet, +x east, +z south) and the site's building
footprints. Output: a class raster on a 1 ft grid - asphalt, concrete, lawn, bed (mulch/planting), canopy (trees hide the
ground), roof/unknown - as run-length rows, plus per-class shares. Pixels never leave this step; only classes do.
Rules (HSV, fixed): grey + dark = asphalt; grey + light = concrete; green + bright = lawn; green + dark = canopy;
warm hue + saturated + not bright = bed. Building footprints (+2 ft) are excluded. A 3x3 majority filter and a minimum
patch size clean the result."""
import json, math, sys
import numpy as np
from PIL import Image
from scipy import ndimage

CLASSES = ['unknown', 'asphalt', 'concrete', 'lawn', 'canopy', 'bed']


def read_world(pgw):
    a, d, b, e, c, f = [float(v) for v in open(pgw).read().split()]
    return a, e, c, f                      # px size x, px size z, x of first pixel centre, z of first pixel centre


def classify(img_path, pgw_path, bounds, footprints, cell=1.0):
    im = np.asarray(Image.open(img_path).convert('RGB')).astype(np.float32) / 255.0
    sx, sz, x0, z0 = read_world(pgw_path)
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    mx, mn = im.max(-1), im.min(-1); V = mx; S = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
    hue = np.zeros_like(V); d = np.maximum(mx - mn, 1e-6)
    hue = np.where(mx == r, (60 * ((g - b) / d)) % 360, hue)
    hue = np.where(mx == g, 60 * ((b - r) / d) + 120, hue)
    hue = np.where(mx == b, 60 * ((r - g) / d) + 240, hue)
    lab = np.zeros(V.shape, np.uint8)
    grey = S < 0.16
    lab[grey & (V < 0.56) & (V > 0.05)] = 1                 # asphalt (incl. shaded)
    lab[grey & (V >= 0.58)] = 2                             # concrete
    green = (hue > 50) & (hue < 200) & (S >= 0.16)
    lab[green & (V >= 0.30)] = 3                            # lawn
    lab[green & (V < 0.30)] = 4                             # canopy / shade
    warm = ((hue < 40) | (hue > 330)) & (S >= 0.28) & (V < 0.62) & (V > 0.10)
    lab[warm] = 5                                           # bed: mulch, soil, flowers
    # resample to the model grid by majority
    xmin, xmax, zmin, zmax = bounds
    C = int(math.ceil((xmax - xmin) / cell)); R = int(math.ceil((zmax - zmin) / cell))
    out = np.zeros((R, C), np.uint8)
    jj = ((np.arange(C) + 0.5) * cell + xmin - x0) / sx
    ii = ((np.arange(R) + 0.5) * cell + zmin - z0) / sz
    k = max(1, int(round(cell / sx)))
    for cls in range(1, len(CLASSES)):
        m = ndimage.uniform_filter((lab == cls).astype(np.float32), size=k)
        I = np.clip(np.round(ii).astype(int), 0, lab.shape[0] - 1); J = np.clip(np.round(jj).astype(int), 0, lab.shape[1] - 1)
        inside = ((ii >= 0) & (ii < lab.shape[0]))[:, None] & ((jj >= 0) & (jj < lab.shape[1]))[None, :]
        share = np.where(inside, m[I][:, J], 0)
        if cls == 1: best = share.copy(); out[:] = np.where(share > 0.35, 1, 0)
        else:
            win = share > np.maximum(best, 0.35); out[win] = cls; best = np.maximum(best, share)
    # buildings are not ground
    import shapely
    from shapely.geometry import Polygon
    gx, gz = np.meshgrid((np.arange(C) + 0.5) * cell + xmin, (np.arange(R) + 0.5) * cell + zmin)
    for fp in footprints:
        P = Polygon(fp).buffer(2.0)
        out[shapely.contains_xy(P, gx, gz)] = 0
    # clean: 3x3 majority, then patches smaller than 12 ft2 go to their surroundings
    from scipy.ndimage import generic_filter
    def maj(v):
        c = np.bincount(v.astype(int), minlength=len(CLASSES)); return int(np.argmax(c))
    out = generic_filter(out, maj, size=5, mode='nearest').astype(np.uint8)
    # concrete on the ground is thin (walks, sidewalks, steps, pads): wide light-grey blobs are roofs - drop what
    # survives an opening with a 9 ft square, except where that blob touches asphalt along a long edge (a pad)
    thick = ndimage.binary_opening(out == 2, structure=np.ones((int(9 / cell), int(9 / cell))))
    out[thick] = 0
    # asphalt on the ground is a network: streets, and driveways joined to them. Compact grey blobs that do not touch
    # the network (dark roofs, the tops of parked trucks) are dropped.
    A = out == 1
    L, n = ndimage.label(A, structure=np.ones((3, 3)))
    if n:
        sizes = ndimage.sum(A, L, range(1, n + 1)) * cell * cell
        net = np.isin(L, np.nonzero(sizes >= 5000)[0] + 1)
        near = ndimage.binary_dilation(net, iterations=int(round(3 / cell)))
        for k in range(1, n + 1):
            if sizes[k - 1] >= 5000: continue
            comp = L == k
            if not (comp & near).any(): out[comp] = 0
    for cls in (1, 2, 5):
        L, n = ndimage.label(out == cls)
        if n:
            sizes = ndimage.sum(np.ones_like(L), L, range(1, n + 1))
            small = np.isin(L, np.nonzero(sizes * cell * cell < (12 if cls == 5 else 40))[0] + 1)
            out[small] = 3 if cls == 5 else 0
    return out, (xmin, zmin, cell)


def rle(row):
    runs, j, n = [], 0, len(row)
    while j < n:
        if row[j]:
            k = j
            while k < n and row[k]: k += 1
            runs += [j, k - j]; j = k
        else: j += 1
    return runs


def main(site, mosaic, pgw, out_path):
    S = json.load(open(f'{site}/scene.json'))
    b = S['site']['bounds']
    fps = [h['outline_poly'] if h.get('outline_poly') else h.get('footprint') for h in S['houses']]
    fps = [f for f in fps if f]
    lab, (x0, z0, cell) = classify(mosaic, pgw, b, fps)
    res = dict(kind='ground-cover', source='aerial orthomosaic (Google Map Tiles, satellite)', status='observed',
               method=__doc__.strip().split('\n')[0], cell=cell, origin=[x0, z0], rows=lab.shape[0], cols=lab.shape[1],
               classes={c: [rle(r_ == k) for r_ in lab] for k, c in enumerate(CLASSES) if k in (1, 2, 3, 5)},
               share={c: round(float((lab == k).mean()), 3) for k, c in enumerate(CLASSES)})
    json.dump(res, open(out_path, 'w'))
    return lab, res


if __name__ == '__main__':
    lab, res = main(*sys.argv[1:5])
    print(res['share'])
