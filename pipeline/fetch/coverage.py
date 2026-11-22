"""Deterministic footprint selection and bounded raster coverage, in model feet.

No network or geometry dependencies. Polygons have a closed outer ring and optional
holes. Selection is from the ORIGINAL target buffer, never neighbors of neighbors.
"""
import math

CELL_FT = 5.2868
MAX_SIDE_FT = 1500.0


def point_segment(p, a, b):
    dx, dz = b[0] - a[0], b[1] - a[1]
    t = max(0, min(1, ((p[0]-a[0])*dx + (p[1]-a[1])*dz) / (dx*dx+dz*dz))) if dx or dz else 0
    return math.hypot(p[0]-a[0]-t*dx, p[1]-a[1]-t*dz)


def edges(ring):
    return zip(ring, ring[1:] + ring[:1])


def in_ring(p, ring):
    inside = False
    for a, b in edges(ring):
        if point_segment(p, a, b) < 1e-7:
            return True
        if (a[1] > p[1]) != (b[1] > p[1]):
            if p[0] < (b[0]-a[0])*(p[1]-a[1])/(b[1]-a[1])+a[0]:
                inside = not inside
    return inside


def contains(p, feature):
    return in_ring(p, feature['polygon']) and not any(in_ring(p, r) for r in feature.get('holes', []))


def cross(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])


def segment_distance(a, b, c, d):
    # Strict crossings plus endpoint tests also cover collinear/zero-length edges.
    if cross(a,b,c)*cross(a,b,d) < 0 and cross(c,d,a)*cross(c,d,b) < 0:
        return 0.0
    return min(point_segment(a,c,d), point_segment(b,c,d), point_segment(c,a,b), point_segment(d,a,b))


def distance(p, feature):
    if contains(p, feature):
        return 0.0
    return min(point_segment(p,a,b) for r in [feature['polygon'], *feature.get('holes', [])] for a,b in edges(r))


def polygon_distance(a, b):
    if any(contains(p,b) for p in a['polygon']) or any(contains(p,a) for p in b['polygon']):
        return 0.0
    return min(segment_distance(p,q,r,s)
               for ar in [a['polygon'], *a.get('holes', [])]
               for br in [b['polygon'], *b.get('holes', [])]
               for p,q in edges(ar) for r,s in edges(br))


def bounds(points):
    return [min(p[0] for p in points), max(p[0] for p in points),
            min(p[1] for p in points), max(p[1] for p in points)]


def area(feature):
    def ring_area(r):
        return abs(sum(a[0]*b[1]-b[0]*a[1] for a,b in edges(r))) / 2
    return ring_area(feature['polygon']) - sum(ring_area(r) for r in feature.get('holes', []))


def aligned_bounds(requested, cell=CELL_FT, max_side=MAX_SIDE_FT):
    if len(requested) != 4 or not all(math.isfinite(v) for v in requested):
        raise ValueError('Bounds must be four finite cell edges: xmin xmax zmin zmax')
    if cell <= 0 or not math.isfinite(cell) or not math.isfinite(max_side) or max_side < cell:
        raise ValueError('Cell size and coverage budget must be positive and finite')
    out = []
    max_cells = int(max_side // cell)
    for lo, hi in (requested[:2], requested[2:]):
        if lo >= hi:
            raise ValueError('Bounds minimum must be less than maximum')
        a, b = math.floor(lo/cell+1e-10), math.ceil(hi/cell-1e-10)
        if b-a > max_cells:
            a = math.floor((lo+hi)/(2*cell)-max_cells/2)
            b = a+max_cells
        out.extend([a*cell, b*cell])
    return out


def grid_spec(requested=None, cell=CELL_FT):
    b = list(requested) if requested is not None else [-60*cell,60*cell,-60*cell,60*cell]
    # Default is the original 120 x 120 grid, with identical sample centers.
    if requested is not None:
        b = aligned_bounds(b, cell)
    cols, rows = round((b[1]-b[0])/cell), round((b[3]-b[2])/cell)
    return dict(bounds_ft=b, rows=rows, cols=cols, cell_size_ft=cell)


def touches(feature, b, tolerance=3.0):
    f = bounds(feature['polygon'])
    return f[0] <= b[0]+tolerance or f[1] >= b[1]-tolerance or f[2] <= b[2]+tolerance or f[3] >= b[3]-tolerance


def plan(features, requested=None, expansions=0, buffer_ft=100, margin_ft=30):
    buildings = features.get('buildings', [])
    notes, intentional = [], []
    target = min(buildings, key=lambda f: distance([0,0],f)) if buildings else None
    if target and distance([0,0],target) > 120:
        notes.append('No mapped footprint within 120 ft of the address; target unresolved.')
        target = None
    if target:
        tb = bounds(target['polygon'])
        zone_bbox = [tb[0]-buffer_ft,tb[1]+buffer_ft,tb[2]-buffer_ft,tb[3]+buffer_ft]
        selected = [f for f in buildings if polygon_distance(target,f) <= buffer_ft+1e-7]
        need = list(zone_bbox)
        for f in selected:
            if f is not target and area(f) > 40000:
                intentional.append(f['id'])
                continue
            b = bounds(f['polygon'])
            need = [min(need[0],b[0]),max(need[1],b[1]),min(need[2],b[2]),max(need[3],b[3])]
        need = [need[0]-margin_ft,need[1]+margin_ft,need[2]-margin_ft,need[3]+margin_ft]
    else:
        selected = []
        need = grid_spec()['bounds_ft']
        zone_bbox = None
        notes.append('No supported mapped target; coverage requires post-analysis verification.')
    if requested is not None:
        need = [min(need[0],requested[0]),max(need[1],requested[1]),min(need[2],requested[2]),max(need[3],requested[3])]
    spec = grid_spec(need)
    b = spec['bounds_ft']
    # Oversized targets can be very asymmetric. Keep the address point inside
    # the budgeted data, rather than clipping around a distant footprint center.
    if target:
        for k in (0,2):
            width=b[k+1]-b[k]
            guard=math.ceil(margin_ft/CELL_FT)*CELL_FT
            if b[k]>-guard:
                b[k],b[k+1]=-guard,width-guard
            elif b[k+1]<guard:
                b[k],b[k+1]=guard-width,guard
    edge = [f['id'] for f in selected if f['id'] not in intentional and touches(f,b)]
    zone_outside = bool(zone_bbox and (zone_bbox[0]<b[0] or zone_bbox[1]>b[1] or zone_bbox[2]<b[2] or zone_bbox[3]>b[3]))
    if zone_outside:
        notes.append('The target buffer exceeds the 1500 ft data budget.')
    unresolved=[]
    if zone_bbox:
        for item in features.get('unresolved_geometries',[]):
            u=item['bounds_ft']
            if u[0]<=zone_bbox[1] and u[1]>=zone_bbox[0] and u[2]<=zone_bbox[3] and u[3]>=zone_bbox[2]:
                unresolved.append(item['id'])
        if unresolved: notes.append('Unresolved mapped geometry intersects the target buffer: '+', '.join(unresolved))
    incomplete = bool(edge or zone_outside or unresolved or not target or features.get('status') != 'ok')
    coverage = dict(lidar_bounds_ft=None,dem_bounds_ft=None,requested_bounds_ft=b,
                    status='incomplete' if incomplete else ('expanded' if expansions else 'complete'),
                    expansions=expansions,edge_touching=edge,notes=notes,
                    target_building_id=target['id'] if target else None,
                    selected_building_ids=[f['id'] for f in selected],
                    section_cut_ids=intentional,buffer_ft=buffer_ft,max_side_ft=MAX_SIDE_FT,
                    verification='mapped_preselection; point-cloud support not yet verified')
    return spec, coverage
