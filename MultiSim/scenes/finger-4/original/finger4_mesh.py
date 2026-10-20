# -*- coding: utf-8 -*-
"""
Geometry and meshing for Finger4.py -- pure Python, no SOFA, so the
standalone params_editor.py can use it too (element counts).

All feature dimensions are given in millimetres. Along each axis the
features' boundaries (outer faces, inner wall faces, channel, duct, the
symmetry midplane) split the axis into segments, and each segment is cut
into the whole number of equal elements that comes closest to the
standard element size: an 18 mm wall with 10 mm elements becomes 2 x 9 mm,
a 24 mm wall 2 x 12 mm, a 4 mm wall stays 1 x 4 mm. Every feature boundary
is therefore a grid plane, and nothing is rounded to a block grid.

The grid is a rectilinear (non-uniform) hexahedral grid; each solid cell
is split into 6 tetrahedra with one fixed (Kuhn) template, which keeps the
tetrahedral mesh conforming across neighbouring cells. Surfaces (the
pressure cavity, the outer skin, the interior visual) are made of the
solid cells' boundary faces, so their vertices are FEM nodes and the
BarycentricMapping onto the tetrahedra is exact.

Units: the returned geometry is in metres (SI), like the scene.
"""
import math

# Kuhn decomposition of a hexahedron, local corners
#   0=(0,0,0) 1=(1,0,0) 2=(1,1,0) 3=(0,1,0) 4=(0,0,1) 5=(1,0,1) 6=(1,1,1) 7=(0,1,1)
KUHN_TETS = [(0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6), (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6)]

MM = 1e-3


def split_segment(lo, hi, element_size, min_count=1):
    """Grid coordinates from lo to hi (inclusive) in the whole number of
    equal steps closest to `element_size` (at least `min_count`)."""
    n = max(min_count, 1, int(round((hi - lo) / element_size)))
    return [lo + (hi - lo) * i / n for i in range(n + 1)]


def axis_grid(breakpoints, element_size, walls=(), min_layers=1):
    """Sorted grid coordinates through all `breakpoints` (features'
    boundaries), each gap subdivided by split_segment -- into at least
    `min_layers` elements where the gap lies within one of the `walls`
    intervals (linear tetrahedra need a few layers through a wall to bend
    realistically)."""
    pts = sorted(set(round(b, 9) for b in breakpoints))
    coords = [pts[0]]
    for lo, hi in zip(pts, pts[1:]):
        if hi - lo < 1e-9:
            continue
        # A wall split into pieces (e.g. by the trunk's edge) gets its minimum
        # number of layers in total, shared out by the pieces' lengths.
        need = 1
        for a, b in walls:
            if a - 1e-9 <= lo and hi <= b + 1e-9:
                need = max(need, math.ceil(min_layers * (hi - lo) / (b - a) - 1e-9))
        coords += split_segment(lo, hi, element_size, need)[1:]
    return coords


def symmetric_grid(half_breakpoints, element_size, walls=(), min_layers=1):
    """Grid mirror-symmetric about 0 (which is always a grid plane), from
    the breakpoints (and wall intervals) of the positive half."""
    half = axis_grid([0.0] + [b for b in half_breakpoints if b > 1e-9], element_size, walls, min_layers)
    return [-c for c in reversed(half[1:])] + half


def _merge_into(value, others, tolerance):
    """`value`, or the nearest of `others` if that is within `tolerance`."""
    close = min(others, key=lambda o: abs(o - value))
    return close if abs(close - value) <= tolerance else value


def nearest(coords, value):
    return min(coords, key=lambda c: abs(c - value))


class Geometry:
    """Resolved chamber geometry (millimetres) and its grid. See
    Finger4.py's parameter table for what each parameter means."""

    def __init__(self, p):
        h = max(0.5, float(p['element_size']))
        self.h = h
        L = max(3 * h, float(p['length']))
        W = max(3 * h, float(p['width']))
        H = max(3 * h, float(p['height']))
        wx = min(max(0.1, float(p['wall_end'])), L / 2 - 0.5)
        wy = min(max(0.1, float(p['wall_side'])), W / 2 - 0.5)
        wb = max(0.1, float(p['wall_bottom']))
        wt = max(0.1, float(p['wall_top']))
        if wb + wt > H - 0.5:
            scale = (H - 0.5) / (wb + wt)
            wb, wt = wb * scale, wt * scale
        self.L, self.W, self.H, self.wx, self.wy, self.wb, self.wt = L, W, H, wx, wy, wb, wt

        cavity_w = W - 2 * wy
        ch = max(0.0, min(float(p['channel_height']), H - wb - wt))
        cw = float(p['channel_width'])
        cw = cavity_w if cw <= 0 or cw >= cavity_w - 1e-6 else max(0.1, cw)
        self.ch, self.cw = ch, cw
        self.full_width_channel = cw >= cavity_w - 1e-6
        cwall = max(0.1, float(p['channel_wall']))
        sep = max(0.0, float(p['channel_separation']))
        self.has_channel = ch > 1e-6
        self.has_duct = self.has_channel and sep > 1e-6
        self.half_sep = sep / 2 if self.has_duct else 0.0
        # The duct joining neighbouring chambers always spans the chambers' full
        # width (whatever the channel's width); channel_wall is its roof thickness.
        self.duct_roof = min(wb + ch + cwall, H)  # duct solid spans z = 0 .. duct_roof
        self.duct_half_w = W / 2
        self.far_x = L / 2 + self.half_sep         # far-end symmetry plane (if a duct)

        # Clamping the proximal end face models the first chamber of a finger:
        # its -X end wall is closed (no duct, no far-end plane there) and held.
        # It replaces the X symmetry plane (they exclude each other; if both
        # are on, the clamp wins).
        self.proximal_end = bool(p.get('clamp_proximal_end', False))
        self.symmetry_x = bool(p['symmetry_x']) and not self.proximal_end
        self.symmetry_y = bool(p['symmetry_y'])

        # Grids. X and Y are always mirror-symmetric about 0 (the chamber is),
        # so the midplanes are grid planes whether or not symmetry is used.
        xb = [L / 2 - wx, L / 2]
        if self.has_duct:
            xb.append(self.far_x)
        yb = [W / 2 - wy, W / 2]
        if not self.full_width_channel:
            yb.append(cw / 2)
        # The clamped base footprint (trunk) is a grid line too, so the clamp
        # covers exactly the given trunk length and width.
        tl = max(0.0, float(p['trunk_length']))
        tw = max(0.0, float(p['trunk_width']))
        # Exact, even when that leaves a thin layer of elements next to another
        # feature: rounding the trunk onto a nearby line changed the result
        # more (e.g. 15 -> 16 mm moved the clamp onto the end wall's root)
        # than the thin elements did. Only (nearly) coincident lines merge.
        if p.get('clamp_to_trunk', True):
            x_extent = self.far_x if self.has_duct else L / 2
            if 0 < tl / 2 < x_extent:
                xb.append(_merge_into(tl / 2, [0.0] + xb, 0.01))
            if 0 < tw / 2 < W / 2:
                yb.append(_merge_into(tw / 2, [0.0] + yb, 0.01))
        zb = [0.0, wb, H - wt, H]
        if self.has_channel:
            zb.append(wb + ch)
            if self.has_duct:
                zb.append(min(self.duct_roof, H))
        layers = max(1, int(p.get('min_wall_layers', 1)))
        self.min_wall_layers = layers
        x_walls = [(L / 2 - wx, L / 2)]
        y_walls = [(W / 2 - wy, W / 2)]
        z_walls = [(0.0, wb), (H - wt, H)]
        if self.has_duct:
            z_walls.append((wb + ch, self.duct_roof))
        self.xs = symmetric_grid(xb, h, x_walls, layers)
        self.ys = symmetric_grid(yb, h, y_walls, layers)
        self.zs = axis_grid(zb, h, z_walls, layers)

        # Symmetry buffer: extra material past an exploited midplane, snapped
        # to the grid (whole elements), at most the half itself.
        buf = max(0.0, float(p['symmetry_buffer']))
        neg_x = [c for c in self.xs if c <= 1e-9]
        neg_y = [c for c in self.ys if c <= 1e-9]
        self.buffer_x = -nearest(neg_x, -min(buf, L / 2)) if self.symmetry_x else None
        self.buffer_y = -nearest(neg_y, -min(buf, W / 2)) if self.symmetry_y else None
        if self.symmetry_x:
            x_lo = -self.buffer_x
        elif self.proximal_end:
            x_lo = -L / 2
        else:
            x_lo = -(self.far_x if self.has_duct else L / 2)
        y_lo = -self.buffer_y if self.symmetry_y else -W / 2
        self.xs = [c for c in self.xs if c >= x_lo - 1e-9]
        self.ys = [c for c in self.ys if c >= y_lo - 1e-9]

        # Trunk (clamped base footprint), centred: on grid lines (see above).
        self.trunk_x = (nearest(self.xs, -tl / 2), nearest(self.xs, tl / 2))
        self.trunk_y = (nearest(self.ys, -tw / 2), nearest(self.ys, tw / 2))
        if self.symmetry_x:
            # The buffer's floor stands in for the trunk past the cut.
            self.trunk_x = (min(self.trunk_x[0], -self.buffer_x), self.trunk_x[1])
        if self.symmetry_y:
            self.trunk_y = (min(self.trunk_y[0], -self.buffer_y), self.trunk_y[1])

    # -- cell classification (by cell centre, mm) -----------------------
    def in_cavity(self, x, y, z):
        return (abs(x) < self.L / 2 - self.wx and abs(y) < self.W / 2 - self.wy and
                self.wb < z < self.H - self.wt)

    def in_bore(self, x, y, z):
        reach = self.far_x if self.has_duct else self.L / 2
        x_min = -(self.L / 2 - self.wx) if self.proximal_end else -reach  # closed proximal end wall
        return (self.has_channel and abs(y) < self.cw / 2 and self.wb < z < self.wb + self.ch and
                x_min < x < reach)

    def is_air(self, x, y, z):
        return self.in_cavity(x, y, z) or self.in_bore(x, y, z)

    def is_solid(self, x, y, z):
        in_box = abs(x) < self.L / 2 and abs(y) < self.W / 2 and 0 < z < self.H
        in_duct = (self.has_duct and self.L / 2 <= abs(x) < self.far_x and abs(y) < self.duct_half_w and
                   0 < z < self.duct_roof and (x > 0 or not self.proximal_end))
        return (in_box or in_duct) and not self.is_air(x, y, z)

    def summary(self):
        def steps(cs):
            d = sorted(set(round(b - a, 3) for a, b in zip(cs, cs[1:])))
            return ", ".join(f"{v:g}" for v in d)
        return (f"grid {len(self.xs) - 1} x {len(self.ys) - 1} x {len(self.zs) - 1} cells; "
                f"element edges X: {steps(self.xs)} mm, Y: {steps(self.ys)} mm, Z: {steps(self.zs)} mm; "
                f"trunk {2 * self.trunk_x[1]:g} x {2 * self.trunk_y[1]:g} mm")


def build(p, z_offset_mm=0.0, outer_skip_normals=(), inner_skip_normals=()):
    """Meshes the chamber described by params `p` (millimetres).

    Returns a dict (all coordinates in metres):
      geometry   : the Geometry (mm)
      positions  : FEM nodes [[x, y, z], ...]
      tetrahedra : [[i, j, k, l], ...]
      cavity     : (points, triangles) closed pressure surface, outward normals
      outer      : (points, triangles) outer skin (visual)
      inner      : (points, triangles) cavity walls seen from inside (visual)
      bottom     : triangles (node indices) of the bottom face, z = 0
      far_nodes  : node indices on the +X far-end plane (and -X, see far_nodes_neg)
    `outer_skip_normals` / `inner_skip_normals`: e.g. {(0, -1, 0)} leaves those
    faces out of the visuals (cutaway).
    """
    g = Geometry(p)
    xs, ys, zs = g.xs, g.ys, g.zs
    nx, ny, nz = len(xs) - 1, len(ys) - 1, len(zs) - 1
    zoff = z_offset_mm

    def centre(i, j, k):
        return (xs[i] + xs[i + 1]) / 2, (ys[j] + ys[j + 1]) / 2, (zs[k] + zs[k + 1]) / 2

    solid = {}
    air = {}
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                c = centre(i, j, k)
                if g.is_solid(*c):
                    solid[(i, j, k)] = True
                elif g.is_air(*c):
                    air[(i, j, k)] = True

    # Nodes: corners of solid cells.
    node_index = {}
    positions = []

    def node(i, j, k):
        key = (i, j, k)
        idx = node_index.get(key)
        if idx is None:
            idx = len(positions)
            node_index[key] = idx
            positions.append([xs[i] * MM, ys[j] * MM, (zs[k] + zoff) * MM])
        return idx

    # The tetrahedron template is mirrored in cells on the negative side of
    # the X and Y midplanes, so the mesh itself is mirror-symmetric (a model
    # cut at a midplane is then exactly half of the full one); diagonals on
    # the midplane faces still agree, so the mesh stays conforming.
    corner_offsets = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
    tetrahedra = []
    for (i, j, k) in sorted(solid):
        cx, cy, _ = centre(i, j, k)
        mx, my = cx < 0, cy < 0
        c = []
        for dx, dy, dz in corner_offsets:
            if mx:
                dx = 1 - dx
            if my:
                dy = 1 - dy
            c.append(node(i + dx, j + dy, k + dz))
        for a, b, cc, d in KUHN_TETS:
            tet = [c[a], c[b], c[cc], c[d]]
            if mx != my:
                tet[2], tet[3] = tet[3], tet[2]  # one reflection flips the orientation back
            tetrahedra.append(tet)

    def is_material_node(key):
        return key in node_index

    # The half that is "real" (shown and pressurised): x >= 0 / y >= 0 when
    # that symmetry is exploited (the buffer past the cut is FEM only).
    i0 = xs.index(min(xs, key=abs)) if g.symmetry_x else 0
    j0 = ys.index(min(ys, key=abs)) if g.symmetry_y else 0

    def kept(i, j):
        return i >= i0 and j >= j0

    # Face of cell (i,j,k) towards direction d (axis a, sign s): its 4 grid
    # corners, counter-clockwise seen from outside the cell (normal = d).
    def face_corners(i, j, k, a, s):
        if a == 0:
            ii = i + 1 if s > 0 else i
            q = [(ii, j, k), (ii, j + 1, k), (ii, j + 1, k + 1), (ii, j, k + 1)]
        elif a == 1:
            jj = j + 1 if s > 0 else j
            q = [(i, jj, k), (i, jj, k + 1), (i + 1, jj, k + 1), (i + 1, jj, k)]
        else:
            kk = k + 1 if s > 0 else k
            q = [(i, j, kk), (i + 1, j, kk), (i + 1, j + 1, kk), (i, j + 1, kk)]
        return q if s > 0 else q[::-1]

    dirs = [(0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)]
    dims = (nx, ny, nz)

    def neighbour(cell, a, s):
        n = list(cell)
        n[a] += s
        return tuple(n), 0 <= n[a] < dims[a]

    def normal_of(a, s):
        v = [0, 0, 0]
        v[a] = s
        return tuple(v)

    def grid_point(key):
        i, j, k = key
        return [xs[i] * MM, ys[j] * MM, (zs[k] + zoff) * MM]

    class Surface:
        def __init__(self):
            self.points, self.tris, self.index = [], [], {}

        def pt(self, key):
            idx = self.index.get(key)
            if idx is None:
                idx = len(self.points)
                self.index[key] = idx
                self.points.append(grid_point(key))
            return idx

        def quad(self, q):
            a, b, c, d = (self.pt(k) for k in q)
            self.tris += [[a, b, c], [a, c, d]]

    cavity, outer, inner = Surface(), Surface(), Surface()
    cut_faces = {}  # plane id -> set of (cell, a, s) air faces on that cut plane

    x_far_i = len(xs) - 1 if g.has_duct else None

    for cell in sorted(solid):
        i, j, k = cell
        for a, s in dirs:
            nb, inside = neighbour(cell, a, s)
            q = face_corners(i, j, k, a, s)
            if inside and nb in solid:
                continue
            on_cut = False
            if not inside:
                # Domain boundary: a symmetry cut (exploited midplane, no
                # buffer) is internal -- not shown; the far-end plane rim is.
                if (a == 0 and s < 0 and g.symmetry_x) or (a == 1 and s < 0 and g.symmetry_y):
                    on_cut = True
            if not kept(i, j):
                continue
            # Faces lying on an exploited midplane (towards the buffer).
            if g.symmetry_x and a == 0 and s < 0 and i == i0:
                on_cut = True
            if g.symmetry_y and a == 1 and s < 0 and j == j0:
                on_cut = True
            if on_cut:
                continue
            if inside and nb in air:
                cavity.quad(q[::-1])     # outward from the air = into this solid cell
                if normal_of(a, -s) not in inner_skip_normals:
                    inner.quad(q[::-1])
            else:
                if normal_of(a, s) not in outer_skip_normals:
                    outer.quad(q)

    # Air faces on the cut planes (exploited midplanes, far-end planes):
    # the pressure surface has to be closed there.
    for cell in sorted(air):
        i, j, k = cell
        if not kept(i, j):
            continue
        if g.symmetry_x and i == i0:
            cut_faces.setdefault('x0', []).append(face_corners(i, j, k, 0, -1))
        if g.symmetry_y and j == j0:
            cut_faces.setdefault('y0', []).append(face_corners(i, j, k, 1, -1))
        if g.has_duct and i == nx - 1:
            cut_faces.setdefault('far+', []).append(face_corners(i, j, k, 0, 1))
        if g.has_duct and not g.symmetry_x and not g.proximal_end and i == 0:
            cut_faces.setdefault('far-', []).append(face_corners(i, j, k, 0, -1))

    for plane, quads in cut_faces.items():
        for loop in _boundary_loops(quads):
            loop = _simplify_loop(loop, is_material_node)
            for tri in _ear_clip(loop, plane, grid_point):
                cavity.tris.append([cavity.pt(key) for key in tri])

    bottom = []
    for (i, j, k) in solid:
        if k == 0:
            q = face_corners(i, j, k, 2, -1)
            a, b, c, d = (node_index[key] for key in q)
            bottom += [[a, b, c], [a, c, d]]

    far_pos = [node_index[key] for key in node_index if g.has_duct and key[0] == nx]
    far_neg = [node_index[key] for key in node_index
               if g.has_duct and not g.symmetry_x and not g.proximal_end and key[0] == 0]

    return {
        'geometry': g,
        'positions': positions,
        'tetrahedra': tetrahedra,
        'cavity': (cavity.points, cavity.tris),
        'outer': (outer.points, outer.tris),
        'inner': (inner.points, inner.tris),
        'bottom': bottom,
        'far_nodes': far_pos,
        'far_nodes_neg': far_neg,
    }


def count_elements(p):
    """(tetrahedra, nodes) for params `p`, without building surfaces."""
    g = Geometry(p)
    xs, ys, zs = g.xs, g.ys, g.zs
    cells = 0
    nodes = set()
    for i in range(len(xs) - 1):
        cx = (xs[i] + xs[i + 1]) / 2
        for j in range(len(ys) - 1):
            cy = (ys[j] + ys[j + 1]) / 2
            for k in range(len(zs) - 1):
                if g.is_solid(cx, cy, (zs[k] + zs[k + 1]) / 2):
                    cells += 1
                    for di in (0, 1):
                        for dj in (0, 1):
                            for dk in (0, 1):
                                nodes.add((i + di, j + dj, k + dk))
    return 6 * cells, len(nodes), g


# -- closing the pressure surface on a cut plane ----------------------------

def _boundary_loops(quads):
    """Boundary loops (lists of grid keys, in the quads' winding) of a set
    of coplanar grid quads (each a list of 4 keys)."""
    edges = {}
    for q in quads:
        for m in range(4):
            e = (q[m], q[(m + 1) % 4])
            if (e[1], e[0]) in edges:
                del edges[(e[1], e[0])]
            else:
                edges[e] = True
    nxt = {}
    for a, b in edges:
        nxt.setdefault(a, []).append(b)
    loops = []
    used = set()
    for start in list(nxt):
        for b0 in nxt[start]:
            if (start, b0) in used:
                continue
            loop = [start]
            a, b = start, b0
            while True:
                used.add((a, b))
                if b == start:
                    break
                loop.append(b)
                cands = [c for c in nxt[b] if (b, c) not in used]
                a, b = b, cands[0]
            loops.append(loop)
    return loops


def _simplify_loop(loop, is_material):
    """Drops loop points that are not FEM nodes (no material there: they
    would be mapped by extrapolation, with spurious pressure forces) --
    except corners, where the loop turns."""
    n = len(loop)
    keep = []
    for m in range(n):
        p0, p1, p2 = loop[m - 1], loop[m], loop[(m + 1) % n]
        d1 = tuple(b - a for a, b in zip(p0, p1))
        d2 = tuple(b - a for a, b in zip(p1, p2))
        straight = all(d1[t] * d2[s] == d1[s] * d2[t] for t in range(3) for s in range(3))
        if not straight or is_material(p1):
            keep.append(p1)
    return keep


def _ear_clip(loop, plane, coord):
    """Triangulates a simple polygon given as grid keys (all on one grid
    plane; `coord(key)` gives its position), keeping its winding.
    Rectilinear polygons with collinear points are fine."""
    axis = 0 if plane in ('x0', 'far+', 'far-') else 1
    pts = [coord(k) for k in loop]
    uv = [(q[1], q[2]) if axis == 0 else (q[0], q[2]) for q in pts]

    def area2(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    total = sum(uv[m][0] * uv[(m + 1) % len(uv)][1] - uv[(m + 1) % len(uv)][0] * uv[m][1]
                for m in range(len(uv)))
    sign = 1 if total > 0 else -1
    idx = list(range(len(loop)))
    tris = []
    guard = 0
    while len(idx) > 3 and guard < 10000:
        guard += 1
        clipped = False
        for m in range(len(idx)):
            a, b, c = idx[m - 1], idx[m], idx[(m + 1) % len(idx)]
            if sign * area2(uv[a], uv[b], uv[c]) <= 1e-18:
                continue  # reflex or collinear
            inside = False
            for o in idx:
                if o in (a, b, c):
                    continue
                p = uv[o]
                if (sign * area2(uv[a], uv[b], p) >= 0 and sign * area2(uv[b], uv[c], p) >= 0 and
                        sign * area2(uv[c], uv[a], p) >= 0):
                    inside = True
                    break
            if inside:
                continue
            tris.append((loop[a], loop[b], loop[c]))
            idx.pop(m)
            clipped = True
            break
        if not clipped:
            break
    if len(idx) == 3:
        a, b, c = idx
        if abs(area2(uv[a], uv[b], uv[c])) > 0:
            tris.append((loop[a], loop[b], loop[c]))
    return tris
