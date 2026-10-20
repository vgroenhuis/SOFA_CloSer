# -*- coding: utf-8 -*-
"""
Geometry and meshing for Finger5.py -- pure Python, no SOFA, so the
standalone params_editor.py can use it too (element counts).

A copy of finger4_mesh.py (../../finger-4/original) for the full finger:
`segments` identical chambers in a row along +X, joined by their channel
ducts (chamber k is centred at x = k * pitch, pitch = length + channel
separation). Only the Y midplane can be exploited as a symmetry plane.
Each chamber's grid and tetrahedra are the single chamber's (mirrored about
the chamber's own centre), so every segment is meshed exactly alike; the
air bore runs through the ducts from chamber to chamber, giving one
connected cavity. The end walls of the first and last chamber are closed.

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
    """Resolved finger geometry (millimetres) and its grid. See Finger5.py's
    parameter table for what each parameter means."""

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
        self.segments = n_seg = max(1, int(p['segments']))

        cavity_w = W - 2 * wy
        ch = max(0.0, min(float(p['channel_height']), H - wb - wt))
        cw = float(p['channel_width'])
        cw = cavity_w if cw <= 0 or cw >= cavity_w - 1e-6 else max(0.1, cw)
        self.ch, self.cw = ch, cw
        self.full_width_channel = cw >= cavity_w - 1e-6
        cwall = max(0.1, float(p['channel_wall']))
        sep = max(0.0, float(p['channel_separation']))
        self.has_channel = ch > 1e-6
        if n_seg > 1 and sep > 1e-6 and not self.has_channel:
            raise ValueError("the chambers need a connecting channel (channel height > 0), "
                             "or no gap between them (channel separation 0)")
        # A duct joins neighbouring chambers when there is a gap between them;
        # with no gap their end walls are one (double) wall.
        self.has_duct = self.has_channel and sep > 1e-6 and n_seg > 1
        self.half_sep = sep / 2 if sep > 1e-6 and n_seg > 1 else 0.0
        self.pitch = L + 2 * self.half_sep
        # The duct joining neighbouring chambers always spans the chambers' full
        # width (whatever the channel's width); channel_wall is its roof thickness.
        self.duct_roof = min(wb + ch + cwall, H)  # duct solid spans z = 0 .. duct_roof
        self.duct_half_w = W / 2
        self.far_x = L / 2 + self.half_sep         # local x of the planes between chambers
        self.x_min = -L / 2                        # the finger's proximal end face
        self.x_max = (n_seg - 1) * self.pitch + L / 2  # distal end face
        # Local x of the collision planes between chambers, and their global x.
        self.plane_x = [(k + 0.5) * self.pitch for k in range(n_seg - 1)] if self.has_duct else []

        self.symmetry_y = bool(p['symmetry_y'])

        # One chamber's grid, mirror-symmetric about its centre, repeated
        # for every chamber.
        xb = [L / 2 - wx, L / 2]
        if self.has_duct:
            xb.append(self.far_x)
        yb = [W / 2 - wy, W / 2]
        if not self.full_width_channel:
            yb.append(cw / 2)
        # The clamped base footprint (trunk, under the first chamber) is a grid
        # line too, so the clamp covers exactly the given trunk length and
        # width. Exact, even when that leaves a thin layer of elements next to
        # another feature (see finger4_mesh.py); only (nearly) coincident
        # lines merge.
        tl = max(0.0, float(p['trunk_length']))
        tw = max(0.0, float(p['trunk_width']))
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
        local_x = symmetric_grid(xb, h, x_walls, layers)
        xs = set()
        for k in range(n_seg):
            for c in local_x:
                x = round(c + k * self.pitch, 9)
                if self.x_min - 1e-9 <= x <= self.x_max + 1e-9:
                    xs.add(x)
        self.xs = sorted(xs)
        self.ys = symmetric_grid(yb, h, y_walls, layers)
        self.zs = axis_grid(zb, h, z_walls, layers)
        if self.symmetry_y:
            self.ys = [c for c in self.ys if c >= -1e-9]

        # Trunk (clamped base footprint), centred under the first chamber.
        self.trunk_x = (nearest(self.xs, -tl / 2), nearest(self.xs, tl / 2))
        self.trunk_y = (nearest(self.ys, -tw / 2), nearest(self.ys, tw / 2))

    def local(self, x):
        """(chamber index, x relative to that chamber's centre) [mm]."""
        k = min(self.segments - 1, max(0, int(math.floor(x / self.pitch + 0.5))))
        return k, x - k * self.pitch

    def _has_neighbour(self, k, xl):
        return self.has_duct and ((xl > 0 and k < self.segments - 1) or (xl < 0 and k > 0))

    # -- cell classification (by cell centre, mm) -----------------------
    def in_cavity(self, x, y, z):
        k, xl = self.local(x)
        return (abs(xl) < self.L / 2 - self.wx and abs(y) < self.W / 2 - self.wy and
                self.wb < z < self.H - self.wt)

    def in_bore(self, x, y, z):
        """The connecting channel: through the end walls and ducts, towards
        the neighbouring chambers only (or through the shared wall when there
        is no gap between them)."""
        if not self.has_channel or self.segments < 2:
            return False
        if not (abs(y) < self.cw / 2 and self.wb < z < self.wb + self.ch):
            return False
        k, xl = self.local(x)
        towards_next = xl > 0 and k < self.segments - 1
        towards_prev = xl < 0 and k > 0
        return abs(xl) < self.L / 2 - self.wx or towards_next or towards_prev

    def is_air(self, x, y, z):
        return self.in_cavity(x, y, z) or self.in_bore(x, y, z)

    def is_solid(self, x, y, z):
        k, xl = self.local(x)
        in_box = abs(xl) < self.L / 2 and abs(y) < self.W / 2 and 0 < z < self.H
        in_duct = (self._has_neighbour(k, xl) and self.L / 2 <= abs(xl) < self.far_x and
                   abs(y) < self.duct_half_w and 0 < z < self.duct_roof)
        return (in_box or in_duct) and not self.is_air(x, y, z)

    def summary(self):
        def steps(cs):
            d = sorted(set(round(b - a, 3) for a, b in zip(cs, cs[1:])))
            return ", ".join(f"{v:g}" for v in d)
        return (f"{self.segments} chambers, {self.x_max - self.x_min:g} mm long; "
                f"grid {len(self.xs) - 1} x {len(self.ys) - 1} x {len(self.zs) - 1} cells; "
                f"element edges X: {steps(self.xs)} mm, Y: {steps(self.ys)} mm, Z: {steps(self.zs)} mm; "
                f"trunk {self.trunk_x[1] - self.trunk_x[0]:g} x {2 * self.trunk_y[1]:g} mm")


def build(p, z_offset_mm=0.0, outer_skip_normals=(), inner_skip_normals=()):
    """Meshes the finger described by params `p` (millimetres).

    Returns a dict (all coordinates in metres):
      geometry   : the Geometry (mm)
      positions  : FEM nodes [[x, y, z], ...]
      tetrahedra : [[i, j, k, l], ...]
      cavity     : (points, triangles) closed pressure surface, outward normals
      outer      : (points, triangles) outer skin (visual)
      inner      : (points, triangles) cavity walls seen from inside (visual)
      bottom     : triangles (node indices) of the bottom face, z = 0
      plane_nodes: per collision plane (between chambers k and k + 1), the
                   node indices of the duct's cross-section on it
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
    # each chamber's own centre (X) and of the Y midplane, so every chamber
    # is meshed alike and mirror-symmetric. The Kuhn template's face
    # diagonals don't change under these reflections, so the mesh stays
    # conforming across every mirror plane.
    corner_offsets = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
    tetrahedra = []
    for (i, j, k) in sorted(solid):
        cx, cy, _ = centre(i, j, k)
        mx, my = g.local(cx)[1] < 0, cy < 0
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

    # With Y symmetry exploited only y >= 0 is meshed; the Y midplane is a cut.
    j0 = 0

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
    cut_faces = {}  # plane id -> list of air faces on that cut plane

    for cell in sorted(solid):
        i, j, k = cell
        for a, s in dirs:
            nb, inside = neighbour(cell, a, s)
            q = face_corners(i, j, k, a, s)
            if inside and nb in solid:
                continue
            # The Y symmetry cut is internal -- not shown.
            if g.symmetry_y and a == 1 and s < 0 and j == j0:
                continue
            if inside and nb in air:
                cavity.quad(q[::-1])     # outward from the air = into this solid cell
                if normal_of(a, -s) not in inner_skip_normals:
                    inner.quad(q[::-1])
            else:
                if normal_of(a, s) not in outer_skip_normals:
                    outer.quad(q)

    # Air faces on the Y symmetry cut: the pressure surface has to be closed there.
    for cell in sorted(air):
        i, j, k = cell
        if g.symmetry_y and j == j0:
            cut_faces.setdefault('y0', []).append(face_corners(i, j, k, 1, -1))

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

    plane_nodes = []
    for x in g.plane_x:
        i = min(range(len(xs)), key=lambda m: abs(xs[m] - x))
        plane_nodes.append([node_index[key] for key in node_index if key[0] == i])

    return {
        'geometry': g,
        'positions': positions,
        'tetrahedra': tetrahedra,
        'cavity': (cavity.points, cavity.tris),
        'outer': (outer.points, outer.tris),
        'inner': (inner.points, inner.tris),
        'bottom': bottom,
        'plane_nodes': plane_nodes,
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
