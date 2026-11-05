"""Menagerie arm -> per-link visual meshes at the zero pose, in Gripper 7's arm frame, decimated."""
import sys, math, json
import numpy as np, mujoco, fast_simplification

xml, out, n_links, my_axes, budget = sys.argv[1], sys.argv[2], int(sys.argv[3]), json.loads(sys.argv[4]), int(sys.argv[5])
m = mujoco.MjModel.from_xml_path(xml)
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
# the arm's hinge joints in order (not the fingers' slides)
hinges = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE][:n_links]
anchors = np.array([d.xanchor[j] for j in hinges]); axes = np.array([d.xaxis[j] for j in hinges])
print('anchors', np.round(anchors, 4).tolist()); print('axes', np.round(axes, 3).tolist())
# a turn about z that brings the second joint's axis onto ours (+-)
want = np.array(my_axes[1], float)
yaw = math.atan2(want[1], want[0]) - math.atan2(axes[1][1], axes[1][0])
R = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
print('yaw', round(math.degrees(yaw), 1))
for k, (a, mine) in enumerate(zip(R @ axes.T .T if False else (axes @ R.T), my_axes)):
    print(k + 1, 'axis', np.round(a, 3), 'mine', mine, 'ok' if abs(abs(np.dot(a, mine)) - 1) < 1e-3 else 'MISMATCH')
print('anchors turned', np.round(anchors @ R.T, 4).tolist())
link_body = [m.jnt_bodyid[j] for j in hinges]  # link k (1..n) moves with joint k
base_body = m.body_parentid[link_body[0]]

def owner(b):  # which link (0 = base) a body rides on
    while b not in link_body and b != base_body and b != 0:
        b = m.body_parentid[b]
    return link_body.index(b) + 1 if b in link_body else 0

groups = {}
for g in range(m.ngeom):
    if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or m.geom_contype[g] or m.geom_conaffinity[g]:
        continue  # visual meshes only
    b = m.geom_bodyid[g]
    k = owner(b)
    if k == 0 and b != base_body:
        continue  # the hand and fingers
    if b != base_body and b not in link_body and k == n_links:
        continue  # (bodies past the last link: the hand)
    mid = m.geom_dataid[g]
    v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(float)
    f = m.mesh_face[m.mesh_faceadr[mid]:m.mesh_faceadr[mid] + m.mesh_facenum[mid]].astype(np.int64)
    w = d.geom_xpos[g] + v @ d.geom_xmat[g].reshape(3, 3).T
    rgba = tuple(np.round(m.mat_rgba[m.geom_matid[g]] if m.geom_matid[g] >= 0 else m.geom_rgba[g], 3))
    key = (k, rgba)
    P, F = groups.get(key, (np.zeros((0, 3)), np.zeros((0, 3), np.int64)))
    groups[key] = (np.vstack([P, w @ R.T]), np.vstack([F, f + len(P)]))

total = sum(len(F) for P, F in groups.values())
arrays, index = {}, []
for i, ((k, rgba), (P, F)) in enumerate(sorted(groups.items())):
    # weld duplicate corners, then decimate to this group's share of the budget (faces)
    Pq = np.round(P / 1e-5).astype(np.int64)
    _, first, inv = np.unique(Pq, axis=0, return_index=True, return_inverse=True)
    P2, F2 = P[first], inv.reshape(-1)[F]
    F2 = F2[(F2[:, 0] != F2[:, 1]) & (F2[:, 1] != F2[:, 2]) & (F2[:, 0] != F2[:, 2])]
    target = max(300, int(budget * len(F) / total))
    if len(F2) > target:
        P3, F3 = fast_simplification.simplify(P2.astype(np.float32), F2.astype(np.int32), 1 - target / len(F2))
    else:
        P3, F3 = P2, F2
    print(f'link {k} rgba {rgba}: {len(F)} -> {len(F3)} faces, {len(P3)} points')
    arrays[f'p{i}'], arrays[f't{i}'] = np.asarray(P3, np.float32), np.asarray(F3, np.int32)
    index.append([k, list(map(float, rgba))])
np.savez_compressed(out, index=json.dumps(index), **arrays)
