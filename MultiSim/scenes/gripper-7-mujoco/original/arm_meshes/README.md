# Arm meshes for Gripper 7

The KUKA LBR iiwa 14 and Franka Emika Panda links come from MuJoCo Menagerie (google-deepmind/mujoco_menagerie):

- `kuka.npz`: `kuka_iiwa_14`, meshes from Drake, BSD 3-Clause (`LICENSE-kuka-iiwa-14`).
- `franka.npz`: `franka_emika_panda`, meshes from Franka Robotics, Apache 2.0 (`LICENSE-franka-panda`).

Changes made by `make_arm_meshes.py`:

- each link's visual meshes are posed at the zero configuration and merged per colour;
- they are decimated, because the viewer streams every vertex each frame:
  - KUKA: to about 9000 faces plus a face per open edge (12400), with pyfqmr keeping the open edges in place, so the
    colours of a link still meet (moved separately, they left jagged gaps around the light band with the lettering);
    meshes under 1000 faces, the band and the lettering, are kept as they are;
  - Franka: to about 10000 faces with fast-simplification (its colours meet well enough; keeping its many open edges
    would cost 80% more vertices);
- they are saved as `p<i>` (points [m], in the arm's base frame) and `t<i>` (triangles), with `index` = `[[link, rgba], ...]`.

Link 0 is the base. Both arms' joint axes at zero lie where Gripper 7's arm models put them, so no turn is needed.

To remake the files:

    uv run --with mujoco --with pyfqmr python make_arm_meshes.py <menagerie>/kuka_iiwa_14/iiwa14.xml kuka.npz 7 '[[0,0,1],[0,1,0],[0,0,1],[0,-1,0],[0,0,1],[0,1,0],[0,0,1]]' 9000 borders
    uv run --with mujoco --with fast-simplification python make_arm_meshes.py <menagerie>/franka_emika_panda/panda.xml franka.npz 7 '[[0,0,1],[0,1,0],[0,0,1],[0,-1,0],[0,0,1],[0,-1,0],[0,0,-1]]' 10000
