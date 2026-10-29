# Arm meshes for Gripper 7

The KUKA LBR iiwa 14 and Franka Emika Panda links come from MuJoCo Menagerie (google-deepmind/mujoco_menagerie):

- `kuka.npz`: `kuka_iiwa_14`, meshes from Drake, BSD 3-Clause (`LICENSE-kuka-iiwa-14`).
- `franka.npz`: `franka_emika_panda`, meshes from Franka Robotics, Apache 2.0 (`LICENSE-franka-panda`).

Changes made by `make_arm_meshes.py`:

- each link's visual meshes are posed at the zero configuration and merged per colour;
- they are decimated to about 9000 faces per arm, because the viewer streams every vertex each frame;
- they are saved as `p<i>` (points [m], in the arm's base frame) and `t<i>` (triangles), with `index` = `[[link, rgba], ...]`.

Link 0 is the base. Both arms' joint axes at zero lie where Gripper 7's arm models put them, so no turn is needed.

To remake the files: `uv run --with mujoco --with fast-simplification python make_arm_meshes.py <menagerie xml> <out.npz> 7 '<axes json>' 9000`
