# -*- coding: utf-8 -*-
"""
Standalone parameter editor for Gripper7Mj.py (Gripper 7 on MuJoCo: a mobile manipulator).

Runs as its own window/process (Tkinter, standard library, no SOFA needed),
reading and writing params.json in the same directory. Every change
auto-saves (text fields on focus-out/Enter, checkboxes immediately). It
does not talk to a running simulation: changes take effect when the
scene is next built (run Gripper7Mj.py again).

Run:
    python params_editor.py
"""
import json
import math
import os
import tkinter as tk
from tkinter import ttk

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARAMS_FILE = os.path.join(_SCRIPT_DIR, 'params.json')

_DEFAULTS = {
    # -- the robot: which gripper, arm and cart --
    'gripper': 'pneunet',        # 'pneunet' (soft fingers), 'parallel' (rigid jaws), 'three_jaw', 'suction'
    'arm': 'ur',                 # 'ur' (UR-like, 6 axes), 'kuka' (LBR iiwa 14), 'franka' (Panda)
    'cart': 'omni',              # 'omni' (omniwheels), 'diff' (two wheels), 'tank' (tracks), 'car' (steering)
    # -- rigid jaws and the suction cup --
    'jaw_stroke': 55.0,          # [mm] each jaw's inner face from the axis, fully open
    'jaw_length': 70.0,          # [mm] below the palm
    'jaw_width': 22.0,           # [mm]
    'jaw_force': 40.0,           # [N] each jaw's grip at 100 kPa (the grip control; the Grip button: 80)
    'suction_force': 30.0,       # [N] the simple cup's pull at full suction (80 kPa and more)
    'finray_length': 80.0,       # [mm] Festo Fin Ray finger (DHAS-GF-80 size): below the jaw
    'finray_base': 30.0,         # [mm] its depth at the jaw (the triangle's base)
    'finray_stiffness': 0.12,    # [N m/rad] each flank hinge (the flanks' bending)
    'mia_torque': 1.0,           # [N m] Mia hand: each motor's largest torque at its finger's base joint
    'tactile_skin': True,        # sensing pads on the fingers' gripping faces (Touchlab / TG0 style e-skin)
    'cup_vacuum': 30.0,          # [kPa] a vacuum cup's (GripWiq, SoftGripping) vacuum at full grip (80 kPa)
    'cup_lip_stiffness': 150.0,  # [N/m] each segment of a vacuum cup's silicone lip
    'sg_spacing': 45.0,          # [mm] SoftGripping parallel base: between the two fingers on each side
    'sg_open_vacuum': 20.0,      # [kPa] SoftGripping: the negative pressure that opens the fingers wide
    'steer_limit': 35.0,         # [deg] the car's largest steering angle
    # -- the fingers --
    'fingers': 4,                # how many, evenly spaced around the end-effector's axis
    'finger_radius': 45.0,       # [mm] from the axis to each finger's inner face (its base layer)
    'finger_azimuth': 0.0,       # [deg] where finger 1 hangs (0: +x, 90: +y)
    # -- each finger's segments [mm] --
    'segments': 6,               # rigid segments (chambers) in each finger
    'length': 8.0,               # one segment, along the finger
    'width': 20.0,               # across it
    'height': 16.0,              # from the base layer to the roof
    'wall_end': 1.5,             # end walls (they set the cavity's cross-section and the mass)
    'wall_side': 1.5,
    'wall_bottom': 3.0,
    'wall_top': 1.5,
    'channel_separation': 2.0,   # gap between neighbouring segments, at the base
    # -- the pointy tips --
    'tips_enabled': True,
    'tip_length': 14.0,          # [mm] beyond the last segment
    'tip_width': 3.0,            # [mm] the point, across the finger
    'tip_thickness': 2.0,        # [mm] the point, from the finger's inner face
    # -- joints --
    'joint_stiffness': 0.34,     # [N m/rad] torsional stiffness of each hinge
    'joint_damping': 0.01,       # [s] stiffness-proportional damping of the joints
    'joint_open_limit': 90.0,    # [deg] largest bend of a hinge (a hard stop beyond it)
    'side_stiffness': 5.0,       # sideways bending of each joint: its stiffness, times joint_stiffness (0: rigid
                                 #   sideways, as in Gripper 4); a silicone finger is stiffer across than along
    'side_limit': 20.0,          # [deg] largest sideways bend of a joint
    'joint_armature': 2e-6,      # [kg m^2] added inertia of each hinge (MuJoCo: keeps the light segments stable)
    'limit_timeconst': 0.002,    # [s] the hinges' limits (MuJoCo solref): small = hard stops; soft ones give way
                                 #   when the gripper is pushed into the floor or a wall
    'pressure_torque_gain': 1.0, # 1: the torque of the pressure on the cavity's cross-section
    # -- the end-effector's motion --
    'tcp_depth': 50.0,           # [mm] the tool point below the palm (the point that moves and turns)
    # -- the arm [mm], UR-like (at zero all links point up; the tool points along the shoulder axis) --
    'arm_mount_x': 250.0,        # where it stands on the platform (UR-like arm): forward of the platform's centre
    'arm_mount_y': 120.0,        #   and to the left (towards the shelves at the start)
    'arm_base_height': 160.0,    # base joint to shoulder joint
    'arm_shoulder_offset': 135.0,  # sideways
    'arm_upper': 450.0,          # shoulder to elbow
    'arm_elbow_offset': 120.0,   # sideways, back
    'arm_fore': 400.0,           # elbow to wrist 1
    'arm_wrist1': 100.0,         # wrist 1 to wrist 2 (sideways)
    'arm_wrist2': 100.0,         # wrist 2 to wrist 3 (up)
    'arm_wrist3': 100.0,         # wrist 3 to the flange (sideways, the tool axis)
    'arm_gain': 3000.0,          # [N m/rad] position gain of the base, shoulder and elbow (wrist: a tenth)
    'ik_rate': 100.0,            # [Hz] inverse kinematics solves per second
    'ik_max_jump': 30.0,         # [deg] a solution that moves a joint more than this at once is refused (a flip)
    # -- the platform [mm] --
    'platform_length': 1000.0,
    'platform_width': 700.0,
    'platform_height': 250.0,    # its deck
    'platform_mass': 60.0,       # [kg]
    'start_x': -600.0,           # where its centre starts (heading +x, along the shelves)
    'start_y': -400.0,
    'drive_speed': 0.5,          # [m/s] largest speed, any direction (omniwheels; the joystick's full stroke)
    'drive_acceleration': 0.6,   # [m/s2]
    'turn_speed': 60.0,          # [deg/s] largest turn rate
    'turn_acceleration': 180.0,  # [deg/s2]
    'flange_acceleration': 1.5,  # [m/s2] largest acceleration of the target
    'flange_speed': 0.4,         # [m/s] its largest speed
    'angular_acceleration': 360.0,  # [deg/s2]
    'angular_speed': 90.0,       # [deg/s]
    'flange_mass': 0.5,          # [kg] the gripper's plate and wrist adapter
    # -- the store [mm] --
    'shelf_distance': 300.0,     # the shelves' front edge (y = this)
    'shelf_length': 1200.0,      # each unit, along x (two bays)
    'shelf_depth': 320.0,
    'deck_height': 100.0,        # the lowest shelf (on the plinth)
    'level_height': 560.0,       # from one shelf's top to the next
    'board_thickness': 20.0,
    'lip_height': 8.0,           # the low edge along the front of every shelf
    'second_unit': True,         # a second shelf unit, beside the first (at +x)
    'crate_length': 440.0,       # the platform's crate, inside, along the platform
    'crate_width': 500.0,        # inside, across it
    'crate_height': 80.0,
    'crate_wall': 6.0,
    'item_gap': 45.0,            # [mm] between neighbouring places on a shelf (room for the gripper)
    'crate_gap': 40.0,           # [mm] between items in a crate (room for the fingers)
    'item_mass_scale': 1.0,      # times each item's mass (see PRODUCTS)
    # -- contacts --
    'friction': 0.8,             # Coulomb friction coefficient of every contact
    'torsional_friction': 0.005, # [m] resistance to twisting about a contact's normal
    'rolling_friction': 0.002,   # [m] round items: resistance to rolling
    'impratio': 10.0,            # MuJoCo: ratio of frictional to normal constraint impedance (higher: less slip)
    'noslip_iterations': 10,     # MuJoCo's no-slip solver passes: without them a held item creeps out (~1 mm/s)
    'finger_collisions': True,   # the fingers touch each other
    'sleep': True,               # resting bodies sleep (MuJoCo islands): cheaper, they wake when touched
    # -- material --
    'density': 1000.0,           # [kg/m^3] of the silicone (the segment's mass: the walls' volume)
    'gravity_enabled': True,
    # -- pressure --
    'ramp_start': 0.0,           # [s] when the pressure starts to rise
    'ramp_time': 0.4,            # [s]
    'dt': 0.02,                  # [s] one frame step (many MuJoCo steps)
    'timestep': 0.001,           # [s] MuJoCo's time step
    'instant': False,
    # -- visuals --
    'wall_alpha': 1.0,
    'joint_alpha': 1.0,
    'item_alpha': 1.0,
    'spot_alpha': 0.45,
    'show_joint_marks': True,
    'show_spots': True,          # the empty places on the shelves, marked in the product's colour
    'theme': 'System',
}

_PALETTES = {
    'Light': dict(bg='#f0f0f0', fg='#000000', entry_bg='#ffffff', entry_fg='#000000',
                  select_bg='#0078d7', select_fg='#ffffff', btn_bg='#e1e1e1',
                  muted_fg='#666666', error_fg='#cc0000', border='#adadad'),
    'Dark': dict(bg='#1e1e1e', fg='#f5f5f5', entry_bg='#333333', entry_fg='#ffffff',
                 select_bg='#0a84c0', select_fg='#ffffff', btn_bg='#3a3a3a',
                 muted_fg='#b8b8b8', error_fg='#ff8080', border='#5a5a5a'),
}


def _system_prefers_dark():
    """Reads the Windows 'Apps use light theme' registry setting. Defaults
    to light (returns False) if unavailable (non-Windows, older Windows,
    or any registry-access failure)."""
    try:
        import winreg
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"SOFTWARE\Microsoft\Windows\CurrentVersion\Themes\Personalize")
        value, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
        return value == 0
    except Exception:
        return False


def _set_titlebar_dark(window, dark):
    """Best-effort dark title bar on Windows 10 1809+/11 (DWM API via
    ctypes, no extra dependency). Silently does nothing if unsupported."""
    try:
        import ctypes
        hwnd = ctypes.windll.user32.GetParent(window.winfo_id())
        value = ctypes.c_int(1 if dark else 0)
        for attr in (20, 19):  # DWMWA_USE_IMMERSIVE_DARK_MODE: 20 (current), 19 (older builds)
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd, attr, ctypes.byref(value), ctypes.sizeof(value))
    except Exception:
        pass

# Drop-downs: (label, json_key, [(value, text), ...])
_CHOICE_FIELDS = [
    ('Gripper', 'gripper', [('pneunet', 'Soft PneuNet fingers (Gripper 5)'),
                            ('softgripping_centric', 'SoftGripping SoftGripper, centric base'),
                            ('softgripping_parallel', 'SoftGripping SoftGripper, parallel base'),
                            ('softgripping_suction', 'SoftGripping SoftGripper, centric with suction cup'),
                            ('gripwiq_st', 'GripWiq SofTouch suction cup, ST'),
                            ('gripwiq_cs', 'GripWiq SofTouch suction cup, CS (larger)'),
                            ('festo_finray', 'Festo Fin Ray fingers (DHAS) on parallel jaws'),
                            ('mia', 'Prensilia Mia hand'),
                            ('parallel', 'Rigid parallel jaws'), ('three_jaw', 'Rigid three-jaw chuck'),
                            ('suction', 'Simple suction cup (adhesion)')]),
    ('Robot arm', 'arm', [('ur', 'UR-like, 6 axes'), ('kuka', 'KUKA LBR iiwa 14, 7 axes'),
                          ('franka', 'Franka Emika Panda, 7 axes')]),
    ('Cart', 'cart', [('omni', 'Omniwheels (any direction)'), ('diff', 'Two-wheel differential drive'),
                      ('tank', 'Tank tracks (skid steering)'), ('car', 'Car-like (front-wheel steering)')]),
]

# Vectors (the web viewer's panel; params.json is edited by hand in the desktop editor): (label, key, length, decimals, unit)
_VECTOR_FIELDS = []

# (label, json_key, unit_scale [json = displayed * scale], decimals, unit)
_FIELDS = [
    ('Jaws: inner face from the axis, open',   'jaw_stroke',          1, 1, 'mm'),
    ('Jaws: length below the palm',            'jaw_length',          1, 1, 'mm'),
    ('Jaws: width',                            'jaw_width',           1, 1, 'mm'),
    ('Jaws: grip force at 100 kPa',            'jaw_force',           1, 1, 'N'),
    ('Simple suction cup: pull at full suction', 'suction_force',     1, 1, 'N'),
    ('Vacuum cups: vacuum at full grip',       'cup_vacuum',          1, 1, 'kPa'),
    ('Vacuum cups: stiffness of a lip segment', 'cup_lip_stiffness',  1, 1, 'N/m'),
    ('SoftGripping parallel: finger spacing',  'sg_spacing',          1, 1, 'mm'),
    ('SoftGripping: opening vacuum',           'sg_open_vacuum',      1, 1, 'kPa'),
    ('Fin Ray finger: length',                 'finray_length',       1, 1, 'mm'),
    ('Fin Ray finger: depth at the jaw',       'finray_base',         1, 1, 'mm'),
    ('Fin Ray finger: flank hinge stiffness',  'finray_stiffness',    1, 3, 'N m/rad'),
    ('Mia hand: motor torque limit',           'mia_torque',          1, 2, 'N m'),
    ('Car: largest steering angle',            'steer_limit',         1, 1, 'deg'),
    ('Number of fingers',                      'fingers',             1, 0, ''),
    ('Fingers: distance of the inner faces from the axis', 'finger_radius', 1, 1, 'mm'),
    ('Direction of finger 1 (0: +x, 90: +y)',  'finger_azimuth',      1, 1, 'deg'),
    ('Segments per finger (rigid chambers)',   'segments',            1, 0, ''),
    ('Segment length (along the finger)',      'length',              1, 2, 'mm'),
    ('Width (across the finger)',              'width',               1, 2, 'mm'),
    ('Height (base layer to roof)',            'height',              1, 2, 'mm'),
    ('End wall thickness',                     'wall_end',            1, 2, 'mm'),
    ('Side wall thickness',                    'wall_side',           1, 2, 'mm'),
    ('Bottom layer thickness',                 'wall_bottom',         1, 2, 'mm'),
    ('Roof thickness',                         'wall_top',            1, 2, 'mm'),
    ('Gap between segments (at the base)',     'channel_separation',  1, 2, 'mm'),
    ('Tip length (beyond the last segment)',   'tip_length',          1, 1, 'mm'),
    ('Tip point width (across the finger)',    'tip_width',           1, 1, 'mm'),
    ('Tip point thickness (from the inner face)', 'tip_thickness',    1, 1, 'mm'),
    ('Joint torsional stiffness',              'joint_stiffness',     1, 4, 'N m/rad'),
    ('Joint damping (stiffness-proportional)', 'joint_damping',       1, 4, 's'),
    ('Joint open limit',                       'joint_open_limit',    1, 1, 'deg'),
    ('Sideways stiffness (times the joint stiffness; 0: rigid)', 'side_stiffness', 1, 2, ''),
    ('Sideways bend limit',                    'side_limit',          1, 1, 'deg'),
    ('Joint armature (added inertia)',         'joint_armature',      1, 7, 'kg m2'),
    ('Joint limit time constant (hardness)',   'limit_timeconst',     1, 4, 's'),
    ('Pressure torque gain',                   'pressure_torque_gain', 1, 3, ''),
    ('Tool point below the palm',              'tcp_depth',           1, 1, 'mm'),
    ('Largest acceleration',                   'flange_acceleration', 1, 3, 'm/s2'),
    ('Largest speed',                          'flange_speed',        1, 3, 'm/s'),
    ('Largest angular acceleration',           'angular_acceleration', 1, 0, 'deg/s2'),
    ('Largest angular speed',                  'angular_speed',       1, 0, 'deg/s'),
    ('Gripper plate and adapter mass',         'flange_mass',         1, 3, 'kg'),
    ('Arm: base, forward of the platform centre', 'arm_mount_x',      1, 0, 'mm'),
    ('Arm: base, left of the platform centre', 'arm_mount_y',         1, 0, 'mm'),
    ('Arm: base joint to shoulder',            'arm_base_height',     1, 0, 'mm'),
    ('Arm: shoulder offset (sideways)',        'arm_shoulder_offset', 1, 0, 'mm'),
    ('Arm: upper arm (shoulder to elbow)',     'arm_upper',           1, 0, 'mm'),
    ('Arm: elbow offset (sideways)',           'arm_elbow_offset',    1, 0, 'mm'),
    ('Arm: forearm (elbow to wrist 1)',        'arm_fore',            1, 0, 'mm'),
    ('Arm: wrist 1 to wrist 2',                'arm_wrist1',          1, 0, 'mm'),
    ('Arm: wrist 2 to wrist 3',                'arm_wrist2',          1, 0, 'mm'),
    ('Arm: wrist 3 to the flange',             'arm_wrist3',          1, 0, 'mm'),
    ('Arm: joint position gain (wrist: a tenth)', 'arm_gain',         1, 0, 'N m/rad'),
    ('Inverse kinematics: solves per second',  'ik_rate',             1, 0, 'Hz'),
    ('Inverse kinematics: largest joint jump', 'ik_max_jump',         1, 1, 'deg'),
    ('Platform length',                        'platform_length',     1, 0, 'mm'),
    ('Platform width',                         'platform_width',      1, 0, 'mm'),
    ('Platform deck height',                   'platform_height',     1, 0, 'mm'),
    ('Platform mass',                          'platform_mass',       1, 1, 'kg'),
    ('Platform start x',                       'start_x',             1, 0, 'mm'),
    ('Platform start y',                       'start_y',             1, 0, 'mm'),
    ('Largest drive speed',                    'drive_speed',         1, 2, 'm/s'),
    ('Drive acceleration',                     'drive_acceleration',  1, 2, 'm/s2'),
    ('Largest turn rate',                      'turn_speed',          1, 0, 'deg/s'),
    ('Turn acceleration',                      'turn_acceleration',   1, 0, 'deg/s2'),
    ('Shelf front edge (y)',                   'shelf_distance',      1, 0, 'mm'),
    ('Shelf unit length (two bays)',           'shelf_length',        1, 0, 'mm'),
    ('Shelf depth',                            'shelf_depth',         1, 0, 'mm'),
    ('Lowest shelf (deck) height',             'deck_height',         1, 0, 'mm'),
    ('Shelf spacing (top to top)',             'level_height',        1, 0, 'mm'),
    ('Shelf board thickness',                  'board_thickness',     1, 1, 'mm'),
    ('Shelf front lip height',                 'lip_height',          1, 1, 'mm'),
    ('Gap between places on a shelf',          'item_gap',            1, 1, 'mm'),
    ('Crate: inside length (along the platform)', 'crate_length',     1, 0, 'mm'),
    ('Crate: inside width (across it)',        'crate_width',         1, 0, 'mm'),
    ('Crate: wall height',                     'crate_height',        1, 0, 'mm'),
    ('Crate: wall thickness',                  'crate_wall',          1, 1, 'mm'),
    ('Gap between items in a crate',           'crate_gap',           1, 1, 'mm'),
    ('Item masses (times the real ones)',      'item_mass_scale',     1, 2, ''),
    ('Friction coefficient',                   'friction',            1, 2, ''),
    ('Torsional friction',                     'torsional_friction',  1, 4, 'm'),
    ('Rolling friction (round items)',         'rolling_friction',    1, 4, 'm'),
    ('Friction impedance ratio (impratio)',    'impratio',            1, 1, ''),
    ('No-slip solver iterations (0: off)',     'noslip_iterations',   1, 0, ''),
    ('Silicone density',                       'density',             1, 1, 'kg/m3'),
    ('Pressure ramp start',                    'ramp_start',          1, 2, 's'),
    ('Pressure ramp time',                     'ramp_time',           1, 2, 's'),
    ('Frame step',                             'dt',                  1, 4, 's'),
    ('MuJoCo time step',                       'timestep',            1, 5, 's'),
    ('Segment opacity',                        'wall_alpha',          1, 2, ''),
    ('Joint mark opacity',                     'joint_alpha',         1, 2, ''),
    ('Item opacity',                           'item_alpha',          1, 2, ''),
    ('Empty place marks opacity',              'spot_alpha',          1, 2, ''),
]

_BOOL_FIELDS = [
    ('Tactile skin on the fingers (Touchlab / TG0 style)', 'tactile_skin', True),
    ('Pointy fingertips', 'tips_enabled', True),
    ('A second shelf unit, beside the first', 'second_unit', True),
    ('Mark the empty places on the shelves', 'show_spots', True),
    ('Apply pressure instantly (not gradual)', 'instant', False),
    ('Enable gravity', 'gravity_enabled', True),
    ('Fingers touch each other', 'finger_collisions', True),
    ('Resting bodies sleep (MuJoCo islands)', 'sleep', True),
    ('Show the hinges as orange marks', 'show_joint_marks', True),
]

def load_params():
    params = dict(_DEFAULTS)
    if os.path.exists(PARAMS_FILE):
        try:
            with open(PARAMS_FILE, 'r') as f:
                loaded = json.load(f)
            params.update({k: v for k, v in loaded.items() if k in params})
        except (OSError, ValueError):
            pass
    return params


def _mesh_stats(params):
    """A summary of the model (no mesh: rigid segments)."""
    try:
        f, n = int(params['fingers']), int(params['segments'])
        length = n * params['length'] + (n - 1) * params['channel_separation']
        tip = f" + {params['tip_length']:.0f} mm tips" if params['tips_enabled'] else ""
        return (f"Gripper {params['gripper']} (soft: {f} fingers of {n} segments, {length:.1f} mm long{tip}), on the "
                f"{params['arm']} arm on the {params['cart']} cart; "
                f"{'two shelf units side by side' if params['second_unit'] else 'one shelf unit'}")
    except Exception as e:
        return f"(not computed: {e})"

class ParamsEditor(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Gripper 7 (MuJoCo) parameters")
        self.resizable(True, True)
        self.attributes('-topmost', True)  # stay visible over the SOFA GUI

        params = load_params()
        self.vars = {}
        self.style = ttk.Style(self)

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        # Scrollable body: there are too many settings now to always fit
        # the window's own height, so the actual content lives in `frame`
        # (unchanged below -- everything still grids into it exactly as
        # before), placed inside a Canvas that scrolls vertically, with a
        # Scrollbar alongside it and the mouse wheel bound for
        # convenience. The window's own size (see the bottom of this
        # method) is now a fixed, reasonable default instead of growing
        # to fit all content, since that's the whole point of scrolling.
        container = ttk.Frame(self)
        container.grid(row=0, column=0, sticky='nsew')
        container.columnconfigure(0, weight=1)
        container.rowconfigure(0, weight=1)

        canvas = tk.Canvas(container, highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient='vertical', command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=0, column=0, sticky='nsew')
        scrollbar.grid(row=0, column=1, sticky='ns')
        self._canvas = canvas

        frame = ttk.Frame(canvas, padding=12)
        frame_window = canvas.create_window((0, 0), window=frame, anchor='nw')
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)
        self._frame = frame

        def _on_frame_configure(event):
            canvas.configure(scrollregion=canvas.bbox('all'))
        frame.bind('<Configure>', _on_frame_configure)

        # Widgets whose text should word-wrap as the window is resized,
        # rather than overflow it -- several of the checkbox labels in
        # particular run to a full sentence or more. Split into "full"
        # width (checkboxes and the note/status/stats labels, which all
        # span both columns) and "half" width (the per-field labels in
        # column 0, sharing the row with an Entry in column 1) -- filled
        # in as those widgets are created below, then kept in sync with
        # the canvas width here.
        self._wrap_full_widgets = []
        self._wrap_half_widgets = []

        def _update_wraplengths(width):
            full_wrap = max(120, width - 28)
            half_wrap = max(80, width // 2 - 20)
            for w in self._wrap_full_widgets:
                w.configure(wraplength=full_wrap)
            for w in self._wrap_half_widgets:
                w.configure(wraplength=half_wrap)
        self._update_wraplengths = _update_wraplengths

        def _on_canvas_configure(event):
            canvas.itemconfig(frame_window, width=event.width)
            _update_wraplengths(event.width)
        canvas.bind('<Configure>', _on_canvas_configure)

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-event.delta / 120), 'units')
        self.bind_all('<MouseWheel>', _on_mousewheel)

        row = 0
        ttk.Label(frame, text="Theme").grid(row=row, column=0, sticky='w', pady=2)
        self.theme_var = tk.StringVar(value=params['theme'])
        theme_box = ttk.Combobox(frame, textvariable=self.theme_var, state='readonly',
                                  width=10, values=['Light', 'Dark', 'System'])
        theme_box.grid(row=row, column=1, sticky='e', pady=2, padx=(8, 0))
        theme_box.bind('<<ComboboxSelected>>', lambda e: (self._apply_theme(), self.save()))
        row += 1

        ttk.Separator(frame, orient='horizontal').grid(
            row=row, column=0, columnspan=2, sticky='ew', pady=6)
        row += 1

        self.choice_vars = {}
        for label, key, choices in _CHOICE_FIELDS:
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky='w', pady=2)
            texts = [t for _, t in choices]
            var = tk.StringVar(value=dict(choices).get(params[key], texts[0]))
            box = ttk.Combobox(frame, textvariable=var, state='readonly', width=28, values=texts)
            box.grid(row=row, column=1, sticky='e', pady=2, padx=(8, 0))
            box.bind('<<ComboboxSelected>>', lambda e: self.save())
            self.choice_vars[key] = (var, choices)
            row += 1

        for label, key, scale, decimals, unit in _FIELDS:
            text = f"{label} [{unit}]" if unit else label
            field_label = ttk.Label(frame, text=text)
            field_label.grid(row=row, column=0, sticky='w', pady=2)
            self._wrap_half_widgets.append(field_label)
            display_value = self._to_display(params[key], scale, decimals)
            var = tk.StringVar(value=str(display_value))
            entry = ttk.Entry(frame, textvariable=var, width=12)
            entry.grid(row=row, column=1, sticky='e', pady=2, padx=(8, 0))
            entry.bind('<FocusOut>', lambda e: self.save())
            entry.bind('<Return>', lambda e: self.save())
            self.vars[key] = (var, scale, decimals)
            row += 1

        self.bool_vars = {}
        self._checkbuttons = []
        for label, key, _default in _BOOL_FIELDS:
            var = tk.BooleanVar(value=bool(params[key]))
            # Plain tk.Checkbutton, not ttk: ttk.Checkbutton has no
            # wraplength option at all (confirmed -- TclError: unknown
            # option "-wraplength" -- not just a config-timing issue),
            # and several of these labels run to a full sentence or more.
            # Styled by hand in _apply_theme() since it won't pick up the
            # ttk style the way ttk.Checkbutton would.
            checkbutton = tk.Checkbutton(frame, text=label, variable=var, command=self.save,
                                          anchor='w', justify='left', highlightthickness=0,
                                          borderwidth=0)
            checkbutton.grid(row=row, column=0, columnspan=2, sticky='we', pady=2)
            self._wrap_full_widgets.append(checkbutton)
            self._checkbuttons.append(checkbutton)
            self.bool_vars[key] = var
            row += 1

        self.stats_label = ttk.Label(frame, text="")
        self.stats_label.grid(row=row, column=0, columnspan=2, sticky='w', pady=(8, 0))
        self._wrap_full_widgets.append(self.stats_label)
        row += 1
        self._update_stats(params)

        row += 1
        self.status = ttk.Label(frame, text=f"Auto-saving to {PARAMS_FILE}")
        self.status.grid(row=row, column=0, columnspan=2, sticky='w', pady=(10, 0))
        self._wrap_full_widgets.append(self.status)

        note = ("Every change auto-saves. Geometry, mesh and material changes need the "
                "SOFA GUI's Reload button to take effect. "
                "Pressure / ramp time / instant "
                "can also be edited live in SOFA's Scene Graph panel (\"PressureRamp\" "
                "node) while the simulation is running, without using this editor.")
        self.note_label = ttk.Label(frame, text=note, justify='left', font=('Segoe UI', 8),
                                     wraplength=280)
        self.note_label.grid(row=row + 1, column=0, columnspan=2,
                              sticky='we', pady=(6, 0))
        self._wrap_full_widgets.append(self.note_label)
        # All the wraplengths above (including this one's initial 280)
        # get kept in sync with the actual canvas width by
        # _on_canvas_configure as the window is resized.

        self._apply_theme()
        # Default to a narrow 400px-wide, 600px-tall window (less of the
        # screen taken up alongside the SOFA GUI); with this many
        # settings the content no longer fits any reasonable fixed
        # height, hence the scrollbar above -- so, unlike the width,
        # height is a fixed reasonable default rather than sized to
        # content. Minimum size is deliberately much smaller than that
        # (fields will overlap/clip below it, relying on the scrollbar
        # for anything cut off) -- an accepted trade-off for letting the
        # window be shrunk small.
        self.geometry("400x600")
        self.minsize(200, 200)

        # The canvas's very first <Configure> firing(s) happen while
        # widgets are still being created above (the geometry manager
        # reacts immediately, it doesn't wait for __init__ to finish), so
        # _wrap_full_widgets/_wrap_half_widgets were still empty when
        # they ran -- nothing after that necessarily re-fires <Configure>
        # just because more widgets were added to the (already-sized)
        # canvas. Apply the wraplengths once more explicitly, now that
        # everything exists and the window has its final size, so they
        # actually take effect instead of silently no-op'ing. Needs a
        # full update() here, not just update_idletasks(): the window
        # hasn't actually been mapped/drawn on screen yet at this point,
        # so winfo_width() would otherwise still report Tk's 1px
        # placeholder instead of the real, laid-out canvas width.
        self.update()
        self._update_wraplengths(canvas.winfo_width())

    @staticmethod
    def _to_display(json_value, scale, decimals):
        value = json_value / scale
        return int(round(value)) if decimals == 0 else round(value, decimals)

    def _update_stats(self, params):
        self.stats_label.config(text=_mesh_stats(params))

    def _apply_theme(self):
        choice = self.theme_var.get()
        dark = (choice == 'Dark') or (choice == 'System' and _system_prefers_dark())
        pal = _PALETTES['Dark' if dark else 'Light']
        self._palette = pal

        style = self.style
        style.theme_use('clam')
        # `clam` doesn't reliably keep a plain `configure(foreground=...)`
        # across widget states (focus/readonly/active/disabled) -- text
        # can silently fall back to its default (near-black) color in some
        # of those states unless the state is covered explicitly via
        # `.map()` too, which is what was causing poor dark-mode contrast.
        style.configure('.', background=pal['bg'], foreground=pal['fg'],
                         fieldbackground=pal['entry_bg'])
        style.map('.', foreground=[('disabled', pal['muted_fg']), ('!disabled', pal['fg'])])
        style.configure('TFrame', background=pal['bg'])
        style.configure('TLabel', background=pal['bg'], foreground=pal['fg'])
        style.configure('TEntry', fieldbackground=pal['entry_bg'], foreground=pal['entry_fg'],
                         insertcolor=pal['fg'], bordercolor=pal['border'])
        style.map('TEntry', foreground=[('disabled', pal['muted_fg']), ('!disabled', pal['entry_fg'])],
                   fieldbackground=[('!disabled', pal['entry_bg'])])
        style.configure('TCombobox', fieldbackground=pal['entry_bg'], foreground=pal['entry_fg'],
                         background=pal['btn_bg'], arrowcolor=pal['fg'])
        style.map('TCombobox', fieldbackground=[('readonly', pal['entry_bg'])],
                   foreground=[('readonly', pal['entry_fg']), ('!disabled', pal['entry_fg'])])
        style.configure('TSeparator', background=pal['border'])
        style.configure('TScrollbar', background=pal['btn_bg'], troughcolor=pal['bg'],
                         bordercolor=pal['border'], arrowcolor=pal['fg'])
        style.map('TScrollbar', background=[('active', pal['select_bg'])])
        # Combobox popdown list isn't a ttk widget -- styled via option db.
        self.option_add('*TCombobox*Listbox.background', pal['entry_bg'])
        self.option_add('*TCombobox*Listbox.foreground', pal['entry_fg'])
        self.option_add('*TCombobox*Listbox.selectBackground', pal['select_bg'])
        self.option_add('*TCombobox*Listbox.selectForeground', pal['select_fg'])

        self.configure(bg=pal['bg'])
        self._canvas.configure(bg=pal['bg'])  # plain Tk widget, not ttk-styled
        for cb in self._checkbuttons:  # plain tk.Checkbutton too -- see why above
            cb.configure(bg=pal['bg'], fg=pal['fg'], selectcolor=pal['entry_bg'],
                         activebackground=pal['bg'], activeforeground=pal['fg'])
        self.status.configure(foreground=pal['muted_fg'])
        self.stats_label.configure(foreground=pal['muted_fg'])
        self.note_label.configure(foreground=pal['muted_fg'])
        _set_titlebar_dark(self, dark)

    def save(self):
        params = load_params()
        try:
            for key, (var, scale, decimals) in self.vars.items():
                value = float(var.get()) * scale
                params[key] = int(round(value)) if decimals == 0 else value
            for key, var in self.bool_vars.items():
                params[key] = bool(var.get())
            for key, (var, choices) in self.choice_vars.items():
                params[key] = next(v for v, t in choices if t == var.get())
            params['theme'] = self.theme_var.get()
        except ValueError as e:
            self.status.config(text=f"Not saved -- invalid value ({e})",
                                foreground=self._palette['error_fg'])
            return

        self._update_stats(params)

        try:
            with open(PARAMS_FILE, 'w') as f:
                json.dump(params, f, indent=2)
        except OSError as e:
            self.status.config(text=f"Save failed: {e}", foreground=self._palette['error_fg'])
            return

        self.status.config(text=f"Saved -- press Reload in the SOFA GUI to apply",
                            foreground=self._palette['muted_fg'])


if __name__ == '__main__':
    ParamsEditor().mainloop()
