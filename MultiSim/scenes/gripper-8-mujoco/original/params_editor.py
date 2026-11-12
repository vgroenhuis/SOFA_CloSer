# -*- coding: utf-8 -*-
"""
Standalone parameter editor for Gripper8Mj.py (Gripper 8 on MuJoCo: a grip test line of every gripper).

Runs as its own window/process (Tkinter, standard library, no SOFA needed),
reading and writing params.json in the same directory. Every change
auto-saves (text fields on focus-out/Enter, checkboxes immediately). It
does not talk to a running simulation: changes take effect when the
scene is next built (run Gripper8Mj.py again).

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
    # -- the test --
    'item': 'g7_egg',            # the grocery item (ITEM_CHOICES): 'chips' (150 g, sized below), 'chips_40',
                                 # 'chips_25', 'chips_15', 'potatoes', 'g7_<product>', 'hs_<product>'
    'spacing': 400.0,            # [mm] between the grippers, along x
    'lift_height': 450.0,        # [mm] the tool point's height while it carries and turns the item
    'travel': 1000.0,            # [mm] how far forward (+y) it carries the item
    'move_speed': 200.0,         # [mm/s] the moves' average speed (smooth starts and stops: the top speed 1.9 times)
    'turn_speed': 60.0,          # [deg/s] the turn about x, on average
    'grip_pressure': 80.0,       # [kPa] the grip (soft fingers: pressure; jaws: 80 = 80 % of Jaw force at 100;
                                 #   cups: full vacuum from 80; the Mia hand: closed from 80)
    'grip_wait': 1.0,            # [s] after the grip has ramped up, before lifting
    'release_wait': 1.5,         # [s] after letting go, before rising
    # -- the deformable items --
    'chips_length': 220.0,       # [mm] the 150 g bag of chips ('chips'), lying flat: along x
    'chips_width': 150.0,        # [mm] across
    'chips_thickness': 90.0,     # [mm] in the middle (thinner towards the edges)
    'chips_mass': 150.0,         # [g] the bag and the chips
    'chips_roundness': 0.6,      # every bag of chips: its long sides' thickness, of the middle's (0.3: flat, 0.9: a
                                 # sausage); the sealed ends are always flat (0.3)
    'chips_stiffness': 100.0,    # [kPa] its Young's modulus (the air inside and the foil: how hard it is to squash)
    'chips_poisson': 0.47,       # its Poisson's ratio: 0.5 keeps the volume exactly (above 0.48 gets stiff to solve)
    'potato_count': 6,           # potatoes in the bag (rows of three, two rows, then a layer above)
    'potato_radius': 28.0,       # [mm] each one (a sphere)
    'potato_mass': 100.0,        # [g] each one
    'net_slack': 15.0,           # [%] how much longer the net's strings are than where the potatoes start
    'item_mass_scale': 1.0,      # times each item's mass
    # -- rigid jaws and the suction cups --
    'jaw_stroke': 55.0,          # [mm] each jaw's inner face from the axis, fully open
    'jaw_length': 70.0,          # [mm] below the palm
    'jaw_width': 22.0,           # [mm]
    'jaw_force': 40.0,           # [N] each jaw's grip at 100 kPa
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
    # -- the soft fingers (PneuNet) --
    'fingers': 4,                # how many, evenly spaced around the end-effector's axis
    'finger_radius': 45.0,       # [mm] from the axis to each finger's inner face (its base layer)
    'finger_azimuth': 0.0,       # [deg] where finger 1 hangs (0: +x, 90: +y)
    'segments': 6,               # rigid segments (chambers) in each finger
    'length': 8.0,               # [mm] one segment, along the finger
    'width': 20.0,               # [mm] across it
    'height': 16.0,              # [mm] from the base layer to the roof
    'wall_end': 1.5,             # [mm] end walls (they set the cavity's cross-section and the mass)
    'wall_side': 1.5,
    'wall_bottom': 3.0,
    'wall_top': 1.5,
    'channel_separation': 2.0,   # [mm] gap between neighbouring segments, at the base
    'tips_enabled': True,
    'tip_length': 14.0,          # [mm] beyond the last segment
    'tip_width': 3.0,            # [mm] the point, across the finger
    'tip_thickness': 2.0,        # [mm] the point, from the finger's inner face
    'joint_stiffness': 0.34,     # [N m/rad] torsional stiffness of each hinge
    'joint_damping': 0.01,       # [s] stiffness-proportional damping of the joints
    'joint_open_limit': 90.0,    # [deg] largest bend of a hinge (a hard stop beyond it)
    'side_stiffness': 5.0,       # sideways bending of each joint: its stiffness, times joint_stiffness
    'side_limit': 20.0,          # [deg] largest sideways bend of a joint
    'joint_armature': 2e-6,      # [kg m^2] added inertia of each hinge (MuJoCo: keeps the light segments stable)
    'limit_timeconst': 0.002,    # [s] the hinges' limits (MuJoCo solref)
    'pressure_torque_gain': 1.0, # 1: the torque of the pressure on the cavity's cross-section
    'tcp_depth': 50.0,           # [mm] the tool point below the palm (the point that moves and turns)
    'flange_mass': 0.5,          # [kg] the gripper's plate and wrist adapter
    # -- contacts --
    'friction': 0.8,             # Coulomb friction coefficient of every contact
    'torsional_friction': 0.005, # [m] resistance to twisting about a contact's normal
    'rolling_friction': 0.002,   # [m] round items: resistance to rolling
    'impratio': 10.0,            # MuJoCo: ratio of frictional to normal constraint impedance (higher: less slip)
    'noslip_iterations': 10,     # MuJoCo's no-slip solver passes: without them a held item creeps out (~1 mm/s)
    'finger_collisions': True,   # the fingers touch each other
    # -- material --
    'density': 1000.0,           # [kg/m^3] of the silicone (the segment's mass: the walls' volume)
    'gravity_enabled': True,
    # -- the grip's ramp --
    'ramp_start': 0.0,           # [s] the grip can't start before this
    'ramp_time': 0.4,            # [s] from 0 to the grip
    'dt': 0.02,                  # [s] one frame step (many MuJoCo steps)
    'timestep': 0.001,           # [s] MuJoCo's time step
    'instant': False,
    # -- visuals --
    'wall_alpha': 1.0,
    'joint_alpha': 1.0,
    'item_alpha': 1.0,
    'show_joint_marks': True,
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
    # (the items: Gripper8Mj.item_choices(); regenerate this list when Gripper 7's products or the shelf change)
    ('Item', 'item', [
        ('chips', 'Bag of chips 150 g (deformable, size as set)'),
        ('chips_40', 'Bag of chips (40 g) (deformable)'),
        ('chips_25', 'Bag of chips (25 g) (deformable)'),
        ('chips_15', 'Mini bag of chips (15 g) (deformable)'),
        ('potatoes', 'Bag of potatoes (deformable)'),
        ('g7_apple', 'Gripper 7: Apple'),
        ('g7_tomato', 'Gripper 7: Tomato'),
        ('g7_egg', 'Gripper 7: Egg'),
        ('g7_can', 'Gripper 7: Soda can'),
        ('g7_juice', 'Gripper 7: Juice box'),
        ('g7_spice', 'Gripper 7: Spice jar'),
        ('g7_yogurt', 'Gripper 7: Yogurt cup'),
        ('g7_tea', 'Gripper 7: Tea box'),
        ('g7_butter', 'Gripper 7: Butter'),
        ('g7_pear', 'Gripper 7: Pear'),
        ('g7_banana', 'Gripper 7: Banana'),
        ('hs_dr_vl_melk', 'Hagelslag: De Ruijter chocoladevlokken melk'),
        ('hs_dr_vl_puur', 'Hagelslag: De Ruijter chocoladevlokken puur'),
        ('hs_dr_vl_feest', 'Hagelslag: De Ruijter vlokfeest'),
        ('hs_dr_vl_wit', 'Hagelslag: De Ruijter witte vlokken'),
        ('hs_dr_roy_vl', 'Hagelslag: De Ruijter Royaal chocoladevlokken'),
        ('hs_dr_roy_ha', 'Hagelslag: De Ruijter Royaal hagel'),
        ('hs_dr_spec', 'Hagelslag: De Ruijter Specials intens puur'),
        ('hs_dr_ha_melk', 'Hagelslag: De Ruijter hagel melk'),
        ('hs_dr_ha_puur', 'Hagelslag: De Ruijter hagel puur'),
        ('hs_dr_ha_feest', 'Hagelslag: De Ruijter hagelfeest'),
        ('hs_jb_minis', "Hagelslag: Jumbo Stroo! mini's"),
        ('hs_jb_unicorn', 'Hagelslag: Jumbo Crunchy Unicorn mix'),
        ('hs_jb_chocobal', 'Hagelslag: Jumbo Choco Bal triomix'),
        ('hs_jb_duomix', 'Hagelslag: Jumbo Crunchy Duo mix'),
        ('hs_jb_ha_melk', 'Hagelslag: Jumbo hagelslag melk'),
        ('hs_jb_ha_puur', 'Hagelslag: Jumbo hagelslag puur'),
        ('hs_jb_duo_ha', 'Hagelslag: Jumbo Duo hagelslag mix'),
        ('hs_jb_vrucht', 'Hagelslag: Jumbo vruchtenhagel mix'),
        ('hs_jb_kruid', 'Hagelslag: Jumbo kruidnootjes mini mix'),
        ('hs_jb_ha_small', 'Hagelslag: Jumbo hagelslag klein'),
        ('hs_jb_xxl_puur', 'Hagelslag: Jumbo hagel XXL puur'),
        ('hs_jb_vl_melk', 'Hagelslag: Jumbo vlokken melk'),
        ('hs_jb_vl_puur', 'Hagelslag: Jumbo vlokken puur'),
        ('hs_jb_vl_duo', 'Hagelslag: Jumbo Duo vlokken mix'),
        ('hs_jb_xxl_mix', 'Hagelslag: Jumbo Duo hagel XXL mix'),
        ('hs_jb_xxl_disco', 'Hagelslag: Jumbo hagel XXL melk & disco'),
        ('hs_vz_melk', 'Hagelslag: Venz hagelslag melk'),
        ('hs_vz_puur', 'Hagelslag: Venz hagelslag puur'),
        ('hs_vz_kwink', 'Hagelslag: Venz Kwinkslag'),
        ('hs_vz_xxl_wit', 'Hagelslag: Venz XXL wit'),
        ('hs_vz_xxl_melk', 'Hagelslag: Venz XXL melk'),
        ('hs_vz_beer', 'Hagelslag: Venz Beer hagelslag'),
        ('hs_vz_kroko', 'Hagelslag: Venz Krokovlokken'),
        ('hs_vz_tijger', 'Hagelslag: Venz Tijgervlokken'),
        ('hs_vz_banaan', 'Hagelslag: Venz bananen split'),
        ('hs_vz_aardbei', 'Hagelslag: Venz aardbeien smoothie'),
        ('hs_dr_bos', 'Hagelslag: De Ruijter bosvruchtenhagel'),
        ('hs_dr_anijs', 'Hagelslag: De Ruijter witte hagel'),
        ('hs_dr_muis_roze', 'Hagelslag: De Ruijter muisjes roze'),
        ('hs_dr_muis_blauw', 'Hagelslag: De Ruijter muisjes blauw'),
        ('hs_choca_klein', 'Hagelslag: Choca (smal pak)'),
        ('hs_choca_vl', 'Hagelslag: Choca vlokken'),
        ('hs_choca_mix', 'Hagelslag: Choca vlok mix'),
        ('hs_dr_vrucht', 'Hagelslag: De Ruijter vruchten hagel'),
        ('hs_korrel_l', 'Hagelslag: Chogal boterhamkorrels licht'),
        ('hs_korrel_d', 'Hagelslag: Chogal boterhamkorrels donker'),
        ('hs_matinettes', 'Hagelslag: Jacques Matinettes (stapel)'),
        ('hs_schudde', 'Hagelslag: Bolletje Schuddebuikjes speculaas'),
        ('hs_jb_spec', 'Hagelslag: Jumbo speculoos pasta'),
        ('hs_jb_spec_cr', 'Hagelslag: Jumbo speculoos crunchy'),
        ('hs_biscoff', 'Hagelslag: Lotus Biscoff creamy'),
        ('hs_biscoff_cr', 'Hagelslag: Lotus Biscoff crunchy'),
        ('hs_jb_melk', 'Hagelslag: Jumbo chocopasta melk'),
        ('hs_jb_puur', 'Hagelslag: Jumbo chocopasta puur'),
        ('hs_jb_wit', 'Hagelslag: Jumbo witte chocopasta'),
        ('hs_jb_bio', 'Hagelslag: Jumbo Bio chocopasta hazelnoot'),
        ('hs_jb_hazel', 'Hagelslag: Jumbo hazelnootpasta'),
        ('hs_jb_duopasta', 'Hagelslag: Jumbo Duo pasta hazelnoot & wit'),
        ('hs_nut_750', 'Hagelslag: Nutella 750 g'),
        ('hs_nut_630', 'Hagelslag: Nutella 630 g'),
        ('hs_nut_blue', 'Hagelslag: Nutella (blauwe sticker)'),
        ('hs_nut_cup1', 'Hagelslag: Nutella glas (klein)'),
        ('hs_nut_box', 'Hagelslag: Nutella cups doos (6 porties)'),
        ('hs_nut_400', 'Hagelslag: Nutella 400 g'),
        ('hs_bonne', 'Hagelslag: Bonne Maman hazelnootpasta'),
        ('hs_sv_dark', 'Hagelslag: So Vegan So Fine (donker)'),
        ('hs_sv_wit', 'Hagelslag: So Vegan So Fine (paars)'),
        ('hs_sv_pist', 'Hagelslag: So Vegan So Fine pistache'),
        ('hs_pen_van', 'Hagelslag: Duo Penotti hazelnoot & vanille'),
        ('hs_pen_cream', 'Hagelslag: Duo Penotti (blauwe deksel)'),
        ('hs_pen_spec', 'Hagelslag: Duo Penotti & speculoos'),
        ('hs_hz_spread', 'Hagelslag: Hazelnut spread'),
        ('hs_orinoko', 'Hagelslag: OriNoko'),
        ('hs_orinoko_b', 'Hagelslag: OriNoko (blauwe rand)'),
        ('hs_jb_rinse', 'Hagelslag: Jumbo rinse appelstroop'),
        ('hs_cs_groen', 'Hagelslag: Stroop (groene deksel)'),
        ('hs_cs_rood', 'Hagelslag: Stroop (rode deksel)'),
        ('hs_frutesse', 'Hagelslag: Frutesse'),
        ('hs_timson', 'Hagelslag: Timson rinse appelstroop'),
        ('hs_tin_geel', 'Hagelslag: Rinse appelstroop (gele blik)'),
        ('hs_tin_rood', 'Hagelslag: Stroop (rood blik)'),
        ('hs_kokos_w', 'Hagelslag: Kokosbrood'),
        ('hs_kokos_r', 'Hagelslag: Kokosbrood (rood)'),
        ('hs_pot_swirl', 'Hagelslag: Pot fruitstroop'),
        ('hs_pot_appel', 'Hagelslag: 100% appel'),
        ('hs_pot_dadel', 'Hagelslag: 100% dadel'),
        ('hs_jb_bio_st', 'Hagelslag: Jumbo Bio appelstroop'),
        ('hs_hero', 'Hagelslag: Hero appelstroop'),
        ('hs_marmite', 'Hagelslag: Marmite'),
    ]),
]

# Vectors (the web viewer's panel; params.json is edited by hand in the desktop editor): (label, key, length, decimals, unit)
_VECTOR_FIELDS = []

# (label, json_key, unit_scale [json = displayed * scale], decimals, unit)
_FIELDS = [
    ('Spacing between the grippers (along x)', 'spacing',            1, 0, 'mm'),
    ('Carrying height of the tool point',     'lift_height',         1, 0, 'mm'),
    ('Carried forward (+y)',                  'travel',              1, 0, 'mm'),
    ('Average speed of the moves',            'move_speed',          1, 0, 'mm/s'),
    ('Average speed of the turn about x',     'turn_speed',          1, 1, 'deg/s'),
    ('Grip (pressure; jaws, cups, hand: 80 = full)', 'grip_pressure', 1, 1, 'kPa'),
    ('Wait after gripping, before lifting',   'grip_wait',           1, 2, 's'),
    ('Wait after letting go, before rising',  'release_wait',        1, 2, 's'),
    ('Bag of chips 150 g: length',            'chips_length',        1, 0, 'mm'),
    ('Bag of chips 150 g: width',             'chips_width',         1, 0, 'mm'),
    ('Bag of chips 150 g: thickness in the middle', 'chips_thickness', 1, 0, 'mm'),
    ('Bag of chips 150 g: mass',              'chips_mass',          1, 0, 'g'),
    ('Bags of chips: roundness (side / middle thickness)', 'chips_roundness', 1, 2, ''),
    ('Bag of chips: stiffness (Young\'s modulus)', 'chips_stiffness', 1, 1, 'kPa'),
    ('Bag of chips: Poisson\'s ratio (0.5: constant volume)', 'chips_poisson', 1, 3, ''),
    ('Bag of potatoes: number of potatoes',   'potato_count',        1, 0, ''),
    ('Bag of potatoes: potato radius',        'potato_radius',       1, 1, 'mm'),
    ('Bag of potatoes: potato mass',          'potato_mass',         1, 0, 'g'),
    ('Bag of potatoes: slack of the net',     'net_slack',           1, 1, '%'),
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
    ('Gripper plate and adapter mass',         'flange_mass',         1, 3, 'kg'),
    ('Item mass (times its own)',   'item_mass_scale',     1, 2, ''),
    ('Friction coefficient',                   'friction',            1, 2, ''),
    ('Torsional friction',                     'torsional_friction',  1, 4, 'm'),
    ('Rolling friction (round items)',      'rolling_friction',    1, 4, 'm'),
    ('Friction impedance ratio (impratio)',    'impratio',            1, 1, ''),
    ('No-slip solver iterations (0: off)',     'noslip_iterations',   1, 0, ''),
    ('Silicone density',                       'density',             1, 1, 'kg/m3'),
    ('Grip ramp: earliest start',                    'ramp_start',          1, 2, 's'),
    ('Grip ramp time',                     'ramp_time',           1, 2, 's'),
    ('Frame step',                             'dt',                  1, 4, 's'),
    ('MuJoCo time step',                       'timestep',            1, 5, 's'),
    ('Segment opacity',                        'wall_alpha',          1, 2, ''),
    ('Joint mark opacity',                     'joint_alpha',         1, 2, ''),
    ('Item opacity',                        'item_alpha',          1, 2, ''),
]

_BOOL_FIELDS = [
    ('Tactile skin on the fingers (Touchlab / TG0 style)', 'tactile_skin', True),
    ('Pointy fingertips', 'tips_enabled', True),
    ('Apply pressure instantly (not gradual)', 'instant', False),
    ('Enable gravity', 'gravity_enabled', True),
    ('Fingers touch each other', 'finger_collisions', True),
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
        return (f"Every gripper in a row (soft fingers: {f} of {n} segments, {length:.1f} mm long{tip}), "
                f"each over a {params['item']}")
    except Exception as e:
        return f"(not computed: {e})"

class ParamsEditor(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Gripper 8 (MuJoCo) parameters")
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
