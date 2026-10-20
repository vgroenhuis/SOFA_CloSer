# -*- coding: utf-8 -*-
"""
Standalone parameter editor for Gripper4.py.

Runs as its own window/process (Tkinter, standard library, no SOFA needed),
reading and writing params.json in the same directory. Every change
auto-saves (text fields on focus-out/Enter, checkboxes immediately). It
does not talk to a running SOFA session: geometry, mesh and material
changes take effect at the next scene (re)build -- use the SOFA GUI's
Reload button. Target pressure, ramp time and instant can also be edited
live in SOFA's Scene Graph panel ("PressureRamp" node).

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
    # -- the fingers --
    'fingers': 4,                # how many, evenly spaced around the end-effector's vertical axis
    'finger_radius': 45.0,       # [mm] from the axis to each finger's inner face (its base layer)
    'finger_azimuth': 45.0,      # [deg] where finger 1 hangs (0: +x, 90: +y, the back in the default view)
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
    'floor_size': 600.0,         # [mm] the floor (it stops the items and the fingers)
    # -- joints --
    'joint_stiffness': 0.34,     # [N m/rad] torsional stiffness of each hinge
    'joint_damping': 0.01,       # [s] stiffness-proportional damping of the joints
    'joint_open_limit': 90.0,    # [deg] largest bend of a hinge (a hard stop beyond it)
    'joint_stop_stiffness': 5.0, # [N m/rad] the hard stops
    'pressure_torque_gain': 1.0, # 1: the torque of the pressure on the cavity's cross-section (see the docstring)
    'flange_acceleration': 0.5,       # [m/s2] largest acceleration of the end-effector
    'flange_speed': 0.25,             # [m/s] its largest speed
    'flange_mass': 0.5,               # [kg] the end-effector (it follows the target on the wrist spring)
    'wrist_stiffness': 3000.0,        # [N/m] the spring between the target and the end-effector (critically damped)
    # -- the items on the floor, and the contacts --
    'item_apple': True, 'item_tomato': True, 'item_egg': True, 'item_can': True, 'item_juice': True,
    'item_spice': True, 'item_yogurt': True, 'item_tea': True, 'item_butter': True,
    'item_mass_scale': 1.0,      # times each item's mass (see ITEMS)
    'item_spacing': 1.0,         # times the distances between the items (see ITEMS)
    'item_yaw': 45.0,            # [deg] the items are turned by this about the vertical (to face the fingers)
    'friction': 0.6,             # Coulomb friction coefficient of every contact
    'contact_distance': 0.5,     # [mm] contacts are kept this far apart
    'alarm_distance': 5.0,       # [mm] pairs closer than this are watched for contact
    'item_alpha': 1.0,
    'bin_enabled': True,
    'bin_x': 0.0, 'bin_y': 290.0,
    'bin_length': 220.0,
    'bin_width': 130.0,
    'bin_height': 70.0,
    'bin_wall': 8.0,
    'bin_alpha': 1.0,
    # -- material --
    'density': 1000.0,           # [kg/m^3] of the silicone (the segment's mass: the walls' volume)
    'gravity_enabled': True,
    # -- pressure --
    'ramp_start': 0.0,           # [s] when the pressure starts to rise
    'ramp_time': 0.5,            # [s]
    'dt': 0.02,                  # [s] simulation time step
    'instant': False,
    # -- visuals --
    'wall_alpha': 1.0,
    'joint_alpha': 1.0,
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

# Vectors (the web viewer's panel; params.json is edited by hand in the desktop editor): (label, key, length, decimals, unit)
_VECTOR_FIELDS = []

# (label, json_key, unit_scale [json = displayed * scale], decimals, unit)
_FIELDS = [
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
    ('Floor size',                             'floor_size',          1, 1, 'mm'),
    ('Joint torsional stiffness',              'joint_stiffness',     1, 4, 'N m/rad'),
    ('Joint damping (stiffness-proportional)', 'joint_damping',       1, 4, 's'),
    ('Joint open limit',                       'joint_open_limit',    1, 1, 'deg'),
    ('Joint stop stiffness',                   'joint_stop_stiffness', 1, 3, 'N m/rad'),
    ('Pressure torque gain',                   'pressure_torque_gain', 1, 3, ''),
    ('Flange acceleration',                    'flange_acceleration', 1, 3, 'm/s2'),
    ('Flange speed',                           'flange_speed',        1, 3, 'm/s'),
    ('End-effector mass',                      'flange_mass',         1, 3, 'kg'),
    ('Wrist stiffness (end-effector on its target)', 'wrist_stiffness', 1, 0, 'N/m'),
    ('Item masses (times the real ones)',      'item_mass_scale',     1, 2, ''),
    ('Item spacing (times the default)',       'item_spacing',        1, 2, ''),
    ('Items turned about the vertical',        'item_yaw',            1, 1, 'deg'),
    ('Friction coefficient',                   'friction',            1, 2, ''),
    ('Contact distance',                       'contact_distance',    1, 2, 'mm'),
    ('Contact watch distance',                 'alarm_distance',      1, 2, 'mm'),
    ('Item opacity',                           'item_alpha',          1, 2, ''),
    ('Large box: centre x',                    'bin_x',               1, 1, 'mm'),
    ('Large box: centre y',                    'bin_y',               1, 1, 'mm'),
    ('Large box: inside length (x)',           'bin_length',          1, 1, 'mm'),
    ('Large box: inside width (y)',            'bin_width',           1, 1, 'mm'),
    ('Large box: wall height',                 'bin_height',          1, 1, 'mm'),
    ('Large box: wall thickness',              'bin_wall',            1, 1, 'mm'),
    ('Large box opacity',                      'bin_alpha',           1, 2, ''),
    ('Silicone density',                       'density',             1, 1, 'kg/m3'),
    ('Pressure ramp start',                    'ramp_start',          1, 2, 's'),
    ('Pressure ramp time',                     'ramp_time',           1, 2, 's'),
    ('Time step',                              'dt',                  1, 4, 's'),
    ('Segment opacity',                        'wall_alpha',          1, 2, ''),
    ('Joint mark opacity',                     'joint_alpha',         1, 2, ''),
]

_BOOL_FIELDS = [
    ('Apple (170 g)', 'item_apple', True),
    ('Tomato (100 g)', 'item_tomato', True),
    ('Egg (60 g)', 'item_egg', True),
    ('Soda can (380 g)', 'item_can', True),
    ('Juice box (220 g)', 'item_juice', True),
    ('Spice jar (120 g)', 'item_spice', True),
    ('Yogurt cup (150 g)', 'item_yogurt', True),
    ('Tea box (90 g)', 'item_tea', True),
    ('Butter (250 g)', 'item_butter', True),
    ('A large open box on the floor', 'bin_enabled', True),
    ('Apply pressure instantly (not gradual)', 'instant', False),
    ('Enable gravity', 'gravity_enabled', True),
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
        items = sum(1 for k, v in params.items() if k.startswith('item_') and v is True)
        return (f"Rigid model: {f} fingers of {n} segments, {length:.1f} mm long ({n - 1} hinges each); "
                f"{items} items on the floor")
    except Exception as e:
        return f"(not computed: {e})"

class ParamsEditor(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Gripper 4 parameters")
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
