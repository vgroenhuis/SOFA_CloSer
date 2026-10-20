# -*- coding: utf-8 -*-
"""
Standalone parameter editor for Finger6.py.

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
    # -- geometry [mm] --
    'segments': 7,               # chambers in the finger
    'element_size': 2.0,         # standard element edge; each feature is split into
                                 # the nearest whole number of equal elements
    'min_wall_layers': 1,        # ... but every wall into at least this many layers (1: single-wall
                                 # chambers, fast; 2 bends more realistically but takes 3-4x as long)
    'length': 8.0,              # one chamber, outer extent along X (the finger's axis)
    'width': 20.0,              # along Y
    'height': 16.0,              # along Z (up)
    'wall_end': 1.5,             # end walls (perpendicular to X), facing the neighbours
    'wall_side': 1.5,            # side walls (perpendicular to Y)
    'wall_bottom': 3.0,          # bottom layer
    'wall_top': 1.5,             # roof
    'channel_height': 2.0,       # connecting channel, from the cavity floor up
    'channel_width': 2.0,        # 0 = the full cavity width
    'channel_wall': 1.5,         # duct roof thickness (the duct spans the full width)
    'channel_separation': 2.0,   # gap between neighbouring chambers, i.e. the duct length
    'trunk_length': 8.0,        # clamped footprint under the first chamber (centred)
    'trunk_width': 20.0,
    'floor_size': 120.0,         # green floor, purely visual
    'base_height': 60.0,         # [mm] the finger's base above the floor (z = 0): high enough that the curling finger clears it
    # -- boundary conditions --
    'clamp_proximal_end': True,       # the first chamber's outer end face is held (the mount)
    'clamp_to_trunk': False,          # the first chamber's base clamped to a rigid trunk
    'symmetry_y': True,
    'prevent_symmetry_crossing': True,
    'collision_planes': True,         # a collision plane between each pair of neighbouring chambers
    # -- rigid rod below the finger: axis along Y, centre at (x, z) [mm], and its contact model --
    'rod_enabled': True,
    'rod_x': 45.0,                    # [mm] along the finger (world X)
    'rod_z': 41.0,                    # [mm] height of the axis above the floor (world Z)
    'rod_radius': 10.0,               # [mm]
    'rod_contact_stiffness': 1.0e5,   # [N/m] per contact node
    'rod_contact_whole_skin': False,  # every node of the outer skin can touch it (False: the underside only)
    'contact_outer_face_only': True,  # ... checked for the end walls' outer faces only (faster)
    'plane_hold_stiffness': 1.0e4,    # [N/m] per duct node, holding each plane to the duct
    'strain_limiting_layer': True,    # inextensible sheet under the base (see the docstring)
    'strain_layer_thickness': 0.1,    # [mm] paper
    'strain_layer_young_modulus': 6.5e9,   # [Pa] paper
    'strain_layer_poisson_ratio': 0.2,
    'strain_layer_density': 750.0,    # [kg/m^3] paper
    'strain_layer_tension_only': True,  # wrinkles rather than carrying compression, like paper
    # -- material --
    'hyperelastic': False,       # neo-Hookean instead of linear corotational
    'young_modulus': 1.0e6,      # [Pa]
    'poisson_ratio': 0.45,
    'density': 1000.0,           # [kg/m^3]
    'gravity_enabled': False,
    # -- pressure --
    'target_pressure': 50000.0,  # [Pa] (30 kPa: a typical PneuNet pressure)
    'ramp_start': 0.0,           # [s] when the pressure starts to rise (e.g. after gravity has settled)
    'ramp_time': 0.5,            # [s]
    'dt': 0.01,                  # [s] simulation time step
    'instant': False,
    # -- visuals --
    'cutaway': False,
    'interior_cutaway': False,
    'wall_alpha': 1.0,
    'interior_alpha': 1.0,
    'channel_end_alpha': 0.5,    # the collision planes
    'strain_layer_alpha': 1.0,
    'rod_alpha': 1.0,
    'mirror_visual_y': True,
    'show_pressure_overlay': False,
    'show_fem_elements': False,
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

# (label, json_key, unit_scale [json = displayed * scale], decimals, unit)
_FIELDS = [
    ('Chambers (segments)',        'segments',         1, 0, ''),
    ('Element size (standard edge)', 'element_size',   1, 2, 'mm'),
    ('Minimum wall layers',        'min_wall_layers',  1, 0, 'elements'),
    ('Chamber length (X)',         'length',           1, 2, 'mm'),
    ('Width (Y)',                  'width',            1, 2, 'mm'),
    ('Height (Z)',                 'height',           1, 2, 'mm'),
    ('End wall thickness (X)',     'wall_end',         1, 2, 'mm'),
    ('Side wall thickness (Y)',    'wall_side',        1, 2, 'mm'),
    ('Bottom layer thickness',     'wall_bottom',      1, 2, 'mm'),
    ('Roof thickness',             'wall_top',         1, 2, 'mm'),
    ('Channel height',             'channel_height',   1, 2, 'mm'),
    ('Channel width (0 = full cavity width)', 'channel_width', 1, 2, 'mm'),
    ('Channel duct roof thickness', 'channel_wall',    1, 2, 'mm'),
    ('Channel separation (gap between chambers)', 'channel_separation', 1, 2, 'mm'),
    ('Trunk length (X, under the first chamber)', 'trunk_length', 1, 2, 'mm'),
    ('Trunk width (Y)',            'trunk_width',      1, 2, 'mm'),
    ('Finger base height above the floor (z = 0)', 'base_height', 1, 2, 'mm'),
    ('Floor size',                 'floor_size',       1, 1, 'mm'),
    ('Collision plane holding stiffness (per duct node)', 'plane_hold_stiffness', 1, 1, 'N/m'),
    ('Strain-limiting layer thickness', 'strain_layer_thickness', 1, 3, 'mm'),
    ("Strain-limiting layer Young's modulus", 'strain_layer_young_modulus', 1e6, 1, 'MPa'),
    ('Strain-limiting layer Poisson ratio', 'strain_layer_poisson_ratio', 1, 3, ''),
    ('Strain-limiting layer density', 'strain_layer_density', 1, 1, 'kg/m^3'),
    ("Young's modulus",  'young_modulus',   1e3,  1, 'kPa'),
    ('Poisson ratio',    'poisson_ratio',   1,    3, ''),
    ('Density',          'density',         1,    1, 'kg/m^3'),
    ('Target pressure',  'target_pressure', 1e3,  2, 'kPa'),
    ('Ramp start (pressure starts rising at)', 'ramp_start', 1, 2, 's'),
    ('Ramp time',        'ramp_time',       1,    2, 's'),
    ('Time step',        'dt',              1e-3, 2, 'ms'),
    ('Wall opacity',     'wall_alpha',      1,    2, '0-1'),
    ('Interior opacity', 'interior_alpha',  1,    2, '0-1'),
    ('Collision plane opacity', 'channel_end_alpha', 1, 2, '0-1'),
    ('Rod opacity', 'rod_alpha', 1, 2, '0-1'),
    ('Rod x (axis along Y)', 'rod_x', 1, 2, 'mm'),
    ('Rod z (height of the axis)', 'rod_z', 1, 2, 'mm'),
    ('Rod radius', 'rod_radius', 1, 2, 'mm'),
    ('Rod contact stiffness (per node)', 'rod_contact_stiffness', 1, 0, 'N/m'),
    ('Strain-limiting layer opacity', 'strain_layer_alpha', 1, 2, '0-1'),
]

# (label, json_key, default) -- simple checkboxes
_BOOL_FIELDS = [
    ('Apply pressure instantly (not gradual)', 'instant', False),
    ('Enable gravity', 'gravity_enabled', False),
    ('Neo-Hookean hyperelastic material (instead of linear corotational)', 'hyperelastic', False),
    ('Exploit Y symmetry (model the +Y half only)', 'symmetry_y', True),
    ('Prevent nodes crossing the Y midplane (one-sided stop)', 'prevent_symmetry_crossing', True),
    ('Mirror visual model across Y symmetry plane', 'mirror_visual_y', True),
    ('Collision plane between neighbouring chambers (end walls touch it)', 'collision_planes', True),
    ('Rigid rod below the finger', 'rod_enabled', True),
    ('Rod contact: every node of the outer skin (off: the underside only, faster)', 'rod_contact_whole_skin', False),
    ('... contact checked for the end walls\' outer faces only (fewer nodes: faster)', 'contact_outer_face_only', True),
    ('Clamp the proximal end face (the mount)', 'clamp_proximal_end', True),
    ('Clamp the first chamber\'s bottom to a rigid trunk', 'clamp_to_trunk', False),
    ('Strain-limiting layer under the base', 'strain_limiting_layer', True),
    ('Strain-limiting layer carries no compression (wrinkles, like paper)', 'strain_layer_tension_only', True),
    ('Cutaway window in outer wall (see into the cavity)', 'cutaway', False),
    ('Cutaway window in interior volume (same side)', 'interior_cutaway', False),
    ('Show pressure indicator (red/green, opaque, no alpha control)', 'show_pressure_overlay', False),
    ('Show FEM elements (SOFA\'s "Show Force Fields" flag)', 'show_fem_elements', False),
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
    """Element count and grid summary, from the scene's own mesher
    (finger6_mesh.py, pure Python)."""
    try:
        import sys
        if _SCRIPT_DIR not in sys.path:
            sys.path.insert(0, _SCRIPT_DIR)
        import finger6_mesh
        tets, nodes, geometry = finger6_mesh.count_elements(params)
        return f"FEM: {tets:,} tetrahedra, {nodes:,} nodes; {geometry.summary()}"
    except Exception as e:  # keep the editor usable whatever the input
        return f"(mesh not computed: {e})"


class ParamsEditor(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Finger 6 parameters")
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
