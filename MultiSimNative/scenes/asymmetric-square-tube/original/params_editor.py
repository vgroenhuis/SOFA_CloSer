# -*- coding: utf-8 -*-
"""
Standalone parameter editor for AsymmetricSquareTube.py.

Runs as its own window/process (Tkinter, part of the standard library --
no SOFA dependency, works with any Python install), reading and writing
params.json in the same directory. Every change auto-saves (text fields
on focus-out/Enter, checkboxes immediately) -- there's no Save button.
It does NOT talk to a running SOFA session directly: every parameter here
only takes effect on the next scene (re)build -- after a change, use the
SOFA GUI's Reload button, or just re-run for a headless session.

Same layout/theming machinery as ../DoublePendulum/params_editor.py
(scrollable body, Light/Dark/System theme, Windows dark-titlebar) --
copied rather than imported, so this project has no dependency on that
one; only the field list and the derived "wall thickness" readout are
specific to this tube.

Run:
    python params_editor.py
"""
import json
import os
import tkinter as tk
from tkinter import ttk

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARAMS_FILE = os.path.join(_SCRIPT_DIR, 'params.json')

_DEFAULTS = {
    'outer_width': 0.1,
    'outer_height': 0.1,
    'divs_x': 10,
    'divs_z': 10,
    'depth': 0.03,
    'young_modulus': 1.0e6,
    'poisson_ratio': 0.45,
    'density': 1100.0,
    'cavity_width': 0.02,
    'cavity_height': 0.02,
    'cavity_center_x': 0.06,
    'cavity_center_z': 0.04,
    'top_pressure': 1.0e5,
    'top_ramp_start': 0.0,
    'top_ramp_time': 0.5,
    'cavity_pressure': 1.0e6,
    'cavity_ramp_start': 1.0,
    'cavity_ramp_time': 5.0,
    'gravity': 9.81,
    'gravity_enabled': True,
    'dt': 0.01,
    'run_duration': 9.0,
    'rayleigh_stiffness': 0.1,
    'rayleigh_mass': 0.1,
    'console_log_enabled': True,
    'block_fill_color': [0.85, 0.55, 0.65],
    'block_alpha': 0.95,
    'block_show_edges': True,
    'block_edge_color': [0.25, 0.12, 0.18],
    'show_platen': True,
    'ambient_color': [0.65, 0.65, 0.65],
    'key_light_direction': [0.3, -0.6, 0.7],
    'key_light_color': [1.0, 1.0, 1.0],
    'fill_light_direction': [-0.3, 0.6, 0.0],
    'fill_light_color': [0.55, 0.55, 0.55],
    'key_light_rotate_enabled': False,
    'key_light_rotate_period': 8.0,
    'show_fem_elements': False,
    'theme': 'System',   # 'Light' / 'Dark' / 'System' -- UI-only preference,
                          # stored here for convenience but never read by
                          # AsymmetricSquareTube.py itself.
}

_PALETTES = {
    'Light': dict(bg='#f0f0f0', fg='#000000', entry_bg='#ffffff', entry_fg='#000000',
                  select_bg='#0078d7', select_fg='#ffffff', btn_bg='#e1e1e1',
                  muted_fg='#666666', error_fg='#cc0000', warn_fg='#a06000', border='#adadad'),
    'Dark': dict(bg='#1e1e1e', fg='#f5f5f5', entry_bg='#333333', entry_fg='#ffffff',
                 select_bg='#0a84c0', select_fg='#ffffff', btn_bg='#3a3a3a',
                 muted_fg='#b8b8b8', error_fg='#ff8080', warn_fg='#e0a030', border='#5a5a5a'),
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
    ('Outer width (X)',               'outer_width',        1,    3, 'm'),
    ('Outer height (Z)',              'outer_height',       1,    3, 'm'),
    ('Divisions (X)',                 'divs_x',             1,    0, 'cells'),
    ('Divisions (Z)',                 'divs_z',             1,    0, 'cells'),
    ('Depth (Y extent)',              'depth',              1,    3, 'm'),
    ("Young's modulus",               'young_modulus',      1e6,  3, 'MPa'),
    ("Poisson's ratio",               'poisson_ratio',      1,    3, ''),
    ('Density',                       'density',            1,    1, 'kg/m^3'),
    ('Cavity width',                  'cavity_width',       1,    3, 'm'),
    ('Cavity height',                 'cavity_height',      1,    3, 'm'),
    ('Cavity center X',               'cavity_center_x',    1,    3, 'm'),
    ('Cavity center Z',               'cavity_center_z',    1,    3, 'm'),
    ('Top pressure',                  'top_pressure',       1e5,  2, 'bar'),
    ('Top pressure ramp start',       'top_ramp_start',     1,    2, 's'),
    ('Top pressure ramp time',        'top_ramp_time',      1,    2, 's'),
    ('Cavity pressure',               'cavity_pressure',    1e5,  2, 'bar'),
    ('Cavity pressure ramp start',    'cavity_ramp_start',  1,    2, 's'),
    ('Cavity pressure ramp time',     'cavity_ramp_time',   1,    2, 's'),
    ('Gravity',                       'gravity',            1,    2, 'm/s^2'),
    ('Timestep',                      'dt',                 1e-3, 2, 'ms'),
    ('Run duration (headless)',       'run_duration',       1,    1, 's'),
    ('Rayleigh stiffness damping',    'rayleigh_stiffness', 1,    3, ''),
    ('Rayleigh mass damping',         'rayleigh_mass',      1,    3, ''),
    ('Block transparency (alpha)',    'block_alpha',        1,    3, '0=clear, 1=opaque'),
    ('Key light rotation period',     'key_light_rotate_period', 1, 2, 's per revolution'),
]

# (label, json_key, length, decimals, unit) -- multi-component properties
# (colors, directions), edited as one comma-separated field ("0.85, 0.55,
# 0.65") instead of one field per component. `length` is how many values
# are expected; save() rejects (without saving) a count that doesn't match
# rather than silently padding/truncating.
_VECTOR_FIELDS = [
    ('Block fill color (R,G,B)',           'block_fill_color',     3, 3, '0-1 each'),
    ('Block edge color (R,G,B)',           'block_edge_color',     3, 3, '0-1 each'),
    ('Ambient light color (R,G,B)',        'ambient_color',        3, 3, '0-1 each'),
    ('Key light direction (X,Y,Z)',        'key_light_direction',  3, 3, ''),
    ('Key light color (R,G,B)',            'key_light_color',      3, 3, '0-1 each'),
    ('Fill light direction (X,Y,Z)',       'fill_light_direction', 3, 3, ''),
    ('Fill light color (R,G,B)',           'fill_light_color',     3, 3, '0-1 each'),
]

# (label, json_key, default) -- simple checkboxes
_BOOL_FIELDS = [
    ('Enable gravity', 'gravity_enabled', True),
    ('Print per-step platen/cavity diagnostics to the console', 'console_log_enabled', True),
    ('Show block edge/outline overlay', 'block_show_edges', True),
    ('Show platen (top pressing plate)', 'show_platen', True),
    ('Show raw FEM tetrahedra (debug)', 'show_fem_elements', False),
    ('Rotate key light around Z (build/inspection aid)', 'key_light_rotate_enabled', False),
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


def _format_number(value, decimals):
    """Rounds `value` to `decimals` places and formats it WITHOUT padding
    with trailing zeros -- 0.5 at 3 decimals displays as "0.5", not
    "0.500". Trailing zeros the rounding itself didn't need just make a
    field's actual precision harder to read at a glance, and re-typing
    over them (or noticing they're not actually there to overwrite) makes
    editing a comma-separated field more fiddly than it needs to be.
    """
    s = f"{round(value, decimals):.{decimals}f}"
    if '.' in s:
        s = s.rstrip('0').rstrip('.')
    return s if s not in ('', '-', '-0') else '0'


def _wall_thicknesses(params):
    """Distance from each cavity edge to the nearest outer edge -- the
    same 4 numbers AsymmetricSquareTube.py's own snap/clamp logic works
    from, surfaced here as a direct sanity check: any of these at or
    below 0 means the requested cavity doesn't fit (it'll get clamped to
    something that does when the scene builds, but not necessarily what
    was intended), and how unequal they are is exactly the asymmetry the
    project is named for.
    """
    width, height = params['outer_width'], params['outer_height']
    cx, cz = params['cavity_center_x'], params['cavity_center_z']
    hw, hh = params['cavity_width'] / 2.0, params['cavity_height'] / 2.0
    return {
        'left': cx - hw,
        'right': width - (cx + hw),
        'bottom': cz - hh,
        'top': height - (cz + hh),
    }


class ParamsEditor(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AsymmetricSquareTube parameters")
        self.resizable(True, True)
        self.attributes('-topmost', True)  # stay visible over the SOFA GUI

        params = load_params()
        self.vars = {}
        self.style = ttk.Style(self)

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

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

        self.vector_vars = {}
        for label, key, length, decimals, unit in _VECTOR_FIELDS:
            text = f"{label} [{unit}]" if unit else label
            field_label = ttk.Label(frame, text=text)
            field_label.grid(row=row, column=0, columnspan=2, sticky='w', pady=(4, 0))
            self._wrap_full_widgets.append(field_label)
            row += 1
            display_value = ', '.join(_format_number(v, decimals) for v in params[key])
            var = tk.StringVar(value=display_value)
            entry = ttk.Entry(frame, textvariable=var)
            entry.grid(row=row, column=0, columnspan=2, sticky='we', pady=(0, 2))
            entry.bind('<FocusOut>', lambda e: self.save())
            entry.bind('<Return>', lambda e: self.save())
            self.vector_vars[key] = (var, length, decimals)
            row += 1

        self.bool_vars = {}
        self._checkbuttons = []
        for label, key, _default in _BOOL_FIELDS:
            var = tk.BooleanVar(value=bool(params[key]))
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

        note = ("Every change auto-saves. All parameters take effect on the next scene "
                "(re)build -- after a change, use the SOFA GUI's Reload button, or just "
                "re-run for a headless session. A cavity that doesn't fit is snapped/"
                "clamped to one that does when the scene builds, not rejected here.")
        self.note_label = ttk.Label(frame, text=note, justify='left', font=('Segoe UI', 8),
                                     wraplength=280)
        self.note_label.grid(row=row + 1, column=0, columnspan=2,
                              sticky='we', pady=(6, 0))
        self._wrap_full_widgets.append(self.note_label)

        self._apply_theme()
        self.geometry("420x620")
        self.minsize(220, 220)

        self.update()
        self._update_wraplengths(canvas.winfo_width())

    @staticmethod
    def _to_display(json_value, scale, decimals):
        value = json_value / scale
        return int(round(value)) if decimals == 0 else _format_number(value, decimals)

    def _update_stats(self, params):
        walls = _wall_thicknesses(params)
        bad = [name for name, t in walls.items() if t <= 0]
        text = (f"Wall thickness -- left: {walls['left'] * 1000:.1f} mm, "
                f"right: {walls['right'] * 1000:.1f} mm, "
                f"bottom: {walls['bottom'] * 1000:.1f} mm, "
                f"top: {walls['top'] * 1000:.1f} mm")
        color = None
        if bad:
            text += f"  -- {', '.join(bad)} <= 0, cavity will be clamped to fit"
            color = self._palette['warn_fg'] if hasattr(self, '_palette') else None
        self.stats_label.config(text=text)
        if color:
            self.stats_label.configure(foreground=color)
        elif hasattr(self, '_palette'):
            self.stats_label.configure(foreground=self._palette['muted_fg'])

    def _apply_theme(self):
        choice = self.theme_var.get()
        dark = (choice == 'Dark') or (choice == 'System' and _system_prefers_dark())
        pal = _PALETTES['Dark' if dark else 'Light']
        self._palette = pal

        style = self.style
        style.theme_use('clam')
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
        self.option_add('*TCombobox*Listbox.background', pal['entry_bg'])
        self.option_add('*TCombobox*Listbox.foreground', pal['entry_fg'])
        self.option_add('*TCombobox*Listbox.selectBackground', pal['select_bg'])
        self.option_add('*TCombobox*Listbox.selectForeground', pal['select_fg'])

        self.configure(bg=pal['bg'])
        self._canvas.configure(bg=pal['bg'])
        for cb in self._checkbuttons:
            cb.configure(bg=pal['bg'], fg=pal['fg'], selectcolor=pal['entry_bg'],
                         activebackground=pal['bg'], activeforeground=pal['fg'])
        self.status.configure(foreground=pal['muted_fg'])
        self.note_label.configure(foreground=pal['muted_fg'])
        _set_titlebar_dark(self, dark)

    def save(self):
        params = dict(_DEFAULTS)
        try:
            for key, (var, scale, decimals) in self.vars.items():
                value = float(var.get()) * scale
                params[key] = int(round(value)) if decimals == 0 else value
            for key, (var, length, decimals) in self.vector_vars.items():
                raw = var.get().split(',')
                if len(raw) != length:
                    raise ValueError(f"{key} needs {length} comma-separated values, got {len(raw)}")
                params[key] = [float(x.strip()) for x in raw]
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

        self.status.config(text="Saved -- press Reload in the SOFA GUI to apply",
                            foreground=self._palette['muted_fg'])


if __name__ == '__main__':
    ParamsEditor().mainloop()
