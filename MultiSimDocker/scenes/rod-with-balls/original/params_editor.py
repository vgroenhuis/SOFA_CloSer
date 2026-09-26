# -*- coding: utf-8 -*-
"""
Standalone parameter editor for RodWithBalls.py.

Runs as its own window/process (Tkinter, part of the standard library --
no SOFA dependency, works with any Python install), reading and writing
params.json in the same directory. Every change auto-saves (text fields
on focus-out/Enter, checkboxes immediately) -- there's no Save button.
It does NOT talk to a running SOFA session directly: every parameter here
(geometry, masses, initial conditions, the spring, and the flags) only
takes effect on the next scene (re)build -- after a change, use the SOFA
GUI's Reload button, or just re-run for a headless session.

Same layout/theming machinery as ../DoublePendulum/params_editor.py
(scrollable body, Light/Dark/System theme, Windows dark-titlebar) --
copied rather than imported, so this project has no dependency on that
one; only the field list and the derived "wall clearance"-style readout
(here: each ball's own initial r) are specific to this project.

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
    'anchor_x': 0.0,
    'anchor_z': 1.0,
    'theta_initial_deg': 90.0,
    'omega_initial': 0.0,
    'ball1_mass_kg': 1.0,
    'ball1_initial_x': -0.5,
    'ball1_initial_z': 1.0,
    'ball2_mass_kg': 1.0,
    'ball2_initial_x': 0.6,
    'ball2_initial_z': 1.0,
    'rod_visual_half_length': 2.0,
    'spring_anchor_x': 0.6,
    'spring_anchor_z': 2.0,
    'spring_rest_length': 1.0,
    'spring_stiffness': 200.0,
    'gravity': 9.81,
    'dt': 0.001,
    'run_duration': 15.0,
    'console_log_enabled': True,
    'show_pivot_marker': True,
    'show_spring_anchor_marker': True,
    'theme': 'System',   # 'Light' / 'Dark' / 'System' -- UI-only preference,
                          # stored here for convenience but never read by
                          # RodWithBalls.py itself.
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
    ('Hinge X',                       'anchor_x',                1,    3, 'm'),
    ('Hinge height (Z)',              'anchor_z',                1,    3, 'm'),
    ('Initial rod angle',             'theta_initial_deg',       1,    1, 'deg, 90=horizontal'),
    ('Initial angular velocity',      'omega_initial',           1,    3, 'rad/s'),
    ('Ball 1 mass',                   'ball1_mass_kg',           1,    3, 'kg'),
    ('Ball 1 initial X',              'ball1_initial_x',         1,    3, 'm'),
    ('Ball 1 initial Z',              'ball1_initial_z',         1,    3, 'm'),
    ('Ball 2 mass',                   'ball2_mass_kg',           1,    3, 'kg'),
    ('Ball 2 initial X',              'ball2_initial_x',         1,    3, 'm'),
    ('Ball 2 initial Z',              'ball2_initial_z',         1,    3, 'm'),
    ('Rod visual half-length',        'rod_visual_half_length',  1,    2, 'm'),
    ('Spring anchor X',               'spring_anchor_x',         1,    3, 'm'),
    ('Spring anchor Z',               'spring_anchor_z',         1,    3, 'm'),
    ('Spring rest length',            'spring_rest_length',      1,    3, 'm'),
    ('Spring stiffness',              'spring_stiffness',        1,    1, 'N/m'),
    ('Gravity',                       'gravity',                 1,    3, 'm/s^2'),
    ('Timestep',                      'dt',                      1e-3, 3, 'ms'),
    ('Run duration (headless)',       'run_duration',            1,    1, 's'),
]

# (label, json_key, default) -- simple checkboxes
_BOOL_FIELDS = [
    ('Print per-step energy to the console', 'console_log_enabled', True),
    ('Show hinge marker', 'show_pivot_marker', True),
    ('Show spring anchor marker', 'show_spring_anchor_marker', True),
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


def _initial_radii(params):
    """Each ball's own signed distance from the hinge at the configured
    initial condition -- the same r1/r2 RodWithBalls.py itself derives
    (see its own module docstring), surfaced here as a direct sanity
    check: a huge |r| usually means theta_initial_deg is close to 0/180
    (sin(theta) in the denominator), not a mistaken ball position.
    """
    theta = math.radians(params['theta_initial_deg'])
    sin_t = math.sin(theta)
    r1 = (params['ball1_initial_x'] - params['anchor_x']) / sin_t
    r2 = (params['ball2_initial_x'] - params['anchor_x']) / sin_t
    return r1, r2


class ParamsEditor(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("RodWithBalls parameters")
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

        self.bool_vars = {}
        self._checkbuttons = []
        for label, key, _default in _BOOL_FIELDS:
            var = tk.BooleanVar(value=bool(params[key]))
            # Plain tk.Checkbutton, not ttk: ttk.Checkbutton has no
            # wraplength option at all, and styled by hand in
            # _apply_theme() since it won't pick up the ttk style the
            # way ttk.Checkbutton would (see ../PneuNetFinger's editor).
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
                "re-run for a headless session. There's nothing here that can be edited "
                "live on an already-running simulation.")
        self.note_label = ttk.Label(frame, text=note, justify='left', font=('Segoe UI', 8),
                                     wraplength=280)
        self.note_label.grid(row=row + 1, column=0, columnspan=2,
                              sticky='we', pady=(6, 0))
        self._wrap_full_widgets.append(self.note_label)

        self._apply_theme()
        self.geometry("420x600")
        self.minsize(220, 220)

        self.update()
        self._update_wraplengths(canvas.winfo_width())

    @staticmethod
    def _to_display(json_value, scale, decimals):
        value = json_value / scale
        return int(round(value)) if decimals == 0 else round(value, decimals)

    def _update_stats(self, params):
        try:
            r1, r2 = _initial_radii(params)
            self.stats_label.config(text=f"Initial r1={r1:.3f} m, r2={r2:.3f} m")
        except ZeroDivisionError:
            self.stats_label.config(text="theta_initial_deg is 0 or 180 -- r1/r2 undefined "
                                          "(sin(theta)=0)")

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
        self.stats_label.configure(foreground=pal['muted_fg'])
        self.note_label.configure(foreground=pal['muted_fg'])
        _set_titlebar_dark(self, dark)

    def save(self):
        params = dict(_DEFAULTS)
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

        self.status.config(text="Saved -- press Reload in the SOFA GUI to apply",
                            foreground=self._palette['muted_fg'])


if __name__ == '__main__':
    ParamsEditor().mainloop()
