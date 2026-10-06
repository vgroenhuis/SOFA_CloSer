# -*- coding: utf-8 -*-
"""
Standalone parameter editor for PneuNetFinger.py.

Runs as its own window/process (Tkinter, part of the standard library --
no SOFA dependency, works with any Python install), reading and writing
params.json in the same directory. Every change auto-saves (text fields
on focus-out/Enter, checkboxes immediately) -- there's no Save button.
It does NOT talk to a running SOFA session directly:
  - Block size / box dimensions / wall thickness / material changes only
    take effect on the next scene (re)build -- after a change, use the
    SOFA GUI's Reload button.
  - Target pressure / ramp time / instant can also be edited live while
    the simulation is already running, directly in the SOFA GUI's Scene
    Graph panel (select the "PressureRamp" node) -- no need for this
    editor, or a reload, for those.

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
    'block_size': 10.0,
    'length_blocks': 6,
    'width_blocks': 10,
    'height_blocks': 8,
    'wall_length_blocks': 1,
    'wall_width_blocks': 1,
    'wall_height_blocks': 1,
    'trunk_length_blocks': 6,
    'trunk_width_blocks': 10,
    'floor_size_blocks': 30,
    'channel_height_blocks': 2,
    'channel_separation_blocks': 0.0,
    'channel_end_constraint': True,
    'channel_end_alpha': 0.0,
    'prevent_channel_end_crossing': True,
    'symmetry_x': False,
    'symmetry_y': False,
    'symmetry_buffer_blocks': 0,
    'prevent_symmetry_crossing': True,
    'mirror_visual_x': True,
    'mirror_visual_y': True,
    'young_modulus': 1.0e6,
    'poisson_ratio': 0.45,
    'density': 1100.0,
    'target_pressure': 1000.0,
    'ramp_time': 2.0,
    'instant': False,
    'gravity_enabled': True,
    'cutaway': True,
    'interior_cutaway': True,
    'wall_alpha': 0.4,
    'interior_alpha': 0.5,
    'show_pressure_overlay': True,
    'show_fem_elements': False,
    'theme': 'System',   # 'Light' / 'Dark' / 'System' -- UI-only preference,
                          # stored here for convenience but never read by
                          # PneuNetFinger.py itself.
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
    ('Block size',                'block_size',          1, 2, 'mm'),
    ('Length (X)',                'length_blocks',       1, 0, 'blocks'),
    ('Width (Y)',                 'width_blocks',        1, 0, 'blocks'),
    ('Height (Z)',                'height_blocks',       1, 0, 'blocks'),
    ('Wall thickness length (X)', 'wall_length_blocks',  1, 0, 'blocks'),
    ('Wall thickness width (Y)', 'wall_width_blocks',   1, 0, 'blocks'),
    ('Wall thickness height (Z)', 'wall_height_blocks',  1, 0, 'blocks'),
    ('Trunk length (X)',            'trunk_length_blocks', 1, 0, 'blocks'),
    ('Trunk width (Y)',             'trunk_width_blocks',  1, 0, 'blocks'),
    ('Floor size',                  'floor_size_blocks',   1, 0, 'blocks'),
    ('Channel opening height',      'channel_height_blocks', 1, 0, 'blocks'),
    ('Channel separation (to neighbor)', 'channel_separation_blocks', 1, 2, 'blocks'),
    ('Symmetry buffer',             'symmetry_buffer_blocks', 1, 0, 'blocks'),
    ("Young's modulus",  'young_modulus',   1e3,  1, 'kPa'),
    ('Poisson ratio',    'poisson_ratio',   1,    3, ''),
    ('Density',          'density',         1,    1, 'kg/m^3'),
    ('Target pressure',  'target_pressure', 1,    0, 'Pa'),
    ('Ramp time',        'ramp_time',       1,    2, 's'),
    ('Wall opacity',     'wall_alpha',      1,    2, '0-1'),
    ('Interior opacity', 'interior_alpha',  1,    2, '0-1'),
    ('Channel end-plane opacity', 'channel_end_alpha', 1, 2, '0-1'),
]

# (label, json_key, default) -- simple checkboxes
_BOOL_FIELDS = [
    ('Apply pressure instantly (not gradual)', 'instant', False),
    ('Enable gravity', 'gravity_enabled', True),
    ('Exploit X symmetry (halves FEM elements)', 'symmetry_x', False),
    ('Exploit Y symmetry (halves FEM elements)', 'symmetry_y', False),
    ('Prevent nodes crossing symmetry planes (one-sided stop, no collision model; '
     'guards both X/Y midplanes regardless of whether their symmetry is exploited)',
     'prevent_symmetry_crossing', True),
    ('Mirror visual model across X symmetry plane', 'mirror_visual_x', True),
    ('Mirror visual model across Y symmetry plane', 'mirror_visual_y', True),
    ('Constrain channel duct far end as a rigid plane (free to translate X / rotate Y)',
     'channel_end_constraint', True),
    ('Prevent chamber wall bulging past the channel duct far-end plane',
     'prevent_channel_end_crossing', True),
    ('Cutaway window in outer wall (see into the cavity)', 'cutaway', True),
    ('Cutaway window in interior volume (same side)', 'interior_cutaway', True),
    ('Show pressure indicator (red/green, opaque, no alpha control)',
     'show_pressure_overlay', True),
    ('Show FEM elements (SOFA\'s "Show Force Fields" flag; set here since '
     'Reload resets it in the GUI)', 'show_fem_elements', False),
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


def _count_fem_elements(params):
    """Number of tetrahedral FEM elements the current geometry would
    produce: 6 per shell grid cell (Kuhn decomposition) within whatever
    i/j/k sub-range is actually built (the full range, or -- per active
    symmetry_x/y flag -- one half extended by symmetry_buffer_blocks
    extra cells past the plane), shell cells being all cells in that
    sub-range minus the unmeshed interior (cavity) ones and minus the
    channel-opening cells carved from the -X/+X end walls. Mirrors the
    axis-resolution and range logic in PneuNetFinger.py's
    _resolve_geometry/build_hollow_cube_mesh -- duplicated (rather than
    imported) so this editor keeps working with any Python install, no
    SOFA bindings required.
    """
    def resolve_axis(extent_blocks, wall_blocks, symmetric):
        divs = max(1, int(round(extent_blocks)))
        wall = max(1, int(round(wall_blocks)))
        min_divs = 2 * wall + 1  # need at least 1 interior (cavity) cell
        divs = max(divs, min_divs)
        if symmetric:
            divs += divs % 2  # round up to even
        return divs, wall

    def axis_range(divs, symmetric, buf):
        if not symmetric:
            return 0, divs
        return max(0, divs // 2 - buf), divs

    def interval_len(lo, hi):
        return max(0, hi - lo)

    def interior_len(lo, hi, wall, divs):
        return interval_len(max(lo, wall), min(hi, divs - wall))

    def end_wall_len(lo, hi, wall, divs):
        # Cells within [lo, hi) that are in the -axis or +axis wall band
        # (i < wall or i >= divs - wall) -- the X-end walls a channel
        # opening carves into.
        return (interval_len(max(lo, 0), min(hi, wall)) +
                interval_len(max(lo, divs - wall), min(hi, divs)))

    buf = max(0, int(round(params['symmetry_buffer_blocks'])))
    divs_x, wall_x = resolve_axis(params['length_blocks'], params['wall_length_blocks'], params['symmetry_x'])
    divs_y, wall_y = resolve_axis(params['width_blocks'], params['wall_width_blocks'], params['symmetry_y'])
    divs_z, wall_z = resolve_axis(params['height_blocks'], params['wall_height_blocks'], False)

    i_lo, i_hi = axis_range(divs_x, params['symmetry_x'], buf)
    j_lo, j_hi = axis_range(divs_y, params['symmetry_y'], buf)
    k_lo, k_hi = 0, divs_z

    total_cells = interval_len(i_lo, i_hi) * interval_len(j_lo, j_hi) * interval_len(k_lo, k_hi)
    interior_cells = (interior_len(i_lo, i_hi, wall_x, divs_x) *
                       interior_len(j_lo, j_hi, wall_y, divs_y) *
                       interior_len(k_lo, k_hi, wall_z, divs_z))

    channel_h = max(0, min(int(round(params['channel_height_blocks'])), divs_z - 2 * wall_z))
    channel_cells = (end_wall_len(i_lo, i_hi, wall_x, divs_x) *
                      interior_len(j_lo, j_hi, wall_y, divs_y) *
                      interval_len(wall_z, wall_z + channel_h))

    # Connecting-channel duct (see build_hollow_cube_mesh's docstring for
    # `channel_separation_blocks`): a short hollow-walled extension per
    # built end wall, same wall/bore pattern as the main shell's own
    # k-band test but over its own short duct_k_hi span, welded onto the
    # shell at whichever j-range is actually built (j_lo..j_hi, same as
    # the main shell -- includes symmetry_buffer_blocks, unlike the
    # cavity/visual surfaces' own buffer-free width).
    half_sep = max(0.0, params['channel_separation_blocks']) / 2.0
    duct_cells = 0
    if channel_h > 0 and half_sep > 1e-9:
        duct_k_hi = wall_z + channel_h + wall_z
        # Subdivided into layers no longer than one block along X (see
        # build_hollow_cube_mesh's docstring) -- half_sep is already in
        # block units here, so this is the same ceil() it uses.
        duct_divs_x = max(1, math.ceil(half_sep - 1e-9))
        num_sides = 1 if params['symmetry_x'] else 2
        duct_cells = duct_divs_x * num_sides * (
            interval_len(j_lo, j_hi) * duct_k_hi -
            interior_len(j_lo, j_hi, wall_y, divs_y) * channel_h)

    return (total_cells - interior_cells - channel_cells + duct_cells) * 6


class ParamsEditor(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PneuNetFingerInflation parameters")
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

        note = ("Every change auto-saves. Block size / box dimensions / wall thickness / "
                "material changes need the SOFA GUI's Reload button to take effect. "
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
        n = _count_fem_elements(params)
        self.stats_label.config(text=f"FEM elements: {n:,}")

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

        self.status.config(text=f"Saved -- press Reload in the SOFA GUI to apply",
                            foreground=self._palette['muted_fg'])


if __name__ == '__main__':
    ParamsEditor().mainloop()
