"""Field labels and units of params.json, for the web viewer's parameter
panel (sofaweb reads these tables without importing this file). Edit
params.json by hand for runSofa."""

# (label, json_key, unit_scale [json = displayed * scale], decimals, unit)
_FIELDS = [
    ('Length (X)',                 'length',             1,    1, 'mm'),
    ('Width (Y)',                  'width',              1,    1, 'mm'),
    ('Height (Z)',                 'height',             1,    1, 'mm'),
    ('Element size',               'element_size',       1,    2, 'mm'),
    ("Young's modulus",            'young_modulus',      1e3,  1, 'kPa'),
    ('Poisson ratio',              'poisson_ratio',      1,    3, ''),
    ('Density',                    'density',            1,    1, 'kg/m^3'),
    ('Damping (Rayleigh stiffness)', 'rayleigh_stiffness', 1,  4, 's'),
    ('Damping (Rayleigh mass)',    'rayleigh_mass',      1,    3, '1/s'),
    ('Push force (sideways, while held)', 'side_force',  1,    3, 'N'),
    ('Flick force',                'flick_force',        1,    3, 'N'),
    ('Flick duration',             'flick_duration',     1e-3, 1, 'ms'),
    ('Force arrow length',         'arrow_scale',        1,    0, 'mm/N'),
    ('Time step',                  'dt',                 1e-3, 2, 'ms'),
    ('Beam surface opacity',       'beam_alpha',         1,    2, '0-1'),
    ('Clamp block opacity',        'clamp_alpha',        1,    2, '0-1'),
    ('Force arrow opacity',        'arrow_alpha',        1,    2, '0-1'),
]

_VECTOR_FIELDS = []

_BOOL_FIELDS = []
