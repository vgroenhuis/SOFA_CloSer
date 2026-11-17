# Snack shelf model

A two-bay supermarket shelf of chips and nuts, modelled from five photos (not included): the left bay (100 cm) gwoon
nut tubs and bags of nuts, the right bay (125 cm) Pringles (a pyramid on top), Lay's / Doritos paper tins, crisps
tubes and bags of chips. Built with Hagelslag's pipeline (~/hagelslag/hagelslag_shelf), adapted.

## Files
- `shelf.glb`: the visual model with embedded textures. Metres, +Y up, +Z towards the aisle, origin on the floor at
  the back-left corner. Node `p_<id>` holds all packages of one product; `fx_*` are the fixtures.
- `placements.json`: one entry per package (715): product, kind, bay, shelf, centre and box size in metres.
- `meta.json`: per-product name, size, price, counts.
- `catalog.py`: the products (photo crops: a quad in the 1126 x 2000 photo, sizes in cm) and the shelf layout;
  `build.py` regenerates the above from it and `tex/`; `textures.py` regenerates `tex/` from the photos
  (SNACK_PHOTOS: their folder, named 1.jpg ... 5.jpg).

## Rebuild
    uv run --no-project --with trimesh --with pillow --with numpy --with scipy python build.py
    (FONTS=<folder with DejaVuSans*.ttf> if not in /usr/share/fonts/truetype/dejavu)

## Accuracy
Sizes are estimates scaled from the photos (a Pringles can 7.5 x 23.5 cm; 165 g, per spar.nl), not measured; shelf
heights and depths likewise. Stock behind the front row is assumed.
