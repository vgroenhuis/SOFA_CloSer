# Hagelslag shelf model

Two-bay supermarket shelf, modelled from photos.

## Files
- `shelf.glb`: visual model with embedded textures. Metres, +Y up, +Z towards the aisle, origin on the floor at the back-left corner.
  Node `p_<id>` holds ALL packages of one product merged into one mesh. Nodes `fx_*` are fixtures (back panel, wire shelves, rails, price tags, glass dividers).
- `placements.json`: one entry per package (985): product, kind, bay, shelf, centre and bounding-box size in metres. Use this to spawn separate rigid bodies.
- `meta.json`: per-product name, size in cm, price, counts.
- `catalog.py`: products, sizes and shelf layout. `build.py`: regenerates all of the above from `catalog.py` and `tex/`.
- `tex/`: rectified package fronts, one per product.
- `textures.py`: regenerates `tex/` from the original photos, which are not included. Not needed for a normal rebuild.
- `index.html`: the web viewer (expects `shelf.b64.txt`, base64 of the GLB, and `meta.json`).

## Rebuild
    pip install trimesh pillow numpy
    python3 build.py

## Accuracy
- Sizes are estimates scaled from photos, assuming each bay is 100 cm wide. Not measured.
- Shelf heights are the least certain; the top shelf (175 cm) could be nearer 160 cm.
- Stock depth behind the front row is assumed.

## Not simulation-ready
- No mass, friction, or collision geometry. Wire shelves are thin boxes.
- Jars are textured lathes, bags are tilted boxes (`placements.json` gives their untilted boxes), cartons have gable tops.
