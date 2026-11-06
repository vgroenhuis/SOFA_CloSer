"""Renders docs/choices/<key>_<value>.jpg: a picture of every gripper, arm and cart in the Robot drop-downs, for the
viewer's overview (sofaweb_scene.py's choice_gallery).

Start the scene's server first (from scenes/hagelslag-mujoco):
    PYTHONPATH=../_base python -m sofaweb serve sofaweb_scene.py --host 127.0.0.1 --port 8195
then:
    uv run --no-project --with playwright --with pillow python docs/make_choice_images.py [http://127.0.0.1:8195]
(Playwright needs a Chromium: `playwright install chromium`.) The server's parameters are changed one by one and set
back to the defaults at the end. Each picture shows only that part, framed by its bounding box, on a plain background.
"""
import io
import os
import sys
import time

from PIL import Image
from playwright.sync_api import sync_playwright

URL = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8195").rstrip("/")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "choices")
SIZE = (480, 360)  # the saved picture (rendered at twice this, then scaled down)

GRIPPER = r"/world/angles/bodies/(?!skin/|joint_marks/)"
ARM = r"/world/robot/arm/"
CART = r"/world/robot/platform/"
# parameter: (the models to show, view direction from the part towards the camera); every option of its drop-down
PARTS = {"gripper": (GRIPPER, (0.75, -1.0, -0.3)), "arm": (ARM, (0.9, -1.0, 0.45)), "cart": (CART, (0.8, -1.0, 0.5))}

FRAME = """async ([pattern, dir]) => {
    const THREE = await import('three');
    const re = new RegExp(pattern);
    const box = new THREE.Box3();
    const own = new Set();
    for (const e of window.sofaweb.models.values()) {
        const show = re.test(e.info.name);
        e.object.visible = show;
        own.add(e.object);
        if (show) box.expandByObject(e.object);
    }
    // nothing else: the manipulator's target marker and gizmo, the gauge, ...
    window.sofaweb.scene.traverse((o) => {
        if ((o.isMesh || o.isLine || o.isPoints || o.isSprite) && !own.has(o) && !own.has(o.parent)) o.visible = false;
    });
    const c = box.getCenter(new THREE.Vector3()), r = box.getSize(new THREE.Vector3()).length() / 2;
    const cam = window.sofaweb.camera, n = Math.hypot(...dir);
    const d = 1.08 * r / Math.sin(Math.min(cam.fov, cam.fov * cam.aspect) * Math.PI / 360);
    const set = (id, v) => { const el = document.getElementById(id); el.value = v.map((x) => x.toFixed(4)).join(', ');
                             el.dispatchEvent(new Event('change')); };
    set('d-camera-target', [c.x, c.y, c.z]);
    set('d-camera-position', [c.x + d * dir[0] / n, c.y + d * dir[1] / n, c.z + d * dir[2] / n]);
    return [c.x, c.y, c.z, r];
}"""


def wait_running(page, generation):
    for _ in range(120):
        info = page.request.get(URL + "/api/info").json()
        if info["status"] == "running" and info["generation"] > generation:
            page.request.post(URL + "/api/pause")  # (held still: a cart may creep)
            return info["generation"]
        time.sleep(1)
    raise RuntimeError("the scene didn't rebuild")


def main():
    os.makedirs(OUT, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        page = browser.new_page(viewport={"width": SIZE[0], "height": SIZE[1]}, device_scale_factor=2)
        page.set_default_timeout(300000)
        generation = page.request.get(URL + "/api/info").json()["generation"]
        defaults = page.request.post(URL + "/api/params/default").json()  # (rebuilds with the defaults)
        generation = wait_running(page, generation)
        page.goto(URL + "/")
        page.wait_for_function("window.sofaweb && window.sofaweb.models.size > 50")
        page.add_style_tag(content="#top-controls, #bottom, #manip-height, #manip-drive, #manip-row, #banner, "
                                   "#play-hint, #side-panels, #hud, #charts { display: none !important; }")
        choices = {f["key"]: [v for v, _ in f["choices"]] for sec in page.request.get(URL + "/api/scene").json()["spec"]
                   for f in sec["fields"] if f["key"] in PARTS}
        current = {}
        for key, value, (models, direction) in [(k, v, PARTS[k]) for k in PARTS for v in choices[k]]:
            want = {**{k: defaults[k] for k in PARTS}, key: value}
            if want != current:
                if any(defaults.get(k) != v for k, v in want.items()) or current:
                    page.request.post(URL + "/api/params", data={"changes": want})
                    generation = wait_running(page, generation)
                current = want
                time.sleep(6)
            # Framed again until it stays put: after a rebuild the models arrive over a few seconds (and arrive visible)
            last = None
            for _ in range(20):
                framed = page.evaluate(FRAME, [models, list(direction)])
                if last and framed[3] > 0.01 and max(abs(u - v) for u, v in zip(framed, last)) < 1e-3:
                    break
                last = framed
                time.sleep(2)
            else:
                raise RuntimeError(f"{key} {value}: the view didn't settle")
            time.sleep(1)
            png = page.screenshot()
            Image.open(io.BytesIO(png)).convert("RGB").resize(SIZE, Image.LANCZOS).save(
                os.path.join(OUT, f"{key}_{value}.jpg"), quality=85, optimize=True)
            print(f"{key}_{value}.jpg", flush=True)
        page.request.post(URL + "/api/params", data={"changes": {k: defaults[k] for k in PARTS}})
        browser.close()


if __name__ == "__main__":
    main()
