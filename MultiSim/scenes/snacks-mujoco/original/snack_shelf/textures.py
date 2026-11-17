"""tex/<product>.jpg: each product's front, rectified from the photos (catalog.py's quads), and contact.jpg."""
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw
from catalog import P
PHOTOS = os.environ.get('SNACK_PHOTOS', '/tmp/claude-1001/-home-groenhuis/6bccabff-c297-46bd-889c-b5007c8a5018/images/')
cache = {}
def load(k):
    if k not in cache:
        cache[k] = cv2.imread(os.path.join(PHOTOS, f'{k}.jpg'))
    return cache[k]
os.makedirs('tex', exist_ok=True)
for pid, x in P.items():
    im = load(x['img'])
    q = np.float32(x['quad'])
    h = 384; w = max(64, int(round(h * x['w'] / x['h'])))
    if w > 384: w = 384; h = max(64, int(round(w * x['h'] / x['w'])))
    M = cv2.getPerspectiveTransform(q, np.float32([[0, 0], [w, 0], [w, h], [0, h]]))
    out = cv2.warpPerspective(im, M, (w, h), flags=cv2.INTER_AREA)
    cv2.imwrite(f'tex/{pid}.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, 85])
ids = list(P); cols = 12; cw = 120; ch = 190
sheet = Image.new('RGB', (cols * cw, (len(ids) // cols + 1) * ch), 'white'); d = ImageDraw.Draw(sheet)
for i, pid in enumerate(ids):
    t = Image.open(f'tex/{pid}.jpg'); t.thumbnail((cw - 6, ch - 22))
    x, y = (i % cols) * cw + 3, (i // cols) * ch + 18
    sheet.paste(t, (x, y)); d.text((x, y - 15), pid[:18], fill='black')
sheet.save('contact.jpg', quality=88)
print(len(ids), sheet.size)
