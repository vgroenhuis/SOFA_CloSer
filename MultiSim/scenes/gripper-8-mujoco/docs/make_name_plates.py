"""Renders original/name_plates/<gripper>.png: the name plate on the floor in front of each gripper of the line.
A gripper without a plate simply has none, so run this again after adding a gripper to Gripper8Mj.GRIPPER_CHOICES:

    uv run --no-project --with pillow python docs/make_name_plates.py [path/to/DejaVuSans-Bold.ttf]

(from scenes/gripper-8-mujoco; any bold sans-serif TrueType font will do.)
"""
import ast
import os
import sys

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, '..', 'original', 'Gripper8Mj.py')
OUT = os.path.join(HERE, '..', 'original', 'name_plates')
FONT = sys.argv[1] if len(sys.argv) > 1 else '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
SIZE = (720, 180)  # the plate is 4 x 1 (360 x 90 mm on the floor)


def gripper_choices():
    tree = ast.parse(open(SCRIPT, encoding='utf-8').read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], 'id', '') == 'GRIPPER_CHOICES':
            return ast.literal_eval(node.value)
    raise SystemExit('no GRIPPER_CHOICES in Gripper8Mj.py')


def lines_of(text, font, draw, width):
    """The text in at most two lines that fit `width`, split at a space or before a bracket."""
    words, best = text.replace(' (', ' |(').split(' '), None
    for k in range(1, len(words)):
        a, b = ' '.join(words[:k]).replace('|', ''), ' '.join(words[k:]).replace('|', '')
        w = max(draw.textlength(a, font=font), draw.textlength(b, font=font))
        if best is None or w < best[0]:
            best = (w, [a, b])
    one = text.replace('|', '')
    if draw.textlength(one, font=font) <= width or best is None:
        return [one]
    return best[1]


def main():
    os.makedirs(OUT, exist_ok=True)
    for i, (key, label) in enumerate(gripper_choices()):
        img = Image.new('RGB', SIZE, (250, 250, 246))
        d = ImageDraw.Draw(img)
        d.rectangle([4, 4, SIZE[0] - 5, SIZE[1] - 5], outline=(40, 44, 52), width=6)
        d.rectangle([4, 4, 110, SIZE[1] - 5], fill=(40, 44, 52))
        num = ImageFont.truetype(FONT, 84)
        d.text((57, SIZE[1] / 2), str(i + 1), font=num, fill=(250, 200, 30), anchor='mm')
        for size in (60, 54, 48, 42, 36, 32):
            font = ImageFont.truetype(FONT, size)
            lines = lines_of(label, font, d, SIZE[0] - 160)
            if all(d.textlength(t, font=font) <= SIZE[0] - 160 for t in lines) and len(lines) * size * 1.2 < 160:
                break
        y0 = SIZE[1] / 2 - (len(lines) - 1) * size * 0.6
        for k, t in enumerate(lines):
            d.text((130, y0 + k * size * 1.2), t, font=font, fill=(30, 32, 38), anchor='lm')
        img.save(os.path.join(OUT, f'{key}.png'), optimize=True)
        print(f'{key}.png: {" / ".join(lines)}')


if __name__ == '__main__':
    main()
