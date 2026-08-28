#!/usr/bin/env python3
"""
Build an OT-SVG color font from per-letter transparent PNGs.

Input:  a folder of A.png ... Z.png, each exported from a square Figma frame
        with the baseline at a fixed height and a transparent background.
Output: a .ttf with an embedded SVG table (each glyph = one base64 PNG).

Usage:
    python build_font.py ./letters ./SprinkleFont.ttf
"""

import base64
import io
import sys
from pathlib import Path

from PIL import Image
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib.tables.S_V_G_ import table_S_V_G_

# ---------------------------------------------------------------- settings --

UPM = 1000               # em square. Don't change unless you know why.
BASELINE_RATIO = 0.75    # baseline position in the source frame (750/1000)
SIDEBEARING = 45         # blank space each side of the ink, in font units

# Per-letter overrides, in font units. Diagonal and open letters (A V W X Y,
# L T Z) leave visual air that the ink box doesn't measure, so they want
# less. Use (left, right) for asymmetric letters. Tune by eye.
SIDEBEARING_OVERRIDES = {
    "A": (25, 25),
    "F": (45, 25),
    "J": (30, 35),
    "L": (45, 25),
    "T": (25, 25),
    "V": (25, 25),
    "W": (30, 30),
    "X": (30, 30),
    "Y": (25, 25),
    "Z": (35, 35),
}
WORD_SPACE = 260         # width of the space character
FAMILY = "Sprinkles"
STYLE = "Regular"
VERSION = "1.000"

# Downscale source PNGs before embedding. 1.0 = full size.
# Lower this if the font file comes out too heavy.
IMAGE_SCALE = 1.0

# Background keying, for PNGs with a solid background baked in (e.g. shot on
# white paper). "auto" skips it when the file already has real transparency,
# which is what a Figma frame with no fill exports. Force with True / False.
KEY_BACKGROUND = "auto"
KEY_THRESHOLD = 40
EDGE_FEATHER = 0.6       # px of blur on the alpha edge; 0 to disable

# ------------------------------------------------------------------ helpers --


def key_background(img):
    """Flood-fill the background to transparent, starting from each corner."""
    import numpy as np
    from PIL import ImageDraw, ImageFilter

    rgb = img.convert("RGB")
    magic = (255, 0, 255)
    for corner in [(0, 0), (rgb.width - 1, 0),
                   (0, rgb.height - 1), (rgb.width - 1, rgb.height - 1)]:
        ImageDraw.floodfill(rgb, corner, magic, thresh=KEY_THRESHOLD)

    arr = np.array(rgb)
    mask = ((arr[:, :, 0] == 255) & (arr[:, :, 1] == 0) & (arr[:, :, 2] == 255))
    alpha = Image.fromarray(np.where(mask, 0, 255).astype("uint8"))
    if EDGE_FEATHER:
        alpha = alpha.filter(ImageFilter.GaussianBlur(EDGE_FEATHER))

    out = img.convert("RGBA")
    # respect any transparency the file already had
    existing = np.array(out.getchannel("A"))
    combined = np.minimum(existing, np.array(alpha))
    out.putalpha(Image.fromarray(combined.astype("uint8")))
    return out, 100.0 * mask.mean()


def load_letter(path):
    """Return (cropped RGBA image, ink bbox, source frame size)."""
    img = Image.open(path).convert("RGBA")
    if img.width != img.height:
        print(f"  ! {path.name} is {img.width}x{img.height}, not square")

    removed = None
    needs_key = KEY_BACKGROUND
    if needs_key == "auto":
        # if the four corners are already transparent, the export is clean
        w, h = img.size
        a = img.getchannel("A")
        corners = [a.getpixel(p) for p in
                   [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]]
        needs_key = max(corners) > 8

    if needs_key:
        img, removed = key_background(img)
        if removed < 20:
            print(f"  ! {path.name}: only {removed:.0f}% keyed — "
                  "background may not be uniform, try a higher KEY_THRESHOLD")

    bbox = img.getchannel("A").getbbox()
    if bbox is None:
        raise ValueError(f"{path.name} is fully transparent — check your export")
    return img.crop(bbox), bbox, img.size


def encode_png(img):
    """Optionally downscale, then return base64 PNG."""
    if IMAGE_SCALE != 1.0:
        w = max(1, round(img.width * IMAGE_SCALE))
        h = max(1, round(img.height * IMAGE_SCALE))
        img = img.resize((w, h), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def svg_doc(gid, b64, x, y, w, h):
    """One SVG document for the OT-SVG table.

    NOTE: SVG-table coordinates put the origin on the glyph origin with
    y pointing DOWN, so anything above the baseline has a negative y.
    """
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'xmlns:xlink="http://www.w3.org/1999/xlink" version="1.1">'
        f'<g id="glyph{gid}">'
        f'<image x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
        f'xlink:href="data:image/png;base64,{b64}"/>'
        "</g></svg>"
    )


def fallback_box(width, top, bottom):
    """A hollow rectangle drawn as the non-color fallback outline.

    Apps without OT-SVG support show this instead of nothing. Swap the body
    of this function for `return TTGlyphPen(None).glyph()` if you'd rather
    they render blank.
    """
    pen = TTGlyphPen(None)
    inset = 12
    x0, x1 = inset, max(inset + 1, width - inset)
    y0, y1 = bottom, top
    # outer contour
    pen.moveTo((x0, y0))
    pen.lineTo((x0, y1))
    pen.lineTo((x1, y1))
    pen.lineTo((x1, y0))
    pen.closePath()
    # inner contour (counter-clockwise) to hollow it out
    b = 28
    ix0, iy0, ix1, iy1 = x0 + b, y0 + b, x1 - b, y1 - b
    if ix1 > ix0 and iy1 > iy0:
        pen.moveTo((ix0, iy0))
        pen.lineTo((ix1, iy0))
        pen.lineTo((ix1, iy1))
        pen.lineTo((ix0, iy1))
        pen.closePath()
    return pen.glyph()


# --------------------------------------------------------------------- main --


def build(src_dir, out_path):
    src_dir = Path(src_dir)
    letters = [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
               if (src_dir / f"{c}.png").exists()]
    missing = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ") - set(letters)
    if missing:
        print(f"Missing: {' '.join(sorted(missing))}")
    if not letters:
        sys.exit(f"No letter PNGs found in {src_dir}")

    glyph_order = [".notdef", "space"] + letters
    glyphs, metrics, svgs = {}, {}, []

    glyphs[".notdef"] = TTGlyphPen(None).glyph()
    metrics[".notdef"] = (WORD_SPACE, 0)
    glyphs["space"] = TTGlyphPen(None).glyph()
    metrics["space"] = (WORD_SPACE, 0)

    max_top, min_bottom = 0, 0

    for name in letters:
        gid = glyph_order.index(name)
        img, bbox, (fw, fh) = load_letter(src_dir / f"{name}.png")

        scale = UPM / fw                 # source pixels -> font units
        baseline_px = fh * BASELINE_RATIO
        x0, y0, x1, y1 = bbox

        w = (x1 - x0) * scale
        h = (y1 - y0) * scale
        # y measured down from the baseline, so above-baseline is negative
        top = (y0 - baseline_px) * scale
        bottom = (y1 - baseline_px) * scale

        sb = SIDEBEARING_OVERRIDES.get(name, SIDEBEARING)
        lsb, rsb = sb if isinstance(sb, tuple) else (sb, sb)

        advance = round(w + lsb + rsb)
        metrics[name] = (advance, lsb)
        glyphs[name] = fallback_box(advance, round(-top), round(-bottom))

        svgs.append(svg_doc(gid, encode_png(img), lsb, top, w, h))
        max_top = max(max_top, -top)
        min_bottom = min(min_bottom, -bottom)
        print(f"  {name}  advance {advance:4d}  cap {(-top):.0f}")

    ascender = round(max(max_top, UPM * 0.75) * 1.05)
    descender = round(min(min_bottom, -UPM * 0.12))

    fb = FontBuilder(UPM, isTTF=True)
    fb.setupGlyphOrder(glyph_order)

    # Map both cases to the same glyphs so lowercase typing still works.
    cmap = {0x20: "space"}
    for name in letters:
        cmap[ord(name)] = name
        cmap[ord(name.lower())] = name
    fb.setupCharacterMap(cmap)

    fb.setupGlyf(glyphs)
    fb.setupHorizontalMetrics(metrics)
    fb.setupHorizontalHeader(ascent=ascender, descent=descender)
    fb.setupNameTable({
        "familyName": FAMILY,
        "styleName": STYLE,
        "psName": f"{FAMILY.replace(' ', '')}-{STYLE}",
        "version": VERSION,
    })
    fb.setupOS2(sTypoAscender=ascender, sTypoDescender=descender,
                usWinAscent=ascender, usWinDescent=abs(descender))
    fb.setupPost()

    svg_table = table_S_V_G_()
    svg_table.docList = [(doc, glyph_order.index(n), glyph_order.index(n))
                         for doc, n in zip(svgs, letters)]
    fb.font["SVG "] = svg_table

    fb.save(out_path)
    kb = Path(out_path).stat().st_size / 1024
    print(f"\n{out_path}  ({kb:.0f} KB, {len(letters)} letters)")


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "./letters"
    out = sys.argv[2] if len(sys.argv) > 2 else "./SprinkleFont.ttf"
    build(src, out)