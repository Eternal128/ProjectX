"""Draw the website's plates (flat SVG drawings in the palette's colours).

    python src/figures/site_plates.py

Writes site/assets/img/*.svg. These stand in for photographs until real archival
pictures can be fetched (see site/README.md).
"""
from __future__ import annotations

import math
import random
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "site" / "assets" / "img"
INK, MUTED, DEEP, BONE, SURF, SURF2 = "#38450f", "#5d683a", "#232b09", "#ecebe7", "#e1dfd9", "#d5d3cb"


def svg(w, h, body, bg=None):
    rect = f'<rect width="{w}" height="{h}" fill="{bg}"/>' if bg else ""
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">{rect}{body}</svg>\n'


def poly(points, fill):
    return f'<path d="M{" L".join(f"{x:.1f} {y:.1f}" for x, y in points)} Z" fill="{fill}"/>'


def ring(cx, cy, rx, ry, t, fill):
    return (f'<path fill-rule="evenodd" fill="{fill}" d="M{cx-rx} {cy} a{rx} {ry} 0 1 0 {2*rx} 0 a{rx} {ry} 0 1 0 {-2*rx} 0 Z '
            f'M{cx-rx+t} {cy} a{rx-t} {ry-t} 0 1 0 {2*(rx-t)} 0 a{rx-t} {ry-t} 0 1 0 {-2*(rx-t)} 0 Z"/>')


def leaf(x, y, angle, length, width, fill):
    L, W = length, width
    d = f"M0 0 C{L*.3:.1f} {-W:.1f} {L*.72:.1f} {-W*.8:.1f} {L:.1f} 0 C{L*.72:.1f} {W*.8:.1f} {L*.3:.1f} {W:.1f} 0 0 Z"
    mid = f'<path d="M{L*.08:.1f} 0 L{L*.9:.1f} 0" stroke="{DEEP}" stroke-opacity=".35" stroke-width="2"/>'
    return f'<g transform="translate({x:.1f} {y:.1f}) rotate({angle:.1f})"><path d="{d}" fill="{fill}"/>{mid}</g>'


def kylix():
    b = []
    b.append(ring(150, 236, 92, 40, 20, INK))           # handles
    b.append(ring(850, 236, 92, 40, 20, INK))
    b.append(f'<path d="M140 210 C230 420 770 420 860 210 Z" fill="{INK}"/>')  # bowl
    b.append(f'<path d="M168 262 C300 372 700 372 832 262 L842 244 C700 330 300 330 158 244 Z" fill="{DEEP}"/>')  # band
    for i in range(23):                                   # tongue pattern under the band
        x = 238 + i * 23.5
        y = 318 + 26 * math.sin(math.pi * i / 22) ** 0.6
        b.append(f'<path d="M{x:.1f} {y-16:.1f} q8 22 16 0 Z" fill="{MUTED}"/>')
    b.append(f'<ellipse cx="500" cy="210" rx="360" ry="34" fill="{MUTED}"/>')  # interior seen over the lip
    b.append(f'<ellipse cx="500" cy="214" rx="300" ry="22" fill="{DEEP}" opacity=".55"/>')
    b.append(f'<path d="M470 386 L530 386 L522 470 L478 470 Z" fill="{INK}"/>')  # stem
    b.append(f'<ellipse cx="500" cy="482" rx="150" ry="24" fill="{INK}"/>')       # foot
    b.append(f'<rect x="350" y="482" width="300" height="14" fill="{DEEP}"/>')
    return svg(1000, 560, "".join(b), SURF2)


def columns():
    b = [f'<rect x="0" y="500" width="1000" height="60" fill="{INK}"/>',
         f'<rect x="40" y="478" width="920" height="22" fill="{MUTED}"/>',
         f'<rect x="70" y="460" width="860" height="18" fill="{INK}"/>']
    tops = [70, 70, 210, 70, 300, 70]
    for i, top in enumerate(tops):
        x = 110 + i * 136
        w = 84
        if top == 70:
            b.append(f'<rect x="{x-18}" y="{top-26}" width="{w+36}" height="20" fill="{INK}"/>')      # abacus
            b.append(poly([(x-12, top-6), (x+w+12, top-6), (x+w, top+14), (x, top+14)], DEEP))      # echinus
            shaft_top = top + 14
            b.append(poly([(x, shaft_top), (x+w, shaft_top), (x+w+6, 460), (x-6, 460)], INK))
        else:
            jag = [(x-4, top+12), (x+18, top), (x+34, top+16), (x+52, top+4), (x+70, top+18), (x+w+4, top+8)]
            b.append(poly(jag + [(x+w+6, 460), (x-6, 460)], INK))
            shaft_top = top + 18
        for f in range(1, 6):                                   # flutes
            fx = x + f * w / 6
            b.append(f'<rect x="{fx-3:.1f}" y="{shaft_top+4}" width="6" height="{456-shaft_top}" fill="{MUTED}"/>')
    b.append(poly([(780, 500), (930, 500), (920, 470), (800, 476)], DEEP))  # fallen drum
    return svg(1000, 560, "".join(b), SURF2)


def laurel():
    rnd = random.Random(7)
    b = []
    pts = []
    for i in range(80):                                      # stem along a gentle S-curve, bottom to top
        t = i / 79
        x = 330 + 120 * math.sin(t * 2.6 + 0.4) - 60 * t
        y = 1180 - t * 1060
        pts.append((x, y))
    left, right = [], []
    for i, (x, y) in enumerate(pts):
        j = min(i + 1, len(pts) - 1)
        dx, dy = pts[j][0] - pts[max(i - 1, 0)][0], pts[j][1] - pts[max(i - 1, 0)][1]
        n = math.hypot(dx, dy) or 1
        w = 9 * (1 - i / 90)
        left.append((x - dy / n * w, y + dx / n * w))
        right.append((x + dy / n * w, y - dx / n * w))
    b.append(poly(left + right[::-1], DEEP))
    for i in range(6, 78, 6):                                 # leaf pairs
        x, y = pts[i]
        dx, dy = pts[i + 1][0] - pts[i - 1][0], pts[i + 1][1] - pts[i - 1][1]
        base = math.degrees(math.atan2(dy, dx))
        size = 1.0 - i / 140
        for side, tone in ((-1, INK), (1, MUTED)):
            ang = base + side * (38 + rnd.uniform(-6, 6))
            b.append(leaf(x, y, ang, 300 * size, 70 * size, tone))
        if i % 15 == 1:
            b.append(f'<circle cx="{x+rnd.uniform(-30,30):.1f}" cy="{y+22:.1f}" r="13" fill="{DEEP}"/>')
            b.append(f'<circle cx="{x+rnd.uniform(-34,34):.1f}" cy="{y+46:.1f}" r="11" fill="{DEEP}"/>')
    return svg(720, 1200, "".join(b), SURF2)


def tripod():
    b = [ring(118, 92, 46, 46, 14, INK), ring(242, 92, 46, 46, 14, INK),
         f'<rect x="50" y="128" width="260" height="16" fill="{DEEP}"/>',
         f'<path d="M58 144 C58 250 302 250 302 144 Z" fill="{INK}"/>',
         poly([(96, 210), (122, 214), (70, 384), (44, 384)], INK),
         poly([(170, 228), (190, 228), (192, 392), (168, 392)], DEEP),
         poly([(238, 214), (264, 210), (316, 384), (290, 384)], INK),
         f'<ellipse cx="57" cy="388" rx="26" ry="8" fill="{DEEP}"/>',
         f'<ellipse cx="303" cy="388" rx="26" ry="8" fill="{DEEP}"/>',
         f'<rect x="80" y="300" width="200" height="12" fill="{MUTED}"/>']
    return svg(360, 400, "".join(b), SURF2)


def pile():
    rnd = random.Random(3)
    b = []
    for i in range(52):                                       # a stack of sheets
        y = 560 - i * 8.6
        w = rnd.uniform(520, 600)
        x = 150 + rnd.uniform(-18, 18) + (8 * math.sin(i / 5))
        tone = BONE if i % 3 else SURF2
        ang = rnd.uniform(-0.8, 0.8)
        b.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="7" fill="{tone}" '
                 f'transform="rotate({ang:.2f} {x+w/2:.1f} {y:.1f})"/>')
    for x, y, a in ((560, 112, -9), (300, 96, 7)):             # loose sheets on top
        b.append(f'<rect x="{x}" y="{y}" width="260" height="7" fill="{BONE}" transform="rotate({a} {x+130} {y})"/>')
    b.append(f'<rect x="0" y="568" width="960" height="32" fill="{DEEP}"/>')
    return svg(960, 600, "".join(b), INK)


def hourglass():
    b = [f'<rect x="90" y="60" width="420" height="44" fill="{INK}"/>',
         f'<rect x="90" y="796" width="420" height="44" fill="{INK}"/>',
         f'<rect x="120" y="104" width="22" height="692" fill="{DEEP}"/>',
         f'<rect x="458" y="104" width="22" height="692" fill="{DEEP}"/>',
         f'<path d="M180 104 C180 330 288 400 292 450 C288 500 180 570 180 796 L420 796 C420 570 312 500 308 450 C312 400 420 330 420 104 Z" fill="{BONE}"/>',
         f'<path d="M232 300 C250 360 290 400 296 440 L304 440 C310 400 350 360 368 300 Z" fill="{MUTED}"/>',
         f'<rect x="297" y="440" width="6" height="250" fill="{MUTED}"/>',
         f'<path d="M190 796 C210 700 270 676 300 676 C330 676 390 700 410 796 Z" fill="{MUTED}"/>']
    return svg(600, 900, "".join(b))


def balance():
    b = [poly([(330, 820), (570, 820), (520, 770), (380, 770)], INK),
         f'<rect x="438" y="190" width="24" height="580" fill="{INK}"/>',
         f'<circle cx="450" cy="180" r="26" fill="{DEEP}"/>',
         f'<g transform="rotate(-7 450 196)"><rect x="110" y="186" width="680" height="20" fill="{INK}"/>'
         f'<circle cx="122" cy="196" r="16" fill="{DEEP}"/><circle cx="778" cy="196" r="16" fill="{DEEP}"/></g>']
    for cx, top, bowl in ((160, 238, 560), (740, 168, 470)):   # left side heavier, hangs lower
        b.append(poly([(cx - 2, top), (cx + 2, top), (cx - 92, bowl), (cx - 98, bowl)], DEEP))
        b.append(poly([(cx - 2, top), (cx + 2, top), (cx + 98, bowl), (cx + 92, bowl)], DEEP))
        b.append(f'<path d="M{cx-120} {bowl} C{cx-110} {bowl+70} {cx+110} {bowl+70} {cx+120} {bowl} Z" fill="{INK}"/>')
    for i in range(5):                                        # sheets in the heavier pan
        b.append(f'<rect x="{90 + (i % 2) * 8}" y="{548 - i * 11}" width="132" height="8" fill="{MUTED}"/>')
    b.append(f'<circle cx="740" cy="452" r="18" fill="{MUTED}"/>')
    return svg(900, 900, "".join(b))


def dividers():
    b = [poly([(292, 150), (308, 150), (520, 840), (506, 846)], INK),
         poly([(292, 150), (308, 150), (94, 846), (80, 840)], DEEP),
         f'<circle cx="300" cy="150" r="40" fill="{INK}"/>',
         f'<circle cx="300" cy="150" r="14" fill="{MUTED}"/>',
         f'<rect x="288" y="40" width="24" height="76" fill="{INK}"/>',
         poly([(160, 560), (440, 560), (434, 584), (166, 584)], MUTED)]
    return svg(600, 900, "".join(b))


def stone():
    rnd = random.Random(11)
    b = [poly([(70, 120), (300, 70), (440, 96), (520, 190), (540, 760), (60, 770)], MUTED)]
    def band(y0, rows, kind):
        for r in range(rows):
            y = y0 + r * 30
            x = 110
            while x < 480:
                if kind == 0:
                    s = rnd.choice(["c", "r", "t"])
                    if s == "c":
                        b.append(f'<circle cx="{x+8}" cy="{y+8}" r="8" fill="{DEEP}"/>')
                    elif s == "r":
                        b.append(f'<rect x="{x}" y="{y}" width="14" height="16" fill="{DEEP}"/>')
                    else:
                        b.append(poly([(x, y + 16), (x + 16, y + 16), (x + 8, y)], DEEP))
                    x += 26
                else:
                    w = rnd.uniform(14, 60) if kind == 2 else rnd.uniform(8, 22)
                    b.append(f'<rect x="{x:.1f}" y="{y+6}" width="{min(w, 480-x):.1f}" height="{8 if kind == 2 else 6}" fill="{DEEP}"/>')
                    x += w + (12 if kind == 2 else 7)
    band(200, 6, 0)
    band(400, 6, 1)
    band(590, 5, 2)
    return svg(600, 840, "".join(b))


def mark():
    return svg(64, 64, leaf(14, 50, -62, 40, 10, BONE) + leaf(30, 52, -24, 34, 9, BONE))


def favicon():
    return svg(64, 64, leaf(14, 50, -62, 40, 10, INK) + leaf(30, 52, -24, 34, 9, MUTED), BONE)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in [("kylix", kylix), ("columns", columns), ("laurel", laurel), ("tripod", tripod),
                     ("pile", pile), ("hourglass", hourglass), ("balance", balance),
                     ("dividers", dividers), ("stone", stone), ("mark", mark), ("favicon", favicon)]:
        (OUT / f"{name}.svg").write_text(fn())
        print("wrote", OUT / f"{name}.svg")


if __name__ == "__main__":
    main()
