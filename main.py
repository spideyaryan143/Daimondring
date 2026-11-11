# Diamond Ring - Web Ready Version
# JIN2 (@jin_22ee), 2026. Free to use for personal projects, see LICENSE.txt.

import asyncio
import sys
import time
from pathlib import Path

import numpy as np
import pygame

TAU = np.pi * 2
LOOP = 20.0         # length of one run (s)
BUILD = 3.2         # dust fades in around here
TURN = 12.0         # seconds per full turn
FLOOR = -0.78

POOLS = ["1il!;:", "rtfjcvxz=+", "aeoun0{}[]<>", "08B$#&%Q"]   # dark -> bright
CHARS = " .,:~+*^01|/\\-_#()x" + "".join(POOLS)
AT = {c: i for i, c in enumerate(CHARS)}
FILL = np.array([[AT[p[k % len(p)]] for k in range(12)] for p in POOLS])

BAND, PRONG, GEM, GLINT, FIRE_R, FIRE_B, FIRE_G, DUST, REFL, STAR = range(10)


def smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


# ring geometry

R_BAND, TUBE = 0.55, 0.065


def band():
    a, b = [x.ravel() for x in np.meshgrid(np.linspace(0, TAU, 520, endpoint=False), np.linspace(0, TAU, 44, endpoint=False))]
    rr = TUBE * (1 - 0.35 * np.clip(np.sin(a), 0, 1) ** 6)      # thinner under the stone
    c = np.stack([np.cos(a), np.sin(a), np.zeros_like(a)], 1)
    n = c * np.cos(b)[:, None] + np.array([0, 0, 1.0]) * np.sin(b)[:, None]
    p = c * R_BAND + n * np.stack([rr, rr, rr * 1.5], 1)
    return p, n / np.linalg.norm(n, axis=1, keepdims=True)


GIRDLE_Y, GIRDLE_R, TABLE_Y, TABLE_R, CULET_Y = 0.88, 0.25, 1.00, 0.135, 0.66


def diamond():
    # flat facets, one normal per facet
    P, N, F = [], [], []
    k = 0

    def facet(corners, n):
        nonlocal k
        u, v = [x.ravel() for x in np.meshgrid(np.linspace(0, 1, 16), np.linspace(0, 1, 16))]
        keep = u + v <= 1
        a, b, c = corners
        pts = a + (b - a) * u[keep, None] + (c - a) * v[keep, None]
        P.append(pts)
        N.append(np.tile(n / np.linalg.norm(n), (len(pts), 1)))
        F.append(np.full(len(pts), k))
        k += 1

    m = 16
    for i in range(m):
        a0, a1 = TAU * i / m, TAU * (i + 1) / m
        g0 = np.array([GIRDLE_R * np.cos(a0), GIRDLE_Y, GIRDLE_R * np.sin(a0)])
        g1 = np.array([GIRDLE_R * np.cos(a1), GIRDLE_Y, GIRDLE_R * np.sin(a1)])
        t0 = np.array([TABLE_R * np.cos(a0), TABLE_Y, TABLE_R * np.sin(a0)])
        t1 = np.array([TABLE_R * np.cos(a1), TABLE_Y, TABLE_R * np.sin(a1)])
        culet = np.array([0, CULET_Y, 0])
        crown = np.cross(g1 - g0, t0 - g0)
        crown *= np.sign(crown[1]) or 1
        facet((g0, g1, t0), crown)
        facet((g1, t1, t0), crown * [1, 1.15, 1])
        pav = np.cross(culet - g0, g1 - g0)
        pav *= -np.sign(pav[1]) or 1
        facet((g0, g1, culet), pav)
        facet((np.array([0, TABLE_Y, 0]), t0, t1), np.array([0, 1.0, 0]))      # table
    return np.concatenate(P), np.concatenate(N), np.concatenate(F)


def prongs():
    P, N = [], []
    for i in range(4):
        a = TAU * (i + 0.5) / 4
        base = np.array([0.08 * np.cos(a), R_BAND + 0.02, 0.08 * np.sin(a)])
        tip = np.array([(GIRDLE_R + 0.015) * np.cos(a), TABLE_Y - 0.03, (GIRDLE_R + 0.015) * np.sin(a)])
        u, phi = [x.ravel() for x in np.meshgrid(np.linspace(0, 1, 40), np.linspace(0, TAU, 12, endpoint=False))]
        d = (tip - base) / np.linalg.norm(tip - base)
        s = np.cross(d, [0, 0, 1.0])
        s /= np.linalg.norm(s)
        w = np.cross(s, d)
        n = np.cos(phi)[:, None] * s + np.sin(phi)[:, None] * w
        P.append(base + u[:, None] * (tip - base) + n * 0.022)
        N.append(n)
    return np.concatenate(P), np.concatenate(N)


BAND_P, BAND_N = band()
GEM_P, GEM_N, GEM_F = diamond()
PRONG_P, PRONG_N = prongs()
RING_P = np.concatenate([BAND_P, PRONG_P, GEM_P])
RING_N = np.concatenate([BAND_N, PRONG_N, GEM_N])
RING_PART = np.concatenate([np.zeros(len(BAND_P), int), np.ones(len(PRONG_P), int), 10 + GEM_F])
RING_KIND = np.concatenate([np.full(len(BAND_P), BAND), np.full(len(PRONG_P), PRONG), np.full(len(GEM_P), GEM)])
GEM_F_ALL = np.concatenate([np.zeros(len(BAND_P) + len(PRONG_P), int), GEM_F])


# lights, motion

def lights(t):
    key = np.array([0.0, 0.8, 0.6])
    rim = np.array([np.cos(t * 0.9), 0.5, np.sin(t * 0.9)])
    return key / np.linalg.norm(key), rim / np.linalg.norm(rim)


def spin(t):
    return TAU * max(t - DRAW[1] - 0.6, 0) / TURN


def rot_y(p, a):
    c, s = np.cos(a), np.sin(a)
    return np.stack([c * p[:, 0] + s * p[:, 2], p[:, 1], -s * p[:, 0] + c * p[:, 2]], 1)


def draw_order():
    # band bottom -> top on both sides, then prongs, then the stone from the culet up
    p = RING_P
    up_band = np.abs(np.angle(np.exp(1j * (np.arctan2(p[:, 1], p[:, 0]) + np.pi / 2)))) / np.pi
    prong = 1.0 + (p[:, 1] - R_BAND) / (TABLE_Y - R_BAND) * 0.25
    gem = 1.25 + (p[:, 1] - CULET_Y) / (TABLE_Y - CULET_Y) * 0.45
    return np.select([RING_KIND == BAND, RING_KIND == PRONG], [up_band, prong], gem)


ORDER = draw_order()        # 0 .. 1.7
DRAW = (0.3, 5.6)           # drawing starts / ends (s)


def reveal(t):
    head = (t - DRAW[0]) / (DRAW[1] - DRAW[0]) * ORDER.max()
    k = smooth((head - ORDER) / 0.06)
    return k, np.abs(head - ORDER) < 0.035      # visibility, drawing tip


# particles

SPK = np.random.default_rng(13).random((2600, 5))
SPK_IDX = (SPK[:, 0] * len(RING_P)).astype(int)
SPK_BORN = DRAW[0] + ORDER[SPK_IDX] / ORDER.max() * (DRAW[1] - DRAW[0])


def sparks(t):
    # thrown off the drawing tip, pulled down, fading out
    age = t - SPK_BORN
    life = 0.9 + 0.9 * SPK[:, 1]
    alive = (age > 0) & (age < life)
    a = SPK[:, 2] * TAU
    v = np.stack([np.cos(a) * (0.4 + 0.6 * SPK[:, 3]), 0.5 + 0.9 * SPK[:, 4], np.sin(a) * (0.4 + 0.6 * SPK[:, 3])], 1)
    p = RING_P[SPK_IDX] + v * age[:, None] - np.array([0, 1.0, 0]) * (0.5 * 2.2 * age ** 2)[:, None]
    glow = np.clip(1 - age / life, 0, 1) ** 1.3 * alive * (0.6 + 0.4 * np.sin(age * 30 + a))
    return p, glow, np.where(SPK[:, 3] > 0.7, GLINT, DUST)


def flash(t):
    return np.exp(-((t - DRAW[1] - 0.15) / 0.18) ** 2)


BURST = np.random.default_rng(14).random((700, 3))


def burst(t):
    # expanding ring of light when the drawing finishes
    k = t - DRAW[1] - 0.1
    if k < 0 or k > 2.2:
        return np.zeros((0, 3)), np.zeros(0), np.zeros(0, int)
    a = BURST[:, 0] * TAU
    r = 0.85 + 2.4 * (1 - np.exp(-k * 2.0)) * (0.97 + 0.06 * BURST[:, 1])
    p = np.stack([np.cos(a) * r, 0.42 + np.sin(a) * r, (BURST[:, 2] - 0.5) * 0.05 - 0.4], 1)
    glow = np.clip(1 - k / 2.2, 0, 1) ** 1.5
    kinds = np.select([BURST[:, 1] < 0.25, BURST[:, 1] < 0.45, BURST[:, 1] < 0.6], [FIRE_R, FIRE_B, FIRE_G], GLINT)
    return p, np.full(len(a), glow), kinds


HALO = np.random.default_rng(15).random((900, 2))


def halos(t):
    # two tilted orbits
    on = smooth((t - DRAW[1] - 0.3) / 1.5)
    if on <= 0:
        return np.zeros((0, 3)), np.zeros(0), np.zeros(0, int)
    n = len(HALO) // 2
    P = []
    for i, tilt in enumerate((0.55, -0.55)):
        a = HALO[i * n:(i + 1) * n, 0] * TAU + t * (0.9 if i == 0 else -0.7)
        x, y, z = np.cos(a) * 1.05, np.sin(a) * 0.32, np.sin(a) * 0.95
        c, s = np.cos(tilt), np.sin(tilt)
        P.append(np.stack([x * c - y * s, x * s + y * c + 0.42, z], 1))
    p = np.concatenate(P)
    head = (HALO[:, 0] * 37 + t * 1.3) % 1.0
    glow = on * (0.15 + 0.85 * np.clip(1 - head * 3, 0, 1)) * (0.6 + 0.4 * np.sin(t * 6 + HALO[:, 1] * TAU))
    return p, glow, np.where(HALO[:, 1] > 0.5, DUST, GLINT)


DUSTS = np.random.default_rng(5).random((520, 4))


def dust(t):
    d = DUSTS
    a = d[:, 0] * TAU + t * (0.3 + 0.5 * d[:, 1])
    r = 0.9 + 0.8 * d[:, 2]
    y = ((d[:, 3] * 2.4 + t * 0.08) % 2.4) - 0.8
    p = np.stack([np.cos(a) * r, y, np.sin(a) * r * 0.7], 1)
    tw = 0.25 + 0.75 * np.clip(np.sin(t * (2 + 3 * d[:, 1]) + d[:, 0] * 40), 0, 1) ** 4
    return p, tw * smooth((t - BUILD + 0.5) / 1.5)


STARS = np.random.default_rng(2).random((90, 3))


# camera, scene

def look(P):
    target, eye = np.array([0, 0.22, 0]), np.array([0.0, 0.85, 4.7])
    fwd = (target - eye) / np.linalg.norm(target - eye)
    right = np.cross(fwd, [0, 1.0, 0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    rel = P - eye
    return rel @ right, rel @ up, rel @ fwd, -fwd


def scene(t):
    yaw = spin(t)
    body = rot_y(RING_P, yaw)
    normal = rot_y(RING_N, yaw)
    k, tip = reveal(t)

    key, rim = lights(t)
    view = look(np.zeros((1, 3)))[3]
    diff = np.clip(normal @ key, 0, 1)
    half_k = (key + view) / np.linalg.norm(key + view)
    half_r = (rim + view) / np.linalg.norm(rim + view)
    spec = np.clip(normal @ half_k, 0, 1) ** 30 + 0.9 * np.clip(normal @ half_r, 0, 1) ** 24
    metal = RING_KIND != GEM
    shade = np.where(metal, 0.5 + 0.35 * diff + 0.9 * spec, 0.8 + 0.2 * diff + 1.4 * spec)
    kind = RING_KIND.copy()
    glint = (~metal) & (spec > 0.18)
    fire = (GEM_F_ALL + int(t * 7)) % 4
    kind = np.where(glint, np.select([fire == 0, fire == 1, fire == 2], [GLINT, FIRE_R, FIRE_B], FIRE_G), kind)
    kind = np.where(metal & (spec > 0.5), GLINT, kind)
    part = np.where(k < 0.96, -2, RING_PART)
    shade = np.where(k < 0.96, np.where(tip, 1.2, 0.0), shade)
    kind = np.where(k < 0.96, np.where(tip, GLINT, DUST), kind)

    refl = body.copy()                                  # floor reflection
    refl[:, 1] = 2 * FLOOR - refl[:, 1] - 0.06
    rs = np.where(k >= 0.96, 0.22 * np.clip(shade, 0, 1.2), 0)

    dp, ds = dust(t)
    stars = np.stack([STARS[:, 0] * 10 - 5, STARS[:, 1] * 6 - 1.5, np.full(len(STARS), -6.0)], 1)
    sp, ss, st = sparks(t)
    bp, bs, bt = burst(t)
    hp, hs, ht = halos(t)
    shade = shade * (1 + 0.8 * flash(t))
    P = np.concatenate([body, refl, dp, stars, sp, bp, hp])
    shade = np.concatenate([shade, rs, ds, 0.1 + 0.1 * np.sin(t * 2 + STARS[:, 2] * TAU), ss, bs, hs])
    tag = np.concatenate([kind, np.full(len(refl), REFL), np.full(len(dp), DUST), np.full(len(stars), STAR), st, bt, ht])
    pp = np.concatenate([part, np.full(len(P) - len(part), -1)])
    return P, shade, tag, pp, {"deg": float(np.degrees(yaw) % 360), "glints": int(glint.sum())}


def outline_letter(gx, gy):
    ax, ay = np.abs(gx), np.abs(gy)
    c = np.where(ax > 2 * ay, AT["|"], np.where(ay > 2 * ax, np.where(gy > 0, AT["_"], AT["-"]),
                 np.where(gx * gy > 0, AT["/"], AT["\\"])))
    return np.where((ax + ay) == 0, AT["+"], c)


def frame(t, cols, rows, aspect):
    P, shade, T, K, info = scene(t)
    x, y, z, _ = look(P)
    F = 2.9
    zz = np.maximum(z, 0.2)
    sx = (0.5 + x / zz * F * 0.5) * cols
    sy = (0.5 - y / zz * F * 0.5 * aspect) * rows
    order = np.argsort(-z)                              # far to near
    sx, sy, z, shade, T, K = sx[order], sy[order], z[order], shade[order], T[order], K[order]
    ix, iy = sx.astype(int), sy.astype(int)
    ok = (ix >= 0) & (ix < cols) & (iy >= 0) & (iy < rows) & (z > 0.2)

    part = np.full((rows, cols), -1)
    depth = np.full((rows, cols), 99.0)
    lum = np.zeros((rows, cols))
    tag = np.zeros((rows, cols), int)
    b = ok & (K >= 0)
    part[iy[b], ix[b]], depth[iy[b], ix[b]] = K[b], z[b]
    lum[iy[b], ix[b]], tag[iy[b], ix[b]] = shade[b], T[b]
    body = part >= 0

    # outlines: cells next to empty space or to a different part further back
    pp = np.pad(part, 1, constant_values=-1)
    pd = np.pad(depth, 1, constant_values=99)
    edge = np.zeros_like(body)
    gx = np.zeros((rows, cols))
    gy = np.zeros((rows, cols))
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == dy == 0:
                continue
            nb = pp[1 + dy:rows + 1 + dy, 1 + dx:cols + 1 + dx]
            nz = pd[1 + dy:rows + 1 + dy, 1 + dx:cols + 1 + dx]
            out = body & ((nb < 0) | ((nb != part) & (nz > depth + 0.02)))
            gx += dx * out
            gy += dy * out
            if dx == 0 or dy == 0:
                edge |= out

    r, c = np.mgrid[0:rows, 0:cols]
    h = (r * 7919 + c * 104729) % 9973
    level = np.clip((lum * 3.2).astype(int), 0, 3)
    inside = FILL[level, (h + (t * 4 + h % 11).astype(int)) % 12]
    sparkle = body & np.isin(tag, [GLINT, FIRE_R, FIRE_B, FIRE_G])
    char = np.where(body, np.where(edge, outline_letter(gx, gy), inside), 0)
    char = np.where(sparkle, AT["*"], char)
    lum = np.where(body, np.where(edge, np.maximum(0.75, lum), np.clip(lum, 0.3, 1)), 0)

    # particles, reflection, stars
    fx = ok & (K < 0) & (shade > 0.02)
    hidden = body[iy[fx], ix[fx]] & (depth[iy[fx], ix[fx]] < z[fx])
    fi, fj, fs, ft = (a[fx][~hidden] for a in (iy, ix, shade, T))
    keep = ~body[fi, fj]
    fi, fj, fs, ft = fi[keep], fj[keep], fs[keep], ft[keep]
    refl = FILL[np.clip((fs * 6).astype(int), 0, 1), (fi * 31 + fj) % 12]
    mote = np.where(fs > 0.7, AT["*"], np.where(fs > 0.35, AT["+"], AT["."]))
    ember = np.where((fi + fj) % 3 == 0, AT["&"], np.where((fi + fj) % 3 == 1, AT["8"], AT["%"]))
    letter = np.select([ft == REFL, ft == DUST, ft == STAR], [refl, mote, AT["."]], ember)
    char[fi, fj], lum[fi, fj], tag[fi, fj] = letter, np.clip(fs, 0, 1), ft

    # rays from the six brightest glints
    score = np.where(sparkle, lum + (h % 7) * 0.01, 0)
    flat = np.argsort(score.ravel())[::-1][:6]
    for n, idx in enumerate(flat):
        if score.ravel()[idx] <= 0:
            break
        cy, cx = divmod(int(idx), cols)
        pulse = 0.5 + 0.5 * np.sin(t * 7 + n * 1.7)
        reach = int(5 + 9 * pulse)
        for dist in range(1, reach + 1):
            fade = 1 - dist / (reach + 1)
            for (dy, dx, ch) in ((0, 1, "-"), (0, -1, "-"), (1, 0, "|"), (-1, 0, "|")):
                yy, xx = cy + dy * dist, cx + dx * dist * 2
                if 0 <= yy < rows and 0 <= xx < cols and lum[yy, xx] < fade:
                    char[yy, xx], lum[yy, xx], tag[yy, xx] = AT[ch], fade, GLINT
            if dist <= reach // 2:
                for (dy, dx, ch) in ((1, 1, "\\"), (-1, -1, "\\"), (1, -1, "/"), (-1, 1, "/")):
                    yy, xx = cy + dy * dist, cx + dx * dist
                    if 0 <= yy < rows and 0 <= xx < cols and lum[yy, xx] < fade * 0.7:
                        char[yy, xx], lum[yy, xx], tag[yy, xx] = AT[ch], fade * 0.7, GLINT
        char[cy, cx], lum[cy, cx] = AT["*"], 1.0
    return char, lum, tag, info, int(body.sum())


# window

TITLE = "Diamond Ring - JIN2"
SIGN = "@jin_22ee · 2026"
CW, CH, FONT_PX = 6, 9, 10
W, H = 1080, 960
COLS, ROWS = W // CW, H // CH
ASPECT = (COLS * CW) / (ROWS * CH)
TRAIL = 0.5
COLORS = np.array([[214, 220, 232], [232, 236, 244], [226, 238, 255], [255, 255, 255], [255, 170, 200],
                   [160, 210, 255], [255, 230, 160], [255, 214, 140], [150, 150, 165], [120, 120, 140]],
                  np.float32) / 255
DIM, PINK = (110, 104, 120), (255, 122, 182)
MONO = "consolas,menlo,dejavusansmono,couriernew,monospace"


def hud(info):
    return f"rotation {info['deg']:06.2f}°   glints {info['glints']:4d}"


def glyphs():
    font = pygame.font.SysFont(MONO, FONT_PX)
    atlas = []
    for ch in CHARS:
        s = pygame.Surface((CW, CH))
        g = font.render(ch, True, (255, 255, 255))
        s.blit(g, ((CW - g.get_width()) // 2, (CH - g.get_height()) // 2))
        atlas.append(pygame.surfarray.array3d(s)[:, :, 0].T.astype(np.float32) / 255)
    return np.stack(atlas)


class Canvas:
    def __init__(self, atlas):
        self.atlas = (atlas * 255).astype(np.uint8)
        self.shape = pygame.Surface((COLS * CW, ROWS * CH))
        self.trail = pygame.Surface((W, H))
        self.fade = (int(TRAIL * 255),) * 3

    def reset(self):
        self.trail.fill((0, 0, 0))

    def draw(self, t):
        char, lum, tag, info, count = frame(t, COLS, ROWS, ASPECT)
        mask = self.atlas[char.T].transpose(0, 3, 1, 2).reshape(COLS * CW, ROWS * CH)
        pygame.surfarray.blit_array(self.shape, np.repeat(mask[..., None], 3, 2))
        col = (COLORS[np.clip(tag, 0, len(COLORS) - 1)] * np.clip(lum, 0, 1)[..., None] * 255).astype(np.uint8)
        tint = pygame.transform.scale(pygame.surfarray.make_surface(col.transpose(1, 0, 2)), self.shape.get_size())
        self.shape.blit(tint, (0, 0), special_flags=pygame.BLEND_MULT)
        self.trail.fill(self.fade, special_flags=pygame.BLEND_MULT)     # previous frame fades
        self.trail.blit(self.shape, (0, 0), special_flags=pygame.BLEND_MAX)
        return self.trail, info, count


def blur(surf, k, radius):
    small = pygame.transform.smoothscale(surf, (W // k, H // k))
    if hasattr(pygame.transform, "gaussian_blur"):
        small = pygame.transform.gaussian_blur(small, max(1, radius // k))
    return pygame.transform.smoothscale(small, (W, H))


def glow_layer(surf, amount):
    layer = surf.copy()
    layer.fill((int(255 * amount),) * 3, special_flags=pygame.BLEND_MULT)
    return layer


def window_size():
    # shrink on small / scaled displays
    info = pygame.display.Info()
    k = 1.0
    if info.current_w > 0 and info.current_h > 0:
        k = min(1.0, 0.85 * info.current_h / H, 0.95 * info.current_w / W)
    return int(W * k), int(H * k)


async def main():
    pygame.init()
    size = window_size()
    window = pygame.display.set_mode(size)
    pygame.display.set_caption(TITLE)
    screen = pygame.Surface((W, H))
    small = pygame.font.SysFont(MONO, 17)
    canvas, clock = Canvas(glyphs()), pygame.time.Clock()
    t, paused = 0.0, False
    art, info, count = canvas.draw(t)
    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                pygame.quit()
                return
            if e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE:
                paused = not paused
            if e.type == pygame.KEYDOWN and e.key == pygame.K_r:
                t = 0.0
                canvas.reset()
            if e.type == pygame.KEYDOWN and e.key == pygame.K_s:
                pygame.image.save(screen, str(Path(__file__).with_name(f"diamond_ring_{int(time.time())}.png")))
        dt = clock.tick(30) / 1000
        if not paused:
            t += min(dt, 1 / 15)
            if t > LOOP:
                t = 0.0
                canvas.reset()
            art, info, count = canvas.draw(t)
        screen.fill((0, 0, 0))
        screen.blit(art, (0, 0))
        screen.blit(glow_layer(blur(art, 4, 5), 0.75), (0, 0), special_flags=pygame.BLEND_ADD)
        screen.blit(glow_layer(blur(art, 8, 22), 0.45), (0, 0), special_flags=pygame.BLEND_ADD)
        y = H - 70
        screen.blit(small.render(SIGN, True, (120, 112, 128)), (64, y - 26))
        screen.blit(small.render(hud(info), True, DIM), (64, y))
        screen.blit(small.render(f"letters {count:05,d}", True, DIM), (64, y + 26))
        tt = small.render(f"{'●' if int(t * 2) % 2 == 0 else ' '} t {t:05.2f}s", True, PINK)
        screen.blit(tt, (W - 64 - tt.get_width(), 30))
        window.blit(screen if size == (W, H) else pygame.transform.smoothscale(screen, size), (0, 0))
        pygame.display.flip()

        # REQUIRED FOR WEB BROWSERS
        await asyncio.sleep(0)


if __name__ == "__main__":
    asyncio.run(main())
  
