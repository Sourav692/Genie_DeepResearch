"""Generate a 640x360 thumbnail for the Databricks App overview page."""

from PIL import Image, ImageDraw, ImageFont
import math, os

W, H = 640, 360

img = Image.new("RGB", (W, H))
draw = ImageDraw.Draw(img)

# Background gradient: dark navy to deep blue
for y in range(H):
    t = y / H
    r = int(15 * (1 - t) + 30 * t)
    g = int(23 * (1 - t) + 58 * t)
    b = int(42 * (1 - t) + 138 * t)
    draw.line([(0, y), (W, y)], fill=(r, g, b))

# Subtle radial glow (top-right)
cx, cy, radius = 500, 60, 220
for ry in range(max(0, cy - radius), min(H, cy + radius)):
    for rx in range(max(0, cx - radius), min(W, cx + radius)):
        d = math.sqrt((rx - cx) ** 2 + (ry - cy) ** 2)
        if d < radius:
            alpha = 0.08 * (1 - d / radius) ** 2
            pr, pg, pb = img.getpixel((rx, ry))
            nr = min(255, int(pr + 100 * alpha))
            ng = min(255, int(pg + 160 * alpha))
            nb = min(255, int(pb + 255 * alpha))
            img.putpixel((rx, ry), (nr, ng, nb))

# Second glow (bottom-left)
cx2, cy2, radius2 = 100, 320, 180
for ry in range(max(0, cy2 - radius2), min(H, cy2 + radius2)):
    for rx in range(max(0, cx2 - radius2), min(W, cx2 + radius2)):
        d = math.sqrt((rx - cx2) ** 2 + (ry - cy2) ** 2)
        if d < radius2:
            alpha = 0.06 * (1 - d / radius2) ** 2
            pr, pg, pb = img.getpixel((rx, ry))
            nr = min(255, int(pr + 59 * alpha))
            ng = min(255, int(pg + 130 * alpha))
            nb = min(255, int(pb + 246 * alpha))
            img.putpixel((rx, ry), (nr, ng, nb))

# Decorative grid dots
for gx in range(30, W, 40):
    for gy in range(30, H, 40):
        d_center = math.sqrt((gx - W / 2) ** 2 + (gy - H / 2) ** 2)
        opacity = max(0, 0.12 - d_center / 2000)
        if opacity > 0:
            pr, pg, pb = img.getpixel((gx, gy))
            nr = min(255, int(pr + 255 * opacity))
            ng = min(255, int(pg + 255 * opacity))
            nb = min(255, int(pb + 255 * opacity))
            draw.ellipse([gx - 1, gy - 1, gx + 1, gy + 1], fill=(nr, ng, nb))

# Icon box (rounded rectangle with sparkle symbol)
icon_x, icon_y, icon_s = W // 2 - 24, 80, 48
draw.rounded_rectangle(
    [icon_x, icon_y, icon_x + icon_s, icon_y + icon_s],
    radius=12,
    fill=(37, 99, 235),
    outline=(59, 130, 246),
    width=1,
)

# Sparkle inside the icon (3 diamond shapes)
def draw_sparkle(draw, cx, cy, size, fill):
    pts = [(cx, cy - size), (cx + size * 0.3, cy), (cx, cy + size), (cx - size * 0.3, cy)]
    draw.polygon(pts, fill=fill)
    pts2 = [(cx - size * 0.7, cy), (cx, cy - size * 0.3), (cx + size * 0.7, cy), (cx, cy + size * 0.3)]
    draw.polygon(pts2, fill=fill)

draw_sparkle(draw, W // 2, icon_y + icon_s // 2, 14, (255, 255, 255))
draw_sparkle(draw, W // 2 + 20, icon_y + 8, 5, (147, 197, 253))

# Load a font (fallback to default)
def get_font(size, bold=False):
    paths = [
        "/System/Library/Fonts/SFNSDisplay.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/HelveticaNeue.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in paths:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()

font_title = get_font(28, bold=True)
font_sub = get_font(14)
font_badge = get_font(11)

# Title
title = "Genie Deep Research"
bbox = draw.textbbox((0, 0), title, font=font_title)
tw = bbox[2] - bbox[0]
draw.text(((W - tw) // 2, 148), title, fill=(255, 255, 255), font=font_title)

# Subtitle
sub = "AI-Powered Multi-Agent Analytics"
bbox2 = draw.textbbox((0, 0), sub, font=font_sub)
sw = bbox2[2] - bbox2[0]
draw.text(((W - sw) // 2, 188), sub, fill=(148, 163, 184), font=font_sub)

# Feature pills
pills = ["Supervisor Agent", "Parallel Execution", "Quality Check"]
pill_y = 230
total_width = sum(draw.textbbox((0, 0), p, font=font_badge)[2] - draw.textbbox((0, 0), p, font=font_badge)[0] + 28 for p in pills) + 12 * (len(pills) - 1)
px = (W - total_width) // 2

for pill_text in pills:
    tb = draw.textbbox((0, 0), pill_text, font=font_badge)
    pw = tb[2] - tb[0] + 28
    ph = 26
    draw.rounded_rectangle(
        [px, pill_y, px + pw, pill_y + ph],
        radius=13,
        fill=(30, 58, 95),
        outline=(59, 130, 246, 80),
        width=1,
    )
    draw.text((px + 14, pill_y + 6), pill_text, fill=(147, 197, 253), font=font_badge)
    px += pw + 12

# Bottom accent line
draw.rounded_rectangle([W // 2 - 40, 290, W // 2 + 40, 293], radius=2, fill=(37, 99, 235))

# "Databricks App" label at bottom
db_text = "DATABRICKS APP"
bbox3 = draw.textbbox((0, 0), db_text, font=font_badge)
dw = bbox3[2] - bbox3[0]
draw.text(((W - dw) // 2, 306), db_text, fill=(100, 116, 139), font=font_badge)

out_path = os.path.join(os.path.dirname(__file__), "app_thumbnail.png")
img.save(out_path, "PNG", optimize=True)

file_size = os.path.getsize(out_path)
print(f"Saved: {out_path} ({file_size / 1024:.1f} KB, {W}x{H})")
