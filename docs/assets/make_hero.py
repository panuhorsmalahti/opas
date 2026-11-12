"""Generate docs/assets/opas-hero.svg, the README header image.

Run: python3 docs/assets/make_hero.py   (standard library only)
"""

import math
import random
from pathlib import Path

W, H = 1600, 760
OUT = Path(__file__).with_name("opas-hero.svg")
FONT = "Inter, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "'JetBrains Mono', 'SF Mono', Menlo, Consolas, monospace"

BG0, BG1 = "#0b1020", "#121a33"
CARD, CARD_EDGE = "#151e3a", "#2a3766"
TEXT, MUTED, FAINT = "#eef2ff", "#a8b3d6", "#6c7aa8"
TEAL, VIOLET, PINK, AMBER = "#2dd4bf", "#8b5cf6", "#f472b6", "#fbbf24"

out: list[str] = []


def add(s: str) -> None:
    out.append(s)


def text(x, y, s, size=16, fill=TEXT, weight=400, anchor="start", family=FONT, extra=""):
    add(f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" font-weight="{weight}" '
        f'fill="{fill}" text-anchor="{anchor}" {extra}>{s}</text>')


def card(x, y, w, h, title, step):
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="22" fill="url(#cardFill)" '
        f'stroke="{CARD_EDGE}" stroke-width="1.5" filter="url(#shadow)"/>')
    add(f'<circle cx="{x + 34}" cy="{y + 36}" r="15" fill="url(#accent)"/>')
    text(x + 34, y + 41.5, str(step), 15, "#0b1020", 800, "middle")
    text(x + 60, y + 42, title, 19, TEXT, 700)


def heat(v):
    """Map 0..1 to a dark-blue → teal → violet → pink colour ramp."""
    stops = [(0.0, (22, 32, 66)), (0.35, (45, 212, 191)), (0.7, (139, 92, 246)), (1.0, (244, 114, 182))]
    for (a, ca), (b, cb) in zip(stops, stops[1:]):
        if v <= b:
            t = (v - a) / (b - a)
            c = [round(ca[i] + (cb[i] - ca[i]) * t) for i in range(3)]
            return f"rgb({c[0]},{c[1]},{c[2]})"
    return "rgb(244,114,182)"


def arrow(x1, y1, x2, y2, bend=0.0):
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2 + bend
    add(f'<path d="M{x1},{y1} Q{mx},{my} {x2},{y2}" fill="none" stroke="url(#flow)" '
        f'stroke-width="3.5" stroke-linecap="round" marker-end="url(#head)" filter="url(#glowSoft)"/>')


# --- document, defs, background --------------------------------------------------------
add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
    f'role="img" aria-labelledby="t d">')
add('<title id="t">opas: routing LLM requests on model activations</title>')
add('<desc id="d">A request in any of four APIs goes through one prefill pass of an encoder LLM; '
    'the hidden state of one layer feeds a linear probe that predicts success and cost per model tier; '
    'opas routes to the tier with the best success minus lambda times cost.</desc>')
add(f"""<defs>
  <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0" stop-color="{BG0}"/><stop offset="1" stop-color="{BG1}"/>
  </linearGradient>
  <radialGradient id="glowA" cx="0.3" cy="0.35" r="0.6">
    <stop offset="0" stop-color="{VIOLET}" stop-opacity="0.22"/><stop offset="1" stop-color="{VIOLET}" stop-opacity="0"/>
  </radialGradient>
  <radialGradient id="glowB" cx="0.8" cy="0.8" r="0.5">
    <stop offset="0" stop-color="{TEAL}" stop-opacity="0.16"/><stop offset="1" stop-color="{TEAL}" stop-opacity="0"/>
  </radialGradient>
  <linearGradient id="cardFill" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0" stop-color="#18224a"/><stop offset="1" stop-color="{CARD}"/>
  </linearGradient>
  <linearGradient id="accent" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0" stop-color="{TEAL}"/><stop offset="1" stop-color="{VIOLET}"/>
  </linearGradient>
  <linearGradient id="flow" x1="0" y1="0" x2="1" y2="0">
    <stop offset="0" stop-color="{TEAL}"/><stop offset="1" stop-color="{VIOLET}"/>
  </linearGradient>
  <linearGradient id="bar" x1="0" y1="0" x2="1" y2="0">
    <stop offset="0" stop-color="{TEAL}"/><stop offset="1" stop-color="{VIOLET}"/>
  </linearGradient>
  <linearGradient id="chosen" x1="0" y1="0" x2="1" y2="0">
    <stop offset="0" stop-color="#1f3b5c"/><stop offset="1" stop-color="#3b2a6b"/>
  </linearGradient>
  <marker id="cacheHead" viewBox="0 0 10 10" refX="4" refY="5" markerWidth="4.2" markerHeight="4.2" orient="auto">
    <path d="M0,0 L10,5 L0,10 z" fill="{AMBER}"/>
  </marker>
  <marker id="head" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="9" markerHeight="9" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="{VIOLET}"/>
  </marker>
  <filter id="shadow" x="-10%" y="-10%" width="120%" height="125%">
    <feDropShadow dx="0" dy="10" stdDeviation="14" flood-color="#000" flood-opacity="0.45"/>
  </filter>
  <filter id="glow" x="-50%" y="-200%" width="200%" height="500%">
    <feGaussianBlur stdDeviation="6" result="b"/>
    <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
  </filter>
  <filter id="glowSoft" x="-20%" y="-50%" width="140%" height="200%">
    <feGaussianBlur stdDeviation="2.5" result="b"/>
    <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
  </filter>
  <pattern id="grid" width="32" height="32" patternUnits="userSpaceOnUse">
    <path d="M32 0 H0 V32" fill="none" stroke="#ffffff" stroke-opacity="0.035" stroke-width="1"/>
  </pattern>
</defs>""")
add(f'<rect width="{W}" height="{H}" rx="28" fill="url(#bg)"/>')
add(f'<rect width="{W}" height="{H}" rx="28" fill="url(#grid)"/>')
add(f'<rect width="{W}" height="{H}" rx="28" fill="url(#glowA)"/>')
add(f'<rect width="{W}" height="{H}" rx="28" fill="url(#glowB)"/>')

# --- title ---------------------------------------------------------------------------
text(56, 82, "opas", 46, TEXT, 800, extra='letter-spacing="-1"')
text(176, 82, "route every LLM request by reading a model's activations", 22, MUTED, 400)
text(56, 116, "one prefill pass · no tokens generated · predicts success, cost and response time · you choose the balance",
     16, FAINT, 400)

TOP, BOT = 150, 690
CH = BOT - TOP

# --- 1. request ------------------------------------------------------------------------
x1, w1 = 40, 300
card(x1, TOP, w1, CH, "Request", 1)
text(x1 + 24, TOP + 84, "any of four APIs, unchanged", 13.5, FAINT)
apis = [("OpenAI Chat", "/v1/chat/completions"), ("Anthropic Messages", "/v1/messages"),
        ("Open Responses", "/v1/responses"), ("Gemini Interactions", "/v1beta/interactions")]
for i, (name, path) in enumerate(apis):
    y = TOP + 104 + i * 58
    add(f'<rect x="{x1 + 22}" y="{y}" width="{w1 - 44}" height="46" rx="12" fill="#0f1733" stroke="{CARD_EDGE}"/>')
    add(f'<circle cx="{x1 + 42}" cy="{y + 23}" r="5" fill="{[TEAL, VIOLET, PINK, AMBER][i]}"/>')
    text(x1 + 56, y + 21, name, 14.5, TEXT, 600)
    text(x1 + 56, y + 37, path, 11.5, FAINT, 400, family=MONO)
# chat bubble
by = TOP + 352
add(f'<rect x="{x1 + 22}" y="{by}" width="{w1 - 44}" height="132" rx="16" fill="#1d2a57" stroke="#34457f"/>')
add(f'<path d="M{x1 + 50},{by + 132} l10,16 l10,-16 z" fill="#1d2a57" stroke="#34457f"/>')
add(f'<rect x="{x1 + 52}" y="{by + 127}" width="22" height="8" fill="#1d2a57"/>')
text(x1 + 40, by + 30, "user", 12, TEAL, 700, extra='letter-spacing="1"')
for k, line in enumerate(["Rewrite our image-resize", "module in Rust, compile", "it to WASM, and call it",
                          "from the Node.js server."]):
    text(x1 + 40, by + 56 + k * 21, line, 15, TEXT, 400)
text(x1 + 24, BOT - 18, '"model": "auto"', 13, MUTED, 400, family=MONO)

# --- 2. encoder ------------------------------------------------------------------------
x2, w2 = 400, 400
card(x2, TOP, w2, CH, "Open-weight model reads it", 2)
text(x2 + 24, TOP + 84, "Qwen3.5-9B · one pass over the prompt, no output", 13.5, FAINT)
rng = random.Random(7)
n_layers, n_cells = 11, 22
lx, lw = x2 + 52, 250
cell = lw / n_cells
row_h, gap = 20, 8
stack_top = TOP + 112
chosen_layer = 3  # index from the top of the drawn stack
for r in range(n_layers):
    y = stack_top + r * (row_h + gap)
    depth = 1 - r / (n_layers - 1)  # 1 at the top (deep layers), 0 at the bottom
    phase = rng.uniform(0, 6.28)
    for c in range(n_cells):
        v = 0.5 + 0.5 * math.sin(c * 0.55 + phase) * (0.35 + 0.65 * depth) + rng.uniform(-0.18, 0.18)
        v = min(1, max(0, v * (0.55 + 0.45 * depth)))
        add(f'<rect x="{lx + c * cell:.1f}" y="{y}" width="{cell - 2:.1f}" height="{row_h}" rx="3" fill="{heat(v)}" '
            f'opacity="{0.55 + 0.45 * depth:.2f}"/>')
    label = "L32" if r == 0 else ("L1" if r == n_layers - 1 else "")
    if label:
        text(lx - 12, y + 16, label, 11.5, FAINT, 600, "end", family=MONO)
# highlight the probed layer
hy = stack_top + chosen_layer * (row_h + gap)
add(f'<rect x="{lx - 6}" y="{hy - 5}" width="{lw + 8}" height="{row_h + 10}" rx="8" fill="none" '
    f'stroke="{AMBER}" stroke-width="2.5" filter="url(#glow)"/>')
text(lx - 12, hy + 16, "L24", 11.5, AMBER, 700, "end", family=MONO)
# tokens flowing in at the bottom
ty = stack_top + n_layers * (row_h + gap) + 14
for k, tok in enumerate(["Rewrite", "image", "Rust", "…", "WASM", "&lt;think&gt;"]):
    tx = lx - 6 + k * 50
    add(f'<rect x="{tx}" y="{ty}" width="46" height="22" rx="6" fill="#0f1733" stroke="{CARD_EDGE}"/>')
    text(tx + 23, ty + 15, tok, 10.5, MUTED, 500, "middle", family=MONO)
text(x2 + w2 / 2, ty + 50, "its layer activations are the input", 12.5, FAINT, 400, "middle")
text(x2 + w2 / 2, ty + 69, "to a trained model that suggests the best model", 12.5, FAINT, 400, "middle")
# hidden-state vector pulled out of L24
vx, vy = x2 + w2 - 50, hy - 70
add(f'<rect x="{vx - 6}" y="{vy - 6}" width="30" height="{12 * 15 + 12}" rx="9" fill="#0f1733" stroke="{AMBER}" stroke-opacity="0.7"/>')
for k in range(12):
    add(f'<rect x="{vx}" y="{vy + k * 15}" width="18" height="12" rx="3" fill="{heat(rng.uniform(0.25, 1))}"/>')
text(vx + 9, vy - 14, "h", 17, AMBER, 700, "middle", family=MONO)
add(f'<path d="M{lx + lw + 4},{hy + row_h / 2} C{lx + lw + 20},{hy + row_h / 2} {vx - 26},{vy + 76} {vx - 9},{vy + 76}" '
    f'fill="none" stroke="{AMBER}" stroke-width="2" stroke-dasharray="4 4"/>')
text(vx + 9, vy + 12 * 15 + 30, "4096-d", 11.5, AMBER, 600, "middle", family=MONO)

# --- 3. probe ----------------------------------------------------------------------------
x3, w3 = 860, 330
card(x3, TOP, w3, CH, "Trained predictor", 3)
text(x3 + 24, TOP + 84, "rates every candidate from the activations", 13.5, FAINT)
tiers = [("small", 0.58, 1, "2s"), ("medium", 0.79, 2, "6s"), ("large", 0.86, 3, "12s"), ("xlarge", 0.91, 4, "20s")]
hy0 = TOP + 124
text(x3 + 24, hy0, "tier", 12, FAINT, 700, extra='letter-spacing="1"')
text(x3 + 100, hy0, "success", 12, FAINT, 700, extra='letter-spacing="1"')
text(x3 + 262, hy0, "cost", 12, FAINT, 700, "end", extra='letter-spacing="1"')
text(x3 + w3 - 24, hy0, "time", 12, FAINT, 700, "end", extra='letter-spacing="1"')
for i, (name, p, cost, secs) in enumerate(tiers):
    y = TOP + 146 + i * 84
    text(x3 + 24, y + 22, name, 15.5, TEXT, 600)
    bw = 82
    add(f'<rect x="{x3 + 100}" y="{y + 8}" width="{bw}" height="18" rx="9" fill="#0f1733"/>')
    add(f'<rect x="{x3 + 100}" y="{y + 8}" width="{bw * p:.1f}" height="18" rx="9" fill="url(#bar)"/>')
    text(x3 + 100 + bw + 8, y + 22, f"{p:.2f}", 12.5, MUTED, 600, family=MONO)
    text(x3 + 262, y + 22, "$" * cost, 13, AMBER, 700, "end", family=MONO)
    text(x3 + w3 - 24, y + 22, secs, 13, TEAL, 700, "end", family=MONO)
    if i < 3:
        add(f'<line x1="{x3 + 24}" y1="{y + 56}" x2="{x3 + w3 - 24}" y2="{y + 56}" stroke="{CARD_EDGE}" stroke-opacity="0.6"/>')
add(f'<rect x="{x3 + 22}" y="{BOT - 84}" width="{w3 - 44}" height="36" rx="10" fill="#0f1733" stroke="{TEAL}" stroke-opacity="0.5"/>')
text(x3 + w3 / 2, BOT - 61, "lightweight predictor included in the repo", 12.5, TEAL, 600, "middle")
text(x3 + w3 / 2, BOT - 22, "cost and time use your prices and speeds", 12.5, FAINT, 400, "middle")

# --- 4. route ----------------------------------------------------------------------------
x4, w4 = 1250, 310
card(x4, TOP, w4, CH, "Route", 4)
text(x4 + 24, TOP + 84, "you set the balance per request", 13.5, FAINT)
add(f'<rect x="{x4 + 22}" y="{TOP + 102}" width="{w4 - 44}" height="112" rx="12" fill="#0f1733" stroke="{CARD_EDGE}"/>')
for i, (label, pct, color) in enumerate([("quality", 60, VIOLET), ("speed", 20, TEAL), ("cost", 20, AMBER)]):
    y = TOP + 126 + i * 30
    text(x4 + 40, y + 5, label, 13, TEXT, 600)
    bx, bw = x4 + 112, 120
    add(f'<rect x="{bx}" y="{y - 6}" width="{bw}" height="12" rx="6" fill="#1b2650"/>')
    add(f'<rect x="{bx}" y="{y - 6}" width="{bw * pct / 60:.1f}" height="12" rx="6" fill="{color}"/>')
    text(x4 + w4 - 40, y + 5, f"{pct}%", 13, MUTED, 700, "end", family=MONO)
models = [("small", "claude-haiku", False), ("medium", "your-model", True), ("large", "claude-opus", False),
          ("xlarge", "claude-fable", False)]
for i, (tier, model, chosen) in enumerate(models):
    y = TOP + 230 + i * 56
    fill = "url(#chosen)" if chosen else "#0f1733"
    stroke = TEAL if chosen else CARD_EDGE
    add(f'<rect x="{x4 + 22}" y="{y}" width="{w4 - 44}" height="46" rx="13" fill="{fill}" stroke="{stroke}" '
        f'stroke-width="{2.5 if chosen else 1}" {"filter=\"url(#glowSoft)\"" if chosen else ""}/>')
    text(x4 + 42, y + 20, tier, 12, TEAL if chosen else FAINT, 700, extra='letter-spacing="1"')
    text(x4 + 42, y + 38, model, 15, TEXT if chosen else MUTED, 600, family=MONO)
    if chosen:
        add(f'<circle cx="{x4 + w4 - 48}" cy="{y + 23}" r="13" fill="{TEAL}"/>')
        add(f'<path d="M{x4 + w4 - 54},{y + 23} l4,5 l9,-10" fill="none" stroke="#0b1020" stroke-width="3" '
            f'stroke-linecap="round" stroke-linejoin="round"/>')
# cache-aware badge
cy = BOT - 78
add(f'<rect x="{x4 + 22}" y="{cy}" width="{w4 - 44}" height="56" rx="13" fill="#0f1733" stroke="{AMBER}" stroke-opacity="0.55"/>')
# Circular "reuse" arrow: a 290° arc whose arrowhead marker follows the arc's tangent.
icx, icy, ir = x4 + 46, cy + 28, 10
a0, a1 = math.radians(-60), math.radians(230)
sx, sy = icx + ir * math.cos(a0), icy + ir * math.sin(a0)
ex, ey = icx + ir * math.cos(a1), icy + ir * math.sin(a1)
add(f'<path d="M{sx:.2f},{sy:.2f} A{ir},{ir} 0 1,1 {ex:.2f},{ey:.2f}" fill="none" stroke="{AMBER}" '
    f'stroke-width="2.4" stroke-linecap="round" marker-end="url(#cacheHead)"/>')
text(x4 + 70, cy + 24, "cache-aware", 14, AMBER, 700)
text(x4 + 70, cy + 43, "warm prompt cache costs less", 12, MUTED, 400)

# --- arrows between stages -------------------------------------------------------------
arrow(x1 + w1 + 6, TOP + CH / 2 - 40, x2 - 12, TOP + CH / 2 - 40, -6)
arrow(vx + 32, vy + 76, x3 - 10, TOP + 300, 10)
arrow(x3 + w3 + 8, TOP + 300, x4 - 10, TOP + 309, -6)

text(W / 2, H - 30, "OpenAI · Anthropic · Open Responses · Gemini Interactions   ·   "
     "quality / speed / cost weights per request   ·   cache-aware", 14, FAINT, 500, "middle", extra='letter-spacing="0.3"')
add("</svg>")
OUT.write_text("\n".join(out))
print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.1f} KB)")
