# Solo Choir Studio — DESIGN.md

The design system the harmony-line editor is built against. Convention: a single markdown design doc
(à la awesome-design-md) an implementer/agent reads to keep the UI consistent. Aesthetic reference:
Google **Magenta** — clean, technical, functional; the **data visualisation is the interface**.

## 1. Visual theme & atmosphere
**Hybrid, light & airy** — the Vercel/Geist register throughout. White chrome (hairline borders,
monospace pill labels) and a **light frosted-glass editor canvas** (translucent white, blurred) that
floats over a **minimal animated gradient**: grey + cyan, two soft blobs drifting in OPPOSITE
directions (~16s / ~21s), glowing up through the frosted editor. Harmony lines are deepened for
contrast on the light glass. No skeuomorphism, no decorative gradients beyond that one ambient field,
**no 3D**. Precise instrument, not a toy.

## 2. Colour palette & roles
| Role | Hex | Use |
|---|---|---|
**Chrome (light):**
| Canvas | `#ffffff` | app background |
| Canvas-tint | `#fafafa` | subtle panels / pill labels |
| Gradient | `cyan rgba(34,211,238,.24) + grey rgba(100,116,139,.20)`, two blobs, opposite drift 16s/21s | the one ambient field |
| Line/border | `#eaeaea` | hairlines, dividers, pill outlines |
| Ink | `#1a1a1a` | primary text |
| Ink-muted | `#888888` | labels, secondary text |
| Accent | `#0070f3` | interactive (hover/active/links), the ONE accent (Vercel blue) |
| Warn | `#f5a623` | "discard edits?" / destructive confirm |

**Editor glass (light, frosted, translucent):** `background rgba(255,255,255,.42)` + `backdrop-filter
blur(16px) saturate(1.1)` + `1px rgba(255,255,255,.7)` border + a soft `0 8px 30px rgba(40,60,100,.08)`
shadow. The grey↔cyan gradient glows through it. Labels are muted ink (`#888`); the harmony lines use
the **deepened** palette (dark-on-light):
| **Voice — Melody** | `#1a1a1a` | lead line (near-black, thickest) |
| **Voice — Sop** | `#16a34a` | green |
| **Voice — Alto** | `#9333ea` | purple (distinct from Tenor red) |
| **Voice — Tenor** | `#dc2626` | red |
| **Voice — Bass** | `#2563eb` | blue |
Voice colours are the only saturated hues; chrome + glass stay greyscale + the single blue accent.

## 3. Typography
- **Mono** (`ui-monospace, "SF Mono", Menlo, monospace`) for all data/labels/values (notes, times,
  Hz, voice names) — it reads as instrumentation.
- **Sans** (`system-ui, -apple-system, sans-serif`) for prose/buttons.
- Sizes: 12px labels, 13px body, 11px micro-labels; letter-spacing `.04em` on uppercase labels.

## 4. Components
- **Button**: 1px `#2a2a30` border, transparent bg, `#e8e8e8` text, 8px radius, 8×14 padding; hover →
  border `#20c4b8`; active/pressed → bg `rgba(32,196,184,.10)`, border + text `#20c4b8`.
- **Toggle / segmented** (snap mode, style): same border; selected segment uses the accent rule.
- **Record**: a small filled dot; recording → it blinks the warn colour.
- **Note dot**: 3px filled circle in the voice colour; **selected/dragging** → 5px + white ring.
- **Chord-tone band**: a horizontal shaded row `rgba(255,255,255,.04)` behind the grid at chord-tone
  pitches.

## 5. Layout & whitespace
- One screen, no scrolling: a **top bar** (Record · Arrange · style · snap-mode · Render · Export),
  the **editor canvas** filling the rest, a thin **status line** at the bottom.
- Generous margins (≥24px) around the canvas; gridlines faint so the lines dominate.
- The canvas owns ≥75% of the height.

## 6. Depth & elevation
Almost flat. Surfaces are white on near-white (`#ffffff` / `#fafafa`) separated by `#eaeaea`
hairlines. At most one very soft shadow on a floating control; no heavy elevation. Depth comes from
the single ambient radial glow + whitespace, not shadows.

## 7. Do's & don'ts
- DO let the line graph be the hero; keep controls to one row.
- DO use the voice colours consistently everywhere (legend, lines, dots, stems list).
- DON'T add gradients, glassmorphism, rounded "cards", icons-for-decoration, or animation beyond a
  subtle drag/active transition (≤120ms).
- DON'T introduce a second accent colour; the teal is the only interactive hue.

## 8. Responsive behaviour
Desktop-only (pywebview window, min 960×640). The canvas re-fits on resize (recompute the time/pitch
→ pixel mapping); the top bar wraps to two rows below ~1040px.

## 9. Agent-prompt guide
> "Build against studio/DESIGN.md: near-white canvas (#ffffff) with ONE minimal animated gradient
> (grey + cyan, two soft blobs drifting opposite ways ~16s/21s) behind everything; the editor is a
> LIGHT frosted glass (rgba(255,255,255,.42) + backdrop blur) the gradient glows through. One blue
> accent (#0070f3), `#eaeaea` hairlines, monospace micro-labels in pill boxes, five deepened voice
> colours (Melody near-black, Sop #16a34a, Alto #9333ea, Tenor #dc2626, Bass #2563eb) for the harmony
> lines only. The frosted line-canvas is the hero and fills the screen; one row of controls on top, a
> thin status line at the bottom. Flat, hairlines not heavy shadows, no gradients beyond the one
> ambient field, no 3D, ≤120ms transitions. Vercel/Geist register."
