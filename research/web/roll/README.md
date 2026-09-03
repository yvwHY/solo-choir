# Solo Choir — QR Score-Roll (`web/roll/`)

A take-home web page: scan a QR from the **studio** app and a vertical "score-roll" of your
arrangement opens; **swipe up/down to play** (finger = a music-box crank) with a choir soundfont.

## How it works (pure front-end — no backend)
- The whole arrangement is encoded into the URL's **`#` fragment** (`score-codec.js`:
  quantise → delta → varint → deflate-raw → base64url). One static page serves unlimited cards;
  nothing is stored server-side.
- `index.html` decodes the fragment, draws the roll, and plays notes via the vendored choir soundfont
  in `sf/` (WebAudioFont, GM 52 Choir Aahs — see `sf/CREDITS.md`).
- Cap: ≤ ~30 s / ≤ ~300 notes (studio refuses longer — a QR/URL stays small and scannable).

## Files
- `index.html` — the roll page (decode + vertical render + bidirectional scrub + sustain playback).
- `score-codec.js` — shared encoder/decoder (also loaded by the studio app to build the QR URL).
- `sf/` — vendored WebAudioFont player + Choir Aahs preset (so the page is self-contained, no CDN).
- `_codectest.html` — dev round-trip test (not part of the deployed experience).

## Deploy (make it openable on visitors' phones)
The page is host-agnostic static files. Publish the repo's `web/` directory to any static host:
- **GitHub Pages / Netlify / Cloudflare Pages / Render (Static Site)** — point it at `web/` (or the repo
  root) and note the resulting public URL for `web/roll/index.html`.

Then tell the studio app that URL so the QR points at the public page (not just this Mac):
- Edit **`studio/ui/index.html`** → set the `ROLL_BASE` constant in the "Make card" handler, e.g.
  `const ROLL_BASE = "https://yourname.github.io/solo-choir/web/roll/index.html";`
- Empty `ROLL_BASE` (default) uses `location.origin + "/web/roll/index.html"` — handy for **local
  testing on this Mac** (the studio server serves `web/` from the repo root), but a visitor's phone
  can't reach `127.0.0.1`, so set `ROLL_BASE` before the exhibition.
