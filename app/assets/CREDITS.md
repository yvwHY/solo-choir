# Asset credits

## head.glb
- **Type:** smooth mannequin head + bust (featureless — no eye/mouth cavities).
- **Source:** Sketchfab, **CC0 / Public Domain** (user-provided).
- **License:** CC0 — no attribution required.
- **Source URL:** _(add the Sketchfab page URL here if you want it on record)_

The loader in `ui/solo_choir_ui_v10.html` picks the largest mesh and bakes its
node transform, so the model's original (Z-up) orientation is corrected
automatically. Orientation/size can be tweaked via `MODEL_ROT_Y`, `MODEL_TILT`,
`MODEL_TARGET_H`.

## Bundled web libraries & font (Task 3.2 — vendored locally for offline/exhibition reliability)
- **three.min.js** — three.js r128. **MIT License** (© three.js authors). From cdnjs.
- **GLTFLoader.js** — three.js r128 examples loader. **MIT License**.
- **space-grotesk-*.woff2 + font.css** — Space Grotesk (Florian Karsten). **SIL Open Font License 1.1**. woff2 subsets fetched from Google Fonts; `font.css` is the Google-served @font-face CSS with URLs rewritten to `/app/assets/`.

All four are permissively licensed (MIT / OFL) — safe to commit (unlike the git-ignored Beatrice engine + trained models). Previously loaded from CDNs; vendored so a dead exhibition network can't blank the 3D stage or drop the font.
