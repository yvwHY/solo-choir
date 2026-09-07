# Asset credits

## head.glb (not included in this repository)
- **What:** a smooth male base mesh (head + bust), used by the early interface
  `ui/solo_choir_ui_v10.html` as the 3D head, and by the Fusion 360 scripts in
  `hardware/mask_frame/` as a fitting reference. The performing application
  (`app/respond_shell.py` → `ui/respond_live.html`) does not load it.
- **Source:** "CCO_ Male_base_mesh_standing" by iamsunroy on Sketchfab,
  https://sketchfab.com/3d-models/cco--male-base-mesh-standing-8340f5cc2d08497abfdb42aba8462a33
  (downloaded 16 June 2026).
- **Licence:** **Sketchfab Standard** (the licence registered on the model page,
  despite the "CCO" in its title). Standard permits use inside a project but
  does not permit making the file available to others as a stand-alone file, so
  it is **not committed here**. To use it, download it from the page above and
  save it as `app/assets/head.glb`; if the file is absent, the v10 interface
  falls back to its procedural head automatically.

The loader in `ui/solo_choir_ui_v10.html` picks the largest mesh and bakes its
node transform, so the model's original (Z-up) orientation is corrected
automatically. Orientation/size can be tweaked via `MODEL_ROT_Y`, `MODEL_TILT`,
`MODEL_TARGET_H`.

## Bundled web libraries & font (Task 3.2 — vendored locally for offline/exhibition reliability)
- **three.min.js** — three.js r128. **MIT License** (© three.js authors). From cdnjs.
- **GLTFLoader.js** — three.js r128 examples loader. **MIT License**.
- **space-grotesk-*.woff2 + font.css** — Space Grotesk (Florian Karsten). **SIL Open Font License 1.1**. woff2 subsets fetched from Google Fonts; `font.css` is the Google-served @font-face CSS with URLs rewritten to `/app/assets/`.

The three bundled libraries/fonts are permissively licensed (MIT / OFL) — safe to commit (unlike the git-ignored Beatrice engine, trained models, and head.glb). Previously loaded from CDNs; vendored so a dead exhibition network can't blank the 3D stage or drop the font.
