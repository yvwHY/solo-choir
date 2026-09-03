"""CPDL scraper: Partsongs x SATB subset -> multi-track MIDI corpus.

Feasibility verified 2026-07-23 (worklog §T cont. 2): ~25k SATB works, ~80%
carry a full-score MIDI (one track per voice), 2/3 auto-pairable. This pulls
the homophonic-leaning subset first (partsongs: sectional repeats, tonal
harmony -- best transfer to pop) at a polite 1.5 s/request.

Resume-safe: already-downloaded files are skipped; page scan restarts cheaply
(batched 50/request). Run:
  python cpdl_scrape.py <out_dir> [--category Partsongs] [--limit N]
Output: <out_dir>/*.mid|*.midi + manifest.jsonl (page title, files, pageid).
"""

import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://www.cpdl.org/wiki/api.php"
UA = {"User-Agent": "SoloChoir-research-corpus/0.1 (MA thesis; polite; yvw.liao@gmail.com)"}
SLEEP = 1.5
# .mxl (compressed MusicXML) included: modern partsong editions often carry
# mxl instead of MIDI, and mxl names its parts -- cleaner for voice pairing
MEDIA_RE = re.compile(r"\[\[(?:Media|File):\s*([^|\]]+?\.(?:midi?|mxl))\s*[|\]]", re.I)


def api(params):
    params = dict(params, format="json")
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def iter_satb_pages(category):
    """Yield (pageid, title, wikitext) for category members that are also SATB."""
    cont = {}
    while True:
        res = api({
            "action": "query", "generator": "categorymembers",
            "gcmtitle": f"Category:{category}", "gcmnamespace": 0, "gcmlimit": 50,
            "prop": "categories|revisions", "clcategories": "Category:SATB",
            "cllimit": "max", "rvprop": "content", "rvslots": "main", **cont,
        })
        for p in res.get("query", {}).get("pages", {}).values():
            if "categories" not in p:  # not SATB
                continue
            try:
                text = p["revisions"][0]["slots"]["main"]["*"]
            except (KeyError, IndexError):
                continue
            yield p["pageid"], p["title"], text
        if "continue" not in res:
            return
        cont = res["continue"]
        time.sleep(SLEEP)


def file_urls(names):
    """File names -> {name: url} via imageinfo (batch 50)."""
    out = {}
    for i in range(0, len(names), 50):
        batch = names[i : i + 50]
        res = api({
            "action": "query", "prop": "imageinfo", "iiprop": "url",
            "titles": "|".join("File:" + n for n in batch),
        })
        norm = {n["to"]: n["from"] for n in res.get("query", {}).get("normalized", [])}
        for p in res.get("query", {}).get("pages", {}).values():
            if "imageinfo" in p:
                name = norm.get(p["title"], p["title"])
                out[name.split(":", 1)[-1]] = p["imageinfo"][0]["url"]
        time.sleep(SLEEP)
    return out


def main():
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    category = sys.argv[sys.argv.index("--category") + 1] if "--category" in sys.argv else "Partsongs"
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    manifest = out_dir / "manifest.jsonl"
    done_pages = set()
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            done_pages.add(json.loads(line)["pageid"])
    log = lambda *a: print(*a, flush=True)

    n_pages = n_files = 0
    with manifest.open("a") as mf:
        for pageid, title, text in iter_satb_pages(category):
            if limit and n_pages >= limit:
                break
            n_pages += 1
            if pageid in done_pages:
                continue
            names = list(dict.fromkeys(m.strip().replace(" ", "_")
                                       for m in MEDIA_RE.findall(text)))
            urls = file_urls(names) if names else {}
            saved = []
            for name, url in urls.items():
                dest = out_dir / f"{pageid}_{name}"
                if not dest.exists():
                    try:
                        req = urllib.request.Request(url, headers=UA)
                        with urllib.request.urlopen(req, timeout=60) as r:
                            dest.write_bytes(r.read())
                        time.sleep(SLEEP)
                    except Exception as e:
                        log(f"  DL FAIL {name}: {e}")
                        continue
                saved.append(dest.name)
                n_files += 1
            mf.write(json.dumps({"pageid": pageid, "title": title, "files": saved},
                                ensure_ascii=False) + "\n")
            mf.flush()
            if n_pages % 25 == 0:
                log(f"[{n_pages} pages scanned, {n_files} files downloaded]")
    log(f"DONE: {n_pages} pages, {n_files} MIDI files -> {out_dir}")


if __name__ == "__main__":
    main()
