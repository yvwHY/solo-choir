import json, re, time, urllib.parse, urllib.request

API = "https://www.cpdl.org/wiki/api.php"
UA = {"User-Agent": "SoloChoir-research-probe/0.1 (yvw.liao@gmail.com)"}

titles = """A - 14 : Ps. 144 : Mon Dieu, mon Roi - et Acclamation (Remi Studer)
A - 5 : Ps. 111 et Acclamation (Remi Studer)
A - Avent I : Psaume et Acclamation (Remi Studer)
A - Avent II : Psaume 71 (Remi Studer)
E ben ch'indegno aspiri (Andrea Rota)
E d'una viduvella (Filippo Azzaiolo)
E Dio per questo fa (Jacquet de Berchem)
E fattasi Reina (Giulio Renaldi)
Haager Messe (Rudolf Pflaum)
Hab's Je gethan (Heinrich Finck)
Habanera (from 'Carmen') (Georges Bizet)
Hacia Belén va una burra (Traditional)
M'amie a bien le regard (Orlando di Lasso)
M'amie a eu de dieu le don (Pierre Clereau)
M'y larrez vous (Roquelay)
M-O-T-H-E-R (Theodore Morse)
S každým úderem (Miroslav Raichl)
S'all'hor che piu sperai (Pomponio Nenna)
S'ha feito de nuey (José Lera)
S'i' 'l dissi mai, ch'i' venga in odio a quella (Bartolomeo Tromboncino)""".splitlines()

results = []
for t in titles:
    q = urllib.parse.urlencode({"action":"parse","page":t,"prop":"wikitext","format":"json"})
    req = urllib.request.Request(f"{API}?{q}", headers=UA)
    try:
        d = json.load(urllib.request.urlopen(req, timeout=30))
        wt = d["parse"]["wikitext"]["*"]
    except Exception as e:
        results.append({"title": t, "error": str(e)}); time.sleep(1); continue
    # MIDI links: {{filepath:...mid}} or [[Media:...mid]] or external .mid links
    mids = re.findall(r'(?i)([^\s\|\[\]{}=]+\.(?:mid|midi))', wt)
    mids = list(dict.fromkeys(mids))
    # genre/era hints
    genre = re.findall(r'\{\{Genre\|([^}]*)\}\}', wt)
    pub = re.findall(r"'''First published:'''\s*([^\n<]*)", wt)
    composer_era = re.findall(r'\{\{Composer\|([^}]*)\}\}', wt)
    results.append({"title": t, "n_midi": len(mids), "midi": mids[:8],
                    "genre": genre, "first_pub": pub[:1]})
    time.sleep(1)

print(json.dumps(results, ensure_ascii=False, indent=1))
