#!/usr/bin/env python3
"""Lequel des indices de profondeur resiste aux rotations de la main ?

Cinq poses, et un retour camera pendant toute la mesure : sans voir ce que
MediaPipe voit, impossible de savoir si la main sort du cadre, si le squelette
decroche, ou si c'est l'image qui s'est degradee.

Apres la pose 1, une CIBLE marque la position du poignet. Les poses 2 a 4 se
font en gardant le poignet dessus : c'est le seul moyen de ne pas confondre
rotation et deplacement. L'ecart est affiche en direct.

    python3 orientation.py                 # 8 s par pose
    python3 orientation.py --duree 12 --port 8080
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bras_main as BM                                      # noqa: E402

PALME = (0, 1, 2, 5, 9, 13, 17)
FOCALE = 640.0          # approchee ; seuls les rapports comptent ici
MARGE = 0.03            # bord d'image au-dela duquel la main est trop pres

ETAT = {"jpeg": None, "info": {}, "verrou": threading.Lock()}


def indices(lm, monde, w, h):
    """Tous les candidats, calcules sur la meme image."""
    import cv2
    import numpy as np

    d = {}
    xy = np.array([[p.x * w, p.y * h] for p in lm], np.float64)
    pal = xy[list(PALME)]

    d["racine_aire"] = math.sqrt(max(
        cv2.contourArea(cv2.convexHull(pal.astype(np.float32))), 1.0))
    d["paume_px"] = float(np.mean([np.hypot(*(xy[k] - xy[0]))
                                   for k in (5, 9, 13, 17)]))
    pp = np.sqrt(((pal[:, None, :] - pal[None, :, :]) ** 2).sum(-1))
    paires = np.sort(pp[np.triu_indices(len(pal), 1)])
    d["diametre_palme"] = float(paires[-1])
    d["moy_3_max"] = float(paires[-3:].mean())
    tout = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    d["etendue_main"] = float(tout.max())
    # temoin de deplacement : le POIGNET, centre des deux rotations
    d["_cx"] = float(xy[0, 0])
    d["_cy"] = float(xy[0, 1])

    if monde is None:
        return d
    p3 = np.array([[q.x, q.y, q.z] for q in monde], np.float64)
    if not np.isfinite(p3).all():
        return d

    K = np.array([[FOCALE, 0.0, w / 2.0], [0.0, FOCALE, h / 2.0],
                  [0.0, 0.0, 1.0]], np.float64)
    z5 = np.zeros(5)
    try:
        ok, rv, tv = cv2.solvePnP(p3, xy, K, z5, flags=cv2.SOLVEPNP_SQPNP)
        if ok:
            d["pnp_metrique"] = 1000.0 * float(tv.ravel()[2])
    except cv2.error:
        pass
    t3 = float(np.sqrt((p3 ** 2).sum(1).mean()))
    if t3 > 1e-6:
        try:
            ok, rv, tv = cv2.solvePnP(p3 / t3, xy, K, z5,
                                      flags=cv2.SOLVEPNP_SQPNP)
            if ok:
                d["pnp_forme"] = float(tv.ravel()[2])
        except cv2.error:
            pass
    try:
        ok, rv, tv = cv2.solvePnP(p3[list(PALME)], pal, K, z5,
                                  flags=cv2.SOLVEPNP_SQPNP)
        if ok:
            d["pnp_palme"] = 1000.0 * float(tv.ravel()[2])
    except cv2.error:
        pass

    a, bb = p3[5] - p3[0], p3[17] - p3[0]
    n = np.cross(a, bb)
    nn = float(np.linalg.norm(n))
    if nn > 1e-9:
        cos_t = max(abs(float(n[2])) / nn, 0.25)
        d["aire_sur_cos"] = d["racine_aire"] / math.sqrt(cos_t)
        d["paume_sur_cos"] = d["paume_px"] / cos_t
    return d


def annoter(cv2, np, img, lm, mp, cible, tol):
    """Dessine ce que MediaPipe voit, plus la cible de poignet."""
    h, w = img.shape[:2]
    m = int(MARGE * min(w, h))
    cv2.rectangle(img, (m, m), (w - m, h - m), (70, 70, 70), 1)

    if lm is not None:
        pts = [(int(p.x * w), int(p.y * h)) for p in lm]
        for a, b in mp.solutions.hands.HAND_CONNECTIONS:
            cv2.line(img, pts[a], pts[b], (0, 220, 0), 2)
        for p in pts:
            cv2.circle(img, p, 3, (0, 0, 230), -1)
        # l'enveloppe de paume, dont sortent la moitie des candidats
        pal = np.array([pts[i] for i in PALME], np.int32)
        cv2.polylines(img, [cv2.convexHull(pal)], True, (255, 190, 0), 2)
        cv2.circle(img, pts[0], 8, (255, 255, 255), 2)

    if cible is not None:
        cx, cy = int(cible[0]), int(cible[1])
        ok = (lm is not None
              and math.hypot(lm[0].x * w - cible[0], lm[0].y * h - cible[1]) < tol)
        coul = (0, 230, 0) if ok else (0, 140, 255)
        cv2.circle(img, (cx, cy), int(tol), coul, 2)
        cv2.line(img, (cx - 12, cy), (cx + 12, cy), coul, 2)
        cv2.line(img, (cx, cy - 12), (cx, cy + 12), coul, 2)
    return img


def recolter(main, duree, titre, detail, cible, tol):
    import numpy as np
    print("\n%s" % titre)
    for l in detail.split("\n"):
        print("   %s" % l)

    def publier(phase, reste, lm, extra=None):
        img = main.image
        if img is None:
            return
        a = annoter(main.cv2, np, img.copy(), lm, main.mp, cible, tol)
        ok, buf = main.cv2.imencode(".jpg", a,
                                    [main.cv2.IMWRITE_JPEG_QUALITY, 72])
        d = {"pose": titre, "consigne": detail.replace("\n", " "),
             "phase": phase, "reste": "%.1f s" % max(reste, 0.0)}
        if lm is None:
            d["main"] = "PAS DETECTEE"
        else:
            h, w = img.shape[:2]
            d["main"] = "vue"
            d["poignet"] = "x %.0f  y %.0f" % (lm[0].x * w, lm[0].y * h)
            bord = min(min(p.x, 1 - p.x, p.y, 1 - p.y) for p in lm)
            d["marge au bord"] = ("%.0f %% %s" % (100 * bord,
                                                  "TROP PRES" if bord < MARGE else ""))
            if cible is not None:
                e = math.hypot(lm[0].x * w - cible[0], lm[0].y * h - cible[1])
                d["ecart a la cible"] = "%.0f px %s" % (
                    e, "OK" if e < tol else "REVIENS SUR LA CIBLE")
        if extra:
            d.update(extra)
        with ETAT["verrou"]:
            if ok:
                ETAT["jpeg"] = buf.tobytes()
            ETAT["info"] = d

    fin_attente = time.time() + 4.0
    while time.time() < fin_attente:
        main.lire()
        publier("placement", fin_attente - time.time(), main.reperes)
        time.sleep(0.02)

    print("   mesure %.0f s" % duree, flush=True)
    lignes, vus, tot = [], 0, 0
    fin = time.time() + duree
    while time.time() < fin:
        ok, img = main.cap.read()
        if not ok or img is None or not img.size:
            continue
        tot += 1
        if main.flip is not None:
            img = main.cv2.flip(img, main.flip)
        main.image = img
        res = main.hands.process(main.cv2.cvtColor(img, main.cv2.COLOR_BGR2RGB))
        lm = res.multi_hand_landmarks[0].landmark if res.multi_hand_landmarks else None
        main.reperes = lm
        if lm is not None:
            vus += 1
            mw = getattr(res, "multi_hand_world_landmarks", None)
            h, w = img.shape[:2]
            lignes.append(indices(lm, mw[0].landmark if mw else None, w, h))
        publier("MESURE EN COURS", fin - time.time(), lm,
                {"images retenues": "%d / %d" % (vus, tot)})
    print("   %d images sur %d (%.0f %%)" % (vus, tot, 100.0 * vus / max(tot, 1)))
    return lignes


def _med(v):
    w = sorted(v)
    return w[len(w) // 2]


def _etendue(v):
    import numpy as np
    a = np.array(v, float)
    return float(np.percentile(a, 95) - np.percentile(a, 5))


PAGE = b"""<!doctype html><meta charset=utf-8>
<title>mesure d'orientation</title>
<style>body{margin:0;background:#14181a;color:#e7ecee;
font:14px/1.5 ui-monospace,Menlo,monospace;display:flex;flex-direction:column;
align-items:center;gap:10px;padding:14px}
h2{margin:0;font-size:17px;font-weight:600;text-align:center}
p#c{margin:0;max-width:640px;text-align:center;color:#9fb0b6}
img{max-width:100%;border:1px solid #333c41;border-radius:4px}
table{border-collapse:collapse;min-width:330px}
td{padding:3px 12px;border-bottom:1px solid #262c30}
td:last-child{text-align:right;color:#4fcfc6}
.al td:last-child{color:#ff9a4d}</style>
<h2 id=t>en attente</h2><p id=c></p><img id=v><table id=m></table>
<script>
function t(){document.getElementById('v').src='image.jpg?'+Date.now();
fetch('info').then(r=>r.json()).then(d=>{
document.getElementById('t').textContent=(d.phase||'')+'  -  '+(d.pose||'');
document.getElementById('c').textContent=d.consigne||'';
delete d.pose;delete d.consigne;delete d.phase;
document.getElementById('m').innerHTML=Object.entries(d).map(([k,v])=>
'<tr'+(/TROP|REVIENS|PAS DET/.test(v)?' class=al':'')+'><td>'+k+
'</td><td>'+v+'</td></tr>').join('')});}
setInterval(t,150);t();
</script>"""


class _H(BaseHTTPRequestHandler):
    def _envoyer(self, corps, mime):
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(corps)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(corps)

    def do_GET(self):  # noqa: N802
        c = self.path.split("?")[0]
        if c in ("/", "/index.html"):
            self._envoyer(PAGE, "text/html; charset=utf-8")
        elif c == "/image.jpg":
            with ETAT["verrou"]:
                j = ETAT["jpeg"]
            self._envoyer(j or b"", "image/jpeg")
        elif c == "/info":
            with ETAT["verrou"]:
                d = dict(ETAT["info"])
            self._envoyer(json.dumps(d).encode("utf-8"), "application/json")
        else:
            self.send_error(404)

    def log_message(self, *a):
        pass


def principal(args):
    main = BM.Main(device=args.device, flip=BM.REGLAGES["flip"])
    BM.regler_exposition(main)

    srv = ThreadingHTTPServer(("0.0.0.0", args.port), _H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    from bras_viewer import _ip_locale
    print("\n>>> OUVRE LA VUE CAMERA :  http://%s:%d/" % (_ip_locale(), args.port))
    print(">>> garde-la sous les yeux pendant toute la mesure.\n")
    time.sleep(1.5)

    d = args.duree
    tol = args.tolerance
    consignes = [
        ("face", "1/5  PAUME DE FACE  -  reference",
         "doigts tendus vers le HAUT, paume bien a plat face a l'objectif.\n"
         "C'est cette pose qui fixe la cible de poignet pour la suite."),
        ("pivot45", "2/5  PIVOT 45 deg  (axe des doigts)",
         "doigts toujours vers le HAUT. L'avant-bras tourne sur lui-meme,\n"
         "dans le sens le plus confortable. Paume a mi-chemin, de biais.\n"
         "Le poignet reste DANS LA CIBLE."),
        ("pivot90", "3/5  PIVOT 90 deg  -  main de CHANT",
         "doigts TOUJOURS vers le haut, meme pivot, meme sens : la camera\n"
         "ne voit plus que la tranche de la main. Poignet DANS LA CIBLE."),
        ("bascule", "4/5  BASCULE 45 deg  (axe horizontal)",
         "retour paume de face, puis le poignet PLIE : le bout des doigts se\n"
         "rapproche de l'objectif de 45 deg. Poignet DANS LA CIBLE."),
        ("loin", "5/5  PAUME DE FACE a nouveau, mais LOIN",
         "exactement la pose 1, aussi loin que tu piloterais. Ici la cible\n"
         "ne s'applique plus : c'est la pose qui donne le signal utile."),
    ]

    poses, cible = [], None
    for nom, titre, detail in consignes:
        jeu = recolter(main, d, titre, detail,
                       None if nom == "loin" else cible, tol)
        poses.append((nom, jeu))
        if nom == "face" and len(jeu) >= 5:
            cible = (_med([e["_cx"] for e in jeu]), _med([e["_cy"] for e in jeu]))
            print("   cible de poignet posee en x %.0f  y %.0f" % cible)
    main.fermer()
    srv.shutdown()

    for nom, jeu in poses:
        if len(jeu) < 10:
            raise SystemExit("pose '%s' : trop peu d'images exploitables" % nom)

    cles = []
    for _, jeu in poses:
        for e in jeu:
            for k in e:
                if k not in cles:
                    cles.append(k)
    cles = [k for k in cles if all(any(k in e for e in j) for _, j in poses)]
    visibles = [k for k in cles if not k.startswith("_")]

    chemin = os.path.expanduser("~/orientation.csv")
    with open(chemin, "w") as f:
        f.write("pose," + ",".join(cles) + "\n")
        for nom, jeu in poses:
            for e in jeu:
                f.write(nom + "," + ",".join(
                    ("%.6g" % e[k]) if k in e else "" for k in cles) + "\n")

    centres = {n: (_med([e["_cx"] for e in j]), _med([e["_cy"] for e in j]))
               for n, j in poses}
    bouge = max(math.hypot(centres[n][0] - centres["face"][0],
                           centres[n][1] - centres["face"][1])
                for n in ("pivot45", "pivot90", "bascule"))
    print("\npoignet : %.0f px d'ecart maximal a la cible (tolerance %.0f px)"
          % (bouge, tol))
    print("\n%d images. Detail dans %s" % (sum(len(j) for _, j in poses), chemin))
    print("Lance maintenant :  python3 ~/analyse.py")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--duree", type=float, default=8.0, help="secondes par pose")
    p.add_argument("--tolerance", type=float, default=35.0,
                   help="rayon de la cible de poignet, en pixels")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--device", type=int, default=-1)
    raise SystemExit(principal(p.parse_args()))
