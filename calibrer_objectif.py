#!/usr/bin/env python3
"""Mesure la distorsion de l'objectif, au damier.

Pourquoi. L'objectif est tres grand angle et sa distorsion se voit a l'oeil
nu : les montants de porte sont courbes. Mesure faite a la main, paume a
distance constante deplacee lateralement, la profondeur estimee lit
640 / 500 / 650 mm du bord gauche au bord droit - soit pres de 30 % de faux
recul pour un simple deplacement lateral.

La cause est geometrique. L'objectif comprime la peripherie : plus la main
s'eloigne du centre, plus elle parait petite. Or c'est justement sa taille
apparente - la travee de la paume - qui sert a estimer la distance.

La correction de perspective deja en place ne touche pas a ca : elle corrige
l'effet de la DISTANCE sur la position apparente, pas celui de l'OBJECTIF
sur le rayon apparent. Les deux se cumulent, et celui-ci est le plus gros.

Ce que fait ce programme. Il collecte une quinzaine de vues d'un damier sous
des angles et des positions varies, puis resout le modele de l'objectif
(focale, centre optique, coefficients radiaux et tangentiels) et l'ecrit
dans ~/bras_objectif.json. bras_main le lit s'il existe et redresse les 21
points de la main a chaque image - quelques microsecondes.

Deux damiers, 9 x 6 coins interieurs tous les deux :

  damier_calibration.pdf  a imprimer a 100 %, reglet de controle inclus, a
                          COLLER SUR UN SUPPORT RIGIDE - un damier tenu a la
                          main gondole et toute la mesure part avec lui.
  damier_ecran.png        a afficher plein ecran. Un ecran est plus plat que
                          n'importe quel carton. Mesurer alors une case au
                          reglet SUR L'ECRAN et la passer en --case, puisque
                          sa taille depend de la definition d'affichage.
                          Tourner la camera plutot que le damier : pour
                          mesurer l'objectif, seule la position relative
                          compte.

    python3 calibrer_objectif.py
    python3 calibrer_objectif.py --vues 20 --port 8080
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

COINS = (9, 6)          # coins interieurs : colonnes, lignes
CASE_MM = 20.0
FICHIER = "~/bras_objectif.json"

PAGE = b"""<!doctype html><html><head><meta charset="utf-8">
<title>calibration objectif</title>
<style>body{background:#111;color:#eee;font:14px system-ui;margin:0;
text-align:center}img{max-width:100%;display:block;margin:0 auto}
#t{padding:10px;font:600 15px ui-monospace,monospace}</style></head>
<body><div id="t">...</div><img id="v" src="vue.jpg">
<script>
setInterval(function(){
  var i=new Image(); i.src="vue.jpg?"+Date.now();
  (i.decode?i.decode():Promise.resolve()).then(function(){
    document.getElementById("v").src=i.src;});
  fetch("etat",{cache:"no-store"}).then(function(r){return r.json();})
    .then(function(e){document.getElementById("t").textContent=e.texte;});
}, 200);
</script></body></html>"""


class _Vue:
    def __init__(self):
        self.verrou = threading.Lock()
        self.jpeg = b""
        self.texte = "demarrage"

    def poser(self, jpeg, texte):
        with self.verrou:
            if jpeg:
                self.jpeg = jpeg
            self.texte = texte

    def lire(self):
        with self.verrou:
            return self.jpeg, self.texte


_vue = _Vue()


class _H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def do_GET(self):
        chemin = self.path.split("?")[0]
        if chemin.startswith("/vue"):
            corps, _ = _vue.lire()
            t = "image/jpeg"
        elif chemin.startswith("/etat"):
            corps = json.dumps({"texte": _vue.lire()[1]}).encode()
            t = "application/json"
        else:
            corps, t = PAGE, "text/html; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", t)
        self.send_header("Content-Length", str(len(corps)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(corps)


def principal(a) -> int:
    import cv2
    import numpy as np

    serveur = ThreadingHTTPServer(("0.0.0.0", a.port), _H)
    threading.Thread(target=serveur.serve_forever, daemon=True).start()
    print("vue camera : http://<adresse de la carte>:%d/\n" % a.port)

    cap = cv2.VideoCapture(a.device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.largeur)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.hauteur)
    if not cap.isOpened():
        raise SystemExit("camera /dev/video%d inaccessible" % a.device)

    # Exposition automatique le temps de la calibration. bras_main laisse la
    # camera en manuel, calee sur une scene ou la main occupe le cadre ; un
    # damier, a fortiori affiche sur un ecran, n'a rien a voir avec ca.
    # Ici le flou de bouge et la saturation comptent plus que la stabilite.
    if not a.manuel:
        os.system("v4l2-ctl -d /dev/video%d --set-ctrl=auto_exposure=3"
                  " >/dev/null 2>&1" % a.device)
        time.sleep(0.8)

    # grille des coins dans le repere du damier, en millimetres
    grille = np.zeros((COINS[0] * COINS[1], 3), np.float32)
    grille[:, :2] = np.mgrid[0:COINS[0], 0:COINS[1]].T.reshape(-1, 2) * a.case

    monde, image, vignettes = [], [], []
    dernier = 0.0
    forme = None
    print("Montre le damier a la camera. Une vue est prise quand il est vu")
    print("en entier et immobile. Varie les angles ET les positions : coins")
    print("de l'image compris, c'est la que la distorsion se mesure.\n")

    try:
        while len(monde) < a.vues:
            ok, img = cap.read()
            if not ok:
                continue
            forme = img.shape[1::-1]
            gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            trouve, coins = cv2.findChessboardCorners(
                gris, COINS,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE +
                cv2.CALIB_CB_FAST_CHECK)
            vue = img.copy()
            t = time.time()
            etat = "damier non vu - approche-le, eclaire-le, evite les reflets"
            if trouve:
                coins = cv2.cornerSubPix(
                    gris, coins, (11, 11), (-1, -1),
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                cv2.drawChessboardCorners(vue, COINS, coins, trouve)
                centre = coins.reshape(-1, 2).mean(axis=0)
                neuf = all(np.hypot(*(centre - c)) > a.ecart for c in vignettes)
                if not neuf:
                    etat = "deja vu d'ici - deplace le damier ailleurs"
                elif t - dernier < 1.0:
                    etat = "..."
                else:
                    monde.append(grille.copy())
                    image.append(coins)
                    vignettes.append(centre)
                    dernier = t
                    etat = "VUE %d / %d prise" % (len(monde), a.vues)
                    print("   %s  (centre %4.0f, %4.0f)"
                          % (etat, centre[0], centre[1]))
            for c in vignettes:
                cv2.circle(vue, (int(c[0]), int(c[1])), 6, (0, 200, 255), 2)
            ok2, buf = cv2.imencode(".jpg", vue, [cv2.IMWRITE_JPEG_QUALITY, 70])
            _vue.poser(buf.tobytes() if ok2 else None,
                       "%d / %d   %s" % (len(monde), a.vues, etat))
    except KeyboardInterrupt:
        print("\ninterrompu")
    finally:
        cap.release()

    if len(monde) < 6:
        raise SystemExit("trop peu de vues (%d) : il en faut au moins six"
                         % len(monde))

    print("\nresolution du modele sur %d vues..." % len(monde))
    err, K, dist, rvecs, tvecs = cv2.calibrateCamera(monde, image, forme,
                                                     None, None)
    print("  erreur de reprojection : %.3f pixel" % err)
    if err > 1.0:
        print("  ATTENTION : au-dela de 1 pixel la calibration est douteuse.")
        print("  Damier gondole, flou de bouge, ou vues trop semblables.")

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    print("  focale   : %.1f x %.1f pixels" % (fx, fy))
    print("  centre   : %.1f, %.1f  (image %d x %d)"
          % (cx, cy, forme[0], forme[1]))
    import math
    champ = 2 * math.degrees(math.atan(forme[0] / (2 * fx)))
    print("  champ horizontal : %.0f degres" % champ)
    print("  distorsion : k1=%.4f k2=%.4f p1=%.5f p2=%.5f k3=%.4f"
          % tuple(dist.ravel()[:5]))

    # de combien un point du bord se deplace-t-il une fois redresse ?
    import numpy as np2
    bords = np2.array([[[forme[0] * f, forme[1] / 2]]
                       for f in (0.05, 0.25, 0.5, 0.75, 0.95)], np2.float32)
    droits = cv2.undistortPoints(bords, K, dist, P=K).reshape(-1, 2)
    print("\n  deplacement d'un point une fois redresse, sur l'axe median :")
    for (brut, droit) in zip(bords.reshape(-1, 2), droits):
        print("     x = %4.0f  ->  %6.1f   (%+.1f px)"
              % (brut[0], droit[0], droit[0] - brut[0]))

    chemin = os.path.expanduser(FICHIER)
    with open(chemin, "w", encoding="utf-8") as f:
        json.dump({"K": K.tolist(), "dist": dist.ravel().tolist(),
                   "largeur": forme[0], "hauteur": forme[1],
                   "erreur_px": float(err), "vues": len(monde),
                   "case_mm": a.case}, f, indent=1)
    print("\necrit dans %s" % FICHIER)
    print("bras_main le lira au prochain lancement.")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--largeur", type=int, default=640)
    p.add_argument("--hauteur", type=int, default=480)
    p.add_argument("--vues", type=int, default=15)
    p.add_argument("--case", type=float, default=CASE_MM,
                   help="cote d'une case en mm, mesure sur l'impression")
    p.add_argument("--ecart", type=float, default=60.0,
                   help="distance mini entre deux vues, en pixels")
    p.add_argument("--manuel", action="store_true",
                   help="ne pas toucher a l'exposition de la camera")
    p.add_argument("--port", type=int, default=8080)
    sys.exit(principal(p.parse_args()))
