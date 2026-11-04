#!/usr/bin/env python3
"""Serveur local qui rejoue en 3D la pose courante du bras.

La carte sert le simulateur et une route /etat qui rend les angles servo
envoyes au STM32. La page, servie par la carte, bascule toute seule en suivi
direct : les curseurs se verrouillent et le modele suit le bras.

Ouvert depuis claude.ai, le meme fichier reste pilotable au curseur : /etat
n'y repond pas, la page le constate et ne change rien.

    from bras_viewer import Visualiseur
    v = Visualiseur()
    v.demarrer()                      # http://<ip de la carte>:8080
    v.publier(sol.servos, ouverture=12.0)

Demonstration sans materiel :   python3 bras_viewer.py --demo
Service seul :                  python3 bras_viewer.py
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional, Sequence

RACINE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "viewer")
POSE_REPOS = [90.0, 0.0, 0.0, 0.0, 90.0]


class _Etat:
    def __init__(self):
        self.verrou = threading.Lock()
        self.servos = list(POSE_REPOS)
        self.ouverture = 0.0
        self.source = "au repos"
        self.horodatage = time.time()
        self.jpeg = None          # derniere image camera annotee
        self.mesures = {}         # x, y, distance, R, Z, pince
        self.reglages = {}        # molettes posees depuis la page

    def lire(self) -> dict:
        with self.verrou:
            return {"servos": list(self.servos), "ouverture": self.ouverture,
                    "source": self.source, "mesures": dict(self.mesures),
                    "reglages": dict(self.reglages),
                    "age": round(time.time() - self.horodatage, 2)}

    def ecrire(self, servos: Sequence[float], ouverture: Optional[float],
               source: Optional[str]) -> None:
        with self.verrou:
            self.servos = [float(v) for v in list(servos)[:5]]
            if ouverture is not None:
                self.ouverture = float(ouverture)
            if source is not None:
                self.source = source
            self.horodatage = time.time()

    def image(self, jpeg, mesures=None) -> None:
        """Publie l'image camera annotee et les grandeurs mesurees."""
        with self.verrou:
            if jpeg is not None:
                self.jpeg = jpeg
            if mesures is not None:
                self.mesures = dict(mesures)

    def regler(self, d) -> None:
        with self.verrou:
            self.reglages.update({k: float(v) for k, v in d.items()})

    def lire_reglages(self) -> dict:
        with self.verrou:
            return dict(self.reglages)

    def lire_image(self):
        with self.verrou:
            return self.jpeg


_etat = _Etat()


class _Handler(SimpleHTTPRequestHandler):
    racine = RACINE

    def __init__(self, *a, **kw):
        super().__init__(*a, directory=self.racine, **kw)

    def do_GET(self):  # noqa: N802
        if self.path.split("?")[0].rstrip("/") == "/camera.jpg":
            corps = _etat.lire_image() or b""
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(corps)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(corps)
            return
        if self.path.split("?")[0].rstrip("/") == "/etat":
            corps = json.dumps(_etat.lire()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(corps)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(corps)
            return
        super().do_GET()

    def guess_type(self, path):
        """Declare l'encodage. Sans ca le navigateur lit l'UTF-8 en Latin-1
        et affiche les accents en charabia."""
        t = super().guess_type(path)
        if t in ("text/html", "text/plain", "text/css",
                 "application/javascript", "text/javascript"):
            return t + "; charset=utf-8"
        return t

    def do_POST(self):  # noqa: N802
        """Pose une molette depuis la page, sans relancer le programme."""
        if self.path.split("?")[0].rstrip("/") != "/reglage":
            self.send_error(404)
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            _etat.regler(json.loads(self.rfile.read(n).decode("utf-8")))
        except Exception as e:
            self.send_error(400, str(e))
            return
        corps = json.dumps(_etat.lire_reglages()).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(corps)))
        self.end_headers()
        self.wfile.write(corps)

    def log_message(self, *a):  # silence
        pass


def _ip_locale() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


class Visualiseur:
    """Sert le simulateur et publie la pose courante."""

    @staticmethod
    def publier_image(jpeg, mesures=None):
        _etat.image(jpeg, mesures)

    @staticmethod
    def reglages():
        return _etat.lire_reglages()

    @staticmethod
    def poser_reglages(d):
        _etat.regler(d)

    def __init__(self, port: int = 8080, racine: str = RACINE):
        if not os.path.isdir(racine):
            raise FileNotFoundError(
                f"dossier {racine} introuvable - il doit contenir index.html, "
                "geometry.js, collide.js et three.min.js")
        _Handler.racine = racine
        self.port = port
        self.serveur: Optional[ThreadingHTTPServer] = None
        self.fil: Optional[threading.Thread] = None

    def demarrer(self, silencieux: bool = False) -> str:
        self.serveur = ThreadingHTTPServer(("0.0.0.0", self.port), _Handler)
        self.fil = threading.Thread(target=self.serveur.serve_forever, daemon=True)
        self.fil.start()
        url = f"http://{_ip_locale()}:{self.port}/"
        if not silencieux:
            print(f"simulateur servi sur {url}")
        return url

    def arreter(self) -> None:
        if self.serveur:
            self.serveur.shutdown()
            self.serveur.server_close()
            self.serveur = None

    @staticmethod
    def publier(servos: Sequence[float], ouverture: Optional[float] = None,
                source: Optional[str] = None) -> None:
        """Pose a afficher. Appelable depuis n'importe quel fil."""
        _etat.ecrire(servos, ouverture, source)

    def __enter__(self) -> "Visualiseur":
        self.demarrer()
        return self

    def __exit__(self, *exc) -> None:
        self.arreter()


# ------------------------------------------------------------ demonstration

def _demo(port: int) -> int:
    from bras_ik import Bras
    try:
        from bras_collision import Collisions
        filtre = Collisions().verifier
    except Exception:
        filtre = None
        print("bras_collision absent : trajectoire non verifiee")

    b = Bras()
    v = Visualiseur(port=port)
    url = v.demarrer()
    print("ouvre cette adresse depuis ton PC, puis laisse tourner\n")

    etapes = [(-120.0, 150.0, 0.0), (-120.0, 95.0, 0.0), (-120.0, 95.0, 24.0),
              (-120.0, 150.0, 24.0), (-260.0, 190.0, 24.0), (-260.0, 97.0, 24.0),
              (-260.0, 97.0, 0.0), (-260.0, 190.0, 0.0)]
    try:
        while True:
            for rayon, z, ouv in etapes:
                cible = (b.base[0] + rayon, b.base[1], z)
                sol = b.ik(*cible, filtre=filtre)
                if sol is None:
                    print(f"  R={abs(rayon):.0f} Z={z:.0f} : hors d'atteinte")
                    continue
                v.publier(sol.servos, ouverture=ouv,
                          source=f"demo  R {abs(rayon):.0f} mm  Z {z:.0f} mm")
                time.sleep(1.2)
    except KeyboardInterrupt:
        print("\narret")
        v.arreter()
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--demo", action="store_true",
                   help="parcourt une sequence de prise et depose, sans materiel")
    a = p.parse_args()
    if a.demo:
        raise SystemExit(_demo(a.port))
    v = Visualiseur(port=a.port)
    v.demarrer()
    print("Ctrl+C pour arreter. La pose reste au repos tant que rien ne publie.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        v.arreter()
