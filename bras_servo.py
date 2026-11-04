#!/usr/bin/env python3
"""Pilote Python du bras : envoie les angles au STM32 via le Bridge.

Compagnon de sketch/sketch.ino. Le Linux calcule, le MCU applique et garde
les butees, la rampe de vitesse et le watchdog.

    from bras_servo import BrasServo
    from bras_ik import Bras

    b = Bras()
    with BrasServo() as moteurs:
        sol = b.ik(x=-150.0, y=50.0, z=95.0)
        moteurs.envoyer(sol.servos)

Le sketch relache les servos apres 1,5 s sans ordre. Pour tenir une pose,
reenvoyer l'ordre periodiquement : c'est ce que fait `maintenir`.

Autotest :  python3 bras_servo.py --test
"""

from __future__ import annotations

import argparse
import math
import time
from typing import Optional, Sequence

try:
    from arduino.router_bridge import Bridge, RpcError
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "paquet arduino-router-bridge absent.\n"
        "  source ~/hand_env/bin/activate && uv pip install arduino-router-bridge"
    ) from exc

NOMS = ("base", "epaule", "coude", "inclinaison", "rotation", "pince")
COURSE_PINCE_MM = 40.0   # ouverture totale a fond, mesuree en CAO


class BrasServo:
    """Connexion au sketch. S'utilise comme gestionnaire de contexte."""

    def __init__(self, address: Optional[str] = None, timeout: float = 5.0):
        self.bridge = Bridge(address=address) if address else Bridge()
        if not self.bridge.connect(timeout=timeout):
            raise ConnectionError(
                "pas de connexion au routeur Arduino.\n"
                "  - l'utilisateur doit etre dans le groupe arduino-router\n"
                "  - le sketch doit tourner : arduino-cli compile --fqbn arduino:zephyr:unoq --upload sketch/"
            )
        self.derniers: Optional[tuple] = None

    # ------------------------------------------------------------- contexte

    def __enter__(self) -> "BrasServo":
        return self

    def __exit__(self, *exc) -> None:
        self.fermer()

    def fermer(self) -> None:
        try:
            self.relacher()
        except Exception:
            pass
        self.bridge.disconnect()

    # -------------------------------------------------------------- ordres

    def ping(self, timeout: float = 2.0) -> str:
        return self.bridge.call("ping", timeout=timeout)

    def envoyer(self, servos: Sequence[float], timeout: float = 1.0) -> None:
        """Cinq angles servo, dans l'ordre de bras_params.json."""
        if len(servos) < 5:
            raise ValueError("il faut cinq angles")
        v = [float(max(0.0, min(180.0, s))) for s in servos[:5]]
        self.bridge.call("set_angles", *v, timeout=timeout)
        self.derniers = tuple(v)

    def pince_deg(self, angle: float, timeout: float = 1.0) -> None:
        self.bridge.call("set_pince", float(max(0.0, min(180.0, angle))), timeout=timeout)

    def pince_mm(self, ouverture: float, timeout: float = 1.0) -> None:
        """Ouverture en mm entre les faces des mors, convertie en angle servo."""
        f = max(0.0, min(1.0, ouverture / COURSE_PINCE_MM))
        self.pince_deg(180.0 * f, timeout=timeout)

    def lire(self, index: int, timeout: float = 1.0) -> float:
        """Angle reellement applique par le MCU, apres rampe."""
        return float(self.bridge.call("get_angle", int(index), timeout=timeout))

    def lire_tout(self, timeout: float = 1.0) -> list:
        return [self.lire(i, timeout) for i in range(6)]

    def relacher(self, timeout: float = 1.0) -> None:
        """Coupe le signal : les servos ne forcent plus."""
        self.bridge.call("relacher", timeout=timeout)
        self.derniers = None

    def maintenir(self, duree: float, periode: float = 0.5) -> None:
        """Reenvoie la derniere pose pour tenir le watchdog du MCU en vie."""
        if self.derniers is None:
            raise RuntimeError("aucune pose a maintenir")
        fin = time.time() + duree
        while time.time() < fin:
            self.envoyer(self.derniers)
            time.sleep(periode)

    def aller_a(self, servos: Sequence[float], duree: float = 1.5,
                periode: float = 0.04) -> None:
        """Interpolation lineaire depuis la derniere pose, cote Linux.

        La rampe du MCU limite deja la vitesse ; ceci sert a imposer une duree
        de trajet, par exemple pour synchroniser plusieurs axes.
        """
        depart = list(self.derniers) if self.derniers else list(self.lire_tout()[:5])
        arrivee = [float(s) for s in servos[:5]]
        n = max(1, int(duree / periode))
        for k in range(1, n + 1):
            f = k / n
            self.envoyer([d + (a - d) * f for d, a in zip(depart, arrivee)])
            time.sleep(periode)


# ----------------------------------------------------------------- autotest

def _test(args) -> int:
    print("connexion au routeur...")
    with BrasServo() as m:
        print("  ", m.ping())

        from bras_ik import Bras
        b = Bras()

        cibles = [(-100.0, 0.0), (-200.0, 0.0), (-280.0, 0.0)]
        for rayon, _ in cibles:
            sol = b.ik(b.base[0] + rayon, b.base[1], b.table_z + 10.0)
            if sol is None:
                print(f"  R={abs(rayon):.0f} mm : hors d'atteinte")
                continue
            s = sol.arrondi(1)
            print(f"  R={abs(rayon):.0f} mm -> " +
                  "  ".join(f"{n}={v:.1f}" for n, v in zip(NOMS, s)))
            if not args.sec:
                m.aller_a(s, duree=1.5)
                m.maintenir(0.6)
                applique = m.lire_tout()
                print("      applique par le MCU : " +
                      "  ".join(f"{n}={v:.1f}" for n, v in zip(NOMS, applique)))

        if not args.sec:
            print("  ouverture de la pince")
            for mm in (0, 20, 40, 0):
                m.pince_mm(mm)
                time.sleep(0.6)

        print("  relachement")
    print("termine")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--test", action="store_true", help="autotest complet")
    p.add_argument("--sec", action="store_true",
                   help="ne bouge aucun servo, calcule et affiche seulement")
    a = p.parse_args()
    if not a.test:
        p.print_help()
        raise SystemExit(0)
    raise SystemExit(_test(a))
