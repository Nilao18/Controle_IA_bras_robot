#!/usr/bin/env python3
"""Cinematique directe et inverse du bras 5 axes.

Pur Python standard : aucune dependance, tourne dans hand_env (3.11) comme
ailleurs. Les parametres viennent de bras_params.json, le meme fichier que
celui qui alimente le simulateur 3D.

Repere : celui de l'assemblage Fusion 360. Origine au coin du boitier, Z vers
le haut, plan sagittal du bras en Y = 50. Le plan de travail (dessus du
couvercle) est a Z = 85.

Convention d'angle : chaque axe est pilote par un angle servo 0-180 deg.
L'angle geometrique vaut  sens * (servo - ref), ou ref est l'angle servo qui
correspond a la pose CAO telle qu'exportee (bras replie, pince vers le haut).

Usage typique :

    from bras_ik import Bras
    b = Bras()
    sol = b.ik(x=-150.0, y=50.0, z=100.0)     # pince verticale vers le bas
    if sol:
        print(sol.servos)                      # (base, epaule, coude, incl, rot)
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

PARAMS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bras_params.json")

# Orientations de pince usuelles, exprimees en rotation totale
# (somme des angles geometriques epaule + coude + inclinaison).
PINCE_HAUT = 0.0      # comme en pose CAO
PINCE_BAS = 180.0     # retournee, axe vers le bas : position de prise


def _rot(px: float, pz: float, cx: float, cz: float, a_deg: float) -> Tuple[float, float]:
    """Rotation d'un point autour d'un axe horizontal (Y), dans le plan sagittal.

    Meme convention que Three.js : une rotation positive amene +X vers -Z.
    """
    a = math.radians(a_deg)
    ca, sa = math.cos(a), math.sin(a)
    ux, uz = px - cx, pz - cz
    return cx + ux * ca + uz * sa, cz - ux * sa + uz * ca


def _norm180(a: float) -> float:
    """Ramene un angle dans ]-180, 180]."""
    a = (a + 180.0) % 360.0 - 180.0
    return 180.0 if a == -180.0 else a


@dataclass(frozen=True)
class Pose:
    """Une solution de cinematique inverse."""

    servos: Tuple[float, float, float, float, float]
    angles: Tuple[float, float, float, float, float]  # angles geometriques
    coude_en_haut: bool
    epaule: Tuple[float, float]   # position de l'axe du coude, plan sagittal
    coude: Tuple[float, float]    # position de l'axe du poignet, plan sagittal

    def arrondi(self, n: int = 1) -> Tuple[float, ...]:
        return tuple(round(s, n) for s in self.servos)


class HorsAtteinte(Exception):
    """La cible n'est atteignable par aucune configuration."""


class Bras:
    def __init__(self, params_path: str = PARAMS_PATH):
        with open(params_path, "r", encoding="utf-8") as f:
            p = json.load(f)
        self.p = p

        self.base = (p["base"]["x"], p["base"]["y"])
        self.sh = (p["epaule"]["x"], p["epaule"]["z"])
        self.el = (p["coude"]["x"], p["coude"]["z"])
        self.wr = (p["poignet"]["x"], p["poignet"]["z"])
        self.rot_x = p["rotation"]["x"]
        self.grasp_z = p["grasp_z"]
        self.table_z = p["plan_travail_z"]

        self.axes = p["axes"]

        # longueurs deduites
        self.l_bras = math.dist(self.sh, self.el)
        self.l_avant = math.dist(self.el, self.wr)

        # directions des segments en pose CAO
        self.dir_bras0 = math.degrees(math.atan2(self.el[1] - self.sh[1],
                                                 self.el[0] - self.sh[0]))
        self.dir_avant0 = math.degrees(math.atan2(self.wr[1] - self.el[1],
                                                  self.wr[0] - self.el[0]))

        # vecteur poignet -> centre des mors, en pose CAO
        self.outil = (self.rot_x - self.wr[0], self.grasp_z - self.wr[1])
        self.l_outil = math.hypot(*self.outil)
        self.dir_outil0 = math.degrees(math.atan2(self.outil[1], self.outil[0]))

    # ------------------------------------------------------------------ servo

    def servo_vers_angle(self, i: int, servo: float) -> float:
        a = self.axes[i]
        return a["sens"] * (servo - a["ref"])

    def angle_vers_servo(self, i: int, angle: float) -> float:
        a = self.axes[i]
        return a["ref"] + angle / a["sens"]

    def dans_butees(self, i: int, servo: float, marge: float = 1e-6) -> bool:
        a = self.axes[i]
        return a["min"] - marge <= servo <= a["max"] + marge

    def _servo_valide(self, i: int, angle: float) -> Optional[float]:
        """Angle servo dans les butees, en essayant les tours equivalents.

        Un angle geometrique de -180 et de +180 designent la meme pose : selon
        l'axe, un seul des deux tombe dans la course du servo.
        """
        for tour in (0.0, -360.0, 360.0):
            s = self.angle_vers_servo(i, angle + tour)
            if self.dans_butees(i, s):
                return s
        return None

    # --------------------------------------------------------------- directe

    def fk(self, servos: Sequence[float]) -> dict:
        """Cinematique directe. Prend cinq angles servo, rend les positions."""
        a = [self.servo_vers_angle(i, s) for i, s in enumerate(servos)]
        a1, a2, a3, a4, a5 = a

        coude = _rot(self.el[0], self.el[1], self.sh[0], self.sh[1], a2)
        poignet = _rot(*_rot(self.wr[0], self.wr[1], self.el[0], self.el[1], a3),
                       self.sh[0], self.sh[1], a2)
        tcp_s = _rot(*_rot(*_rot(self.rot_x, self.grasp_z, self.wr[0], self.wr[1], a4),
                           self.el[0], self.el[1], a3),
                     self.sh[0], self.sh[1], a2)

        r = tcp_s[0] - self.base[0]
        ca, sa = math.cos(math.radians(a1)), math.sin(math.radians(a1))
        return {
            "x": self.base[0] + r * ca,
            "y": self.base[1] + r * sa,
            "z": tcp_s[1],
            "r": r,
            "epaule": self.sh,
            "coude": coude,
            "poignet": poignet,
            "orientation": a2 + a3 + a4,
            "rotation_pince": a5,
        }

    # --------------------------------------------------------------- inverse

    def ik_toutes(self, x: float, y: float, z: float,
                  orientation: float = PINCE_BAS,
                  rotation_pince: Optional[float] = None,
                  filtre=None) -> List[Pose]:
        """Toutes les solutions valides, de la plus a la moins souhaitable.

        orientation : rotation totale de la pince, en degres. 180 = axe de la
        pince vers le bas (prise), 0 = vers le haut (pose CAO repliee).
        rotation_pince : angle servo du 5e axe ; defaut = milieu de course.
        filtre : appelable prenant les cinq angles servo et rendant None si la
        pose est acceptable, sinon un motif de rejet. Passer la methode
        `verifier` de bras_collision.Collisions pour ecarter les collisions.
        """
        dx, dy = x - self.base[0], y - self.base[1]
        rho = math.hypot(dx, dy)
        cap = math.degrees(math.atan2(dy, dx)) if rho > 1e-9 else 0.0

        # le TCP peut etre devant (r > 0) ou derriere (r < 0) l'axe de base
        branches_base = [(rho, cap), (-rho, _norm180(cap + 180.0))]

        # vecteur outil apres rotation totale
        d_outil = self.dir_outil0 - orientation
        ox = self.l_outil * math.cos(math.radians(d_outil))
        oz = self.l_outil * math.sin(math.radians(d_outil))

        s_rot = (rotation_pince if rotation_pince is not None
                 else (self.axes[4]["min"] + self.axes[4]["max"]) / 2.0)

        sols: List[Pose] = []
        for r, a1 in branches_base:
            s1 = self.angle_vers_servo(0, a1)
            if not self.dans_butees(0, s1):
                continue

            # position du poignet dans le plan sagittal
            wx = self.base[0] + r - ox
            wz = z - oz

            ddx, ddz = wx - self.sh[0], wz - self.sh[1]
            dist = math.hypot(ddx, ddz)
            if dist > self.l_bras + self.l_avant or dist < abs(self.l_bras - self.l_avant):
                continue
            if dist < 1e-9:
                continue

            cos_alpha = (self.l_bras ** 2 + dist ** 2 - self.l_avant ** 2) / (2 * self.l_bras * dist)
            cos_alpha = max(-1.0, min(1.0, cos_alpha))
            alpha = math.degrees(math.acos(cos_alpha))
            psi = math.degrees(math.atan2(ddz, ddx))

            for signe in (+1, -1):            # coude en haut puis en bas
                dir_bras = psi + signe * alpha
                a2 = _norm180(self.dir_bras0 - dir_bras)

                cx = self.sh[0] + self.l_bras * math.cos(math.radians(dir_bras))
                cz = self.sh[1] + self.l_bras * math.sin(math.radians(dir_bras))
                dir_avant = math.degrees(math.atan2(wz - cz, wx - cx))
                a3 = _norm180(self.dir_avant0 - a2 - dir_avant)

                a4 = _norm180(orientation - a2 - a3)

                s2_ = self._servo_valide(1, a2)
                s3_ = self._servo_valide(2, a3)
                s4_ = self._servo_valide(3, a4)
                if s2_ is None or s3_ is None or s4_ is None:
                    continue
                servos = [s1, s2_, s3_, s4_, s_rot]

                angles = tuple(self.servo_vers_angle(i, s) for i, s in enumerate(servos))
                sols.append(Pose(servos=tuple(servos), angles=angles,
                                 coude_en_haut=(signe > 0),
                                 epaule=(cx, cz), coude=(wx, wz)))

        # verification : on ne renvoie que ce qui repasse par la directe
        bonnes = []
        for s in sols:
            f = self.fk(s.servos)
            if math.dist((f["x"], f["y"], f["z"]), (x, y, z)) < 0.05:
                bonnes.append(s)

        # on prefere le coude en haut, puis le bras le moins sollicite
        bonnes.sort(key=lambda s: (not s.coude_en_haut, -s.epaule[1]))

        if filtre is not None:
            bonnes = [s for s in bonnes if filtre(s.servos) is None]
        return bonnes

    def ik(self, x: float, y: float, z: float,
           orientation: float = PINCE_BAS,
           rotation_pince: Optional[float] = None,
           filtre=None) -> Optional[Pose]:
        """Meilleure solution, ou None si la cible n'est pas atteignable."""
        sols = self.ik_toutes(x, y, z, orientation, rotation_pince, filtre)
        return sols[0] if sols else None

    def ik_orientable(self, x: float, y: float, z: float,
                      orientation: float = PINCE_BAS,
                      tolerance: float = 30.0, pas: float = 5.0,
                      rotation_pince: Optional[float] = None,
                      filtre=None) -> Optional[Tuple[Pose, float]]:
        """Comme ik, mais laisse la pince s'incliner si c'est le prix a payer.

        Essaie l'orientation demandee, puis s'en ecarte par pas croissants des
        deux cotes jusqu'a `tolerance`. Rend (pose, ecart) ou None.

        Exiger la verticale stricte reduit beaucoup le volume utile : la pince
        fait 117 mm sous l'axe du poignet, et l'inclinaison n'a que 180 deg de
        course. Tolerer 30 deg multiplie la hauteur exploitable par sept.
        """
        ecarts = [0.0]
        k = pas
        while k <= tolerance + 1e-9:
            ecarts.extend((-k, k))
            k += pas
        for d in ecarts:
            sol = self.ik(x, y, z, orientation + d, rotation_pince, filtre)
            if sol is not None:
                return sol, d
        return None


# ---------------------------------------------------------------- autotest

def _autotest() -> int:
    b = Bras()
    print(f"bras      {b.l_bras:7.2f} mm   (epaule -> coude)")
    print(f"avant-bras{b.l_avant:7.2f} mm   (coude -> poignet)")
    print(f"outil     {b.l_outil:7.2f} mm   (poignet -> centre des mors)")
    print()

    # aller-retour : une pose servo -> directe -> inverse -> la meme pose
    import itertools
    essais = resolues = identiques = 0
    pire_ecart = 0.0
    for s2, s3, s4 in itertools.product(range(0, 181, 15), repeat=3):
        servos = [90, s2, s3, s4, 90]
        f = b.fk(servos)
        if f["z"] < 0 or abs(f["r"]) > 400:
            continue
        essais += 1
        sols = b.ik_toutes(f["x"], f["y"], f["z"], orientation=f["orientation"])
        if not sols:
            continue
        resolues += 1
        ecart = min(math.dist((b.fk(s.servos)["x"], b.fk(s.servos)["y"], b.fk(s.servos)["z"]),
                              (f["x"], f["y"], f["z"])) for s in sols)
        pire_ecart = max(pire_ecart, ecart)
        if any(max(abs(_norm180(a - c)) for a, c in zip(s.servos, servos)) < 0.1
               for s in sols):
            identiques += 1
    print(f"aller-retour : {resolues}/{essais} cibles resolues")
    print(f"               {identiques} retombent sur la pose d'origine, "
          f"{resolues - identiques} sur une autre configuration equivalente")
    print(f"ecart de position maximal : {pire_ecart:.4f} mm")
    print()

    # balayage du plan de travail, pince verticale vers le bas
    z = b.table_z + 10
    atteints = []
    for rayon in range(40, 381, 10):
        cible = (b.base[0] - rayon, b.base[1], z)
        if b.ik(*cible):
            atteints.append(rayon)
    if atteints:
        print(f"pince verticale a Z = {z:.0f} mm : rayons atteignables "
              f"{min(atteints)} a {max(atteints)} mm "
              f"({len(atteints)} sur 35 testes)")

    exemple = b.ik(b.base[0] - 200.0, b.base[1], b.table_z + 10)
    if exemple:
        print()
        print("exemple  cible R = 200 mm, Z = 95 mm, pince verticale :")
        noms = [a["nom"] for a in b.axes]
        for nom, s in zip(noms, exemple.servos):
            print(f"   {nom:24s} {s:7.2f} deg")
        f = b.fk(exemple.servos)
        print(f"   verification directe : X {f['x']:.2f}  Y {f['y']:.2f}  Z {f['z']:.2f}")
    return 0 if resolues == essais and pire_ecart < 0.05 else 1


if __name__ == "__main__":
    raise SystemExit(_autotest())
