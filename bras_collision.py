#!/usr/bin/env python3
"""Detection de collision du bras, maillage contre maillage.

Memes donnees que le simulateur 3D : chaque piece porte un nuage de points de
surface et un volume plein voxelise a 2 mm, erode d'un voxel pour absorber les
jeux d'emboitement. Un point de surface d'une piece qui tombe dans le volume
plein d'une autre signale une interpenetration reelle.

numpy est utilise s'il est present (environ 1 ms par pose) ; sinon le module
retombe sur du Python pur, correct mais une trentaine de fois plus lent.

    from bras_collision import Collisions
    c = Collisions()
    c.verifier([90, 76.4, 61.1, 164.7, 90])     # -> None si la pose est saine
"""

from __future__ import annotations

import base64
import json
import math
import os
from typing import List, Optional, Sequence, Tuple

try:
    import numpy as _np
except ImportError:                                   # pragma: no cover
    _np = None

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bras_collision.json")
PARAMS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bras_params.json")

# paires de pieces a tester ; les pieces solidaires ne s'y trouvent pas
PAIRES = [
    ("L0", "L2"), ("L0", "L3"), ("L0", "L4"), ("L0", "L5"), ("L0", "L6"), ("L0", "L7"),
    ("L1", "L2"), ("L1", "L3"), ("L1", "L4"), ("L1", "L5"), ("L1", "L6"), ("L1", "L7"),
    ("L2", "L3"), ("L2", "L4"), ("L2", "L5"), ("L2", "L6"), ("L2", "L7"),
    ("L3", "L4"), ("L3", "L5"), ("L3", "L6"), ("L3", "L7"),
    ("L4", "L5"), ("L4", "L6"), ("L4", "L7"),
]
MOBILES = ("L2", "L3", "L4", "L5", "L6", "L7")
MIN_POINTS = 2          # nombre de points penetrants avant de declarer la collision


def _mat_rot_y(deg: float):
    a = math.radians(deg); c, s = math.cos(a), math.sin(a)
    return ((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c))


def _mat_rot_z(deg: float):
    a = math.radians(deg); c, s = math.cos(a), math.sin(a)
    return ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))


def _mv(R, v):
    return (R[0][0]*v[0] + R[0][1]*v[1] + R[0][2]*v[2],
            R[1][0]*v[0] + R[1][1]*v[1] + R[1][2]*v[2],
            R[2][0]*v[0] + R[2][1]*v[1] + R[2][2]*v[2])


def _mm(A, B):
    return tuple(tuple(sum(A[i][k]*B[k][j] for k in range(3)) for j in range(3))
                 for i in range(3))


def _pivot(R, centre):
    """Rotation autour d'un point : rend (R, t) tel que p' = R p + t."""
    return R, tuple(centre[i] - _mv(R, centre)[i] for i in range(3))


def _compose(A, B):
    """(Ra,ta) o (Rb,tb)"""
    return _mm(A[0], B[0]), tuple(_mv(A[0], B[1])[i] + A[1][i] for i in range(3))


def _coins(lo, hi):
    """Les huit sommets du pave [lo, hi]."""
    c = [(lo[0] if i & 1 else hi[0],
          lo[1] if i & 2 else hi[1],
          lo[2] if i & 4 else hi[2]) for i in range(8)]
    return _np.asarray(c, dtype=_np.float64) if _np is not None else c


def _boite_monde(coins, R, t):
    """Boite alignee sur les axes qui contient le pave local, une fois tourne."""
    if _np is not None:
        m = coins @ _np.asarray(R).T + _np.asarray(t)
        return m.min(axis=0), m.max(axis=0)
    p = [tuple(_mv(R, c)[i] + t[i] for i in range(3)) for c in coins]
    return ([min(q[i] for q in p) for i in range(3)],
            [max(q[i] for q in p) for i in range(3)])


def _chevauche(a, b, marge=0.0):
    """Deux boites alignees se chevauchent-elles ?"""
    (a0, a1), (b0, b1) = a, b
    return (a0[0] - marge <= b1[0] and b0[0] - marge <= a1[0] and
            a0[1] - marge <= b1[1] and b0[1] - marge <= a1[1] and
            a0[2] - marge <= b1[2] and b0[2] - marge <= a1[2])


class Collisions:
    def __init__(self, data_path: str = DATA_PATH, params_path: str = PARAMS_PATH):
        with open(data_path, "r", encoding="utf-8") as f:
            d = json.load(f)
        self.scale = d["scale"]
        self.vox = d["vox"]
        self.pieces = {}
        for k, p in d["pieces"].items():
            brut = base64.b64decode(p["points"])
            n = p["n"]
            if _np is not None:
                pts = _np.frombuffer(brut, dtype="<i2")[:n*3].reshape(n, 3).astype(_np.float64) / self.scale
                bits = _np.unpackbits(_np.frombuffer(base64.b64decode(p["volume"]), dtype=_np.uint8),
                                      bitorder="little")
                dim = p["dim"]
                vol = bits[:dim[0]*dim[1]*dim[2]].reshape(dim).astype(bool)
            else:
                import struct
                vals = struct.unpack("<%dh" % (n*3), brut[:n*6])
                pts = [tuple(vals[i*3+j]/self.scale for j in range(3)) for i in range(n)]
                vol = base64.b64decode(p["volume"])
            self.pieces[k] = {"nom": p["nom"], "n": n, "pts": pts,
                              "lo": p["lo"], "dim": p["dim"], "vol": vol}

        # Boites englobantes locales, et leurs huit sommets.
        #
        # Le controle coutait 26 ms sur la carte : 24 paires testees dans les
        # deux sens, soit 48 transformations de nuage complet. Or deux pieces
        # eloignees n'ont aucune chance de se penetrer. On transforme donc
        # d'abord les huit sommets du pave englobant - une soixantaine de
        # points pour tout le bras - et on ne touche aux nuages que pour les
        # paires dont les boites se chevauchent vraiment.
        #
        # Un pave tourne plutot qu'une sphere : les segments de bras sont
        # longs et minces, et leur sphere englobante est enorme alors que
        # leur pave reste juste.
        for k, p in self.pieces.items():
            if _np is not None:
                lo_p = p["pts"].min(axis=0)
                hi_p = p["pts"].max(axis=0)
            else:
                lo_p = [min(q[i] for q in p["pts"]) for i in range(3)]
                hi_p = [max(q[i] for q in p["pts"]) for i in range(3)]
            lo_v = list(p["lo"])
            hi_v = [p["lo"][i] + p["dim"][i] * self.vox for i in range(3)]
            p["coins_pts"] = _coins(lo_p, hi_p)
            p["coins_vol"] = _coins(lo_v, hi_v)

        with open(params_path, "r", encoding="utf-8") as f:
            q = json.load(f)
        self.base = (q["base"]["x"], q["base"]["y"])
        self.sh = (q["epaule"]["x"], q["epaule"]["z"])
        self.el = (q["coude"]["x"], q["coude"]["z"])
        self.wr = (q["poignet"]["x"], q["poignet"]["z"])
        self.rot = (q["rotation"]["x"], q["rotation"]["y"])
        self.axes = q["axes"]
        c = q["couvercle"]
        self.boitier = (c["x0"], c["x1"], c["y0"], c["y1"], c["z_boitier"])
        self.course_pince = q["pince"]["course_totale"]

    # ---------------------------------------------------------- geometrie

    def _reperes(self, angles: Sequence[float], ouverture: float):
        a1, a2, a3, a4, a5 = angles
        I = (((1.0,0.0,0.0),(0.0,1.0,0.0),(0.0,0.0,1.0)), (0.0,0.0,0.0))
        T = {"L0": I}
        T["L1"] = _pivot(_mat_rot_z(a1), (self.base[0], self.base[1], 0.0))
        T["L2"] = _compose(T["L1"], _pivot(_mat_rot_y(a2), (self.sh[0], 0.0, self.sh[1])))
        T["L3"] = _compose(T["L2"], _pivot(_mat_rot_y(a3), (self.el[0], 0.0, self.el[1])))
        T["L4"] = _compose(T["L3"], _pivot(_mat_rot_y(a4), (self.wr[0], 0.0, self.wr[1])))
        T["L5"] = _compose(T["L4"], _pivot(_mat_rot_z(a5), (self.rot[0], self.rot[1], 0.0)))
        dy = ouverture / 2.0
        ident = ((1.0,0.0,0.0),(0.0,1.0,0.0),(0.0,0.0,1.0))
        T["L6"] = _compose(T["L5"], (ident, (0.0, -dy, 0.0)))
        T["L7"] = _compose(T["L5"], (ident, (0.0,  dy, 0.0)))
        return T

    def _dedans(self, cible: str, pts_monde, R, t) -> int:
        """Combien de points monde tombent dans le volume plein de `cible`."""
        p = self.pieces[cible]
        lo, dim, vol = p["lo"], p["dim"], p["vol"]
        if _np is not None:
            loc = (pts_monde - _np.asarray(t)) @ _np.asarray(R)
            idx = _np.floor((loc - _np.asarray(lo)) / self.vox).astype(_np.int64)
            ok = _np.all((idx >= 0) & (idx < _np.asarray(dim)), axis=1)
            if not ok.any():
                return 0
            i = idx[ok]
            return int(vol[i[:, 0], i[:, 1], i[:, 2]].sum())
        n = 0
        for w in pts_monde:
            d = (w[0]-t[0], w[1]-t[1], w[2]-t[2])
            lx = R[0][0]*d[0] + R[1][0]*d[1] + R[2][0]*d[2]
            ly = R[0][1]*d[0] + R[1][1]*d[1] + R[2][1]*d[2]
            lz = R[0][2]*d[0] + R[1][2]*d[1] + R[2][2]*d[2]
            ix = int(math.floor((lx - lo[0]) / self.vox))
            if ix < 0 or ix >= dim[0]: continue
            iy = int(math.floor((ly - lo[1]) / self.vox))
            if iy < 0 or iy >= dim[1]: continue
            iz = int(math.floor((lz - lo[2]) / self.vox))
            if iz < 0 or iz >= dim[2]: continue
            k = (ix*dim[1] + iy)*dim[2] + iz
            if (vol[k >> 3] >> (k & 7)) & 1:
                n += 1
                if n >= MIN_POINTS:
                    return n
        return n

    def _monde(self, cle: str, T):
        R, t = T[cle]
        pts = self.pieces[cle]["pts"]
        if _np is not None:
            return pts @ _np.asarray(R).T + _np.asarray(t)
        return [tuple(_mv(R, p)[i] + t[i] for i in range(3)) for p in pts]

    # ------------------------------------------------------------ controle

    def verifier_angles(self, angles: Sequence[float], ouverture: float = 0.0) -> Optional[str]:
        """Angles geometriques. Rend None si la pose est saine, sinon le contact."""
        T = self._reperes(angles, ouverture)
        bp, bv = {}, {}
        for k, p in self.pieces.items():
            R, t = T[k]
            bp[k] = _boite_monde(p["coins_pts"], R, t)
            bv[k] = _boite_monde(p["coins_vol"], R, t)

        # nuages transformes a la demande : la plupart des paires sont ecartees
        # sur les boites, et leur nuage n'a alors jamais a etre calcule
        W = {}

        def monde(k):
            if k not in W:
                W[k] = self._monde(k, T)
            return W[k]

        self.paires_vues = 0
        for a, b in PAIRES:
            if _chevauche(bp[a], bv[b]):
                self.paires_vues += 1
                if self._dedans(b, monde(a), *T[b]) >= MIN_POINTS:
                    return f"{self.pieces[a]['nom']} / {self.pieces[b]['nom']}"
            if _chevauche(bp[b], bv[a]):
                self.paires_vues += 1
                if self._dedans(a, monde(b), *T[a]) >= MIN_POINTS:
                    return f"{self.pieces[a]['nom']} / {self.pieces[b]['nom']}"
        x0, x1, y0, y1, zb = self.boitier
        boite_boitier = ((x0, y0, -1e6), (x1, y1, zb))
        for k in MOBILES:
            lo, hi = bp[k]
            # ni sous la table, ni dans l'emprise du boitier : rien a compter
            if lo[2] >= 0.0 and not _chevauche(bp[k], boite_boitier):
                continue
            P = monde(k)
            if _np is not None:
                sous = (P[:, 2] < 0).sum()
                dans = ((P[:, 0] > x0) & (P[:, 0] < x1) & (P[:, 1] > y0) &
                        (P[:, 1] < y1) & (P[:, 2] < zb)).sum()
            else:
                sous = sum(1 for p in P if p[2] < 0)
                dans = sum(1 for p in P if x0 < p[0] < x1 and y0 < p[1] < y1 and p[2] < zb)
            if sous >= MIN_POINTS:
                return f"{self.pieces[k]['nom']} / table"
            if dans >= MIN_POINTS:
                return f"{self.pieces[k]['nom']} / boitier"
        return None

    def verifier(self, servos: Sequence[float], ouverture: float = 0.0) -> Optional[str]:
        """Angles servo. Rend None si la pose est saine, sinon le contact."""
        angles = [self.axes[i]["sens"] * (s - self.axes[i]["ref"])
                  for i, s in enumerate(servos[:5])]
        return self.verifier_angles(angles, ouverture)

    def trajet_libre(self, depart: Sequence[float], arrivee: Sequence[float],
                     pas: int = 12, ouverture: float = 0.0) -> Optional[str]:
        """Interpole entre deux poses servo et verifie chaque etape."""
        for k in range(pas + 1):
            f = k / pas
            inter = [d + (a - d) * f for d, a in zip(depart[:5], arrivee[:5])]
            c = self.verifier(inter, ouverture)
            if c:
                return f"{c} (a {100*f:.0f} % du trajet)"
        return None


def _autotest() -> int:
    import time
    from bras_ik import Bras

    c = Collisions()
    b = Bras()
    print("numpy" if _np is not None else "Python pur", "- donnees chargees")

    t0 = time.perf_counter()
    for _ in range(50):
        c.verifier([90, 76.4, 61.1, 164.7, 90])
    print(f"cout d'un controle : {(time.perf_counter()-t0)/50*1000:.2f} ms\n")

    # (libelle, servo ou angles, attend-on une collision ?)
    cas = [("pose pliee",      ("servo", [90, 0, 0, 0, 90]),            False),
           ("prise R200",      ("servo", [90, 76.4, 61.1, 164.7, 90]),  False),
           ("bras tendu",      ("servo", [90, 0, 180, 0, 90]),          False),
           ("epaule a fond",   ("servo", [90, 180, 0, 0, 90]),          True),
           ("coude a l'envers",("angle", [0, -87, -63, 150, 0]),        True),
           ("poignet rabattu", ("angle", [0, 0, 0, 30, 0]),             True)]
    bon = 0
    for nom, (mode, v), attendu in cas:
        r = c.verifier(v) if mode == "servo" else c.verifier_angles(v)
        ok = (r is not None) == attendu
        bon += ok
        etat = ("collision : " + r) if r else "OK"
        print(f"  {nom:17s} {etat:42s} {'v' if ok else 'X  attendu ' + ('collision' if attendu else 'OK')}")

    print("\nbalayage du plan de travail, pince verticale :")
    libres = []
    for rayon in range(60, 381, 10):
        sol = b.ik(b.base[0] - rayon, b.base[1], b.table_z + 10.0)
        if sol and c.verifier(sol.servos) is None:
            libres.append(rayon)
    if libres:
        print(f"  rayons sains : {min(libres)} a {max(libres)} mm ({len(libres)} sur 33)")
    return 0 if bon == len(cas) else 1


if __name__ == "__main__":
    raise SystemExit(_autotest())
