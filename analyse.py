#!/usr/bin/env python3
"""Separe la distance de l'orientation dans ~/orientation.csv.

Tenir le poignet immobile pendant quatre rotations est illusoire : la derniere
mesure accusait 4 cm de derive, et le "bruit" relevait alors du mouvement de
la main, pas de la detection.

Mais chaque indice vaut  k * g(orientation) / distance. En logarithme :

    log f_i(image) = log k_i + o_i(pose) - log d(image)

et le terme de distance est COMMUN a tous les indices d'une meme image. Avec
dix indices par image et des centaines d'images, le systeme est largement
surdetermine : on resout d(image) et o_i(pose) ensemble, et la derive du
poignet se retrouve entierement dans d, ou elle ne gene plus personne.

Ce que l'ajustement rend :
  orientation : de combien l'indice s'effondre sous rotation, en plus du
                plus stable de tous, qui sert de reference et lit zero. La
                distance etant inconnue, seul ce relatif est observable -
                mais c'est exactement ce qu'il faut pour choisir.
  bruit       : dispersion restante une fois la distance et l'orientation
                expliquees. C'est le vrai bruit de detection.

    python3 analyse.py
"""

from __future__ import annotations

import os
import sys

import numpy as np


def lire(chemin):
    with open(chemin) as f:
        entete = f.readline().strip().split(",")
        lignes = [l.strip().split(",") for l in f if l.strip()]
    cles = [k for k in entete[1:] if not k.startswith("_")]
    idx = [entete.index(k) for k in cles]
    poses, valeurs = [], []
    for l in lignes:
        try:
            v = [float(l[i]) for i in idx]
        except (ValueError, IndexError):
            continue
        if not all(x > 0 for x in v):
            continue
        poses.append(l[0])
        valeurs.append(v)
    return cles, poses, np.array(valeurs, float)


def ajuster(LOG, poses, n_iter=120):
    M, N = LOG.shape
    P = poses.max() + 1
    b = np.zeros(M)
    o = np.zeros((N, P))
    k = LOG.mean(axis=0)
    for _ in range(n_iter):
        b = (LOG - (k[None, :] + o[:, poses].T)).mean(axis=1)
        R = LOG - (k[None, :] + b[:, None])
        for p in range(P):
            o[:, p] = R[poses == p].mean(axis=0)
        o -= o[:, [0]]                       # la pose 1 est la reference
        # Jauge. Un raccourci de perspective ne peut que RETRECIR la mesure,
        # jamais l'agrandir : la vraie reponse est donc negative ou nulle pour
        # tous les indices. On cale donc chaque pose sur son indice le plus
        # stable, qui lit zero ; les autres lisent de combien ils s'effondrent
        # EN PLUS. Caler sur la moyenne, comme je l'avais fait d'abord, faisait
        # passer le meilleur indice pour le pire des que la majorite s'ecroulait.
        o -= o.max(axis=0)[None, :]
        k = (LOG - b[:, None] - o[:, poses].T).mean(axis=0)
    residus = LOG - (k[None, :] + b[:, None] + o[:, poses].T)
    return b, o, residus


def principal():
    chemin = os.path.expanduser("~/orientation.csv")
    if not os.path.exists(chemin):
        raise SystemExit("%s introuvable : lance d'abord orientation.py" % chemin)
    cles, noms, V = lire(chemin)
    if len(V) < 50:
        raise SystemExit("trop peu de lignes exploitables (%d)" % len(V))

    ordre = []
    for n in noms:
        if n not in ordre:
            ordre.append(n)
    if "face" in ordre:
        ordre.remove("face")
        ordre.insert(0, "face")
    poses = np.array([ordre.index(n) for n in noms])

    LOG = np.log(V)
    # certains indices sont des DISTANCES (ils croissent avec l'eloignement),
    # les autres des TAILLES (elles decroissent). On les remet dans le meme
    # sens, sinon ils se combattent dans l'ajustement.
    ref = LOG[:, 0] - LOG[:, 0].mean()
    sens = []
    for j in range(LOG.shape[1]):
        c = np.corrcoef(ref, LOG[:, j] - LOG[:, j].mean())[0, 1]
        sens.append(-1.0 if c < 0 else 1.0)
    sens = np.array(sens)
    LOG = LOG * sens[None, :]

    b, o, R = ajuster(LOG, poses)

    print("\n%d images, %d indices, %d poses : %s"
          % (len(V), len(cles), len(ordre), ", ".join(ordre)))
    inverses = [cles[j] for j in range(len(cles)) if sens[j] < 0]
    if inverses:
        print("indices remis dans le sens des tailles : %s" % ", ".join(inverses))

    etendue_d = float(np.exp(b.max() - b.min()))
    print("la distance reconstruite varie d'un facteur %.2f sur l'ensemble"
          % etendue_d)

    print("\n  candidat          bruit    %s   pire   verdict"
          % "  ".join("%-8s" % p[:8] for p in ordre[1:]))
    print("  " + "-" * (34 + 10 * (len(ordre) - 1)))

    lignes = []
    for j, k in enumerate(cles):
        bruit = 100.0 * float(R[:, j].std())
        par_pose = [100.0 * abs(float(o[j, p])) for p in range(1, len(ordre))]
        pertinentes = [par_pose[i] for i, p in enumerate(ordre[1:])
                       if p != "loin"]
        pire = max([bruit] + pertinentes)
        lignes.append((pire, k, bruit, par_pose))

    for pire, k, bruit, par_pose in sorted(lignes):
        verdict = ("solide" if pire < 8 else
                   "acceptable" if pire < 15 else
                   "fragile" if pire < 30 else "a ecarter")
        print("  %-16s %5.1f %%  %s  %5.1f %%  %s"
              % (k, bruit,
                 "  ".join("%+6.1f %%" % v for v in par_pose),
                 pire, verdict))

    # ---- balayage de la correction d'inclinaison -----------------------
    # aire_sur_cos vaut racine_aire / sqrt(cos) : le rapport des deux colonnes
    # redonne donc cos exactement. sqrt(cos) sous-corrigeait, cos sur-corrigeait
    # - l'exposant juste est entre les deux et n'a jamais ete essaye. On le
    # cherche contre la distance deja estimee ci-dessus, qui absorbe la derive
    # du poignet.
    if "racine_aire" in cles and "aire_sur_cos" in cles:
        ia, ic = cles.index("racine_aire"), cles.index("aire_sur_cos")
        cos = np.clip((V[:, ia] / V[:, ic]) ** 2, 1e-3, 1.0)
        bases = [n for n in ("racine_aire", "paume_px", "diametre_palme",
                             "etendue_main", "moy_3_max") if n in cles]
        print("\n  correction d'inclinaison : indice / cos^alpha")
        print("  (cos reconstruit depuis les colonnes, distance deja retiree)")
        print("\n  base              alpha   %s   pire"
              % "  ".join("%-8s" % q[:8] for q in ordre[1:-1]))
        print("  " + "-" * (30 + 10 * (len(ordre) - 2)))
        for nom in bases:
            jb = cles.index(nom)
            meilleur = None
            for alpha in np.arange(0.0, 1.55, 0.05):
                L = np.log(V[:, jb]) - alpha * np.log(cos) + b
                ref = np.median(L[poses == 0])
                ec = [abs(float(np.median(L[poses == p]) - ref))
                      for p in range(1, len(ordre)) if ordre[p] != "loin"]
                pire = max(ec)
                if meilleur is None or pire < meilleur[0]:
                    meilleur = (pire, alpha, ec)
            pire, alpha, ec = meilleur
            print("  %-16s %5.2f   %s  %5.1f %%"
                  % (nom, alpha,
                     "  ".join("%+6.1f %%" % (100 * e) for e in ec), 100 * pire))

    print("\n  bruit       : dispersion residuelle, distance et orientation")
    print("                deja expliquees. C'est le vrai bruit de detection.")
    print("  colonnes de pose : de combien l'indice s'effondre dans cette")
    print("                orientation, en plus du plus stable de tous.")
    print("  pire        : la plus grande des erreurs, hors pose lointaine.")
    print("                C'est elle qui decide.")
    return 0


if __name__ == "__main__":
    sys.exit(principal())
