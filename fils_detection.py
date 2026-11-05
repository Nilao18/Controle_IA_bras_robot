#!/usr/bin/env python3
"""Mesure l'effet du nombre de fils de calcul de MediaPipe.

Pourquoi ce script existe. Sur la carte, la detection coute 84 ms par image
et le processus n'occupe que 1,35 coeur sur quatre : MediaPipe calcule sur un
seul. Ce n'est pas un reglage malheureux mais une constante de compilation,
dans mediapipe/calculators/tensor/inference_calculator_utils.cc :

    #if defined(MEDIAPIPE_ANDROID) || defined(MEDIAPIPE_IOS) || ...
      return std::clamp(NumCPUCores() / 2, 1, 4);
    #else
      return 1;        // Linux : un seul fil, en dur
    #endif

Sur Android MediaPipe prend la moitie des coeurs ; sur Linux il en prend un.
La carte est un Linux aarch64 a quatre coeurs, elle tombe donc du mauvais
cote.

Le graphe sait pourtant recevoir la consigne : le noeud InferenceCalculator
porte `delegate { xnnpack {} }` et GetXnnpackNumThreads() lit
`delegate.xnnpack.num_threads` des qu'il vaut autre chose que -1. Mais ce
champ est imbrique a trois niveaux, alors que le `calculator_params` de
l'API Python n'accepte que `{calculateur}.{champ}`, un seul niveau. On
intercepte donc _modify_calculator_options, qui recoit justement le graphe
canonique juste avant la construction, et on y pose le champ.

Le banc mesure sur UNE SEULE image, gardee une fois pour toutes : sans cela
on comparerait des poses de main differentes plutot que des reglages.

    python3 fils_detection.py
    python3 fils_detection.py --fils 1 2 3 4 --tours 60
"""

from __future__ import annotations

import argparse
import sys
import time


def forcer_fils(n: int):
    """Impose n fils aux noeuds d'inference. Rend de quoi tout remettre.

    Les deux formes de portage d'options sont traitees : l'extension
    historique `options`, utilisee par le graphe des mains, et le
    `node_options` empaquete en Any des graphes plus recents.
    """
    import mediapipe.python.solution_base as sb
    from mediapipe.calculators.tensor import inference_calculator_pb2 as icp

    ext = icp.InferenceCalculatorOptions.ext
    original = sb.SolutionBase._modify_calculator_options

    def patche(self, cfg, params):
        original(self, cfg, params)
        for node in cfg.node:
            if "InferenceCalculator" not in node.calculator:
                continue
            pose = False
            for brut in node.node_options:
                if brut.Is(icp.InferenceCalculatorOptions.DESCRIPTOR):
                    o = icp.InferenceCalculatorOptions()
                    brut.Unpack(o)
                    o.delegate.xnnpack.num_threads = n
                    brut.Pack(o)
                    pose = True
            if not pose:
                o = node.options.Extensions[ext]
                o.delegate.xnnpack.num_threads = n
            patche.noeuds.append(node.name or node.calculator)

    patche.noeuds = []
    sb.SolutionBase._modify_calculator_options = patche
    return patche, (lambda: setattr(sb.SolutionBase,
                                    "_modify_calculator_options", original))


def image_avec_main(cv2, mp, device: int, delai: float = 20.0):
    """Attend une image ou une main est vue, et la garde."""
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise SystemExit("camera /dev/video%d inaccessible" % device)
    det = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1,
                                   model_complexity=0,
                                   min_detection_confidence=0.5)
    print("montre ta main a la camera...")
    fin = time.time() + delai
    garde = None
    while time.time() < fin:
        ok, img = cap.read()
        if not ok:
            continue
        res = det.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if res.multi_hand_landmarks:
            garde = img.copy()
            break
    cap.release()
    det.close()
    if garde is None:
        raise SystemExit("aucune main vue en %.0f s : recommence" % delai)
    print("image retenue, main detectee\n")
    return garde


def mesurer(cv2, mp, img, fils, tours):
    """Cout moyen d'une detection, le graphe etant force a `fils` fils."""
    remettre = None
    noeuds = []
    if fils > 0:
        patche, remettre = forcer_fils(fils)
    try:
        det = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1,
                                       model_complexity=0,
                                       min_detection_confidence=0.5,
                                       min_tracking_confidence=0.3)
        if fils > 0:
            noeuds = list(patche.noeuds)
    finally:
        if remettre:
            remettre()
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for _ in range(5):                      # on amorce le suivi
        det.process(rgb)
    t0 = time.perf_counter()
    for _ in range(tours):
        det.process(rgb)
    ms = (time.perf_counter() - t0) / tours * 1000.0
    det.close()
    return ms, noeuds


def principal(a) -> int:
    import cv2
    import mediapipe as mp
    print("mediapipe %s\n" % mp.__version__)

    img = image_avec_main(cv2, mp, a.device)

    base, _ = mesurer(cv2, mp, img, 0, a.tours)
    print("  %-22s %6.1f ms   (reference)" % ("defaut MediaPipe", base))

    premier = True
    for n in a.fils:
        try:
            ms, noeuds = mesurer(cv2, mp, img, n, a.tours)
        except Exception as e:
            print("  %-22s echec : %s" % ("%d fils" % n, e))
            continue
        if premier and noeuds:
            premier = False
            print("  (noeuds d'inference touches : %s)" % ", ".join(noeuds))
        print("  %-22s %6.1f ms   x%.2f" % ("%d fils" % n, ms, base / ms))

    print("\nSi aucun reglage ne change le temps, le graphe n'a pas ete")
    print("modifie : le nom des champs a bouge dans cette version, et il")
    print("faudra regarder la liste des noeuds ci-dessus.")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--fils", type=int, nargs="+", default=[2, 3, 4])
    p.add_argument("--tours", type=int, default=40)
    sys.exit(principal(p.parse_args()))
