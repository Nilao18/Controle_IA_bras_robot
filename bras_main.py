#!/usr/bin/env python3
"""Pilotage du bras par la main, via MediaPipe.

Chaine complete : camera -> MediaPipe -> coordonnees cylindriques -> bras_ik
-> filtre de collision -> simulateur 3D et, si demande, servos du STM32.

    gauche / droite   -> rotation de base
    haut / bas        -> hauteur du point de prise
    main proche/loin  -> distance a l'axe de base
    pouce / index     -> ouverture de la pince

La profondeur vient de la travee de paume corrigee de l'inclinaison, en
cos^0.20 : parmi dix indices mesures sur cette camera dans cinq orientations,
c'est celui dont la rotation de la main derange le moins la mesure.

Par defaut rien ne bouge physiquement : seul le simulateur est alimente.

    python3 bras_main.py              # simulateur seul
    python3 bras_main.py --servos     # + envoi au STM32
    python3 bras_main.py --mesure     # n'affiche que ce que voit la camera
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import subprocess
import sys
import time
from typing import Optional, Sequence, Tuple

# --------------------------------------------------------------- reglages

REGLAGES = {
    # part de l'image exploitee ; au-dela on sature
    # orientation de l'image : 0 vertical, 1 horizontal, -1 180 deg, None aucune.
    # Surtout pas de choix automatique : les quatre se valent pour la detection,
    # donc le tirage changeait d'un lancement a l'autre et avec lui le sens des
    # commandes. On fixe, et --flip permet d'essayer.
    "flip": 0,

    # La profondeur vient de la racine de l'aire de la paume, en pixels.
    #
    # Mesure sur 221 images de cette camera, main immobile : cet indice est le
    # moins bruite des neuf candidats testes (3,1 % d'etendue p5-p95, contre
    # 4,4 % pour une travee de paume et 14,0 % pour l'ajustement de pose 3D
    # qui tournait ici). Et il est trois fois plus robuste a l'inclinaison :
    # a 60 degres il varie de 10 % quand une travee varie de 29 %, parce que
    # la base du pouce entre dans l'enveloppe a mesure que la paume se
    # raccourcit. Il est aussi insensible a la pince (0,5 %), les doigts
    # n'etant pas dans l'enveloppe.
    #
    # Les ajustements de pose 3D ont ete abandonnes : sur cette camera ils
    # sont deux a quatre fois plus bruites, pour un signal identique.
    "paume_ids": (0, 1, 2, 5, 9, 13, 17),
    # exposant de la correction d'inclinaison, trouve par balayage sur 351
    # images de cette camera, dans cinq orientations de main tenues au meme
    # endroit. L'aire de paume y perdait 88 % sous bascule du poignet ; la
    # travee corrigee en cos^0.20 ne perd plus que 9,8 % au pire des trois
    # rotations, pour 1,7 % de bruit de detection.
    # Deux travees perpendiculaires, et on garde la plus grande.
    #
    # La travee poignet -> bases des doigts est verticale quand les doigts
    # pointent en haut : pivoter l'avant-bras ne la raccourcit pas. La largeur
    # de paume (index -> auriculaire) est horizontale : plier le poignet ne la
    # raccourcit pas. Chacune est aveugle a l'axe qui gene l'autre, donc la
    # plus grande des deux est toujours celle que la rotation epargne.
    #
    # Mesure, meme metrique pour les deux : sur les rotations realistes, 24 %
    # d'etendue contre 37 % pour la correction en cosinus qu'elle remplace ;
    # sur toute la grille, 37 % contre 95 %. Et a 45 degres de poignet penche,
    # le cas qui genait, +0,6 % au lieu de -26 %.
    #
    # L'exposant adoucit le maximum : a 8, le passage d'une travee a l'autre
    # se fait sur une quinzaine de pour-cent au lieu d'un saut.
    "travee_exposant": 8.0,
    # distance, en metres, a laquelle correspond la pose de calibration. Les
    # millimetres affiches sont desormais rapportes a elle.
    "ref_distance_m": 0.45,
    # periode d'envoi de l'image camera au simulateur, en secondes. Encoder
    # un JPEG coute quelques millisecondes : a 7 images par seconde on ne peut
    # pas se le permettre a chaque image sans perdre de la cadence.
    "periode_vue_s": 0.25,
    # L'exposition etait reglee une seule fois au demarrage, en manuel. Quand
    # la lumiere de la piece baisse - un debut de soiree suffit - l'image
    # s'assombrit, les reperes se degradent, et la profondeur part sans que
    # rien n'ait change dans le code. On surveille donc la luminosite et on
    # refait le reglage quand elle derive.
    "cible_luminosite": 112.0,
    "derive_luminosite": 22.0,
    "periode_luminosite_s": 6.0,
    # La luminosite etait mesuree sur l'image ENTIERE, main comprise : une
    # main qui approche de l'objectif eclaircit l'image, et le reglage se
    # declenchait sur elle. Cinq rattrapages en une minute, chacun vidant les
    # tampons - c'etait la latence. On ne mesure donc que les images SANS
    # main, et jamais plus d'une fois par minute.
    "expo_sans_main": True,
    "expo_periode_mini_s": 60.0,
    # convertit la racine d'aire en metres : distance = cette constante / aire.
    # Valeur approchee, la calibration proche/loin absorbe l'echelle.
    "aire_echelle": 63.0,
    # mediane glissante sur la distance, en images. Une mediane elimine une
    # valeur isolee au lieu de l'etaler comme le ferait un passe-bas. Mesure :
    # 3 images ramenent le balayage du rayon au repos de 12,6 a 7,3 mm, pour
    # 550 ms de reponse en plus. Au-dela de 5, on ne gagne plus rien.
    "mediane_profondeur": 3,
    # marge, en fraction d'image, sous laquelle on refuse de mesurer la
    # profondeur. Quand la main approche du bord, l'enveloppe de la paume se
    # tronque, son aire chute, et le bras recule alors qu'on avance. On prefere
    # tenir la derniere distance connue plutot que de suivre une mesure fausse.
    # On ne rejette que les reperes VRAIMENT hors image, pas une bande de
    # securite : avec la pleine hauteur comme plage de pilotage, atteindre le
    # bas demande d'amener la main au bord, et une marge de 3 % rejetait alors
    # des centaines d'images d'affilee - la profondeur restait figee sur sa
    # derniere valeur, ce qui donnait un bras bloque puis sautant.
    "marge_cadre": 0.0,
    # course de profondeur minimale acceptee a la calibration. En dessous, le
    # gain devient tel que le bruit sature les butees de rayon : c'est ce qui
    # rendait le bras inutilisable quand la calibration ratait.
    "course_mini_m": 0.15,
    # temps de placement avant chaque mesure de calibration. Trois secondes
    # ne suffisaient pas pour se mettre en position.
    "calibration_placement_s": 6.0,
    "calibration_mesure_s": 3.0,

    # garde-fou anti-saut : vitesse maximale credible d'une main. A 7 images par
    # seconde cela autorise 214 mm d'une image a l'autre, soit tous les gestes
    # jusqu'a 0,6 m/s sans un seul faux rejet, et aucune des aberrations
    # relevees. Apres 2 refus d'affilee on cede : la main a vraiment bouge.
    "saut_v_max": 1.5,
    "saut_suite_max": 2,

    # hysteresis : sous ce seuil on ne bouge pas du tout. Evite qu'un servo
    # fremisse en corrigeant sans cesse un dixieme de degre.
    # Seuil de changement, en degres servo, repris du seul systeme publie qui
    # pilote un bras a partir d'une simple webcam (Biomimetics 2026, 88 % de
    # reussite sur 42 essais) : un axe ne bouge que si sa consigne s'ecarte de
    # plus de 2 degres de sa position. C'est la derniere barriere, posee au
    # plus pres du servo, apres tous les autres filtrages.
    "zone_morte_deg": 2.0,
    # moyenne glissante sur les angles envoyes, en images. Meme source : trois
    # images. Elle n'arrondit que les marches laissees par le seuil ci-dessus.
    "moyenne_servo": 3,

    "x_min": 0.20, "x_max": 0.80,
    # pleine hauteur d'image : toute la course verticale du bras tient entre
    # le haut et le bas de l'image, sans etape de calibration.
    # La pince doit atteindre la table AVANT que le poignet ne sorte du cadre.
    # Le point pilote est la base du majeur ; le poignet est en dessous, donc
    # on arrete la course a 80 % de la hauteur plutot qu'au bord.
    "y_min": 0.12, "y_max": 0.80,

    # rotation de base, en degres servo
    "base_centre": 90.0, "base_amplitude": 55.0, "base_sens": -1,

    # hauteur du point de prise, en mm
    "z_bas": 88.0, "z_haut": 200.0,
    "z_sens": 1,          # 1 : main en haut de l'image = pince en haut

    # distance a l'axe, en mm
    "r_min": 110.0, "r_max": 310.0,
    "r_sens": 1,          # 1 : main proche de la camera = bras tendu

    # ouverture de pince, rapport ecart(pouce,index) / taille de main
    "pince_fermee": 0.30,

    # inclinaison de la pince toleree autour de la verticale, en degres.
    # 0 = verticale stricte, mais le volume utile n'a plus que 10 mm de haut.
    "tolerance_pince": 40.0,

    # Lissage. Le filtre un euro seul ne convient pas ici : il relache sa
    # coupure selon la vitesse ESTIMEE, et a 7 images par seconde sur un signal
    # bruite cette vitesse est du bruit. Il s'ouvrait donc en permanence sur du
    # bruit - nerveux et lent a la fois - et ne tenait le bras immobile que
    # 30 % du temps. Une zone morte glissante tient 96 % du temps et repond
    # plus vite : le tremblement naturel reste dans la zone morte et ne produit
    # aucun mouvement, un vrai geste pousse l'ancre sans retard.
    #
    # Seuils par axe, dans l'unite de l'axe. Plus grand = plus ferme, mais la
    # visee devient grossiere. Mesure sur 24 essais a 8 mm de bruit : 16 mm
    # ramene les sauts visibles de 11 % des images a 4 %, pour 120 ms de plus.
    "ancre_base_deg": 2.5,
    "ancre_r_mm": 25.0,
    "ancre_z_mm": 20.0,
    # derive lente DANS la zone morte. A zero, l'hysteresis est franche : sous
    # le seuil le bras ne bouge pas du tout, et aucune derive lente ne passe.
    # C'est au prix d'un decalage residuel qui peut atteindre le seuil.
    "ancre_rattrapage": 0.0,
    # passe-bas leger APRES l'ancre, juste pour arrondir ses marches
    "ancre_coupure": 0.6,
    "ancre_beta": 0.02,


    # variation maximale d'un servo entre deux images, en degres
    # Variation maximale d'un servo entre deux images. A 5 degres et 7 images
    # par seconde, cela plafonnait a 35 deg/s : un balayage de base de 90 deg
    # prenait 2,6 s. Le STM32 rampe deja a 75 deg/s de son cote, c'est lui la
    # vraie limite mecanique - inutile de brider deux fois.
    "vitesse_max_deg": 12.0,

    # on garde la derniere pose pendant ce temps apres la perte de la main
    "tolerance_perte_s": 0.7,
}

NOMS = ("base", "epaule", "coude", "inclinaison", "rotation")


def _borne(v, a, b):
    return a if v < a else (b if v > b else v)


def _interp(v, a, b, va, vb):
    if b == a:
        return va
    return va + (vb - va) * _borne((v - a) / (b - a), 0.0, 1.0)


class UnEuro:
    """Filtre « un euro » : lisse fort quand la main est posee, lache du lest
    quand elle bouge vite. C'est le bon outil a 9 images par seconde, ou un
    lissage exponentiel fixe oblige a choisir entre tremblement et retard."""

    def __init__(self, coupure_min=0.7, beta=0.025, coupure_derivee=1.0):
        self.cmin, self.beta, self.cd = coupure_min, beta, coupure_derivee
        self.x = None
        self.dx = 0.0
        self.t = None

    @staticmethod
    def _alpha(coupure, dt):
        tau = 1.0 / (2.0 * math.pi * coupure)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: float, t: float) -> float:
        if self.x is None:
            self.x, self.t = x, t
            return x
        dt = max(1e-3, t - self.t)
        self.t = t
        d = (x - self.x) / dt
        a_d = self._alpha(self.cd, dt)
        self.dx += a_d * (d - self.dx)
        coupure = self.cmin + self.beta * abs(self.dx)
        a = self._alpha(coupure, dt)
        self.x += a * (x - self.x)
        return self.x


def _nom_v4l2(n: int) -> str:
    try:
        with open(f"/sys/class/video4linux/video{n}/name", "r") as f:
            return f.read().strip()
    except OSError:
        return ""


# noeuds a ecarter : ce sont les codecs materiels Qualcomm. Ils s'ouvrent et
# rendent des images, mais ce ne sont pas des cameras.
ENCODEURS = ("venus", "vidc", "iris", "qcom", "codec", "decoder", "encoder")
CAMERAS = ("arducam", "uvc", "usb", "camera", "webcam", "imx", "ov")


def regler_exposition(main, cible=100.0, bavard=True):
    """Amene la luminosite moyenne sur la cible par correction proportionnelle.

    Hors saturation la luminosite est a peu pres proportionnelle au temps de
    pose : mesurer puis multiplier par cible/mesure converge en trois ou
    quatre essais, et resiste aux mesures aberrantes mieux qu'une dichotomie,
    qui part definitivement du mauvais cote des la premiere lecture fausse.

    auto_exposure=1 veut dire MANUEL en V4L2, pas automatique. On reste en
    manuel : une exposition qui varie en pleine demonstration ferait respirer
    la detection.
    """
    be = main.bornes("exposure_time_absolute")
    if be is None:
        if bavard:
            print("controle d'exposition introuvable - reglage laisse tel quel")
        return
    lo, hi = max(be[0], 1), be[1]
    bg = main.bornes("gain")
    gain = 0 if bg is None else bg[0]

    # la compensation de contre-jour fausse tout : on la coupe si elle existe
    main.ctrl(auto_exposure=1, backlight_compensation=0)
    if bg is not None:
        main.ctrl(gain=gain)

    def mesurer(e):
        main.ctrl(exposure_time_absolute=int(e))
        time.sleep(0.25)
        main.vider(6)
        return main.luminosite(4)

    e = min(max(be[1] // 20, lo), hi)
    lum = mesurer(e)
    for _ in range(6):
        if abs(lum - cible) <= 8:
            break
        facteur = cible / max(lum, 2.0)
        facteur = min(max(facteur, 0.2), 5.0)
        suivant = min(max(int(round(e * facteur)), lo), hi)
        if suivant == e:
            break
        e = suivant
        lum = mesurer(e)

    # encore trop sombre a fond de course : on monte le gain par paliers
    if bg is not None:
        while lum < cible - 20 and e >= hi and gain < bg[1]:
            gain = min(bg[1], gain + max(5, (bg[1] - bg[0]) // 5))
            main.ctrl(gain=gain)
            time.sleep(0.2)
            main.vider(6)
            lum = main.luminosite(4)

    if bavard:
        print("exposition %d (sur %d-%d), gain %d -> luminosite %.0f / 255"
              % (e, lo, hi, gain, lum))
        if lum > cible + 40:
            print("  encore surexpose au minimum : la scene est tres lumineuse,")
            print("  essaie --expo %d ou reduis l'eclairage direct" % lo)


def _expo_auto(n):
    """Remet la camera en exposition automatique, le temps du reperage.

    Sans ca, un reglage manuel laisse par une execution precedente peut rendre
    l'image blanche ou noire, et le test de contraste rejetterait la camera.
    """
    try:
        subprocess.run(["v4l2-ctl", "-d", "/dev/video%d" % n,
                        "--set-ctrl=auto_exposure=3,gain=0"],
                       check=False, capture_output=True)
    except FileNotFoundError:
        pass


def trouver_camera(largeur=640, hauteur=480, bavard=True):
    """Premier /dev/videoN qui est vraiment une camera et rend une vraie image.

    Le nom ecarte les codecs materiels. Le contraste ecarte les noeuds qui ne
    rendent qu'une image uniforme. Si rien ne passe le contraste, on retient
    quand meme le premier noeud qui a donne une image : mieux vaut une camera
    mal exposee que pas de camera du tout.
    """
    import cv2
    noeuds = sorted(int(c[len("/dev/video"):]) for c in glob.glob("/dev/video*")
                    if c[len("/dev/video"):].isdigit())
    if bavard and noeuds:
        print("peripheriques video :")
        for n in noeuds:
            print("   /dev/video%d  %s" % (n, _nom_v4l2(n)))

    def rang(n):
        nom = _nom_v4l2(n).lower()
        if any(k in nom for k in ENCODEURS):
            return 2
        if any(k in nom for k in CAMERAS):
            return 0
        return 1

    repli = None
    for n in sorted(noeuds, key=lambda n: (rang(n), n)):
        if rang(n) == 2:
            continue
        _expo_auto(n)
        cap = cv2.VideoCapture(n, cv2.CAP_V4L2)
        try:
            if not cap.isOpened():
                continue
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, largeur)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, hauteur)
            meilleur = 0.0
            for _ in range(8):
                ok, img = cap.read()
                if not ok or img is None or not img.size:
                    continue
                if repli is None:
                    repli = n
                meilleur = max(meilleur, float(img.std()))
                if meilleur >= 3.0:
                    if bavard:
                        print("camera retenue : /dev/video%d  %s  (%dx%d, contraste %.0f)"
                              % (n, _nom_v4l2(n), img.shape[1], img.shape[0], meilleur))
                    return n
        finally:
            cap.release()

    if repli is not None:
        if bavard:
            print("aucune image contrastee ; on retient /dev/video%d par defaut." % repli)
            print("  si l'image est blanche ou noire, l'exposition est a revoir :")
            print("  v4l2-ctl -d /dev/video%d --set-ctrl=auto_exposure=3,gain=0" % repli)
        return repli

    raise SystemExit(
        "aucune camera exploitable. Verifie la liste ci-dessus et force avec "
        "--device N si tu sais lequel est le bon.")


class Ancre:
    """Zone morte glissante : absorbe le tremblement, laisse passer le geste.

    La sortie ne bouge que si l'entree s'en ecarte de plus de `seuil`, et elle
    se place alors juste au bord de la zone morte. Un tremblement de main reste
    dedans et ne produit donc exactement aucun mouvement, au lieu d'etre
    simplement atenue comme le ferait un passe-bas. Le rattrapage comble le
    decalage residuel sans reintroduire de nervosite.
    """

    def __init__(self, seuil, rattrapage=0.0):
        self.seuil, self.rattrapage = seuil, rattrapage
        self.y = None

    def __call__(self, x):
        if self.y is None:
            self.y = x
            return x
        d = x - self.y
        if abs(d) > self.seuil:
            self.y += d - math.copysign(self.seuil, d)
        else:
            self.y += self.rattrapage * d
        return self.y


class Lisseur:
    """Ancre, puis un passe-bas leger pour arrondir ses marches."""

    def __init__(self, seuil, rattrapage, coupure, beta):
        self.ancre = Ancre(seuil, rattrapage)
        self.filtre = UnEuro(coupure, beta)

    def __call__(self, x, t):
        return self.filtre(self.ancre(x), t)


class Porte:
    """Rejette ce qu'une main ne peut physiquement pas faire.

    Un filtre passe-bas etale une aberration au lieu de l'eliminer : une seule
    image a 1757 mm contamine la sortie pendant une seconde. Comparer a une
    mediane glissante ne marche pas non plus : pendant un deplacement reel la
    mediane est en retard, et le vrai mouvement se fait rejeter. Le seul
    critere sain est la vitesse : au-dela de ce qu'une main peut parcourir
    entre deux images, c'est une erreur de mesure, pas un geste.
    """

    def __init__(self, v_max=1.5, suite_max=2):
        self.v_max, self.suite_max = v_max, suite_max
        self.reset()

    def reset(self):
        self.dernier = None
        self.t = None
        self.suite = 0
        self.rejets = 0

    def __call__(self, d, t):
        if self.dernier is not None and self.t is not None:
            dt = max(1e-3, t - self.t)
            if abs(d - self.dernier) > self.v_max * dt and self.suite < self.suite_max:
                self.suite += 1
                self.rejets += 1
                return self.dernier
        self.suite = 0
        self.dernier, self.t = d, t
        return d


def _pince_3d(monde, lm, w, h):
    """Ecartement pouce-index, rapporte a la taille de la main.

    Sans dimension, donc independant de la distance. Calcule sur le modele 3D
    quand il est disponible : le rapport y est invariant par rotation, alors
    que le meme rapport mesure dans l'image varie de -41 % a +48 % selon
    l'orientation, a ecartement pourtant constant. C'est ce qui faisait
    s'ouvrir et se fermer la pince toute seule.
    """
    # Le modele 3D rend ce rapport invariant par rotation, mais les bouts de
    # doigts sont ce que MediaPipe estime le plus mal en profondeur : a
    # l'essai, la pince est devenue plus instable qu'en 2D, pas moins. On
    # garde donc la mesure dans l'image, qui est bruitee mais franche.
    ecart = math.hypot((lm[4].x - lm[8].x) * w, (lm[4].y - lm[8].y) * h)
    taille = sum(math.hypot((lm[0].x - lm[k].x) * w, (lm[0].y - lm[k].y) * h)
                 for k in (5, 9, 13, 17)) / 4.0
    return ecart / max(taille, 1e-6)


def _travees(lm, w, h, marge=0.0):
    """Travee poignet -> bases des doigts, et largeur de paume, en pixels.

    Les deux sont perpendiculaires dans le plan de la main. Voir REGLAGES
    pour la raison : chacune resiste a l'une des deux rotations du poignet.
    """
    for i in (0, 5, 9, 13, 17):
        if not (-marge <= lm[i].x <= 1.0 + marge
                and -marge <= lm[i].y <= 1.0 + marge):
            return None, None      # repere extrapole hors image : non fiable
    a = sum(math.hypot((lm[k].x - lm[0].x) * w, (lm[k].y - lm[0].y) * h)
            for k in (5, 9, 13, 17)) / 4.0
    b = math.hypot((lm[17].x - lm[5].x) * w, (lm[17].y - lm[5].y) * h)
    if a < 8.0 or b < 5.0:
        return None, None
    return a, b


def _inclinaison(monde):
    """Cosinus de l'inclinaison de paume : affiche seulement, plus utilise."""
    import numpy as np
    if monde is None:
        return None
    p = np.array([[q.x, q.y, q.z] for q in monde], np.float64)
    if not np.isfinite(p).all():
        return None
    n = np.cross(p[5] - p[0], p[17] - p[0])
    nn = float(np.linalg.norm(n))
    return abs(float(n[2])) / nn if nn > 1e-9 else None


class Main:
    def __init__(self, device=-1, largeur=640, hauteur=480,
                 exposition=120, gain=28, flip=0):
        import cv2
        import mediapipe as mp
        self.cv2, self.mp = cv2, mp

        if device < 0:
            device = trouver_camera(largeur, hauteur)
        self.device = device
        # auto_exposure=1 veut dire MANUEL en V4L2 ; 3 veut dire automatique.
        self.ctrl(auto_exposure=1, exposure_time_absolute=exposition, gain=gain)

        self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, largeur)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, hauteur)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        if not self.cap.isOpened():
            raise SystemExit(f"impossible d'ouvrir /dev/video{device}")

        # confiance de suivi basse : on prefere un point un peu bruite a une
        # perte de la main, que le filtre rattrape de toute facon
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False, max_num_hands=1, model_complexity=0,
            min_detection_confidence=0.5, min_tracking_confidence=0.3)
        # 0 = vertical, 1 = horizontal, -1 = 180 deg, None = aucun
        self.flip = flip
        self.image = None
        self.reperes = None
        # si un ajustement de pose echoue, on reutilise la derniere distance
        # valable plutot que de jeter l'image
        self.derniere_distance = None
        self.porte = Porte(v_max=REGLAGES["saut_v_max"],
                           suite_max=REGLAGES["saut_suite_max"])
        self.recentes = []       # mediane glissante sur la distance
        self.hors_cadre = 0      # images ou la paume touchait le bord
        self.cos_t = None        # inclinaison de paume sur la derniere image
        self.ref_a = None        # travees de la pose de calibration
        self.ref_b = None
        self.travee_a = None
        self.travee_b = None
        self.lum = 0.0           # luminosite moyenne de la derniere image
        self.reglages_expo = 0   # nombre de rattrapages d'exposition

    def ctrl(self, **valeurs) -> None:
        """Ecrit des controles V4L2. Silencieux si l'outil ou le controle manque."""
        if not valeurs:
            return
        arg = ",".join("%s=%d" % (k, int(v)) for k, v in valeurs.items())
        try:
            subprocess.run(["v4l2-ctl", "-d", "/dev/video%d" % self.device,
                            "--set-ctrl=" + arg], check=False, capture_output=True)
        except FileNotFoundError:
            pass

    def bornes(self, nom):
        """(min, max) d'un controle V4L2, ou None s'il est introuvable."""
        try:
            r = subprocess.run(["v4l2-ctl", "-d", "/dev/video%d" % self.device,
                                "--list-ctrls"], capture_output=True, text=True)
        except FileNotFoundError:
            return None
        for ligne in r.stdout.splitlines():
            if nom not in ligne:
                continue
            mn = mx = None
            for jeton in ligne.split():
                if jeton.startswith("min="):
                    mn = int(jeton[4:])
                elif jeton.startswith("max="):
                    mx = int(jeton[4:])
            if mn is not None and mx is not None:
                return mn, mx
        return None

    def luminosite(self, n=4):
        """Luminosite moyenne de l'image, 0 a 255."""
        vals = []
        for _ in range(n):
            ok, img = self.cap.read()
            if ok and img is not None and img.size:
                vals.append(float(img.mean()))
        return sum(vals) / len(vals) if vals else 0.0

    def vider(self, n=6):
        """Jette n images. V4L2 garde quatre tampons en file : sans ca, on
        mesure la luminosite d'avant le changement de reglage."""
        for _ in range(n):
            self.cap.read()

    def lire(self) -> Optional[dict]:
        ok, image = self.cap.read()
        if not ok:
            return None
        if self.flip is not None:
            image = self.cv2.flip(image, self.flip)
        self.image = image
        res = self.hands.process(self.cv2.cvtColor(image, self.cv2.COLOR_BGR2RGB))
        if not res.multi_hand_landmarks:
            self.reperes = None
            return None
        lm = res.multi_hand_landmarks[0].landmark
        self.reperes = lm
        h, w = image.shape[:2]
        # quatre travees poignet -> bases des doigts plutot qu'une seule :
        # le bruit de detection se moyenne, et aucune ne s'effondre quand le
        # poing se ferme (contrairement a la largeur de paume 5-17)
        taille = sum(math.hypot((lm[0].x - lm[k].x) * w, (lm[0].y - lm[k].y) * h)
                     for k in (5, 9, 13, 17)) / 4.0
        ecart = math.hypot((lm[4].x - lm[8].x) * w, (lm[4].y - lm[8].y) * h)
        # Base du majeur (9) : le point le plus stable de la main. Le bout
        # de l'index, essaye avant, se deplace vers le pouce quand on pince -
        # fermer la pince faisait alors tourner la base.
        cx = lm[9].x
        cy = lm[9].y

        mw = getattr(res, "multi_hand_world_landmarks", None)
        pince = _pince_3d(mw[0].landmark if mw else None, lm, w, h)
        self.cos_t = _inclinaison(mw[0].landmark if mw else None)
        a, b = _travees(lm, w, h, REGLAGES["marge_cadre"])
        self.travee_a, self.travee_b = a, b
        dist = None
        if a is not None:
            if self.ref_a and self.ref_b:
                p = REGLAGES["travee_exposant"]
                # divise par 2^(1/p) pour que u vaille 1 a la pose de
                # calibration : les millimetres affiches deviennent honnetes
                u = (((a / self.ref_a) ** p + (b / self.ref_b) ** p) ** (1.0 / p)
                     / 2.0 ** (1.0 / p))
                dist = REGLAGES["ref_distance_m"] / max(u, 1e-6)
            else:
                dist = REGLAGES["aire_echelle"] / a
        if dist is None:
            self.hors_cadre += 1
        if dist is not None:
            n = max(1, int(REGLAGES["mediane_profondeur"]))
            self.recentes.append(dist)
            self.recentes = self.recentes[-n:]
            dist = sorted(self.recentes)[len(self.recentes) // 2]
            dist = self.porte(dist, time.time())
        if dist is None:
            dist = self.derniere_distance
        else:
            self.derniere_distance = dist
        if dist is None:
            return None
        return {"x": cx, "y": cy, "taille": taille, "pince": pince,
                "distance": dist}

    def annoter(self, chemin: str) -> None:
        if self.image is None:
            return
        img = self.image.copy()
        if self.reperes is not None:
            h, w = img.shape[:2]
            pts = [(int(p.x * w), int(p.y * h)) for p in self.reperes]
            for a, b in self.mp.solutions.hands.HAND_CONNECTIONS:
                self.cv2.line(img, pts[a], pts[b], (0, 255, 0), 2)
            for p in pts:
                self.cv2.circle(img, p, 3, (0, 0, 255), -1)
        self.cv2.imwrite(chemin, img)

    def fermer(self):
        self.cap.release()


# ------------------------------------------------------------ calibration

def _recolter(main: Main, duree: float, bavard: bool = False) -> dict:
    """Accumule les mesures pendant une duree, par grandeur.

    Avec bavard, affiche la distance vue en direct : sans ca, impossible de
    savoir pendant un balayage si la mesure suit vraiment la main.
    """
    v = {"taille": [], "pince": [], "y": [], "dist": [],
         "tr_a": [], "tr_b": []}
    # entre deux etapes la main fait un grand bond volontaire : sans remise a
    # zero, la porte le prendrait pour une aberration
    main.porte.reset()
    main.recentes = []
    fin = time.time() + duree
    while time.time() < fin:
        m = main.lire()
        if m:
            v["taille"].append(m["taille"])
            if main.travee_a:
                v["tr_a"].append(main.travee_a)
                v["tr_b"].append(main.travee_b)
            v["pince"].append(m["pince"])
            v["y"].append(m["y"])
            v["dist"].append(m["distance"])
            if bavard and len(v["dist"]) % 3 == 0:
                print("\r   %.0f mm   " % (1000 * m["distance"]),
                      end="", flush=True)
    if bavard:
        print("\r" + " " * 20)
    return v


def _mediane(v):
    w = sorted(v)
    return w[len(w) // 2]


def _quantile(v, q):
    w = sorted(v)
    return w[int(q * (len(w) - 1))]


def _correlation(xs, ys):
    """Coefficient de Pearson : dit si la relation vaut quelque chose."""
    n = len(xs)
    if n < 3:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sx = sum((x - mx) ** 2 for x in xs) ** 0.5
    sy = sum((y - my) ** 2 for y in ys) ** 0.5
    if sx < 1e-9 or sy < 1e-9:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def _pente(xs, ys):
    """Moindres carres : pente de ys en fonction de xs."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den > 1e-9 else 0.0


def calibrer(main: Main) -> dict:
    """Quatre temps, et du temps pour se placer.

    Le balayage vertical a disparu : la plage de pilotage est desormais la
    pleine hauteur d'image, il n'y a plus rien a y mesurer. Restent la pose de
    travail, qui fixe la reference et la pince ouverte, puis la pince fermee,
    puis les deux bouts de la course en profondeur.
    """
    attente = REGLAGES["calibration_placement_s"]
    duree = REGLAGES["calibration_mesure_s"]
    print("\ncalibration, quatre temps")
    print("%.0f s pour te placer, puis %.0f s de mesure a chaque fois.\n"
          % (attente, duree))

    def etape(n, consigne, detail):
        print(" %d/4  %s" % (n, consigne))
        print("      %s" % detail)
        reste = attente
        while reste > 0.05:
            main.lire()                 # on garde la detection vivante
            if abs(reste - round(reste)) < 0.03 and round(reste) > 0:
                print("   %d..." % round(reste), flush=True)
            time.sleep(0.1)
            reste -= 0.1
        print("   -- mesure --", flush=True)

    etape(1, "POSE DE TRAVAIL : main OUVERTE, paume vers la camera",
          "doigts bien ecartes, a la distance ou tu piloteras le plus souvent")
    v = _recolter(main, duree)
    if len(v["dist"]) < 5:
        raise SystemExit("main non detectee - verifie l'eclairage et le cadrage")
    ref = _mediane(v["taille"])
    pince_ouverte = _mediane(v["pince"])
    # C'est la seule pose ou la paume est franchement de face : elle fixe donc
    # les deux travees de reference, a partir desquelles la profondeur est
    # calculee ensuite.
    if len(v["tr_a"]) >= 5:
        main.ref_a = _mediane(v["tr_a"])
        main.ref_b = _mediane(v["tr_b"])
        print("      travees de reference : %.0f px (doigts) et %.0f px (paume)"
              % (main.ref_a, main.ref_b))

    etape(2, "PINCE FERMEE : pouce et index qui se TOUCHENT",
          "colle-les vraiment l'un contre l'autre, sans bouger la main")
    v = _recolter(main, duree)
    pince_fermee = _mediane(v["pince"]) if len(v["pince"]) >= 5 else 0.25

    mini = REGLAGES["course_mini_m"]
    etape(3, "LE PLUS PRES : avance la main vers la camera",
          "aussi pres que tu la piloteras, main ouverte")
    v = _recolter(main, duree, bavard=True)
    if len(v["dist"]) < 5:
        raise SystemExit("main perdue pendant la calibration de profondeur")
    d_proche = _quantile(v["dist"], 0.15)

    etape(4, "LE PLUS LOIN : recule la main",
          "aussi loin que tu la piloteras, main ouverte")
    v = _recolter(main, duree, bavard=True)
    if len(v["dist"]) < 5:
        raise SystemExit("main perdue pendant la calibration de profondeur")
    d_loin = _quantile(v["dist"], 0.85)

    if d_loin - d_proche < mini:
        print("\n  course de profondeur de %.0f mm seulement : trop peu pour"
              % (1000 * (d_loin - d_proche)))
        print("  piloter. On elargit a %.0f mm, la profondeur sera approximative."
              % (2000 * mini))
        milieu = (d_loin + d_proche) / 2.0
        d_proche, d_loin = milieu - mini, milieu + mini
    if pince_ouverte - pince_fermee < 0.15:
        print("  attention : course de pince faible (%.2f a %.2f)."
              % (pince_fermee, pince_ouverte))
        pince_fermee = max(0.05, pince_ouverte - 0.4)

    print("\n  taille de reference    : %.0f px" % ref)
    print("  pince                  : %.2f fermee a %.2f ouverte"
          % (pince_fermee, pince_ouverte))
    print("  profondeur pilotee     : %.0f mm (pres) a %.0f mm (loin)"
          % (1000 * d_proche, 1000 * d_loin))
    return {"ref": ref, "pince_ouverte": pince_ouverte,
            "pince_fermee": pince_fermee,
            "d_proche": d_proche, "d_loin": d_loin}


def table_hauteur(resoudre, r_min, r_max, pas=20, marge=1.0, bavard=True):
    """Hauteur atteignable en fonction du rayon, relevee par dichotomie.

    La course verticale depend fortement du rayon : pres de l'axe le bras est
    bride, loin il descend jusqu'a la table. Une plage figee gaspille
    l'essentiel de la course ou propose des cibles inatteignables.
    """
    t0 = time.time()
    table = []
    for R in range(int(r_min), int(r_max) + 1, pas):
        depart = None
        for z in (40.0, 100.0, 160.0, 20.0, 220.0, 260.0):
            if resoudre(R, z)[0] is not None:
                depart = z
                break
        if depart is None:
            continue
        bas, haut = depart, depart
        a, b = 0.0, depart          # plus bas
        for _ in range(9):
            m = (a + b) / 2.0
            if resoudre(R, m)[0] is not None:
                b = bas = m
            else:
                a = m
        a, b = depart, 320.0        # plus haut
        for _ in range(9):
            m = (a + b) / 2.0
            if resoudre(R, m)[0] is not None:
                a = haut = m
            else:
                b = m
        if haut - bas > 15.0:
            table.append((float(R), bas + marge, haut - marge))
    if bavard:
        if table:
            print("course verticale relevee en %.1f s : R %.0f a %.0f mm, "
                  "Z de %.0f a %.0f mm"
                  % (time.time() - t0, table[0][0], table[-1][0],
                     min(t[1] for t in table), max(t[2] for t in table)))
        else:
            print("aucune hauteur atteignable - verifie les butees")
    return table


def bornes_hauteur(table, rayon):
    """Interpole la course verticale disponible au rayon demande."""
    if not table:
        return 90.0, 200.0
    if rayon <= table[0][0]:
        return table[0][1], table[0][2]
    if rayon >= table[-1][0]:
        return table[-1][1], table[-1][2]
    for i in range(len(table) - 1):
        r0, b0, h0 = table[i]
        r1, b1, h1 = table[i + 1]
        if r0 <= rayon <= r1:
            f = (rayon - r0) / (r1 - r0)
            return b0 + (b1 - b0) * f, h0 + (h1 - h0) * f
    return table[-1][1], table[-1][2]


def commandes(m: dict, cal: dict, r: dict, table=None):
    base = r["base_centre"] + r["base_sens"] * _interp(
        m["x"], r["x_min"], r["x_max"], -r["base_amplitude"], r["base_amplitude"])

    # distance en metres, issue de l'ajustement de pose : main pres de la
    # camera -> bras tendu. Aucune correction d'inclinaison a appliquer.
    dist = m["distance"]
    if r["r_sens"] > 0:
        rayon = _interp(dist, cal["d_loin"], cal["d_proche"],
                        r["r_min"], r["r_max"])
    else:
        rayon = _interp(dist, cal["d_loin"], cal["d_proche"],
                        r["r_max"], r["r_min"])

    # la course verticale suit le rayon : toute la hauteur de l'image sert
    bas, haut = bornes_hauteur(table, rayon)
    if r["z_sens"] > 0:
        z = _interp(m["y"], r["y_min"], r["y_max"], haut, bas)
    else:
        z = _interp(m["y"], r["y_min"], r["y_max"], bas, haut)

    ouverture = _interp(m["pince"], cal["pince_fermee"],
                        cal["pince_ouverte"], 0.0, 40.0)
    return base, rayon, z, ouverture, 1000.0 * dist


# ----------------------------------------------------------------- moteur

class Resolveur:
    """Cinematique inverse avec continuite : parmi les solutions possibles,
    on garde celle qui ressemble le plus a la precedente. Sans ca, le bras
    saute d'une configuration a l'autre d'une image sur l'autre."""

    def __init__(self, bras, filtre, tolerance: float, pas: float = 5.0):
        self.b = bras
        self.filtre = filtre
        self.tol = tolerance
        self.pas = pas
        self.precedent: Optional[Sequence[float]] = None
        self.ecart_precedent = 0.0

    def _offsets(self):
        o = [0.0]
        k = self.pas
        while k <= self.tol + 1e-9:
            o.extend((-k, k))
            k += self.pas
        # on prefere la verticale, avec un biais pour l'inclinaison retenue la
        # fois precedente : sans ce biais la pince oscille entre deux solutions
        o.sort(key=lambda d: abs(d) + 0.9 * abs(d - self.ecart_precedent))
        return o

    def __call__(self, x, y, z):
        from bras_ik import PINCE_BAS
        for d in self._offsets():
            sols = self.b.ik_toutes(x, y, z, orientation=PINCE_BAS + d,
                                    filtre=self.filtre)
            if not sols:
                continue
            if self.precedent is not None:
                sols.sort(key=lambda s: max(abs(a - b) for a, b in
                                            zip(s.servos, self.precedent)))
            self.precedent = sols[0].servos
            self.ecart_precedent = d
            return sols[0], d
        return None, 0.0


def _vue_annotee(main):
    """Image camera avec le squelette, encodee en JPEG."""
    img = main.image
    if img is None:
        return None
    img = img.copy()
    if main.reperes is not None:
        h, w = img.shape[:2]
        pts = [(int(p.x * w), int(p.y * h)) for p in main.reperes]
        for a, b in main.mp.solutions.hands.HAND_CONNECTIONS:
            main.cv2.line(img, pts[a], pts[b], (0, 220, 0), 2)
        for i, p in enumerate(pts):
            main.cv2.circle(img, p, 3, (0, 0, 230), -1)
        main.cv2.circle(img, pts[9], 7, (255, 255, 255), 2)   # le point pilote
    ok, buf = main.cv2.imencode(".jpg", img,
                                [main.cv2.IMWRITE_JPEG_QUALITY, 65])
    return buf.tobytes() if ok else None


def _limiter(cible: Sequence[float], courant: Optional[Sequence[float]],
             vmax: float, zone_morte: float = 0.0):
    """Limite la vitesse, et ne bouge pas du tout sous la zone morte.

    Sans elle, chaque axe corrige en permanence un dixieme de degre : un servo
    y repond par un fremissement audible et visible, pour un deplacement que
    personne ne voit.
    """
    if courant is None:
        return list(cible)
    sortie = []
    for t, c in zip(cible, courant):
        d = t - c
        sortie.append(c if abs(d) < zone_morte else c + _borne(d, -vmax, vmax))
    return sortie


ORIENTATIONS = [("aucune", None), ("verticale", 0),
                ("horizontale", 1), ("180 degres", -1)]


def diagnostic(main: "Main", port: int) -> int:
    """Teste les quatre orientations, puis sert la vue annotee au navigateur.

    Sans voir l'image, regler l'eclairage, le cadrage ou l'orientation revient
    a tatonner. Cette page montre exactement ce que MediaPipe recoit.
    """
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    regler_exposition(main)

    voulu = main.flip   # la boucle va l'ecraser : on le garde
    print("\ntest des quatre orientations, 20 images chacune")
    scores = []
    for nom, f in ORIENTATIONS:
        main.flip = f
        vus = 0
        for _ in range(20):
            if main.lire():
                vus += 1
        scores.append((vus, nom, f))
        print(f"   {nom:12s} : main vue {vus:2d}/20")
    scores.sort(reverse=True)
    garde = [x for x in scores if x[2] == voulu]
    meilleur = garde[0] if garde else scores[0]
    main.flip = meilleur[2]
    print(f"\norientation utilisee : {meilleur[1]} ({meilleur[0]}/20)")
    print("   fixee par REGLAGES['flip'] ; les quatre se valent souvent pour la")
    print("   detection, mais elles changent le sens des commandes. --flip pour essayer.")
    if meilleur[0] < 8:
        print("   detection faible dans TOUTES les orientations : regarde l'image,")
        print("   le probleme est l'eclairage, la nettete ou le cadrage.")

    etat = {"jpeg": None, "mesures": {}, "verrou": threading.Lock()}

    PAGE = b"""<!doctype html><meta charset=utf-8>
<title>vue camera</title>
<style>body{margin:0;background:#14181a;color:#e7ecee;
font:14px ui-monospace,Menlo,monospace;display:flex;flex-direction:column;
align-items:center;gap:12px;padding:16px}
img{max-width:100%;border:1px solid #333c41;border-radius:4px}
table{border-collapse:collapse}td{padding:3px 14px;border-bottom:1px solid #262c30}
td:last-child{text-align:right;color:#4fcfc6}</style>
<h2 style="margin:0;font-weight:600">ce que voit MediaPipe</h2>
<img id=v><table id=m></table>
<script>
function t(){document.getElementById('v').src='image.jpg?'+Date.now();
fetch('mesures').then(r=>r.json()).then(d=>{
document.getElementById('m').innerHTML=Object.entries(d)
.map(([k,v])=>'<tr><td>'+k+'</td><td>'+v+'</td></tr>').join('')});}
setInterval(t,200);t();
</script>"""

    class H(BaseHTTPRequestHandler):
        def _envoyer(self, corps, mime):
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(corps)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(corps)

        def do_GET(self):
            chemin = self.path.split("?")[0]
            if chemin in ("/", "/index.html"):
                self._envoyer(PAGE, "text/html; charset=utf-8")
            elif chemin == "/image.jpg":
                with etat["verrou"]:
                    j = etat["jpeg"]
                self._envoyer(j or b"", "image/jpeg")
            elif chemin == "/mesures":
                with etat["verrou"]:
                    d = dict(etat["mesures"])
                self._envoyer(json.dumps(d).encode(), "application/json")
            else:
                self.send_error(404)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("0.0.0.0", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    import socket
    sk = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sk.connect(("10.255.255.255", 1)); ip = sk.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        sk.close()
    print(f"\nvue camera : http://{ip}:{port}/   (Ctrl+C pour arreter)\n")

    vus = tot = 0
    tmin = tmax = None
    try:
        while True:
            m = main.lire()
            tot += 1
            if m:
                vus += 1
                tmin = m["taille"] if tmin is None else min(tmin, m["taille"])
                tmax = m["taille"] if tmax is None else max(tmax, m["taille"])
            if main.image is not None:
                img = main.image.copy()
                if main.reperes is not None:
                    h, w = img.shape[:2]
                    pts = [(int(p.x * w), int(p.y * h)) for p in main.reperes]
                    for a, b in main.mp.solutions.hands.HAND_CONNECTIONS:
                        main.cv2.line(img, pts[a], pts[b], (0, 255, 0), 2)
                    for p in pts:
                        main.cv2.circle(img, p, 4, (0, 0, 255), -1)
                ok, buf = main.cv2.imencode(".jpg", img,
                                            [main.cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok:
                    with etat["verrou"]:
                        etat["jpeg"] = buf.tobytes()
            d = {"orientation": meilleur[1],
                 "main detectee": f"{100*vus/max(tot,1):.0f} %"}
            if main.image is not None:
                d["luminosite"] = f"{main.image.mean():.0f} / 255"
            if m:
                d["x"] = f"{m['x']:.3f}"
                d["y"] = f"{m['y']:.3f}"
                d["taille"] = f"{m['taille']:.1f} px"
                d["ecart / taille"] = f"{m['ecart']/max(m['taille'],1e-6):.2f}"
            if tmin is not None:
                d["taille vue"] = f"{tmin:.0f} a {tmax:.0f} px"
                d["amplitude"] = f"x{tmax/max(tmin,1e-6):.2f}"
            with etat["verrou"]:
                etat["mesures"] = d
    except KeyboardInterrupt:
        print("\narret")
        srv.shutdown()
    finally:
        main.fermer()
    return 0


def principal(args) -> int:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    r = dict(REGLAGES)
    main = Main(device=args.device, flip=REGLAGES["flip"])

    if args.mesure:
        return diagnostic(main, args.port)

    from bras_ik import Bras
    try:
        from bras_collision import Collisions
        filtre = Collisions().verifier
    except Exception as e:
        print(f"bras_collision indisponible ({e}) : trajectoires non verifiees")
        filtre = None

    b = Bras()
    resoudre = Resolveur(b, filtre, r["tolerance_pince"])

    visu = None
    if not args.sans_simu:
        from bras_viewer import Visualiseur
        visu = Visualiseur(port=args.port)
        visu.demarrer()

    moteurs = None
    if args.servos:
        from bras_servo import BrasServo
        moteurs = BrasServo()
        print("servos :", moteurs.ping())
    else:
        print("servos non pilotes (ajouter --servos)")

    regler_exposition(main)
    cal = calibrer(main)

    print()
    table = table_hauteur(lambda R, z: resoudre(b.base[0] - R, b.base[1], z),
                          r["r_min"], r["r_max"])

    def lisseur(seuil):
        return Lisseur(seuil, r["ancre_rattrapage"],
                       r["ancre_coupure"], r["ancre_beta"])

    f_base = lisseur(r["ancre_base_deg"])
    f_r = lisseur(r["ancre_r_mm"])
    f_z = lisseur(r["ancre_z_mm"])
    # la pince, elle, va bien avec un passe-bas : son signal est franc
    f_p = UnEuro(1.2, 0.05)

    envoyes = None
    recents = []            # moyenne glissante sur les angles envoyes
    derniere_main = 0.0
    dernier_controle = 0.0
    dernier_expo = 0.0
    n = n_main = n_ik = 0
    distances = []
    t0 = time.time()
    prochain_apercu = 0.0
    prochaine_vue = 0.0

    print("\nen route. Ctrl+C pour arreter.\n")
    try:
        while True:
            n += 1
            m = main.lire()
            t = time.time()

            if m is None:
                # Pas de main dans l'image : c'est le seul moment ou la
                # luminosite mesure vraiment la piece et non la paume.
                if (main.image is not None
                        and t - dernier_controle > r["periode_luminosite_s"]):
                    dernier_controle = t
                    main.lum = float(main.image.mean())
                    if (abs(main.lum - r["cible_luminosite"]) > r["derive_luminosite"]
                            and t - dernier_expo > r["expo_periode_mini_s"]):
                        dernier_expo = t
                        print("  luminosite %.0f sans main : on refait le reglage"
                              % main.lum)
                        regler_exposition(main, r["cible_luminosite"], bavard=True)
                        main.reglages_expo += 1
                if t - derniere_main > r["tolerance_perte_s"] and visu and envoyes:
                    visu.publier(envoyes, source="main perdue")
                continue

            n_main += 1
            derniere_main = t


            cb, cr, cz, co, dmm = commandes(m, cal, r, table)
            distances.append(dmm)
            cb, cr, cz, co = f_base(cb, t), f_r(cr, t), f_z(cz, t), f_p(co, t)

            # le bras travaille du cote oppose a la pose repliee
            a = math.radians(cb - 90.0)
            x = b.base[0] - cr * math.cos(a)
            y = b.base[1] - cr * math.sin(a)

            sol, ecart = resoudre(x, y, cz)
            if sol is None:
                continue
            n_ik += 1
            envoyes = _limiter(sol.servos, envoyes, r["vitesse_max_deg"],
                               r["zone_morte_deg"])
            recents.append(list(envoyes))
            recents = recents[-max(1, int(r["moyenne_servo"])):]
            sortie = [sum(p[i] for p in recents) / len(recents)
                      for i in range(len(envoyes))]

            if visu and t > prochaine_vue:
                prochaine_vue = t + r["periode_vue_s"]
                visu.publier_image(_vue_annotee(main),
                                   {"x": round(m["x"], 3), "y": round(m["y"], 3),
                                    "distance_mm": round(1000 * m["distance"]),
                                    "R_mm": round(cr), "Z_mm": round(cz),
                                    "base_deg": round(cb, 1),
                                    "pince_mm": round(co),
                                    "cos": round(main.cos_t or 0, 2),
                                    "images_s": round(n / max(t - t0, 1e-3), 1),
                                    "main_pct": round(100 * n_main / max(n, 1))})
            if visu:
                visu.publier(sortie, ouverture=co,
                             source=(f"main  R {cr:.0f}  Z {cz:.0f}  "
                                     f"pince {co:.0f}" +
                                     (f"  inclinee {ecart:+.0f}" if ecart else "")))
            if moteurs:
                moteurs.envoyer(sortie)
                moteurs.pince_mm(co)

            if args.apercu and t > prochain_apercu:
                main.annoter(os.path.expanduser("~/bras_main.jpg"))
                prochain_apercu = t + 1.0

            if n % 30 == 0:
                dt = t - t0
                rec = distances[-300:]
                print(f"  {n/dt:4.1f} img/s  main {100*n_main/n:3.0f}%  "
                      f"IK {100*n_ik/max(n_main,1):3.0f}%  |  "
                      f"R {cr:5.0f}  Z {cz:5.0f}  pince {co:4.0f}  "
                      f"|  main a {min(rec):.0f}-{max(rec):.0f} mm "
                      f"(calibre {1000*cal['d_proche']:.0f}-"
                      f"{1000*cal['d_loin']:.0f})"
                      f"  sauts {main.porte.rejets}"
                      f"  bord {main.hors_cadre}"
                      f"  lum {main.lum:.0f}"
                      f"  cos {main.cos_t or 0:.2f}"
                      f"  expo {main.reglages_expo}"
                      )
    except KeyboardInterrupt:
        print("\narret")
    finally:
        main.fermer()
        if moteurs:
            moteurs.fermer()
        if visu:
            visu.arreter()
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--servos", action="store_true", help="piloter reellement le bras")
    p.add_argument("--sans-simu", action="store_true", help="ne pas servir le simulateur")
    p.add_argument("--apercu", action="store_true",
                   help="ecrire ~/bras_main.jpg annote chaque seconde")
    p.add_argument("--mesure", action="store_true",
                   help="afficher les mesures brutes de la camera, sans rien piloter")
    p.add_argument("--device", type=int, default=-1,
                   help="numero /dev/videoN ; -1 pour chercher automatiquement")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--flip", choices=("aucune", "verticale", "horizontale", "180"),
                   default=None, help="orientation de l'image")
    a = p.parse_args()
    if a.flip is not None:
        REGLAGES["flip"] = {"aucune": None, "verticale": 0,
                            "horizontale": 1, "180": -1}[a.flip]
    raise SystemExit(principal(a))
