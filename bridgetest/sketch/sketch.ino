/*
 * Bras 5 axes + pince - interface servo du STM32
 *
 * Le Linux de la carte calcule les angles, le STM32 les applique. Le MCU
 * garde la main sur trois choses que le Linux ne peut pas franchir :
 *   - les butees materielles, en dur ci-dessous ;
 *   - la vitesse maximale, par rampe ;
 *   - la coupure si le Linux cesse de parler (watchdog).
 *
 * Methodes exposees sur le Bridge :
 *   set_angles(base, epaule, coude, inclinaison, rotation)  -> bool
 *   set_pince(ouverture)                                    -> bool
 *   get_angle(index)                                        -> float
 *   relacher()                                              -> bool
 *   ping()                                                  -> String
 *
 * Les angles sont des angles servo en degres, 0 a 180. L'ordre est celui de
 * bras_params.json. La pince est aussi un angle servo : 0 = fermee.
 */

#include <Arduino_RouterBridge.h>
#include <Servo.h>

#define NB_AXES 6          // 5 axes + pince
#define IDX_PINCE 5

// broches PWM de l'UNO Q : D3 D5 D6 D9 D10 D11
static const int BROCHE[NB_AXES] = { 3, 5, 6, 9, 10, 11 };

// butees materielles. Le Linux ne peut pas les depasser, meme sur bogue.
static const float BUTEE_MIN[NB_AXES] = { 0, 0, 0, 0, 0, 0 };
static const float BUTEE_MAX[NB_AXES] = { 180, 180, 180, 180, 180, 180 };

// calibration de la largeur d'impulsion, par servo
static const int US_MIN[NB_AXES] = { 500, 500, 500, 500, 500, 500 };
static const int US_MAX[NB_AXES] = { 2500, 2500, 2500, 2500, 2500, 2500 };

// pose de repos = pose pliee du simulateur
static const float REPOS[NB_AXES] = { 90, 0, 0, 0, 90, 0 };

static const unsigned long PERIODE_MS = 20;    // 50 Hz
static const float T = 0.020f;                 // PERIODE_MS en secondes
static const unsigned long WATCHDOG_MS = 1500; // silence toléré avant relâche

/* Vitesses et accelerations, par axe et en unites physiques.
 *
 * Avant : un seul PAS_MAX de 1,5 deg par periode, soit 75 deg/s pour tout le
 * monde, et aucune limite d'acceleration. Deux defauts.
 *
 * Un seul plafond condamnait tous les axes a la vitesse du plus charge. A
 * 6 V - la seule tension que les trois modeles admettent ensemble - les
 * servos donnent a vide : MG90S 600 deg/s, MS62 417, XD065 286. Les valeurs
 * ci-dessous sont a 40-45 % de ces vitesses, ce qui laisse la marge que la
 * charge prendra. A verifier bras cable : ce sont des fiches techniques, pas
 * des mesures.
 *
 * Surtout, sans limite d'acceleration le servo passait de l'arret a sa
 * vitesse de pointe en une periode de 20 ms. C'est l'a-coup, et non la
 * vitesse, qui fatigue un bras imprime : jeu dans les engrenages,
 * depassement en fin de course, oscillations que le servo corrige en
 * forcant. Avec une rampe, monter la vitesse de pointe coute peu ; sans
 * elle, la doubler double les a-coups.
 *
 * ordre : base, epaule, coude, inclinaison, rotation, pince
 */
static const float V_MAX[NB_AXES]  = { 180, 120, 180, 250, 250, 300 };  // deg/s
static const float A_MAX[NB_AXES]  = { 1200, 800, 1200, 1700, 1700, 2000 }; // deg/s2

Servo servo[NB_AXES];
static volatile float cible[NB_AXES];
static float courant[NB_AXES];
static float vitesse[NB_AXES];      // deg par periode, etat de la rampe
static bool attache = false;
static unsigned long dernier_ordre = 0;
static unsigned long dernier_cycle = 0;

static float borne(int i, float v) {
  if (v < BUTEE_MIN[i]) return BUTEE_MIN[i];
  if (v > BUTEE_MAX[i]) return BUTEE_MAX[i];
  return v;
}

static void attacher() {
  if (attache) return;
  for (int i = 0; i < NB_AXES; i++) {
    servo[i].attach(BROCHE[i], US_MIN[i], US_MAX[i]);
    vitesse[i] = 0.0f;              // on repart a l'arret, pas sur une
                                    // vitesse datant d'avant le relachement
  }
  attache = true;
}

static void detacher() {
  if (!attache) return;
  for (int i = 0; i < NB_AXES; i++) servo[i].detach();
  attache = false;
}

// --- methodes exposees au Linux -------------------------------------------

bool set_angles(float base, float epaule, float coude,
                float inclinaison, float rotation) {
  const float v[5] = { base, epaule, coude, inclinaison, rotation };
  for (int i = 0; i < 5; i++) cible[i] = borne(i, v[i]);
  dernier_ordre = millis();
  attacher();
  return true;
}

bool set_pince(float ouverture) {
  cible[IDX_PINCE] = borne(IDX_PINCE, ouverture);
  dernier_ordre = millis();
  attacher();
  return true;
}

float get_angle(int index) {
  if (index < 0 || index >= NB_AXES) return -1.0f;
  return courant[index];
}

bool relacher() {
  detacher();
  return true;
}

String ping() {
  return String("bras 5 axes + pince, ") + String(NB_AXES) + String(" servos, pret");
}

// --------------------------------------------------------------------------

void setup() {
  Bridge.begin();
  Monitor.begin();

  for (int i = 0; i < NB_AXES; i++) {
    cible[i] = REPOS[i];
    courant[i] = REPOS[i];
    vitesse[i] = 0.0f;
  }

  Bridge.provide("set_angles", set_angles);
  Bridge.provide("set_pince", set_pince);
  Bridge.provide("get_angle", get_angle);
  Bridge.provide("relacher", relacher);
  Bridge.provide("ping", ping);

  Monitor.println("sketch bras pret - servos detaches tant qu'aucun ordre n'arrive");
}

void loop() {
  const unsigned long t = millis();
  if (t - dernier_cycle < PERIODE_MS) return;
  dernier_cycle = t;

  // watchdog : plus d'ordre depuis un moment, on relache pour ne pas forcer
  if (attache && dernier_ordre != 0 && (t - dernier_ordre) > WATCHDOG_MS) {
    detacher();
    Monitor.println("watchdog : plus d'ordre recu, servos relaches");
    return;
  }
  if (!attache) return;

  /* Rampe trapezoidale, par axe.
   *
   * On ne borne pas seulement la vitesse : on la borne aussi a ce qui permet
   * encore de s'arreter sur la cible, v <= sqrt(2.a.distance). Sans ce
   * terme, l'axe arriverait a pleine vitesse et devrait s'arreter net - le
   * depassement et l'a-coup qu'on cherche justement a supprimer.
   */
  for (int i = 0; i < NB_AXES; i++) {
    const float vmax = V_MAX[i] * T;            // deg par periode
    const float amax = A_MAX[i] * T * T;        // deg par periode au carre
    const float e = cible[i] - courant[i];

    float vise = e;                             // tout rattraper, si possible
    if (vise >  vmax) vise =  vmax;
    if (vise < -vmax) vise = -vmax;
    if (amax > 0.0f) {
      /* Vitesse au-dela de laquelle on ne pourrait plus s'arreter sur la
       * cible. La forme continue sqrt(2.a.d) depasse en pas discret - 1,2
       * degre mesure sur un mouvement d'epaule de 90 degres - parce qu'on
       * parcourt encore une periode entiere a la vitesse courante. Les
       * termes en amax/2 corrigent ce demi-pas. */
      const float frein = sqrtf(2.0f * amax * fabsf(e)
                                + 0.25f * amax * amax) - 0.5f * amax;
      if (vise >  frein) vise =  frein;
      if (vise < -frein) vise = -frein;
      float dv = vise - vitesse[i];
      if (dv >  amax) dv =  amax;
      if (dv < -amax) dv = -amax;
      vitesse[i] += dv;
      if (vitesse[i] * e > 0.0f && fabsf(vitesse[i]) > fabsf(e))
        vitesse[i] = e;              // garde-fou : ne jamais franchir la cible
    } else {
      vitesse[i] = vise;
    }
    courant[i] += vitesse[i];

    const float f = (courant[i] - 0.0f) / 180.0f;
    const int us = US_MIN[i] + (int)(f * (US_MAX[i] - US_MIN[i]) + 0.5f);
    servo[i].writeMicroseconds(us);
  }
}
