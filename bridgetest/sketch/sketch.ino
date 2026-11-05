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
static const float PAS_MAX = 1.5f;             // deg par periode -> 75 deg/s
static const unsigned long WATCHDOG_MS = 1500; // silence toléré avant relâche

Servo servo[NB_AXES];
static volatile float cible[NB_AXES];
static float courant[NB_AXES];
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

  // rampe : on approche la cible d'au plus PAS_MAX par periode
  for (int i = 0; i < NB_AXES; i++) {
    const float c = cible[i];
    float d = c - courant[i];
    if (d > PAS_MAX) d = PAS_MAX;
    else if (d < -PAS_MAX) d = -PAS_MAX;
    courant[i] += d;

    const float f = (courant[i] - 0.0f) / 180.0f;
    const int us = US_MIN[i] + (int)(f * (US_MAX[i] - US_MIN[i]) + 0.5f);
    servo[i].writeMicroseconds(us);
  }
}
