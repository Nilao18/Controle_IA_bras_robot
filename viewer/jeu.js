// Parcours d'essai : prendre un jeton, le poser dans la zone, recommencer.
(function () {
  "use strict";
  var S = window.SIMU;
  if (!S) { console.warn("jeu.js : window.SIMU absent"); return; }
  var THREE = S.THREE, world = S.world, fk = S.fk, qg = S.qg,
      BASE = S.BASE, D = S.D;

  var R_MIN = 100, R_MAX = 300;   // rayon de tirage, en mm
  var ANG = 50;                   // demi-ouverture angulaire, en degres
  var OBJ_R = 14;                 // rayon du jeton
  var OBJ_Z = 25;                 // hauteur de son centre au-dessus de la table
  var ZONE_R = 50;                // rayon de la zone de depose
  var PRISE = 32;                 // distance TCP-jeton acceptee pour saisir
  var FERME = 12, OUVERT = 26;    // seuils d'ouverture de pince, en mm
  var ECART_MINI = 150;           // distance minimale entre jeton et zone

  var etat = "attente", score = 0, essais = 0, depart = 0, chrono = [];

  function tirage() {
    var a = (Math.random() * 2 - 1) * ANG * D;
    var R = R_MIN + Math.random() * (R_MAX - R_MIN);
    return { x: BASE.x - R * Math.cos(a), y: BASE.y - R * Math.sin(a), R: R };
  }

  function tirageEloigne(de) {
    for (var i = 0; i < 60; i++) {
      var p = tirage();
      if (Math.hypot(p.x - de.x, p.y - de.y) >= ECART_MINI) return p;
    }
    return tirage();
  }

  var mJeton = new THREE.Mesh(
    new THREE.SphereGeometry(OBJ_R, 24, 18),
    new THREE.MeshStandardMaterial({ color: 0xe8743b, roughness: 0.45 }));
  var mZone = new THREE.Mesh(
    new THREE.CircleGeometry(ZONE_R, 56),
    new THREE.MeshBasicMaterial({ color: 0x3ba3a0, transparent: true,
                                  opacity: 0.22, depthWrite: false }));
  var mBord = new THREE.Mesh(
    new THREE.RingGeometry(ZONE_R - 2.5, ZONE_R, 56),
    new THREE.MeshBasicMaterial({ color: 0x3ba3a0 }));
  world.add(mJeton); world.add(mZone); world.add(mBord);

  var hud = document.createElement("div");
  hud.id = "jeu";
  hud.style.cssText = "position:fixed;left:14px;top:14px;z-index:60;" +
    "font:600 13px/1.5 ui-monospace,monospace;padding:9px 12px;border-radius:9px;" +
    "background:rgba(14,16,22,.84);color:#f2f4f8;pointer-events:none;" +
    "box-shadow:0 1px 12px rgba(0,0,0,.35)";
  document.body.appendChild(hud);

  var jeton = null, zone = null, tenu = false;

  function nouveau() {
    jeton = tirage();
    zone = tirageEloigne(jeton);
    tenu = false;
    etat = "a_prendre";
    depart = Date.now();
    placer();
  }

  function placer() {
    mJeton.position.set(jeton.x, jeton.y, jeton.z === undefined ? OBJ_Z : jeton.z);
    mZone.position.set(zone.x, zone.y, 0.6);
    mBord.position.set(zone.x, zone.y, 0.8);
  }

  function hud_() {
    var moy = chrono.length
      ? (chrono.reduce(function (a, b) { return a + b; }, 0) / chrono.length / 1000).toFixed(1)
      : "–";
    var l = ["réussis " + score + " / " + essais + "   moyenne " + moy + " s"];
    if (etat === "a_prendre") {
      l.push("pince OUVERTE sur le jeton orange, puis referme");
    } else if (etat === "tenu") {
      l.push("amène-le au-dessus du disque, puis OUVRE");
    }
    l.push("R " + Math.round(jeton.R) + " mm   " +
           ((Date.now() - depart) / 1000).toFixed(1) + " s");
    hud.innerHTML = l.join("<br>");
  }

  function tour() {
    var p = fk(qg(), S.graspZ);
    var dj = Math.hypot(p.x - mJeton.position.x, p.y - mJeton.position.y,
                        p.z - mJeton.position.z);

    if (etat === "a_prendre") {
      if (dj < PRISE && S.ouverture < FERME) { tenu = true; etat = "tenu"; }
    } else if (etat === "tenu") {
      mJeton.position.set(p.x, p.y, Math.max(OBJ_R, p.z));
      if (S.ouverture > OUVERT) {
        essais++;
        var dz = Math.hypot(p.x - zone.x, p.y - zone.y);
        if (dz < ZONE_R) {
          score++;
          chrono.push(Date.now() - depart);
          nouveau();
        } else {                       // lache a cote : le jeton tombe la
          jeton = { x: p.x, y: p.y, R: Math.hypot(p.x - BASE.x, p.y - BASE.y) };
          tenu = false; etat = "a_prendre"; placer();
        }
      }
    }
    mJeton.material.color.setHex(etat === "tenu" ? 0x4ec07a : 0xe8743b);
    hud_();
  }

  nouveau();
  S.surPose(tour);
})();
