import express   from "express";
import cors      from "cors";
import dotenv    from "dotenv";
import OpenAI    from "openai";
import helmet    from "helmet";
import rateLimit from "express-rate-limit";

dotenv.config();

const app    = express();
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

/* ─────────────────────────────────────────────────────────────
   Utilities
───────────────────────────────────────────────────────────── */

function log(endpoint, fields = {}) {
  const ts    = new Date().toISOString().replace("T", " ").slice(0, 19);
  const parts = [`[${ts}]  ${endpoint}`];
  for (const [k, v] of Object.entries(fields)) parts.push(`  ${k}: ${v}`);
  console.log(parts.join("\n"));
}

function makeReqId() {
  return Math.random().toString(36).slice(2, 8);
}

/* ─────────────────────────────────────────────────────────────
   Uncertainty detection — prevents decision-type questions
   when the user expresses not knowing / being unsure
───────────────────────────────────────────────────────────── */
function isUncertain(message) {
  return /weiß nicht|weiss nicht|keine ahnung|unsicher|ich glaube nicht|bin mir nicht sicher|keine vorstellung|weiß ich nicht/i.test(message);
}

function detectUserState(message) {
  const m = message.toLowerCase();
  if (isUncertain(m)) return "UNCERTAIN";
  if (/eigentlich|ich weiß schon|ich müsste|aber/i.test(m)) {
    return "CLEAR";
  }
  if (/halt|irgendwie|einfach|keine zeit|schwierig/i.test(m)) {
    return "AVOIDING";
  }
  if (/weil|deshalb|macht sinn|logisch|klar ist doch/i.test(m)) {
    return "DEFENSIVE";
  }
  return "CLEAR";
}

function isAvoiding(message) {
  return /halt|irgendwie|einfach|keine zeit|schwierig/i.test(message);
}

function isDefensive(message) {
  return /weil|deshalb|macht sinn|logisch|klar ist doch/i.test(message);
}

const ERR = {
  INVALID_BODY:   "INVALID_REQUEST_BODY",
  EMPTY_MESSAGES: "MESSAGES_ARRAY_EMPTY",
  OPENAI_TIMEOUT: "OPENAI_TIMEOUT",
  OPENAI_FAILED:  "AI_RESPONSE_FAILED",
};

function validateMessages(body) {
  if (!body || !Array.isArray(body.messages))
    return { code: ERR.INVALID_BODY,   detail: "body.messages must be an array" };
  if (body.messages.length === 0)
    return { code: ERR.EMPTY_MESSAGES, detail: "messages array must not be empty" };
  return null;
}

async function callOpenAI(messages, options = {}) {
  const { retries = 2, jsonMode = false } = options;

  for (let attempt = 0; attempt <= retries; attempt++) {
    const controller = new AbortController();
    const timer      = setTimeout(() => controller.abort(), 60_000);

    try {
      const completion = await openai.chat.completions.create({
        model:    "gpt-4o-mini",
        messages,
        ...(jsonMode ? { response_format: { type: "json_object" } } : {}),
      }, { signal: controller.signal });

      clearTimeout(timer);
      return completion.choices[0].message.content;

    } catch (err) {
      clearTimeout(timer);
      const isTimeout = err.name === "AbortError" || err.code === "ECONNABORTED";
      if (isTimeout)        throw new Error(ERR.OPENAI_TIMEOUT);
      if (attempt < retries) { console.log(`  [retry] attempt ${attempt + 1}`); continue; }
      throw err;
    }
  }
}

/* ─────────────────────────────────────────────────────────────
   SECURITY MIDDLEWARE
───────────────────────────────────────────────────────────── */

app.use(helmet({ crossOriginResourcePolicy: { policy: "cross-origin" } }));

app.use(cors({
  origin:         "*",
  methods:        ["GET", "POST", "OPTIONS"],
  allowedHeaders: ["Content-Type"],
}));

app.use(express.json({ limit: "50kb" }));

const chatLimiter = rateLimit({
  windowMs: 15 * 60 * 1000, max: 60,
  standardHeaders: true, legacyHeaders: false,
  message: { error: "TOO_MANY_REQUESTS" },
  skip: () => process.env.NODE_ENV === "development",
});

const analyzeLimiter = rateLimit({
  windowMs: 15 * 60 * 1000, max: 10,
  standardHeaders: true, legacyHeaders: false,
  message: { error: "TOO_MANY_REQUESTS" },
  skip: () => process.env.NODE_ENV === "development",
});

/* ─────────────────────────────────────────────────────────────
   QUALITY GUARD — detects low-quality AI responses
───────────────────────────────────────────────────────────── */
function isLowQuality(responseText, lastUserMessage = "") {
  try {
    const clean = responseText.replace(/```json|```/g, "").trim();
    const parsed = JSON.parse(clean);

    const q = (parsed.question || "").toLowerCase().trim();
    const r = (parsed.reflection || "").toLowerCase().trim();
    const state = detectUserState(lastUserMessage);

    // HARD FAILS — immediate rejection
    if (!q || !r) return true;

    // Decision question despite uncertainty — always hard fail
    if (isUncertain(lastUserMessage)) {
      const forbidden = [
        "was wirst du", "was willst du", "was setzt du um",
        "was wirst du tun", "was entscheidest du", "wofür entscheidest du dich",
      ];
      if (forbidden.some(p => q.includes(p))) return true;
      if (q.includes("umsetzen") || q.includes("anfangen") || q.includes("starten")) return true;
    }

    // SCORING
    let score = 0;

    // Too short (+2)
    if (q.length < 15) score += 2;
    if (r.length < 20) score += 2;

    // Generic questions (+2)
    const generic = ["was denkst du", "warum ist das wichtig", "wie fühlst du dich", "was meinst du"];
    if (generic.some(p => q.includes(p))) score += 2;

    // Question too long (+2)
    const wordCount = q.split(" ").filter(Boolean).length;
    if (wordCount > 15) score += 2;

    // Repetitive weak question starters (+1)
    const weakRepetitions = ["warum ist", "was denkst du", "was meinst du"];
    if (weakRepetitions.some(p => q.startsWith(p))) score += 1;

    // No exploratory signal despite uncertainty (+2)
    if (isUncertain(lastUserMessage)) {
      const exploratorySignals = ["warum","woran","was macht","was hält","wenn du","was wäre","was fällt","wodurch","wobei"];
      if (!exploratorySignals.some(p => q.includes(p))) score += 2;
    }

    // Too abstract when avoiding (+2)
    if (isAvoiding(lastUserMessage)) {
      const tooAbstract = ["warum", "was bedeutet", "was denkst du"];
      if (tooAbstract.some(p => q.includes(p))) score += 2;
    }

    // CLEAR — weak question without strong signal (+1)
    if (state === "CLEAR") {
      const weakSignals   = ["woran", "was genau", "wie genau"];
      const strongSignals = ["wovor", "warum", "was vermeidest", "was hält dich"];
      if (weakSignals.some(p => q.includes(p)) && !strongSignals.some(p => q.includes(p))) score += 1;
    }

    // DEFENSIVE — no challenge signal (+2)
    if (isDefensive(lastUserMessage)) {
      const challengeSignals = ["was wäre wenn","stimmt das wirklich","was passiert wenn","was übersiehst du"];
      if (!challengeSignals.some(p => q.includes(p))) score += 2;
    }

    // No tension in question or reflection (+2 each)
    const tension = ["aber","trotzdem","gleichzeitig","wovor","vermeid","entscheid"];
    if (!tension.some(w => q.includes(w))) score += 2;
    if (!tension.some(w => r.includes(w))) score += 0.5;

    // Weak connection between reflection and question (+1)
    const reflectionWords   = r.split(" ").map(w => w.replace(/[^\wäöüß]/g, "")).filter(w => w.length > 4);
    const hasWordOverlap    = reflectionWords.some(word => q.includes(word));
    const semanticBridge    = ["wovor","warum","was hält","was vermeidest","was passiert"];
    const hasSemanticBridge = semanticBridge.some(p => q.includes(p));
    if (!hasWordOverlap && !hasSemanticBridge) score += 0.5;

    // Reflection doesn't match user state (+2 each)

    if (state === "CLEAR") {
      const tensionSignals = ["aber","gleichzeitig","und trotzdem"];
      const altTension     = ["hältst dich zurück","gehst nicht","vermeidest","weißt es","zögerst"];
      if (!tensionSignals.some(w => r.includes(w)) && !altTension.some(w => r.includes(w))) score += 1.5;
    }

    if (state === "DEFENSIVE") {
      const defensiveSignals = ["weil","um zu","damit"];
      const altDefensive     = ["rechtfertigst","begründest","erklärst weg","schützt dich","machst es logisch"];
      if (!defensiveSignals.some(w => r.includes(w)) && !altDefensive.some(w => r.includes(w))) score += 1.5;
    }

    if (state === "AVOIDING") {
      const vagueSignals = ["irgendwie","unklar","nicht greifbar"];
      const altVague     = ["weichst","ausweichst","bleibst unklar","gehst nicht klar darauf","bleibst im unkonkreten"];
      if (!vagueSignals.some(w => r.includes(w)) && !altVague.some(w => r.includes(w))) score += 1.5;
    }

    if (state === "UNCERTAIN") {
      const uncertainSignals = ["nicht sicher","unklar","weißt nicht","nicht klar"];
      const altUncertain     = ["keine antwort","keine richtung","orientierungslos","tastend","suchst noch"];
      if (!uncertainSignals.some(w => r.includes(w)) && !altUncertain.some(w => r.includes(w))) score += 1.5;
    }

    let qualityBoost = 0;
    const hasStrongTension = ["wovor","was vermeidest","was hält dich"].some(p => q.includes(p));
    const hasConnection    = reflectionWords.some(word => q.includes(word));
    if (hasStrongTension && hasConnection) qualityBoost -= 1;

    return (score + qualityBoost) >= 5;

  } catch (e) {
    return true;
  }
}

function isLowQualityResult(result) {
  try {
    const { headline, mirror, tension, shift } = result;
    if (!headline || !mirror || !tension || !shift) return true;
    const text = `${headline} ${mirror} ${tension}`.toLowerCase();
    const generic = [
      "du willst", "du bist", "du hast potenzial",
      "du versuchst", "du möchtest"
    ];
    if (generic.some(p => text.includes(p))) return true;
    if (mirror.length < 40) return true;
    const tensionSignals = ["aber", "trotzdem", "gleichzeitig"];
    if (!tensionSignals.some(w => tension.toLowerCase().includes(w))) return true;
    return false;
  } catch {
    return true;
  }
}
/* ─────────────────────────────────────────────────────────────
   /api/chat — CONVERSATIONAL ONLY
   ─────────────────────────────────────────────────────────────
   Receives the message history and returns the next question
   or reflection. NO JSON output. NO structured result.
   Just one focused question per call.
───────────────────────────────────────────────────────────── */
const CLARITY_SYSTEM_PROMPT = `Du bist Clarity — ein ruhiger, präziser Gesprächspartner.

Du hilfst Menschen, sich selbst klarer zu sehen.
Du bist kein Coach. Du gibst keine Ratschläge. Du motivierst nicht.
Du machst sichtbar, was bereits da ist — auch wenn es unangenehm ist.

━━━━━━━━━━━━━━━━━━━━━━━
KERNPRINZIP
━━━━━━━━━━━━━━━━━━━━━━━

Bedeutung > Wortwahl  
Kontext > einzelne Aussagen  

Du reagierst nicht auf einzelne Wörter, sondern auf Zusammenhänge.

━━━━━━━━━━━━━━━━━━━━━━━
DEINE AUFGABE
━━━━━━━━━━━━━━━━━━━━━━━

Antworte IMMER mit einem JSON-Objekt — nichts anderes:

{
  "reflection": "<1 kurzer Satz (max 18–20 Wörter), beobachtend + leicht interpretierend>",
  "question":   "<1 einzige Frage (max 15 Wörter), direkt, endet mit ?>"
}

━━━━━━━━━━━━━━━━━━━━━━━
REFLECTION (entscheidend)
━━━━━━━━━━━━━━━━━━━━━━━

Die Reflection soll zeigen:
→ „Ich habe verstanden, was hier wirklich passiert“

Regeln:

- 1 Satz
- max 20 Wörter
- darf interpretieren — aber nur basierend auf dem Gesagten
- verbindet mindestens 2 Elemente aus der Antwort
- zeigt Spannung, Widerspruch oder Muster

Beispiele (gut):

"Du willst vorankommen, vermeidest aber die Entscheidung, die das festlegt."
"Du hältst dich beschäftigt, aber nichts davon fühlt sich wirklich wichtig an."
"Du denkst viel darüber nach — aber gehst nicht ins Handeln."

Beispiele (schlecht):

"Du hast gesagt, dass dir Fortschritt wichtig ist."
"Das klingt nach Angst." (zu therapeutisch / zu generisch)
"Gut, dass du das erkennst." (verboten)

━━━━━━━━━━━━━━━━━━━━━━━
FRAGE
━━━━━━━━━━━━━━━━━━━━━━━

Die Frage ist der nächste logische Schritt.

Regeln:

- genau 1 Frage
- max 15 Wörter
- basiert direkt auf der Reflection
- macht etwas sichtbar oder zwingt zu Klarheit
- leicht konfrontierend, aber nicht aggressiv
- keine Doppelfragen

Beispiele (gut):

"Was würde sich ändern, wenn du dich heute festlegst?"
"Wovor schützt dich dieses Verhalten gerade?"
"Was vermeidest du, indem du so weitermachst?"

Beispiele (schlecht):

"Was denkst du darüber?"
"Warum ist das wichtig für dich?"
"Was fühlst du und was meinst du?"

━━━━━━━━━━━━━━━━━━━━━━━
CONNECTION
━━━━━━━━━━━━━━━━━━━━━━━

Jede Antwort muss sich verbunden anfühlen.

Nutze implizite Übergänge wie:
- "Gleichzeitig wirkt es, als..."
- "Und trotzdem..."
- "Was dabei auffällt..."

(keine festen Templates, sondern natürlich integriert)

━━━━━━━━━━━━━━━━━━━━━━━
WICHTIG
━━━━━━━━━━━━━━━━━━━━━━━

- Keine generischen Aussagen
- Keine philosophischen Fragen
- Keine Wiederholung der User-Worte ohne Bedeutung
- Keine künstliche Psychologie

Wenn es sich wie ein Bot anfühlt → falsch  
Wenn es sich wie ein klarer, ehrlicher Mensch anfühlt → richtig

━━━━━━━━━━━━━━━━━━━━━━━
UNSICHERHEIT
━━━━━━━━━━━━━━━━━━━━━━━

Wenn der User Unsicherheit ausdrückt (z.B. "ich weiß nicht", "keine Ahnung", "bin mir nicht sicher"):

- Stelle KEINE Entscheidungsfrage (z.B. "Was wirst du jetzt tun?")
- Stelle stattdessen eine Erkundungsfrage, die hilft die Unklarheit zu beleuchten
- Ziel: Die Unsicherheit selbst sichtbar machen, nicht überspringen

Beispiele bei Unsicherheit (gut):
"Was wäre, wenn du es wüsstest — was würdest du dann sagen?"
"Wovor schützt dich dieses Nicht-Wissen gerade?"
"Was fühlt sich an dieser Unsicherheit vertraut an?"

━━━━━━━━━━━━━━━━━━━━━━━
MEMORY (SEHR WICHTIG)
━━━━━━━━━━━━━━━━━━━━━━━

Du hast Zugriff auf das gesamte Gespräch.

Nutze es aktiv.

Regeln:

- Beziehe dich gelegentlich auf frühere Aussagen
- Verbinde aktuelle Antwort mit vorherigen Antworten
- mache Widersprüche sichtbar
- erkenne Wiederholungen oder Muster

Beispiele:

"Am Anfang klang es so, als wärst du unsicher — jetzt klingt es anders."

"Du hast vorhin gesagt, dass dir X wichtig ist — gleichzeitig wirkt dein Verhalten anders."

"Das taucht jetzt schon zum zweiten Mal auf."

Wichtig:

- Nur nutzen, wenn es sinnvoll ist
- nicht in jeder Antwort
- nicht künstlich einbauen

Ziel:

Die Person soll spüren:
→ "Das baut auf mir auf"`;

app.get("/", (req, res) => res.send("Clarity backend running"));

app.post("/api/chat", chatLimiter, async (req, res) => {
  const endpoint = "POST /api/chat";
  const reqId    = makeReqId();

  const invalid = validateMessages(req.body);
  if (invalid) {
    log(endpoint, { reqId, error: invalid.code });
    return res.status(400).json({ error: invalid.code });
  }

  const { messages } = req.body;
  const lastTwoUser = messages
  .filter(m => m.role === "user" && m.content && m.content.trim())
  .slice(-2);

const memoryContext = lastTwoUser.length
  ? {
      role: "system",
      content: `Letzte Aussagen des Users:\n${lastTwoUser.map(m => m.content).join("\n")}`
    }
  : null;

  // Uncertainty guard — if last user message expresses not knowing,
  // inject a system hint to steer away from decision-type questions
  const lastUserMessage = messages
    .filter(m => m.role === "user")
    .slice(-1)[0]?.content || "";

  const state = detectUserState(lastUserMessage);

  const progressionHint = {
    role: "system",
    content: `
GESPRÄCHSENTWICKLUNG:
Achte darauf, dass sich das Gespräch wirklich weiterentwickelt.
Wenn ein Thema bereits angesprochen wurde:
- gehe tiefer statt es neu zu formulieren
- stelle KEINE ähnliche Frage erneut
Wenn sich ein Muster zeigt:
- benenne es klar
- stelle eine Frage, die es zuspitzt
Vermeide:
- gleiche Satzstruktur
- gleiche Frageform (z.B. mehrfach "Warum...")
Ziel:
Jede Frage fühlt sich wie ein echter nächster Schritt an — nicht wie eine Variation.
`,
  };

  const stateHint = {
    role: "system",
    content: `
USER STATE: ${state}
WICHTIGE ANPASSUNG:
Passe deine Antwort zwingend an diesen Zustand an.
UNCERTAIN:
- mache die Unsicherheit greifbarer
- keine Richtung oder Entscheidung
AVOIDING:
- werde konkret
- bringe den User zu einer klaren Situation
CLEAR:
- erhöhe die Spannung
- stelle eine konfrontierendere Frage
DEFENSIVE:
- hinterfrage die Begründung
- zeige möglichen Selbstschutz
Wenn deine Frage nicht zum Zustand passt, ist sie falsch.
`,
  };

  const uncertaintyHint = isUncertain(lastUserMessage)
    ? [{
        role: "system",
        content: `
HARTE REGEL (HÖCHSTE PRIORITÄT):
Der User hat Unsicherheit ausgedrückt.
Du DARFST KEINE Entscheidungsfrage stellen.
Du DARFST KEINE Frage stellen, die Handlung voraussetzt.
ERLAUBT sind NUR Fragen, die:
- die Unsicherheit genauer machen
- einen vorsichtigen Zugang öffnen
- helfen, etwas greifbarer zu machen
Wenn deine Frage auf Handlung oder Entscheidung abzielt, ist sie FALSCH.
Formuliere die Frage neu.
`,
      }]
    : [];
  log(endpoint, { reqId, messages: messages.length });

  try {
    let text;

for (let attempt = 0; attempt < 2; attempt++) {
  text = await callOpenAI([
  { role: "system", content: CLARITY_SYSTEM_PROMPT },
  progressionHint,
  ...(memoryContext ? [memoryContext] : []),
  ...messages,
  stateHint,
  ...uncertaintyHint,
    ...(attempt === 1 ? [{
      role: "system",
      content: `
Formuliere die Antwort neu und klarer.
Achte darauf:
- Die Reflection zeigt einen echten Zusammenhang
- Die Frage baut logisch darauf auf
- Die Sprache bleibt natürlich und direkt
- Vermeide ähnliche Struktur wie zuvor
WICHTIG:
- Wähle einen leicht anderen Blickwinkel als zuvor
- Hinterfrage ggf. eine andere Facette der Situation
Vermeide generische Formulierungen.
`
    }] : [])
  ]);

  const lowQuality = isLowQuality(text, lastUserMessage);

if (!lowQuality) break;

if (lowQuality && attempt === 0) {
  console.log("⚠️ Low quality response → retrying...");
}
}

if (!text) {
  return res.status(500).json({ error: "EMPTY_AI_RESPONSE" });
}

const finalLowQuality = isLowQuality(text, lastUserMessage);
if (finalLowQuality) {
  console.log("❌ Final response still low quality after retry");
}

res.json({ content: text });
  } catch (err) {
    log(endpoint, { reqId, error: err.message });
    res.status(500).json({ error: ERR.OPENAI_FAILED });
  }
});

/* ─────────────────────────────────────────────────────────────
   /api/analyze — ANALYTICAL ONLY
   ─────────────────────────────────────────────────────────────
   Receives the complete answers array.
   Returns a single structured JSON result — the insight profile.
   Never called mid-conversation. Only called once at the end.
───────────────────────────────────────────────────────────── */
const ANALYSIS_SYSTEM_PROMPT = `Du analysierst die Antworten aus einem geführten Reflexionsgespräch.
Du erhältst ein JSON-Objekt mit einem "answers"-Array (12 Antworten).

Antworte NUR mit validem JSON. Kein Markdown. Kein Text davor oder danach.

SCHRITT 1: TYP-ERKENNUNG (FEST FESTLEGEN)

Lies alle Antworten. Ordne den User exakt EINEM Typ zu.
Dieser Typ ist danach FEST — alle weiteren Schritte basieren ausschließlich auf ihm.

EXPLORER  — Sammelt Optionen statt zu entscheiden.
  Spannung: Entscheidung vs Vermeidung.
  Sprache:  "du weißt, aber entscheidest nicht"
  Schlüsselwörter: vielleicht, könnte, wenn, noch nicht sicher, abwarten.

BUILDER   — Handelt viel, hinterfragt das Wohin selten.
  Spannung: Richtung vs Momentum.
  Sprache:  "du machst, aber hinterfragst nicht"
  Schlüsselwörter: vorankommen, umsetzen, optimieren, Fortschritt.

CREATOR   — Starker Drang zu erschaffen, aber Angst vor Sichtbarkeit.
  Spannung: Sichtbarkeit vs Angst.
  Sprache:  "du hast etwas, aber zeigst es nicht"
  Schlüsselwörter: Ideen, zeigen, noch nicht fertig, noch nicht bereit.

OPTIMIZER — Sieht sofort was nicht stimmt. Hohe Standards, schwer zufrieden.
  Spannung: Perfektion vs Bedeutung.
  Sprache:  "du verbesserst, aber hinterfragst nicht ob es das Richtige ist"
  Schlüsselwörter: besser werden, nicht gut genug, immer noch, Standards.

DRIFTER   — Bewegt sich ohne Richtung.
  Spannung: Bewegung vs Entscheidung.
  Sprache:  "du bist beschäftigt, aber kommst nicht voran"
  Schlüsselwörter: mal schauen, irgendwie, passiert halt, keine Ahnung.

Confidence: 40-65 = unsicher, 66-80 = klar, 81-95 = eindeutig.


SCHRITT 2: ALLE FELDER DURCH DEN TYP GENERIEREN

Der erkannte Typ ist die EINZIGE Linse. Jedes Feld muss den Typ tragen.
Ein Ergebnis, das zu einem anderen Typ passen könnte, ist falsch.

REGEL 1 — EVIDENZBASIERT:
Jede Aussage muss in konkreten Antworten verankert sein.
Keine generischen Interpretationen. Keine Annahmen ohne Grundlage.

REGEL 2 — SUMMARY (Konfrontation + Typ-Linse):
Formel: "Du tust X — nicht weil Y, sondern weil Z."
X = konkretes Verhalten aus den Antworten, durch den Typ gefärbt
Y = Ausrede oder Rationalisierung
Z = unbequeme wahre Ursache, spezifisch für diesen Typ

Typ-spezifische Sprach-Richtung:
  Explorer:  "Du weißt, was du tun würdest — entscheidest dich aber nicht."
  Builder:   "Du machst weiter — ohne zu prüfen, ob es das Richtige ist."
  Creator:   "Du hast etwas zu sagen — aber zeigst es nicht."
  Optimizer: "Du verbesserst ständig — aber nicht das, was wirklich zählt."
  Drifter:   "Du bist beschäftigt — aber entscheidest dich für nichts."

Max 18 Wörter. Direkte du-Ansprache.

REGEL 3 — PATTERN (Muster durch Typ-Linse):
Beschreibe ein konkretes Verhalten aus mindestens 2 Antworten, interpretiert durch den Typ.

Typ-spezifische Muster-Richtung:
  Explorer:  "Du hast mehrfach beschrieben, dass du Optionen offen hältst, statt dich festzulegen."
  Builder:   "Du hast mehrfach beschrieben, dass du weitermachst, ohne die Richtung zu prüfen."
  Creator:   "Du hast mehrfach beschrieben, dass du Ideen zurückhältst, weil sie noch nicht fertig sind."
  Optimizer: "Du hast mehrfach beschrieben, dass du verbesserst, ohne zu fragen ob das Ziel stimmt."
  Drifter:   "Du hast mehrfach beschrieben, dass du reagierst statt entscheidest."

Beginnt mit "Du hast mehrfach..." oder "In deinen Antworten taucht auf..."
Max 20 Wörter.

REGEL 4 — SUGGESTEDACTION (Gegengewicht zum Typ):
Die Handlung muss das Kernmuster des Typs direkt unterbrechen:
  Explorer:  "Entscheide dich heute für eine Option — egal welche."
  Builder:   "Halte heute an und frag: Warum mache ich das?"
  Creator:   "Zeig heute etwas, das du noch zurückgehalten hast."
  Optimizer: "Entscheide heute: Was ist gut genug — und lass den Rest los."
  Drifter:   "Wähle heute eine Richtung — und bleib 7 Tage dabei."

Formulierung: "Tu X — nicht Y." Max 12 Wörter.

REGEL 5 — ENDVALIDIERUNG (PFLICHT vor Output):
1. Könnte dieses Ergebnis zu einem anderen Typ passen? Wenn JA: Neuschreiben.
2. Spiegelt die Sprache die Spannung des Typs? Wenn NEIN: Neuschreiben.
3. Ist das Ergebnis leicht unangenehm, aber wahr? Wenn NEIN: Neuschreiben.
4. Könnte summary oder pattern auf 70% der Menschen zutreffen? Wenn JA: Neuschreiben.

strengths, energySources:
  Nur aus den echten Antworten ableiten.
  Verwende wenn möglich die eigenen Wörter des Users.
  Nie erfinden.


OUTPUT FORMAT

{
  "scores": {
    "Clarity":   <integer 1-100>,
    "Energy":    <integer 1-100>,
    "Strength":  <integer 1-100>,
    "Direction": <integer 1-100>,
    "Action":    <integer 1-100>
  },
  "identityModes": [
    { "type": "<Explorer|Builder|Creator|Optimizer|Drifter>", "confidence": <integer 40-95> }
  ],
  "summary":         "<1 Satz, max 18 Wörter, Formel: Du tust X — nicht weil Y, sondern weil Z. Typ-spezifisch.>",
  "pattern":         "<1 Satz, max 20 Wörter, aus mind. 2 Antworten, durch Typ-Linse, beginnt mit Du hast mehrfach... oder In deinen Antworten...>",
  "strengths":       ["<konkret aus Antworten>", "<konkret>", "<konkret>"],
  "energySources":   ["<konkret aus Antworten>", "<konkret>", "<konkret>"],
  "nextFocus":       "<1 Satz, wichtigster Fokus nächste 30 Tage, aus dem Typ-Muster abgeleitet>",
  "suggestedAction": "<1 Handlung heute, max 12 Wörter, Tu X nicht Y, konterkariert den Typ>"
}

Scores: Basieren auf konkreten Hinweisen. Vermeide runde 10er-Schritte.
identityModes: Immer nur 1 Typ — außer beide liegen >=55 Confidence.`

const SIGNAL_EXTRACTION_PROMPT = `Du liest die Antworten eines Reflexionsgesprächs und extrahierst Signale für die Analyse.
Du erhältst ein JSON-Objekt mit einem "answers"-Array.
Antworte NUR mit validem JSON. Kein Markdown.

{
  "repeated_themes":   ["<Thema das 2+ mal auftaucht>"],
  "avoided_topics":    ["<Was umgangen, relativiert oder schnell verlassen wurde>"],
  "energy_language":   ["<Wörter mit spürbarer Energie oder Abwehr>"],
  "core_tension":      "<Das zentrale Spannungsfeld in 1 Satz>",
  "avoidance_pattern": "<Was konkret vermieden wird und wodurch — 1 Satz>",
  "behavioral_type_signals": {
    "Explorer": <0–10>, "Builder": <0–10>, "Creator": <0–10>,
    "Optimizer": <0–10>, "Drifter": <0–10>
  }
}`;

const CLARITY_RESULT_PROMPT = `
Du bist Clarity — der letzte Spiegel nach einem Gespräch.
Deine Aufgabe:
Verdichte das Gespräch zu einem Ergebnis, das sich exakt nach dieser Person anhört — nicht allgemein.
━━━━━━━━━━━━━━━━━━━━━━━
ZIEL
━━━━━━━━━━━━━━━━━━━━━━━
Der Nutzer soll denken:
"Ja. Genau das bin ich."
NICHT:
"Das könnte auf viele zutreffen."
━━━━━━━━━━━━━━━━━━━━━━━
STRUKTUR (IMMER JSON)
━━━━━━━━━━━━━━━━━━━━━━━
{
  "headline": "<1 kurzer, klarer Satz (max 12 Wörter)>",
  "mirror":   "<1–2 Sätze, konkret, basiert auf echten Aussagen>",
  "tension":  "<1 Satz, zeigt klaren Widerspruch (X aber Y) oder konkretes Ausweichen>",
  "shift":    "<1 Frage, öffnet — keine Lösung>"
}
━━━━━━━━━━━━━━━━━━━━━━━
HARTE REGELN
━━━━━━━━━━━━━━━━━━━━━━━
1. KEINE GENERISCHEN PHRASEN
   ❌ "du willst dich verbessern"
   ❌ "du bist unsicher"
   ❌ "du hast Potenzial"
2. NUTZE KONKRETE SPRACHE AUS DEM GESPRÄCH
   Wenn möglich, übernimm Wörter oder Formulierungen des Users.
3. ZEIGE KONKRETES VERHALTEN
   ❌ abstrakt: "du hältst dich zurück"
   ✅ konkret: "du denkst viel darüber nach — gehst aber nicht los"
4. KEINE ERKLÄRUNGEN
   ❌ "weil du Angst hast"
   ❌ "das liegt daran"
   Nur zeigen.
5. JEDE ZEILE MUSS EINE BEOBACHTUNG SEIN
6. SHIFT IST KEIN COACHING
   ❌ "du könntest jetzt..."
   ❌ "der nächste Schritt ist..."
   ✅ "Wovor schützt dich das gerade?"
7. TENSION IST DER WICHTIGSTE TEIL
   Die tension muss spürbar sein.
   Sie soll leicht unangenehm sein.
   ❌ "du bist unsicher"
   ❌ "du bist noch nicht ganz klar"
   ✅ "du weißt es eigentlich — gehst aber nicht rein"
   ✅ "du denkst darüber nach — vermeidest aber den Schritt"
8. HEADLINE IST KEIN TITEL
   Sie ist eine Beobachtung.
   ❌ "Dein Muster"
   ❌ "Mehr Klarheit gewinnen"
   ✅ "Du weißt mehr, als du gerade zulässt."
   ✅ "Du gehst im Kreis, obwohl du es siehst."
━━━━━━━━━━━━━━━━━━━━━━━
INPUT
━━━━━━━━━━━━━━━━━━━━━━━
Nutze:
- komplette conversation
- Analyse (summary, pattern, type)
Priorität:
1. Wiederholungen
2. Widersprüche
3. Ausweichverhalten
━━━━━━━━━━━━━━━━━━━━━━━
FINAL CHECK (PFLICHT)
━━━━━━━━━━━━━━━━━━━━━━━
Bevor du antwortest, prüfe:
- Könnte das auf viele Menschen passen? → NEU schreiben
- Klingt es wie Coaching? → NEU schreiben
- Ist es konkret genug? → NEU schreiben
- Ist die tension spürbar unangenehm? → wenn nein: neu schreiben
━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT
━━━━━━━━━━━━━━━━━━━━━━━
NUR JSON.
`;
app.post("/api/analyze", analyzeLimiter, async (req, res) => {
  const endpoint = "POST /api/analyze";
  const reqId    = makeReqId();

  const invalid = validateMessages(req.body);
  if (invalid) {
    log(endpoint, { reqId, error: invalid.code });
    return res.status(400).json({ error: invalid.code });
  }

  const { messages } = req.body;
  log(endpoint, { reqId, messages: messages.length });
  const t0 = Date.now();

  try {
    // Step 1: Signal extraction (graceful fallback)
    let signalsBlock = null;
    try {
      const raw = await callOpenAI([
        { role: "system", content: SIGNAL_EXTRACTION_PROMPT },
        ...messages,
      ], { jsonMode: true });
      signalsBlock = raw;
    } catch (e) {
      log(endpoint, { reqId, warning: `signal extraction failed: ${e.message}` });
    }

    // Step 2: Full insight generation
    const analysisMessages = signalsBlock
      ? [
          { role: "system", content: ANALYSIS_SYSTEM_PROMPT },
          ...messages,
          { role: "user", content: `Extrahierte Signale:\n${signalsBlock}` },
        ]
      : [
          { role: "system", content: ANALYSIS_SYSTEM_PROMPT },
          ...messages,
        ];

    const raw     = await callOpenAI(analysisMessages, { jsonMode: true });
    const parsed  = JSON.parse(raw);

    // Step 3: Generate result (mirror / tension / shift)
    let result = null;
    for (let attempt = 0; attempt < 2; attempt++) {
      try {
        const resultMessages = [
          { role: "system", content: CLARITY_RESULT_PROMPT },
          ...messages,
          {
            role: "system",
            content: `
ANALYSE (VERBINDLICH):
Typ: ${parsed.identityModes?.[0]?.type}
Summary: ${parsed.summary}
Pattern: ${parsed.pattern}
WICHTIG:
Das Ergebnis MUSS klar diesen Typ widerspiegeln.
Wenn es auch zu einem anderen Typ passen könnte → falsch.
`
          }
        ];
        const resultRaw    = await callOpenAI(resultMessages, { jsonMode: true });
        const parsedResult = JSON.parse(resultRaw);
        if (!isLowQualityResult(parsedResult)) {
          result = parsedResult;
          break;
        }
        if (attempt === 0) {
          console.log("⚠️ Low quality result → retrying...");
        }
      } catch (e) {
        log(endpoint, { reqId, warning: `result generation failed: ${e.message}` });
      }
    }

    if (!result) {
      console.log("⚠️ Using fallback result");
      result = {
        headline: parsed.summary || "Etwas passt hier nicht ganz.",
        mirror:   parsed.pattern || "In deinen Antworten zeigt sich ein Muster.",
        tension:  "Du siehst es — aber gehst noch nicht wirklich rein.",
        shift:    "Was hält dich gerade davon ab, das wirklich anzugehen?"
      };
    }

    const elapsed = ((Date.now() - t0) / 1000).toFixed(1);

    log(endpoint, { reqId, elapsed: `${elapsed}s`, type: parsed.identityModes?.[0]?.type });
    res.json({
      ...parsed,
      result
    });

  } catch (err) {
    const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
    const code    = err.message === ERR.OPENAI_TIMEOUT ? ERR.OPENAI_TIMEOUT : ERR.OPENAI_FAILED;
    log(endpoint, { reqId, error: code, detail: err.message, elapsed: `${elapsed}s` });
    res.status(500).json({ error: code });
  }
});

/* ─────────────────────────────────────────────────────────────
   Reminder endpoint (mock — no actual email sending)
───────────────────────────────────────────────────────────── */
app.post("/api/reminder", async (req, res) => {
  const reqId    = makeReqId();
  const endpoint = "POST /api/reminder";
  const { email, type = "habit" } = req.body || {};

  if (!email) return res.status(400).json({ error: "EMAIL_REQUIRED" });

  log(endpoint, { reqId, email, type });
  res.json({ ok: true });
});

/* ─────────────────────────────────────────────────────────────
   ANALYTICS — anonymous event counting
   No cookies, no IP logging, no personal data.
   Persisted to events.json on disk (survives restarts, resets on redeploy).
───────────────────────────────────────────────────────────── */
import { readFileSync, writeFileSync, existsSync } from "fs";

const EVENTS_FILE = "./events.json";
const STATS_KEY   = process.env.STATS_KEY || "clarity-stats";

const EVENT_NAMES = new Set([
  "flow_start",
  "question_1",  "question_2",  "question_3",  "question_4",
  "question_5",  "question_6",  "question_7",  "question_8",
  "question_9",  "question_10", "question_11", "question_12",
  "flow_complete",
  "share_opened",
  "share_tapped",
]);

function loadEvents() {
  try {
    if (existsSync(EVENTS_FILE)) return JSON.parse(readFileSync(EVENTS_FILE, "utf8"));
  } catch (_) {}
  return { totals: {}, daily: {} };
}

function saveEvents(data) {
  try { writeFileSync(EVENTS_FILE, JSON.stringify(data), "utf8"); } catch (_) {}
}

let eventData = loadEvents();

// POST /api/event — { name: "flow_start" }
app.post("/api/event", (req, res) => {
  const { name } = req.body || {};
  if (!name || !EVENT_NAMES.has(name)) return res.status(400).json({ error: "INVALID_EVENT" });

  const today = new Date().toISOString().slice(0, 10);
  eventData.totals[name]        = (eventData.totals[name] || 0) + 1;
  eventData.daily[today]        = eventData.daily[today] || {};
  eventData.daily[today][name]  = (eventData.daily[today][name] || 0) + 1;
  saveEvents(eventData);

  res.json({ ok: true });
});

// POST /api/stats — key in body, never in URL
app.post("/api/stats", (req, res) => {
  const { key } = req.body || {};
  if (key !== STATS_KEY) return res.status(401).json({ error: "UNAUTHORIZED" });
  res.json(eventData);
});

/* ─────────────────────────────────────────────────────────────
   Error handling
───────────────────────────────────────────────────────────── */
process.on("unhandledRejection", (reason) => {
  console.error("[unhandledRejection]", reason);
});

const PORT = process.env.PORT || 8080;
app.listen(PORT, "0.0.0.0", () => {
  console.log(`[startup] Clarity backend running on port ${PORT}`);
});
