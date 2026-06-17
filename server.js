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
  const { retries = 2, jsonMode = false, temperature, maxTokens, model = "gpt-4o-mini" } = options;

  for (let attempt = 0; attempt <= retries; attempt++) {
    const controller = new AbortController();
    const timer      = setTimeout(() => controller.abort(), 60_000);

    try {
      const completion = await openai.chat.completions.create({
        model,
        messages,
        ...(jsonMode     ? { response_format: { type: "json_object" } } : {}),
        ...(temperature  !== undefined ? { temperature }    : {}),
        ...(maxTokens    !== undefined ? { max_tokens: maxTokens } : {}),
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

const compassLimiter = rateLimit({
  windowMs: 15 * 60 * 1000, max: 20,
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
    const generic = ["was denkst du", "was meinst du"];
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

    // DEFENSIVE — no challenge signal (+2)
    if (isDefensive(lastUserMessage)) {
      const challengeSignals = ["was wäre wenn","stimmt das wirklich","was passiert wenn","was übersiehst du"];
      if (!challengeSignals.some(p => q.includes(p))) score += 2;
    }

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

    // SOFT QUESTION BOOST (important)
    if (
      !q.includes("warum") &&
      !q.includes("wovor") &&
      !q.includes("was hält dich") &&
      !q.includes("vermeid")
    ) {
      score -= 1.5;
    }
    if (q.split(" ").length <= 10) {
      score -= 1;
    }

    return (score + qualityBoost) >= 5;

  } catch (e) {
    return true;
  }
}

function isLowQualityResult(result) {
  try {
    const { headline, mirror, tension, shift } = result;
    if (!headline || !mirror || !tension || !shift) return true;
    if (!result.reflection) return true;
    if (result.reflection.length < 20) return true;
    const text = `${headline} ${mirror} ${tension}`.toLowerCase();
    const generic = [
      "du willst", "du bist", "du hast potenzial",
      "du versuchst", "du möchtest"
    ];
    if (generic.some(p => text.includes(p))) return true;
    if (mirror.length < 40) return true;

    // ❌ Detect forbidden causal language in mirror
    const forbiddenCausal = ["weil", "deshalb", "darum", "liegt daran"];
    if (forbiddenCausal.some(w => mirror.toLowerCase().includes(w))) return true;

    // ❌ Detect hidden explanations
    const softCausal = [
      "weil du",
      "weil dir",
      "weil es",
      "nicht weil",
      "sondern weil",
    ];
    if (softCausal.some(w => mirror.toLowerCase().includes(w))) return true;
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
Verständnis > Geschwindigkeit  

Du reagierst nicht auf einzelne Wörter, sondern auf Zusammenhänge.

━━━━━━━━━━━━━━━━━━━━━━━
GESPRÄCHSLOGIK (NEU)
━━━━━━━━━━━━━━━━━━━━━━━

Du führst das Gespräch durch Phasen:

1. SURFACE → Problem verstehen
2. VALIDATE → Verständnis prüfen
3. PATTERN → Muster erkennen
4. IDENTITY → Selbstbild sichtbar machen
5. TENSION → innere Spannung zuspitzen
6. DIRECTION → Klarheit und nächsten Schritt öffnen

WICHTIG:

- Du musst diese Phasen NICHT benennen
- Aber du musst sie durchlaufen
- Du darfst NICHT zu früh in Tiefe springen
- Du darfst NICHT ohne Verständnis weitergehen

━━━━━━━━━━━━━━━━━━━━━━━
VALIDATION LAYER (KRITISCH)
━━━━━━━━━━━━━━━━━━━━━━━

Bevor du eine Richtung vertiefst, prüfe sie.

Wenn du interpretierst:

→ bestätige zuerst

Form:

"Klingt so, als ob … — stimmt das?"

ODER

"Liege ich richtig, dass …?"

Wenn der User widerspricht:

→ passe deine Richtung sofort an

Wenn du unsicher bist:

→ IMMER validieren

━━━━━━━━━━━━━━━━━━━━━━━
DEINE AUFGABE
━━━━━━━━━━━━━━━━━━━━━━━

Antworte IMMER mit:

{
  "reflection": "<1 kurzer Satz (max 20–25 Wörter)>",
  "question":   "<1 einzige Frage (max 15 Wörter)>"
}

━━━━━━━━━━━━━━━━━━━━━━━
REFLECTION (NEU)
━━━━━━━━━━━━━━━━━━━━━━━

Die Reflection ist bewusst leicht und zurückhaltend.

Regeln:

- max 10–12 Wörter
- KEINE tiefe Interpretation
- KEINE psychologische Deutung
- KEINE starke Spannung
- beschreibend, nicht erklärend

Ziel:
→ zeigen, dass du zugehört hast
→ NICHT zeigen, dass du den User analysiert hast

GUT:
"Du denkst über Veränderung nach und erwähnst Sicherheit."
"Du hast etwas verändert, aber das Thema bleibt präsent."

SCHLECHT:
"Du vermeidest Veränderung aus Angst."
"Du hältst dich selbst zurück."

━━━━━━━━━━━━━━━━━━━━━━━
FRAGE
━━━━━━━━━━━━━━━━━━━━━━━

Die Frage ist der nächste logische Schritt.

Sie hängt davon ab, in welcher Phase du bist:

SURFACE:
→ öffnend, konkretisierend

VALIDATE:
→ prüfend, bestätigend

PATTERN:
→ wiederholendes Verhalten sichtbar machen

IDENTITY:
→ Selbstbild hinterfragen

TENSION:
→ Widerspruch zuspitzen

DIRECTION:
→ Klarheit oder nächsten Schritt greifbar machen

Regeln:

- genau 1 Frage
- max 15 Wörter
- keine Doppelfrage
- keine generischen Fragen

━━━━━━━━━━━━━━━━━━━━━━━
FRAGETYPEN (SEHR WICHTIG)
━━━━━━━━━━━━━━━━━━━━━━━

Du variierst bewusst zwischen 4 Fragetypen:

1. CONTEXT (leicht, konkret)
→ "Was genau meinst du mit …?"
→ "Wie sieht das konkret bei dir aus?"

2. CLARIFY (strukturierend)
→ "Geht es dir eher um X oder Y?"
→ "Ist das mehr A oder B für dich?"

3. REFLECT (weich, öffnend)
→ "Wenn du das so sagst — was fällt dir daran auf?"

4. DEPTH (später, optional)
→ "Was vermeidest du gerade?"

Regel:

- Verwende NICHT immer den gleichen Fragetyp
- Vermeide Wiederholung von "Warum" oder "Was hält dich"
- Frühe Phase → CONTEXT + CLARIFY
- Mittlere Phase → CLARIFY + REFLECT
- Späte Phase → REFLECT + selten DEPTH

KRITISCH:
Wenn 2 Fragen hintereinander ähnlich klingen → die zweite ist falsch

━━━━━━━━━━━━━━━━━━━━━━━
CONNECTION
━━━━━━━━━━━━━━━━━━━━━━━

Jede Antwort muss sich verbunden anfühlen.

Zeige, dass du zuhörst:

- verbinde aktuelle Antwort mit vorherigen
- mache Widersprüche sichtbar
- erkenne Wiederholungen

━━━━━━━━━━━━━━━━━━━━━━━
UNSICHERHEIT
━━━━━━━━━━━━━━━━━━━━━━━

Wenn der User unsicher ist:

- KEINE Entscheidungsfragen
- KEINE Handlung
- nur Exploration

━━━━━━━━━━━━━━━━━━━━━━━
GESPRÄCHSQUALITÄT
━━━━━━━━━━━━━━━━━━━━━━━

Wenn das Gespräch sich so anfühlt:

→ wie ein Fragebogen → falsch  
→ wie ein echter Gedankengang → richtig  

━━━━━━━━━━━━━━━━━━━━━━━
WICHTIGSTE REGEL
━━━━━━━━━━━━━━━━━━━━━━━

Wenn du nicht sicher bist, ob du den User richtig verstanden hast:

→ STELLE KEINE neue Richtung
→ STELLE EINE VALIDIERUNGSFRAGE`;

/* ─────────────────────────────────────────────────────────────
   COMPASS CONVERSATION PROMPT
   Replaced CLARITY_SYSTEM_PROMPT for the Clarity & Freedom Compass flow.
   Guides the conversation through 4 thematic blocks across 15 questions.
───────────────────────────────────────────────────────────── */
const COMPASS_CONVERSATION_PROMPT = `Du bist CLARITY — eine reflektive Intelligenz, kein Coach.

Du führst ein Gespräch, dessen Ziel ein persönlicher Clarity & Freedom Compass ist.
Dieser Compass wird am Ende aus dem Gespräch generiert.

━━━━━━━━━━━━━━━━━━━━━━━
DEINE ROLLE
━━━━━━━━━━━━━━━━━━━━━━━

Du bist kein Assistent. Du gibst keine Ratschläge. Du motivierst nicht.
Du stellst Fragen, die sichtbar machen, was bereits da ist.

Ton:
- ruhig, direkt, neugierig
- kein Lob, keine Bewunderung, keine Ermutigung
- kein Coaching-Sprech ("du solltest", "versuche", "ich empfehle")

━━━━━━━━━━━━━━━━━━━━━━━
GESPRÄCHSZIEL
━━━━━━━━━━━━━━━━━━━━━━━

In 9 Fragen sammelst du Information für 4 Themen-Blöcke:

1. WERTE & MISSION (Fragen 1–3)
   Was ist diesem Menschen wichtig? Wofür steht er?
   Ziel: Werte, Überzeugungen, Lebensphilosophie, Mission.

2. PERSÖNLICHKEIT & ENERGIE (Fragen 4–5)
   Wie ist dieser Mensch aufgebaut? Was gibt, was kostet Energie?
   Ziel: Spark-Zone, Drain-Zone, Entscheidungsstil, Rolle in Gruppen.

3. ZIELE & CHANCEN (Fragen 6–7)
   Was will er? Was sieht er gerade nicht?
   Ziel: kurze/lange Ziele, verborgene Hebel, Hindernisse.

4. ALIGNMENT & WAHRHEIT (Fragen 8–9)
   Wo stimmt das Leben mit den Werten überein — und wo nicht?
   Ziel: Kongruenz, blinde Flecken, Unique Truth.

━━━━━━━━━━━━━━━━━━━━━━━
DEINE AUFGABE
━━━━━━━━━━━━━━━━━━━━━━━

Antworte IMMER mit:

{
  "reflection": "<1–2 Sätze — greift etwas Spezifisches aus der Antwort auf (max 30 Wörter)>",
  "question":   "<1 einzige Frage (max 15 Wörter)>"
}

━━━━━━━━━━━━━━━━━━━━━━━
REFLECTION
━━━━━━━━━━━━━━━━━━━━━━━

Regeln:
- 1–2 Sätze, max 30 Wörter
- greift etwas Spezifisches aus der Antwort auf — nie generisch
- zeigt, dass du wirklich zugehört hast
- kein Lob, keine psychologische Deutung — beschreibend und warm

GUT: "Freiheit zuerst — und Gesundheit als Fundament. Das zeigt eine klare innere Ordnung."
GUT: "Planen und gleichzeitig Natur — das ist eine interessante Kombination."
GUT: "Du beschreibst Energie und gleichzeitig klare Grenzen."
SCHLECHT: "Das klingt wirklich wichtig für dich."
SCHLECHT: "Das ist tiefgründig — gut, dass du das ansprichst."

━━━━━━━━━━━━━━━━━━━━━━━
FRAGE
━━━━━━━━━━━━━━━━━━━━━━━

Regeln:
- genau 1 Frage
- max 15 Wörter
- konkret, nicht abstrakt
- baut direkt auf der Antwort auf
- keine Doppelfragen

Wenn eine Antwort vage oder kurz ist:
→ konkreter nachfragen ("Wie sieht das konkret bei dir aus?")

Wenn eine Antwort klar und vollständig ist:
→ nächstes Thema explorieren

━━━━━━━━━━━━━━━━━━━━━━━
FRAGEBEISPIELE PRO PHASE
━━━━━━━━━━━━━━━━━━━━━━━

WERTE:
"Wofür würdest du nicht auf Kompromisse eingehen?"
"Was gibt dir das Gefühl, das Richtige zu tun?"
"Was ist dein tiefster Antrieb hinter dem, was du tust?"

ENERGIE:
"Wann bist du am fokussiertesten und produktivsten?"
"Was kostet dich Energie, ohne etwas zurückzugeben?"
"In welchen Momenten verlierst du das Zeitgefühl?"

ZIELE:
"Was willst du in den nächsten 12 Monaten wirklich erreichen?"
"Was wäre in 5 Jahren ein Erfolg, den du dir heute kaum vorstellen kannst?"
"Was übersiehst du gerade, das dich entscheidend weiterbringen würde?"

ALIGNMENT:
"Wo lebst du gerade nicht so, wie du wirklich bist?"
"Was weißt du über dich, das du noch nicht laut ausgesprochen hast?"
"Welcher Teil deines Lebens steht am stärksten im Widerspruch zu deinen Werten?"

━━━━━━━━━━━━━━━━━━━━━━━
WICHTIGSTE REGEL
━━━━━━━━━━━━━━━━━━━━━━━

Stelle KEINE ähnliche Frage wie zuvor.
Jede Frage muss einen neuen Blickwinkel öffnen oder tiefer in ein Thema gehen.

Wenn du unsicher bist, ob du den User richtig verstanden hast:
→ stelle eine konkrete Nachfrage, bevor du das Thema wechselst.`;

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

  // ── 1. QUESTION COUNTER ──────────────────────────────────────
  const questionCount = messages.filter(m => m.role === "assistant").length;

  // ── 4. HARD STOP ─────────────────────────────────────────────
  // Set to 10 so Q9's answer still receives a reflection before the pre-analysis screen
  if (questionCount >= 10) {
    log(endpoint, { reqId, info: "hard stop — questionCount >= 10" });
    return res.json({ done: true });
  }

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

  // ── 2. PHASE STEERING ────────────────────────────────────────
  const phaseHint = {
    role: "system",
    content: `
CURRENT QUESTION: ${questionCount}
COMPASS-PHASEN (VERBINDLICH):
1–3 — WERTE & MISSION
→ Was ist wichtig? Wofür steht dieser Mensch? Lebensphilosophie, Mission.
→ Fragen: offen, einladend, konkret.
4–5 — PERSÖNLICHKEIT & ENERGIE
→ Spark-Zone, Drain-Zone, Entscheidungsstil, Rolle in Gruppen.
→ Fragen: neugierig, beobachtend, konkret.
6–7 — ZIELE & CHANCEN
→ Was will er? Was übersieht er? Kurze und lange Ziele.
→ Fragen: direkt, erforschend.
8–9 — ALIGNMENT & WAHRHEIT
→ Wo stimmt das Leben mit den Werten überein — und wo nicht?
→ Kongruenz, blinde Flecken, tiefste Wahrheit.
→ Fragen: ruhig, leicht konfrontierend.
HARTE REGEL:
Ähnliche Frage wie zuvor → FALSCH
Frage passt nicht zur aktuellen Phase → FALSCH
`,
  };

  // ── 3. REPETITION GUARD ───────────────────────────────────────
  const lastQuestions = messages
    .filter(m => m.role === "assistant")
    .slice(-5)
    .map(m => {
      try {
        const clean  = m.content.replace(/```json|```/g, "").trim();
        const parsed = JSON.parse(clean);
        return parsed.question || m.content;
      } catch {
        return m.content;
      }
    })
    .join("\n");

  const repetitionGuard = {
    role: "system",
    content: `
WIEDERHOLUNGEN VERMEIDEN:
Diese Fragen wurden bereits gestellt:
${lastQuestions}
Stelle KEINE ähnlichen oder leicht umformulierten Fragen.
Jede neue Frage muss:
- einen neuen Blickwinkel einführen
- tiefer gehen ODER die Perspektive wechseln
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
  log(endpoint, { reqId, messages: messages.length, questionCount });

  try {

  let text;
  for (let attempt = 0; attempt < 2; attempt++) {
  text = await callOpenAI([
  { role: "system", content: COMPASS_CONVERSATION_PROMPT },
  phaseHint,
  repetitionGuard,
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
Du erhältst ein JSON-Objekt mit einem "answers"-Array (bis zu 18 Antworten).

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

REGEL 6 — RICHTUNG (NEU):
Formuliere zusätzlich eine klare Entwicklungsrichtung:

→ "Du entwickelst dich weiter, wenn du X statt Y priorisierst."

Max 18 Wörter.
Muss direkt aus dem Typ + Pattern entstehen.

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
  "direction":       "<1 klarer Satz, wohin sich die Person entwickeln sollte>"
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
You are not a coach.
You are not a therapist.
You are a clarity engine.

Your job is to compress a messy human conversation into a sharp, confronting and directional insight.

The user has already reflected deeply.
Do NOT repeat obvious insights.
Do NOT ask questions.
Do NOT explain.

Your output must feel like:
- recognition
- slight discomfort
- direction

━━━━━━━━━━━━━━━━━━━━━━━
INPUT
━━━━━━━━━━━━━━━━━━━━━━━
You receive a conversation transcript of a user exploring a personal tension.

━━━━━━━━━━━━━━━━━━━━━━━
YOUR TASK
━━━━━━━━━━━━━━━━━━━━━━━
Extract and produce:

1. CORE TRUTH
A sharp, confronting statement about what is really happening.
Max 2 sentences.
No soft language.

2. IDENTITY
Who the person actually is when at their best.
1–2 sentences.

3. DRIVER
What truly drives them.
1–2 sentences.

4. PATTERN
The behavior that keeps them stuck.
1–2 sentences.

5. PATH
What actually works for them.
1–2 sentences.

6. COMPASS
A short, memorable directional statement.
Max 2 sentences.

7. VISUAL SIGNAL
Choose dominant and suppressed from:
["security","visibility","perfection","freedom","control","expression"]

━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT (JSON ONLY)
━━━━━━━━━━━━━━━━━━━━━━━
{
  "coreTruth": "",
  "identity": "",
  "driver": "",
  "pattern": "",
  "path": "",
  "compass": "",
  "visual": {
    "dominant": "",
    "suppressed": ""
  }
}

No text before or after the JSON.
`;

// ── Transcript builder ────────────────────────────────────────
// Converts the messages array into a clean, readable conversation log
function buildTranscript(messages) {
  return messages
    .filter(m => m.content && m.content.trim())
    .map(m => `${m.role === "user" ? "User" : "Assistant"}: ${m.content}`)
    .join("\n");
}

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

  // Build clean transcript once — used across all three pipeline steps
  const transcript = buildTranscript(messages);
  const transcriptMsg = { role: "user", content: `TRANSCRIPT:\n${transcript}` };

  try {
    // Step 1: Signal extraction (graceful fallback)
    let signalsBlock = null;
    try {
      const raw = await callOpenAI([
        { role: "system", content: SIGNAL_EXTRACTION_PROMPT },
        transcriptMsg,
      ], { jsonMode: true });
      signalsBlock = raw;
    } catch (e) {
      log(endpoint, { reqId, warning: `signal extraction failed: ${e.message}` });
    }

    // Step 2: Full insight generation
    const analysisMessages = signalsBlock
      ? [
          { role: "system", content: ANALYSIS_SYSTEM_PROMPT },
          transcriptMsg,
          { role: "user", content: `Extrahierte Signale:\n${signalsBlock}` },
        ]
      : [
          { role: "system", content: ANALYSIS_SYSTEM_PROMPT },
          transcriptMsg,
        ];

    const raw     = await callOpenAI(analysisMessages, { jsonMode: true });
    const parsed  = JSON.parse(raw);

    // Step 3: Generate result — clarity engine only, no diluting context
    let result = null;
    for (let attempt = 0; attempt < 2; attempt++) {
      try {
        const resultMessages = [
          { role: "system", content: CLARITY_RESULT_PROMPT },
          transcriptMsg,
        ];
        const resultRaw    = await callOpenAI(resultMessages, { jsonMode: true, temperature: 0.6, maxTokens: 800 });
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
        coreTruth: parsed.summary || "Etwas passt hier noch nicht zusammen.",
        identity:  "Du bist jemand, der mehr weiß, als du gerade zulässt.",
        driver:    parsed.pattern || "Ein Muster zeigt sich in deinen Antworten.",
        pattern:   "Du kreist — aber gehst noch nicht rein.",
        path:      "Weniger Optionen. Mehr Entscheidungen.",
        compass:   "Du weißt es bereits.",
        visual:    { dominant: "security", suppressed: "freedom" },
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
   /api/compass — CLARITY & FREEDOM COMPASS GENERATOR
   ─────────────────────────────────────────────────────────────
   Receives the full conversation transcript (messages array).
   Returns a single structured Compass JSON with 12 sections.
   One AI call. One retry on low quality. No pipeline overhead.
───────────────────────────────────────────────────────────── */

const COMPASS_GENERATION_PROMPT = `Du bist CLARITY — eine reflektive Intelligenz, kein Coach.

Du erhältst das vollständige Transkript eines geführten Gesprächs zwischen einer KI und einem Nutzer.

Deine Aufgabe:
Synthetisiere aus diesem Gespräch einen personalisierten "Clarity & Freedom Compass" mit exakt 12 Sektionen.

━━━━━━━━━━━━━━━━━━━━━━━
DEINE ROLLE
━━━━━━━━━━━━━━━━━━━━━━━

Du bist kein Therapeut. Du gibst keine Ratschläge. Du motivierst nicht.
Du machst sichtbar, was bereits da ist — auch wenn es unangenehm ist.

Jede Aussage muss:
- direkt aus dem Gespräch ableitbar sein (evidenzbasiert, nicht erfunden)
- spezifisch für DIESEN Menschen sein — kein generischer Content
- Spannungen beschreiben, nicht erklären oder motivieren

━━━━━━━━━━━━━━━━━━━━━━━
COPYWRITING-REGELN (VERBINDLICH)
━━━━━━━━━━━━━━━━━━━━━━━

SCHLECHT: "Du überdenkst alles."
GUT:      "Du analysierst weiter, lange nachdem die Entscheidung schon emotional ist."

SCHLECHT: "Du vermeidest Sichtbarkeit aus Angst vor Feedback."
GUT:      "Sichtbarkeit bedeutet für dich Messbarkeit."

VERBOTEN:
- "unlock your potential", "become your best self", "maximize", "transform yourself"
- "du solltest", "versuche", "hier ist wie", jede Ratschlags-Sprache
- Motivationsfloskeln und Coaching-Sprech

PFLICHT für unique_truth, blind_spots, hidden_opportunities:
→ Sätze, die sich leicht unangenehm anfühlen, aber wahr sind.
→ Der Nutzer soll innehalten — nicht motiviert, sondern erkannt werden.

━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT (JSON ONLY)
━━━━━━━━━━━━━━━━━━━━━━━

Antworte NUR mit validem JSON. Kein Markdown. Kein Text davor oder danach.

{
  "core_values": ["<Wert 1>", "<Wert 2>", "<3–8 Kernwerte des Nutzers>"],
  "beliefs": ["<Überzeugungssatz 1>", "<Überzeugungssatz 2>", "<1–3 Sätze, die beschreiben, woran der Nutzer glaubt>"],
  "life_philosophy": "<1–2 Sätze: Warum ist dieser Mensch hier? Wie will er wirken?>",
  "mission_statement": "<1 direkter, unvermeidbarer Satz>",
  "personality_snapshot": {
    "style_primary": "<z.B. D – Dominant / strategisch, entscheidungsstark>",
    "style_secondary": "<z.B. C – Gewissenhaft / präzise, analytisch>",
    "strengths": ["<Stärke 1>", "<Stärke 2>", "<3–6 konkrete Stärken aus dem Gespräch>"],
    "leadership_sentence": "<1 Satz zum Führungs- oder Wirkungsstil>"
  },
  "zone_of_genius": ["<Du bist am stärksten, wenn du ... (3–5 Punkte)>"],
  "zone_of_spark": "<1 Absatz: wie sich die natürliche Stärke und Spark-Zone anfühlt>",
  "spark_zone": ["<Tätigkeit/Umfeld/Ritual das Energie gibt (4–7 Punkte)>"],
  "drain_zone": ["<Tätigkeit/Umfeld die Energie kostet (3–5 Punkte)>"],
  "goals_short_term": ["<Ziel 0–12 Monate (3–5 Punkte)>"],
  "goals_long_term": ["<Vision 3–10 Jahre (3–5 Punkte)>"],
  "core_driver": "<1–2 Sätze: Was treibt diesen Menschen wirklich an?>",
  "hidden_opportunities": ["<Übersehener Hebel 1>", "<3–5 Chancen, die der Nutzer gerade nicht sieht>"],
  "blind_spots": ["<Blinder Fleck 1>", "<3–5 ehrliche, nicht beschämende Punkte>"],
  "unique_truth": "<1–3 Sätze im Schein-vs-Wahrheit-Muster — leicht unangenehm, spezifisch, wahr>",
  "alignment_ok": ["<Bereich der bereits stimmt (3–5 Punkte)>"],
  "alignment_tension": ["<Bereich mit Reibung zwischen Werten und Realität (2–4 Punkte)>"],
  "alignment_direction": "<1–2 Sätze: Was verlangt die Entwicklung dieses Menschen jetzt?>",
  "compass_statement": "<2–4 Sätze: verbindet Werte + Genius-Zone + Vermeidungen + Richtung>",
  "compass_avoid": "<1 Satz: Was soll dieser Mensch vermeiden?>",
  "compass_direction": "<1 Satz: Wohin zeigt sein Kompass?>",
  "habit_stack_10_days": {
    "body": ["<Körper-Habit 1>", "<Körper-Habit 2>"],
    "mind": ["<Geist/Klarheit-Habit 1>", "<Geist/Klarheit-Habit 2>"],
    "focus": ["<Fokus/Arbeit-Habit 1>", "<Fokus/Arbeit-Habit 2>"],
    "relationships": ["<Beziehungs-Habit 1>", "<Beziehungs-Habit 2>"],
    "money": ["<Geld/Struktur-Habit 1>", "<Geld/Struktur-Habit 2>"]
  }
}

━━━━━━━━━━━━━━━━━━━━━━━
PFLICHTREGELN VOR DEM OUTPUT
━━━━━━━━━━━━━━━━━━━━━━━

1. unique_truth: Fühlt sich das wie eine echte Erkenntnis an — oder wie eine Beschreibung? Wenn Beschreibung: neu schreiben.
2. blind_spots: Sind das echte blinde Flecken dieses Menschen — oder könnten sie auf jeden zutreffen? Wenn generisch: neu schreiben.
3. habit_stack_10_days: Sind die Habits aus dem individuellen Compass abgeleitet — oder generische Tipps? Wenn generisch: neu schreiben.
4. compass_statement: Enthält es Werte + Genius + Vermeidung + Richtung? Wenn nicht vollständig: ergänzen.
5. Alle Felder müssen befüllt sein. Kein Feld darf leer, null oder ein leeres Array sein.`;

function isLowQualityCompass(compass) {
  try {
    const required = [
      "core_values", "beliefs", "life_philosophy", "mission_statement",
      "personality_snapshot", "zone_of_genius", "zone_of_spark",
      "spark_zone", "drain_zone", "goals_short_term", "goals_long_term",
      "core_driver", "hidden_opportunities", "blind_spots",
      "unique_truth", "alignment_ok", "alignment_tension", "alignment_direction",
      "compass_statement", "compass_avoid", "compass_direction",
      "habit_stack_10_days",
    ];

    for (const field of required) {
      if (compass[field] === undefined || compass[field] === null) return true;
      if (Array.isArray(compass[field]) && compass[field].length === 0) return true;
      if (typeof compass[field] === "string" && compass[field].trim().length < 5) return true;
    }

    const ps = compass.personality_snapshot;
    if (!ps || !ps.style_primary || !Array.isArray(ps.strengths) || !ps.leadership_sentence) return true;
    if (ps.strengths.length === 0) return true;

    const hs = compass.habit_stack_10_days;
    if (!hs || !hs.body || !hs.mind || !hs.focus || !hs.relationships || !hs.money) return true;
    if ([hs.body, hs.mind, hs.focus, hs.relationships, hs.money].some(a => !Array.isArray(a) || a.length === 0)) return true;

    if (compass.unique_truth.trim().length < 40) return true;
    if (compass.compass_statement.trim().length < 60) return true;

    return false;
  } catch {
    return true;
  }
}

app.post("/api/compass", compassLimiter, async (req, res) => {
  const endpoint = "POST /api/compass";
  const reqId    = makeReqId();

  const invalid = validateMessages(req.body);
  if (invalid) {
    log(endpoint, { reqId, error: invalid.code });
    return res.status(400).json({ error: invalid.code });
  }

  const { messages } = req.body;
  log(endpoint, { reqId, messages: messages.length });
  const t0 = Date.now();

  const transcript    = buildTranscript(messages);
  const transcriptMsg = {
    role:    "user",
    content: `GESPRÄCHS-TRANSKRIPT:\n\n${transcript}`,
  };

  try {
    let compass = null;

    for (let attempt = 0; attempt < 2; attempt++) {
      try {
        const retryHint = attempt === 1
          ? [{
              role:    "system",
              content: `Überprüfe dein JSON vor der Ausgabe:
- Sind ALLE Felder befüllt (kein leeres Array, kein leerer String)?
- Ist unique_truth spezifisch und leicht unangenehm (nicht generisch)?
- Ist habit_stack_10_days individuell auf diesen Menschen zugeschnitten?
- Enthält compass_statement Werte + Genius + Vermeidung + Richtung?
Korrigiere und gib das vollständige, valide JSON erneut aus.`,
            }]
          : [];

        const raw    = await callOpenAI(
          [
            { role: "system", content: COMPASS_GENERATION_PROMPT },
            transcriptMsg,
            ...retryHint,
          ],
          { jsonMode: true, maxTokens: 4000, retries: 0 }
        );
        const parsed = JSON.parse(raw);

        if (!isLowQualityCompass(parsed)) {
          compass = parsed;
          break;
        }

        if (attempt === 0) console.log("⚠️ Low quality compass → retrying...");

      } catch (e) {
        log(endpoint, { reqId, warning: `compass attempt ${attempt} failed: ${e.message}` });
        if (attempt === 1) throw e;
      }
    }

    if (!compass) {
      log(endpoint, { reqId, error: "COMPASS_GENERATION_FAILED" });
      return res.status(500).json({ error: "COMPASS_GENERATION_FAILED" });
    }

    const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
    log(endpoint, { reqId, elapsed: `${elapsed}s` });

    res.json(compass);

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
  "question_13", "question_14", "question_15", "question_16",
  "question_17", "question_18",
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
