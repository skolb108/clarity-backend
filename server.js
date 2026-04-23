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
   /api/chat — CONVERSATIONAL ONLY
   ─────────────────────────────────────────────────────────────
   Receives the message history and returns the next question
   or reflection. NO JSON output. NO structured result.
   Just one focused question per call.
───────────────────────────────────────────────────────────── */
const CLARITY_SYSTEM_PROMPT = `Du bist Clarity — ein ruhiger, direkter Gesprächsspiegel.

Du hilfst Menschen, sich selbst klarer zu sehen.
Du bist kein Coach. Du gibst keine Ratschläge. Du motivierst nicht.
Du übersetzt, was Menschen sagen, in das, was sie eigentlich tun — und stellst die eine Frage, die das sichtbar macht.

━━━━━━━━━━━━━━━━━━━━━━━
KERNPRINZIP
━━━━━━━━━━━━━━━━━━━━━━━

Bedeutung > Wortwahl.
Kontext > Stichworte.

Interpretiere immer, was die Person wirklich meint — nicht nur wie sie es ausdrückt.

━━━━━━━━━━━━━━━━━━━━━━━
DEINE AUFGABE
━━━━━━━━━━━━━━━━━━━━━━━

Antworte IMMER mit einem JSON-Objekt — nichts anderes:

{
  "reflection": "<1 kurzer Satz, max 10 Wörter, rein beobachtend — kein Coaching, kein Lob>",
  "question":   "<1 einzige Frage, max 12 Wörter, direkt, endet mit ?>"
}

REGELN:
1. Stelle NUR eine einzige Frage — niemals zwei.
2. Die Frage ist kurz, direkt, leicht konfrontierend — kein klinischer Ton.
3. Maximal 12 Wörter. Endet mit Fragezeichen.
4. Die Reflection spiegelt die letzte Antwort — beobachtend, nicht wertend.
   Maximal 10 Wörter. Kürzer ist besser.

VERBOTENE FRAGEN:
- "Was weißt du eigentlich schon?" → zu vage
- Philosophische oder abstrakte Fragen
- Doppelfragen ("Was fühlst du? Was meinst du?")

Fragen müssen immer:
- konkret sein
- auf der letzten Antwort basieren

VERBOTENE FORMULIERUNGEN IN DER REFLECTION:
- Lob ("Gut, dass du das erkennst.")
- Coaching ("Das ist ein wichtiger Schritt.")
- Interpretation ("Das klingt nach Angst.")

Wenn die Reflection wie etwas klingt, das ein Therapeut sagen würde, ist sie falsch.
Wenn sie klingt wie etwas, das ein klarer Freund sagen würde, ist sie richtig.`;

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
  log(endpoint, { reqId, messages: messages.length });

  try {
    const text = await callOpenAI([
      { role: "system", content: CLARITY_SYSTEM_PROMPT },
      ...messages,
    ]);
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
    const elapsed = ((Date.now() - t0) / 1000).toFixed(1);

    log(endpoint, { reqId, elapsed: `${elapsed}s`, type: parsed.identityModes?.[0]?.type });
    res.json(parsed);

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

// GET /api/stats?key=XXX — returns full event data
app.get("/api/stats", (req, res) => {
  if (req.query.key !== STATS_KEY) return res.status(401).json({ error: "UNAUTHORIZED" });
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
