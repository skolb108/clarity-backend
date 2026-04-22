import express from "express";
import cors    from "cors";
import dotenv  from "dotenv";
import OpenAI  from "openai";

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
   CORS + JSON
───────────────────────────────────────────────────────────── */
app.use(cors({
  origin:         "*",
  methods:        ["GET", "POST", "OPTIONS"],
  allowedHeaders: ["Content-Type"],
}));
app.use(express.json());

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

app.post("/api/chat", async (req, res) => {
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

━━━━━━━━━━━━━━━━━━━━━━━
SCHRITT 1: TYP-ERKENNUNG
━━━━━━━━━━━━━━━━━━━━━━━

Ordne den User exakt EINEM dieser 5 Typen zu:

EXPLORER  — Sammelt Optionen statt zu entscheiden. Schlüsselwörter: vielleicht, könnte, wenn, noch nicht sicher.
BUILDER   — Handelt viel, hinterfragt das Wohin selten. Schlüsselwörter: vorankommen, umsetzen, optimieren.
CREATOR   — Starker Drang zu erschaffen, aber Angst vor Sichtbarkeit. Schlüsselwörter: Ideen, zeigen, noch nicht bereit.
OPTIMIZER — Sieht sofort was nicht stimmt. Hohe Standards, schwer zufrieden. Schlüsselwörter: besser werden, nicht gut genug.
DRIFTER   — Bewegt sich ohne Richtung. Schlüsselwörter: mal schauen, irgendwie, passiert halt.

Confidence: 40–65 = unsicher, 66–80 = klar, 81–95 = eindeutig.

━━━━━━━━━━━━━━━━━━━━━━━
SCHRITT 2: INSIGHT GENERIEREN
━━━━━━━━━━━━━━━━━━━━━━━

summary:
  Konfrontiert — beschreibt nicht.
  Formel: "Du [konkretes Verhalten] — nicht weil [Ausrede], sondern weil [Wahrheit]."
  Darf sich unangenehm anfühlen. Soll sich wahr anfühlen.
  Max 18 Wörter. Direkte du-Ansprache.

pattern:
  Ein KONKRETES, WIEDERHOLENDES Muster aus den echten Antworten.
  Beginnt mit "Du hast mehrfach..." oder "In deinen Antworten taucht auf..."
  Keine Generalaussagen. Max 20 Wörter.

suggestedAction:
  Eine konkrete Handlung für HEUTE.
  Formulierung: "Tu X — nicht Y." Max 12 Wörter.
  Kein Coaching-Sprech.

strengths, energySources:
  Nur aus den echten Antworten ableiten. Nie erfinden.

━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━

{
  "scores": {
    "Clarity":   <integer 1–100>,
    "Energy":    <integer 1–100>,
    "Strength":  <integer 1–100>,
    "Direction": <integer 1–100>,
    "Action":    <integer 1–100>
  },
  "identityModes": [
    { "type": "<Explorer|Builder|Creator|Optimizer|Drifter>", "confidence": <integer 40–95> }
  ],
  "summary":         "<1 Satz, max 18 Wörter, Konfrontations-Formel>",
  "pattern":         "<1 Satz, max 20 Wörter, beginnt mit 'Du hast mehrfach...' oder 'In deinen Antworten...'>",
  "strengths":       ["<konkret aus Antworten>", "<konkret>", "<konkret>"],
  "energySources":   ["<konkret aus Antworten>", "<konkret>", "<konkret>"],
  "nextFocus":       "<1 Satz — wichtigster Fokus nächste 30 Tage>",
  "suggestedAction": "<1 konkreter Schritt heute, max 12 Wörter>"
}

Scores: Basieren auf konkreten Hinweisen. Vermeide runde 10er-Schritte.
identityModes: Immer nur 1 Typ — außer beide liegen ≥55 Confidence.`;

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

app.post("/api/analyze", async (req, res) => {
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
   Error handling
───────────────────────────────────────────────────────────── */
process.on("unhandledRejection", (reason) => {
  console.error("[unhandledRejection]", reason);
});

const PORT = process.env.PORT || 8080;
app.listen(PORT, "0.0.0.0", () => {
  console.log(`[startup] Clarity backend running on port ${PORT}`);
});
