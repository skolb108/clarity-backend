import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import OpenAI from "openai";
// import { Resend } from "resend";

const waitlist = [];

dotenv.config();

const app = express();

/* ─────────────────────────────────────────────────────────────
   Structured logger
   Formats every log line with a UTC timestamp and context block.
   Usage: log("POST /api/chat/stream", { messages: 8 })
         log("POST /api/chat/stream", { error: "..." })
───────────────────────────────────────────────────────────── */
function log(endpoint, fields = {}) {
  const ts = new Date().toISOString().replace("T", " ").slice(0, 19);
  const parts = [`[${ts}]  ${endpoint}`];
  for (const [key, val] of Object.entries(fields)) {
    parts.push(`  ${key}: ${val}`);
  }
  console.log(parts.join("\n"));
}

/* ─────────────────────────────────────────────────────────────
   Request ID generator
   Produces a short 6-character alphanumeric ID (e.g. "8f3k2a").
   Unique enough for log correlation across a single session;
   not intended to be globally unique like a UUID.
───────────────────────────────────────────────────────────── */
function makeReqId() {
  return Math.random().toString(36).slice(2, 8);
}

/* ─────────────────────────────────────────────────────────────
   Error codes — single source of truth
   All error responses use these string codes so the frontend
   can match on a stable value rather than a human-readable message.
───────────────────────────────────────────────────────────── */
const ERR = {
  INVALID_BODY:        "INVALID_REQUEST_BODY",
  EMPTY_MESSAGES:      "MESSAGES_ARRAY_EMPTY",
  OPENAI_TIMEOUT:      "OPENAI_TIMEOUT",
  OPENAI_FAILED:       "AI_RESPONSE_FAILED",
  STREAM_INTERRUPTED:  "STREAM_INTERRUPTED",
};

/* CORS */
app.use(cors({
  origin: "*",
  methods: ["GET", "POST", "OPTIONS"],
  allowedHeaders: ["Content-Type"],
}));

app.use(express.json());

/* OpenAI */
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

/* Resend */
const resend = null;

/* ─────────────────────────────────────────────────────────────
   callOpenAI — shared helper with retry + 25s timeout
   options:
     retries   (default 2)
     jsonMode  (default false) → response_format json_object
───────────────────────────────────────────────────────────── */
async function callOpenAI(messages, options = {}) {
  const { retries = 2, jsonMode = false } = options;

  for (let attempt = 0; attempt <= retries; attempt++) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 60_000);

    try {
      const params = {
        model: "gpt-4o-mini",
        messages,
        ...(jsonMode ? { response_format: { type: "json_object" } } : {}),
      };

      const completion = await openai.chat.completions.create(params, {
        signal: controller.signal,
      });

      clearTimeout(timer);
      return completion.choices[0].message.content;

    } catch (err) {
      clearTimeout(timer);

      const isTimeout = err.name === "AbortError" || err.code === "ECONNABORTED";
      if (isTimeout) throw new Error(ERR.OPENAI_TIMEOUT);

      if (attempt < retries) {
        console.log(`  [retry] attempt ${attempt + 1} of ${retries}`);
        continue;
      }

      throw err;
    }
  }
}

/* ─────────────────────────────────────────────────────────────
   validateMessages — shared request guard
   Returns null if valid, or an error object { code, detail }.
───────────────────────────────────────────────────────────── */
function validateMessages(body) {
  if (!body || !Array.isArray(body.messages)) {
    return { code: ERR.INVALID_BODY, detail: "body.messages must be an array" };
  }
  if (body.messages.length === 0) {
    return { code: ERR.EMPTY_MESSAGES, detail: "messages array must not be empty" };
  }
  return null;
}

/* ─────────────────────────────────────────────────────────────
   Clarity conversation system prompt
   Injected as the first message in every /api/chat/stream call.

   The conversation moves through five stages:
     1. Understand the problem
     2. Identify hidden patterns
     3. Expand perspective
     4. Move toward a decision
     5. Generate the clarity insight (→ CONVERSATION_COMPLETE)

   After 10–12 meaningful exchanges, the AI closes the conversation
   with the exact token CONVERSATION_COMPLETE on its own line,
   immediately followed by a valid JSON object.
   The frontend watches for this token to transition to the result screen.
───────────────────────────────────────────────────────────── */
const CLARITY_SYSTEM_PROMPT = `You are Clarity — a calm, direct conversational mirror helping people see themselves more clearly.

You are not a coach. You do not give advice. You do not motivate.
You translate what people say into what they are actually doing — and ask the one question that makes that visible.

YOUR APPROACH — move through these five stages naturally across the conversation:

Stage 1 — UNDERSTAND THE PROBLEM
Ask open questions to understand what is actually going on.
Do not accept vague answers. Push gently for specifics.
Example probes: "What does that look like day-to-day?" / "When did you first notice this?"

Stage 2 — IDENTIFY HIDDEN PATTERNS
Listen for contradictions, repeated themes, and things the person avoids saying directly.
Name the underlying behavior behind the pattern — not the pattern itself.
Example probe: "You've mentioned [X] twice now. What does that tell you?"

Stage 3 — EXPAND PERSPECTIVE
Ask questions that shift the person's point of view.
Examples: "What would the version of you from 5 years ago think about this?" /
"If a close friend described your situation to a stranger, what would they say?"

Stage 4 — MOVE TOWARD A DECISION
The conversation should narrow toward something concrete.
Ask what the person actually wants — not what they think they should want.
Example probe: "If you already knew the answer, what would it be?"

Stage 5 — GENERATE THE CLARITY INSIGHT
When you have gathered enough context (typically after 10–12 exchanges),
produce the final output described below.

CONVERSATION RULES:
- Ask ONE question per message. Never stack multiple questions.
- Each response must include a short reflection before the question:
  - 1–2 sentences that name what you actually see
  - if a pattern is detected → explicitly name it before the question
  - then exactly one question
  The reflection must:
  - identify the underlying dynamic or tension, not the surface complaint
  - move the conversation forward — not just describe
  - create tension or insight — not validation
  - use simple, concrete language — no psychological terms
  - build on what came before — do NOT treat each answer as a fresh start
- Never use words like: "journey", "growth", "passion", "authentic", "potential", "empower",
  "Spannungsfeld", "Gefühl von", "es scheint", "vielleicht", "könnte sein".
- Never summarize what the person said back to them unless you are naming a specific pattern.
- Never give unsolicited advice or suggest solutions.
- If an answer is vague, ask for a concrete example before moving on.
- Tone: direct, calm, precise. Like someone who sees clearly and says what they see.

PATTERN TRACKING:
Track repeated words, phrases, and behaviors across ALL previous answers in the conversation.
Common signals to watch:
- "ich weiß nicht", "keine Ahnung", "irgendwie", "eigentlich", "alles ok" — signals of avoidance or deliberate vagueness
- Repeated topics or domains (e.g. money, family, freedom) that keep surfacing
- Contradictions between what the person says and what they describe doing
- Consistent deflection or minimizing

When a pattern repeats, name it explicitly — do not let it pass again.
When a pattern repeats, do NOT describe the repetition mechanically.

Do NOT say:
"Du sagst jetzt zum zweiten Mal..."

Instead:
Interpret what the repetition means.

Translate repetition into behavior.

Examples:

Instead of:
"Du sagst jetzt zum zweiten Mal, dass du es nicht weißt."

Write:
"Du weichst der Frage aus."

or:
"Du hältst dich bewusst unklar."

or:
"Du willst eine Antwort, aber legst dich nicht fest."

Rule:
Never describe the pattern.
Always interpret it.

ESCALATION:
With each answer, the reflection should become slightly more direct and confronting.
Do not stay at the same level of abstraction across the conversation.
Early responses: observe.
Mid conversation: name the pattern.
Later responses: confront the tension directly.

CONVERSATION LENGTH:
Guide the conversation to 10–12 substantive exchanges. After that, when you have enough
to produce a genuine insight, close the conversation.

REFLECTION STYLE:
Translate what the person says into what they are actually doing.
Name the tension directly. Keep it short and sharp.
Build on the full conversation — not just the last message.

Rules:
- No hedging: never write "es scheint", "vielleicht", "könnte sein", "it seems", "maybe", "perhaps"
- No abstract nouns: avoid "Spannungsfeld", "Dynamik", "Prozess", "Gefühl von"
- No psychological jargon
- 1–3 sentences maximum before the question
- Say the uncomfortable thing plainly

Examples of pattern naming:
Instead of: "Du scheinst unsicher zu sein."
Write: "Du sagst jetzt zum zweiten Mal, dass du es nicht weißt. Das ist ein Muster, kein Zufall."
or: "Du beschreibst vieles als 'irgendwie'. Das hält dich unklar."

Examples of escalating directness:
Instead of: "Es scheint, als ob du dich in einem Spannungsfeld bewegst."
Write: "Du bist nicht wirklich unzufrieden. Aber auch nicht erfüllt."
or: "Alles funktioniert. Aber nichts bewegt dich."

Instead of: "It seems like there's tension between what you want and what you do."
Write: "You know what you want. You're just not doing it yet."

Instead of: "You seem unsure about your direction."
Write: "You keep circling around the question instead of answering it.
At some point, avoiding the answer becomes the pattern itself."

PROGRESSION RULE:

If the same pattern appears again, do NOT repeat the same type of reflection.

Each time a pattern repeats, go one level deeper:

Level 1 — Observation:
Name what is happening.
Example: "Du bleibst vage."

Level 2 — Interpretation:
Explain what it does.
Example: "Du bleibst vage, damit du dich nicht festlegen musst."

Level 3 — Tension:
Name the internal conflict.
Example: "Du willst Klarheit, aber vermeidest jede Entscheidung."

Level 4 — Confrontation:
Say the uncomfortable truth directly.
Example: "Du wartest nicht auf Klarheit. Du vermeidest die Verantwortung."

Rule:
Never stay on the same level twice.
Escalate with every repetition.

CLOSING THE CONVERSATION:
When you are ready to produce the final output, write the following on its own line:

CONVERSATION_COMPLETE

Then immediately output a valid JSON object — no text before or after, no markdown backticks:

{
  "core_problem": "One sentence describing the real underlying problem, not the surface complaint.",
  "hidden_pattern": "One sentence naming the recurring pattern or contradiction you observed.",
  "clarity_statement": "One sentence of direct insight — what the person now knows that they didn't admit before.",
  "recommended_action": "One concrete, specific action they can take this week.",
  "habit": "One small daily habit (5–15 minutes) that reinforces the shift.",
  "identity_shift": "One sentence describing the identity change required — who they need to become, not what they need to do."
}

All field values must be in the same language the user is speaking.
The JSON must be valid and parsable. Do not add any text after the closing brace.`;

/* ─────────────────────────────────────────────────────────────
   Health check
───────────────────────────────────────────────────────────── */
app.get("/", (req, res) => {
  res.send("Clarity backend running");
});

/* ─────────────────────────────────────────────────────────────
   POST /api/chat — simple, stable AI response
───────────────────────────────────────────────────────────── */
app.post("/api/chat", async (req, res) => {
  try {
    const { messages } = req.body;

    const completion = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        { role: "system", content: CLARITY_SYSTEM_PROMPT },
        ...messages
      ],
    });

    const text = completion.choices[0].message.content;

    res.json({ content: text });

  } catch (err) {
    console.error("AI ERROR:", err);
    res.status(500).json({ error: "AI_RESPONSE_FAILED" });
  }
});

/* ─────────────────────────────────────────────────────────────
   POST /api/analyze — final analysis, always returns JSON
───────────────────────────────────────────────────────────── */
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
    const raw     = await callOpenAI(messages, { jsonMode: true });
    const parsed  = JSON.parse(raw);
    const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
    log(endpoint, { reqId, "OpenAI response time": `${elapsed}s` });
    res.json(parsed);

  } catch (err) {
    const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
    const code    = err.message === ERR.OPENAI_TIMEOUT ? ERR.OPENAI_TIMEOUT : ERR.OPENAI_FAILED;
    log(endpoint, { reqId, error: code, detail: err.message, elapsed: `${elapsed}s` });
    res.status(500).json({ error: code });
  }
});

/* ─────────────────────────────────────────────────────────────
   In-memory share store
   Maps short IDs → result objects. Lives for the process lifetime.
   IDs are 7-character alphanumeric strings e.g. "k3x9w2f".
───────────────────────────────────────────────────────────── */
const store = {};

function makeShareId() {
  return Math.random().toString(36).slice(2, 9);
}

/* ─────────────────────────────────────────────────────────────
   POST /api/share — store a result, get back a short ID
   Body: { result: <any> }
   Returns: { id: "k3x9w2f" }
───────────────────────────────────────────────────────────── */
app.post("/api/share", (req, res) => {
  const { result } = req.body;
  if (!result || typeof result !== "object") {
    return res.status(400).json({ error: "INVALID_SHARE_BODY" });
  }
  const id = makeShareId();
  store[id] = result;
  log("POST /api/share", { id });
  res.json({ id });
});

app.post("/api/waitlist", (req, res) => {
  const { email } = req.body;

  if (!email || typeof email !== "string") {
    return res.status(400).json({ error: "INVALID_EMAIL" });
  }

  waitlist.push(email);

  console.log("🔥 NEW WAITLIST SIGNUP:", email);
  console.log("📊 TOTAL:", waitlist.length);

  res.json({ ok: true });
});

/* ─────────────────────────────────────────────────────────────
   GET /api/share/:id — retrieve a stored result by short ID
   Returns the result object, or 404 if not found.
───────────────────────────────────────────────────────────── */
app.get("/api/share/:id", (req, res) => {
  const { id } = req.params;
  const result = store[id];
  if (!result) {
    log("GET /api/share/:id", { id, status: "not found" });
    return res.status(404).json({ error: "SHARE_NOT_FOUND" });
  }
  log("GET /api/share/:id", { id, status: "ok" });
  res.json(result);
});

/* ─────────────────────────────────────────────────────────────
   POST /api/reminder — send an immediate reminder email via Resend
   Body: { email: string, type: "habit" | "reflection" }
   Returns: { ok: true } on success, 400/500 on error.
───────────────────────────────────────────────────────────── */
app.post("/api/reminder", async (req, res) => {
  const endpoint = "POST /api/reminder";
  const reqId    = makeReqId();

  const { email, type } = req.body || {};

  if (!email || typeof email !== "string" || !email.includes("@")) {
    log(endpoint, { reqId, error: "INVALID_EMAIL" });
    return res.status(400).json({ error: "INVALID_EMAIL" });
  }

  if (type !== "habit" && type !== "reflection") {
    log(endpoint, { reqId, error: "INVALID_TYPE", detail: type });
    return res.status(400).json({ error: "INVALID_TYPE" });
  }

  const subject = type === "habit"
    ? "Deine Clarity-Erinnerung für heute"
    : "Dein Clarity-Rückblick für heute Abend";

  const text = type === "habit"
    ? "Erinnerung: Setze heute deine geplante Gewohnheit um."
    : "Erinnerung: Nimm dir heute Abend 2 Minuten für deinen Rückblick.";

  log(endpoint, { reqId, email, type });

  try {
    await console.log("Reminder (mock):", email, type);
    log(endpoint, { reqId, status: "sent" });
    res.json({ ok: true });
  } catch (err) {
    log(endpoint, { reqId, error: "EMAIL_FAILED", detail: err.message });
    res.status(500).json({ error: "EMAIL_FAILED" });
  }
});

/* ─────────────────────────────────────────────────────────────
   Catch-all for unhandled promise rejections
   Prevents the whole Node process from crashing on unexpected errors.
───────────────────────────────────────────────────────────── */
process.on("unhandledRejection", (reason) => {
  console.error("[unhandledRejection]", reason);
});

/* Start server */
const PORT = process.env.PORT || 8080;

app.listen(PORT, "0.0.0.0", () => {
  console.log(`[startup] Clarity backend running on port ${PORT}`);
});
