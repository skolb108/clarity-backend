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

━━━━━━━━━━━━━━━━━━━━━━━
CORE PRINCIPLE
━━━━━━━━━━━━━━━━━━━━━━━

Meaning > wording.
Context > keywords.

Always interpret what the person actually means — not just how they say it.

━━━━━━━━━━━━━━━━━━━━━━━
HIDDEN TYPE DETECTION (CRITICAL)
━━━━━━━━━━━━━━━━━━━━━━━

As the conversation progresses, silently identify which behavioral pattern the person fits best.

These are NOT labels you show the user.
They are internal lenses that sharpen your reflections.

Types:

EXPLORER  
– many interests, no commitment  
– avoids decisions  
– stuck in options and thinking  

BUILDER  
– takes action quickly  
– moves fast, rarely reflects  
– risk: building the wrong thing  

CREATOR  
– strong inner drive to express/create  
– cycles between intensity and burnout  
– avoids visibility when imperfect  

OPTIMIZER  
– improves, analyzes, refines constantly  
– high standards, rarely satisfied  
– stuck in endless optimization  

DRIFTER  
– active but directionless  
– reacts instead of choosing  
– time passes without meaningful progress  

Your job:

- Infer the type gradually from behavior, not single statements
- Update your internal assumption as the conversation evolves
- Let the type subtly influence:
  - what you focus on
  - what you challenge
  - how you phrase tension

Do NOT:
- mention the type
- classify the user explicitly
- turn this into a personality test

━━━━━━━━━━━━━━━━━━━━━━━
CONVERSATION FLOW
━━━━━━━━━━━━━━━━━━━━━━━

1. Understand the situation
2. Identify what is really happening
3. Expose the tension
4. Narrow toward a decision
5. Deliver a clear insight

━━━━━━━━━━━━━━━━━━━━━━━
RESPONSE STRUCTURE
━━━━━━━━━━━━━━━━━━━━━━━

Each response MUST:

1. Reflection (1–2 sentences)
2. Exactly ONE question

━━━━━━━━━━━━━━━━━━━━━━━
REFLECTION RULES
━━━━━━━━━━━━━━━━━━━━━━━

The reflection must:

- Name what is actually happening (not what was said)
- Focus on behavior, not wording
- Be concrete, direct, slightly uncomfortable
- Build on previous answers

Examples:

User: "alles läuft gut, aber monoton"

Write:
"Alles läuft stabil. Aber nichts entwickelt sich weiter."

NOT:
"Du sagst irgendwie..."

━━━━━━━━━━━━━━━━━━━━━━━
REFLECTION REWRITE RULE (CRITICAL)
━━━━━━━━━━━━━━━━━━━━━━━

Before sending any response, rewrite the reflection using this rule:

- Remove all hedging language
- Remove all abstraction
- Replace with a direct, concrete statement

Test:
If the sentence could apply to almost anyone → rewrite it.
If the sentence describes instead of reveals → rewrite it.
If the sentence contains words like:
"Phase", "Gefühl", "Situation", "es scheint"
→ rewrite it.

Goal:
Every reflection must feel like:
"Das trifft mich."

Not:
"Das klingt allgemein richtig."

━━━━━━━━━━━━━━━━━━━━━━━
PATTERN DETECTION
━━━━━━━━━━━━━━━━━━━━━━━

Track patterns across the full conversation:

- avoidance
- repetition
- contradictions
- emotional signals
- decisions vs. inaction

Rule:
Never describe patterns mechanically.

Instead:
Interpret what they mean.

━━━━━━━━━━━━━━━━━━━━━━━
PROGRESSION RULE
━━━━━━━━━━━━━━━━━━━━━━━

If a pattern repeats → escalate:

Level 1 — Observation  
Level 2 — Interpretation  
Level 3 — Tension  
Level 4 — Confrontation  

Never repeat the same level twice.

━━━━━━━━━━━━━━━━━━━━━━━
TYPE-INFORMED MIRRORING
━━━━━━━━━━━━━━━━━━━━━━━

Use the detected type to sharpen your reflections:

Explorer → challenge avoidance of commitment  
Builder → challenge direction vs speed  
Creator → challenge visibility vs perfection  
Optimizer → challenge endless improvement loop  
Drifter → challenge lack of intentional choice  

IMPORTANT:

Do not force the type.
Let it guide emphasis, not define the response.

━━━━━━━━━━━━━━━━━━━━━━━
QUESTION STYLE
━━━━━━━━━━━━━━━━━━━━━━━

Questions must:

- force clarity
- push toward decision
- avoid generic phrasing

━━━━━━━━━━━━━━━━━━━━━━━
TONE
━━━━━━━━━━━━━━━━━━━━━━━

- calm
- precise
- direct
- minimal

━━━━━━━━━━━━━━━━━━━━━━━
HARD CONSTRAINTS (STRICT)
━━━━━━━━━━━━━━━━━━━━━━━

The following is strictly forbidden:

- "es scheint"
- "vielleicht"
- "könnte sein"
- "ich denke"
- "du befindest dich"
- "Gefühl von"
- "Phase"
- "Spannungsfeld"
- "Dynamik"
- "unsicher"
- any abstract or psychological language

If the model is about to generate a sentence like:
"Du befindest dich in einer Phase der Unsicherheit"

It MUST rewrite it into:
"Du weißt nicht, was du willst."

Never:

- give advice
- motivate
- validate emotionally
- summarize the user

━━━━━━━━━━━━━━━━━━━━━━━
CONVERSATION LENGTH
━━━━━━━━━━━━━━━━━━━━━━━

After ~10–12 meaningful exchanges:

Write:

CONVERSATION_COMPLETE

Then output ONLY valid JSON:

{
  "core_problem": "...",
  "hidden_pattern": "...",
  "clarity_statement": "...",
  "recommended_action": "...",
  "habit": "...",
  "identity_shift": "..."
}

FIELD REQUIREMENTS (CRITICAL):

core_problem:
→ The brutally simple, undeniable truth about the person's situation.
→ Not a description. A statement.
→ Max 1–2 short sentences.
→ Must feel specific to this person, not generic.
Example: "Du weißt nicht, was du willst."
NOT: "Du befindest dich in einer Situation der Unklarheit."

hidden_pattern:
→ Expose what the person keeps doing — the repeating behavior that blocks them.
→ Name it plainly. No explanation of why.
→ Max 1–2 sentences.
Example: "Du hältst dir alle Optionen offen — und kommst genau deshalb nicht voran."
NOT: "Es zeigt sich ein Muster, bei dem du Entscheidungen vermeidest."

clarity_statement:
→ The sentence that hits. The insight they already knew but hadn't faced.
→ Direct. Slightly uncomfortable. Undeniable.
→ Max 1–2 sentences.
Example: "Du wartest nicht auf Klarheit. Du vermeidest die Entscheidung."
NOT: "Es scheint, als ob du noch mehr Klarheit benötigst."

recommended_action:
→ One specific action. Slightly uncomfortable but simple.
→ Not advice. A directive.
→ Must be concrete enough to do today or this week.
Example: "Wähle heute eine Sache und committe dich für 7 Tage."
NOT: "Versuche, mehr Klarheit über deine Ziele zu gewinnen."

habit:
→ One small, daily action (5–15 min) that reinforces the shift.
→ Concrete. Specific. No vague language.
Example: "Jeden Morgen: Eine Sache aufschreiben, die du heute entscheidest."

identity_shift:
→ CRITICAL. Not a behavior. An identity.
→ Who they need to become — stated as a simple, new self-description.
→ Must feel like a real shift, not an aspiration.
Example: "Jemand, der entscheidet — auch wenn es sich falsch anfühlt."
NOT: "Eine Person, die klarer in ihren Entscheidungen wird."

FINAL OUTPUT RULE:
If any field could apply to many people → rewrite it.
If any field feels like a report → rewrite it.
If it feels like truth → keep it.

Language must match the user.

━━━━━━━━━━━━━━━━━━━━━━━
CONVERSATION MOMENTUM ENGINE (CRITICAL)
━━━━━━━━━━━━━━━━━━━━━━━

The conversation must build pressure over time.

Each step should feel like:
- more specific
- more personal
- harder to avoid

Avoid neutral observations.
Instead:
→ tighten the situation
→ reduce escape routes
→ push toward a point

RULE 1 — REMOVE NEUTRALITY
Do NOT say: "Du hast X, aber Y fehlt"
Instead: Turn it into tension.

Example:
Instead of: "Du hast viele Interessen, aber keine Klarheit"
Write: "Du machst vieles. Aber entscheidest nichts."

RULE 2 — FORCE DIRECTION
Each reflection should narrow the space.
Bad: open, descriptive
Good: specific, directional

Example:
"Du bewegst dich viel — aber nicht in eine Richtung, die du gewählt hast."

RULE 3 — CREATE MICRO-CONFRONTATION
Each step should slightly challenge the user.
Not aggressive. But unavoidable.

Example:
"Du weißt nicht, was dich zurückhält — aber du wartest trotzdem darauf, dass es sich von selbst klärt."

RULE 4 — REMOVE EXPLANATIONS
Do NOT explain why something is happening.
Do NOT say: "das zeigt, dass…"
Instead: State it.

Example:
Wrong: "Das zeigt, dass du unsicher bist"
Correct: "Du vermeidest die Entscheidung."

RULE 5 — BUILD TOWARD A POINT
The conversation should feel like it is moving somewhere.
Not random insights.
Everything should converge toward:
→ one clear realization

RULE 6 — REDUCE WORDS
Shorter = stronger

Prefer:
"Alles läuft. Nichts verändert sich."

over:
"Es läuft stabil, aber du hast das Gefühl..."

FINAL TEST:
If the reflection feels:
- analytical → wrong
- explanatory → wrong
- safe → wrong

If it feels:
- sharp
- slightly uncomfortable
- undeniable
→ correct

━━━━━━━━━━━━━━━━━━━━━━━
FINAL RULE
━━━━━━━━━━━━━━━━━━━━━━━

Do not describe the user.

Expose what they are doing — in plain, undeniable language.

If the reflection sounds like something a therapist would say,
it is wrong.

If it sounds like something a sharp friend would say,
it is correct.`;

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
