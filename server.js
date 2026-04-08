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
    const timer = setTimeout(() => controller.abort(), 25_000);

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
const CLARITY_SYSTEM_PROMPT = `You are Clarity — a calm, perceptive AI mentor helping people gain deep clarity about their life direction.

You are not a life coach. You do not give advice. You do not use motivational language.
You ask precise, uncomfortable questions that reveal what the person already knows but hasn't faced yet.

YOUR APPROACH — move through these five stages naturally across the conversation:

Stage 1 — UNDERSTAND THE PROBLEM
Ask open questions to understand what is actually going on.
Do not accept vague answers. Push gently for specifics.
Example probes: "What does that look like day-to-day?" / "When did you first notice this?"

Stage 2 — IDENTIFY HIDDEN PATTERNS
Listen for contradictions, repeated themes, and things the person avoids saying directly.
Name the pattern when you see it. Be direct but not harsh.
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
- Keep your responses short: 1–3 sentences maximum before your question.
- Never use words like: "journey", "growth", "passion", "authentic", "potential", "empower".
- Never summarize what the person said back to them unless you are naming a specific pattern.
- Never give unsolicited advice or suggest solutions.
- If an answer is vague, ask for a concrete example before moving on.
- Tone: direct, calm, curious, precise. Like a trusted friend who is also a very good thinker.

CONVERSATION LENGTH:
Guide the conversation to 10–12 substantive exchanges. After that, when you have enough
to produce a genuine insight, close the conversation.

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
   POST /api/chat/stream — streaming SSE reflection
   Streams OpenAI tokens to the frontend as Server-Sent Events.

   SSE protocol:
     each token →  data: {"t":"<token>"}\n\n
     on finish  →  data: [DONE]\n\n
     on error   →  data: {"error":"<code>"}\n\n
───────────────────────────────────────────────────────────── */
app.post("/api/chat/stream", async (req, res) => {
  const endpoint = "POST /api/chat/stream";
  const reqId    = makeReqId();

  // ── 1. Validate request ────────────────────────────────────
  const invalid = validateMessages(req.body);
  if (invalid) {
    log(endpoint, { reqId, error: invalid.code, detail: invalid.detail });
    return res.status(400).json({ error: invalid.code });
  }

  const { messages } = req.body;
  log(endpoint, { reqId, messages: messages.length });

  // ── 2. Prepend system prompt ───────────────────────────────
  // The CLARITY_SYSTEM_PROMPT is always the first message so OpenAI
  // always has the full conversation context and stage instructions,
  // regardless of what the frontend sends.
  const messagesWithSystem = [
    { role: "system", content: CLARITY_SYSTEM_PROMPT },
    ...messages,
  ];

  // ── 3. SSE headers — must be set before any write ─────────
  res.setHeader("Content-Type",      "text/event-stream");
  res.setHeader("Cache-Control",     "no-cache, no-transform");
  res.setHeader("Connection",        "keep-alive");
  res.setHeader("X-Accel-Buffering", "no");
  res.flushHeaders();

  // ── 4. Abort controller — 25s timeout + client disconnect ──
  const controller = new AbortController();
  let   completed  = false;

  const timeoutTimer = setTimeout(() => {
    if (!completed) {
      log(endpoint, { reqId, error: ERR.OPENAI_TIMEOUT });
      controller.abort();
      try {
        res.write(`data: ${JSON.stringify({ error: ERR.OPENAI_TIMEOUT })}\n\n`);
        res.end();
      } catch (_) { /* client already gone */ }
    }
  }, 25_000);

  req.on("close", () => {
    if (!completed) {
      log(endpoint, { reqId, info: "client disconnected — aborting upstream request" });
      controller.abort();
      clearTimeout(timeoutTimer);
    }
  });

  // ── 5. Stream from OpenAI ──────────────────────────────────
  const t0 = Date.now();

  try {
    const stream = await openai.chat.completions.create(
      { model: "gpt-4o-mini", messages: messagesWithSystem, stream: true },
      { signal: controller.signal }
    );

    for await (const chunk of stream) {
      if (controller.signal.aborted) break;

      const token = chunk.choices[0]?.delta?.content;
      if (token) {
        res.write(`data: ${JSON.stringify({ t: token })}\n\n`);
      }
    }

    completed = true;
    clearTimeout(timeoutTimer);

    const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
    log(endpoint, { reqId, "OpenAI response time": `${elapsed}s`, status: "done" });

    res.write("data: [DONE]\n\n");
    res.end();

  } catch (err) {
    completed = true;
    clearTimeout(timeoutTimer);

    if (err.name === "AbortError") {
      try { res.end(); } catch (_) { /* already closed */ }
      return;
    }

    const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
    log(endpoint, {
      reqId,
      error:   ERR.OPENAI_FAILED,
      detail:  err.message,
      elapsed: `${elapsed}s`,
    });

    try {
      res.write(`data: ${JSON.stringify({ error: ERR.OPENAI_FAILED })}\n\n`);
      res.end();
    } catch (_) { /* client already gone */ }
  }
});

/* ─────────────────────────────────────────────────────────────
   POST /api/chat — non-streaming reflection (fallback)
───────────────────────────────────────────────────────────── */
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
  const t0 = Date.now();

  try {
    const reply   = await callOpenAI(messages);
    const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
    log(endpoint, { reqId, "OpenAI response time": `${elapsed}s` });
    res.json({ reply });

  } catch (err) {
    const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
    const code    = err.message === ERR.OPENAI_TIMEOUT ? ERR.OPENAI_TIMEOUT : ERR.OPENAI_FAILED;
    log(endpoint, { reqId, error: code, detail: err.message, elapsed: `${elapsed}s` });
    res.status(500).json({ error: code });
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
