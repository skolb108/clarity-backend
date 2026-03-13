import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import OpenAI from "openai";

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

  // ── 1. Validate request ────────────────────────────────────
  const invalid = validateMessages(req.body);
  if (invalid) {
    log(endpoint, { error: invalid.code, detail: invalid.detail });
    return res.status(400).json({ error: invalid.code });
  }

  const { messages } = req.body;
  log(endpoint, { messages: messages.length });

  // ── 2. SSE headers — must be set before any write ─────────
  // X-Accel-Buffering: no  →  disables Railway / nginx proxy buffering
  // Without this, tokens batch up and arrive in large bursts rather than
  // one-by-one, defeating the purpose of streaming.
  res.setHeader("Content-Type",      "text/event-stream");
  res.setHeader("Cache-Control",     "no-cache, no-transform");
  res.setHeader("Connection",        "keep-alive");
  res.setHeader("X-Accel-Buffering", "no");
  res.flushHeaders();

  // ── 3. Abort controller — 25s timeout + client disconnect ──
  const controller = new AbortController();
  let   completed  = false;

  const timeoutTimer = setTimeout(() => {
    if (!completed) {
      log(endpoint, { error: ERR.OPENAI_TIMEOUT });
      controller.abort();
      try {
        res.write(`data: ${JSON.stringify({ error: ERR.OPENAI_TIMEOUT })}\n\n`);
        res.end();
      } catch (_) { /* client already gone */ }
    }
  }, 25_000);

  // Clean up if the client closes the connection mid-stream
  req.on("close", () => {
    if (!completed) {
      log(endpoint, { info: "client disconnected — aborting upstream request" });
      controller.abort();
      clearTimeout(timeoutTimer);
    }
  });

  // ── 4. Stream from OpenAI ──────────────────────────────────
  const t0 = Date.now();

  try {
    const stream = await openai.chat.completions.create(
      { model: "gpt-4o-mini", messages, stream: true },
      { signal: controller.signal }
    );

    for await (const chunk of stream) {
      // Guard: stop writing if the client disconnected mid-stream
      if (controller.signal.aborted) break;

      const token = chunk.choices[0]?.delta?.content;
      if (token) {
        res.write(`data: ${JSON.stringify({ t: token })}\n\n`);
      }
    }

    // Stream completed successfully
    completed = true;
    clearTimeout(timeoutTimer);

    const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
    log(endpoint, { "OpenAI response time": `${elapsed}s`, status: "done" });

    res.write("data: [DONE]\n\n");
    res.end();

  } catch (err) {
    completed = true;
    clearTimeout(timeoutTimer);

    // AbortError is expected on client disconnect or timeout — not a true crash
    if (err.name === "AbortError") {
      try { res.end(); } catch (_) { /* already closed */ }
      return;
    }

    const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
    log(endpoint, {
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

  const invalid = validateMessages(req.body);
  if (invalid) {
    log(endpoint, { error: invalid.code });
    return res.status(400).json({ error: invalid.code });
  }

  const { messages } = req.body;
  log(endpoint, { messages: messages.length });
  const t0 = Date.now();

  try {
    const reply   = await callOpenAI(messages);
    const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
    log(endpoint, { "OpenAI response time": `${elapsed}s` });
    res.json({ reply });

  } catch (err) {
    const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
    const code    = err.message === ERR.OPENAI_TIMEOUT ? ERR.OPENAI_TIMEOUT : ERR.OPENAI_FAILED;
    log(endpoint, { error: code, detail: err.message, elapsed: `${elapsed}s` });
    res.status(500).json({ error: code });
  }
});

/* ─────────────────────────────────────────────────────────────
   POST /api/analyze — final analysis, always returns JSON
───────────────────────────────────────────────────────────── */
app.post("/api/analyze", async (req, res) => {
  const endpoint = "POST /api/analyze";

  const invalid = validateMessages(req.body);
  if (invalid) {
    log(endpoint, { error: invalid.code });
    return res.status(400).json({ error: invalid.code });
  }

  const { messages } = req.body;
  log(endpoint, { messages: messages.length });
  const t0 = Date.now();

  try {
    const raw     = await callOpenAI(messages, { jsonMode: true });
    const parsed  = JSON.parse(raw);
    const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
    log(endpoint, { "OpenAI response time": `${elapsed}s` });
    res.json(parsed);

  } catch (err) {
    const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
    const code    = err.message === ERR.OPENAI_TIMEOUT ? ERR.OPENAI_TIMEOUT : ERR.OPENAI_FAILED;
    log(endpoint, { error: code, detail: err.message, elapsed: `${elapsed}s` });
    res.status(500).json({ error: code });
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
