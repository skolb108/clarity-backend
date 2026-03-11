import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import OpenAI from "openai";

dotenv.config();

const app = express();

/* CORS */
app.use(cors({
  origin: "*",
  methods: ["GET", "POST", "OPTIONS"],
  allowedHeaders: ["Content-Type"]
}));

app.use(express.json());

/* OpenAI */
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
});

/* ─────────────────────────────────────────────────────────────
   callOpenAI — shared helper with retry + 25s timeout
   options:
     retries      (default 2)
     jsonMode     (default false) → enables response_format json_object
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
      if (isTimeout) throw new Error("OpenAI request timeout");

      if (attempt < retries) {
        console.log(`Retrying OpenAI request... (attempt ${attempt + 1} of ${retries})`);
        continue;
      }

      throw err;
    }
  }
}

/* Health check */
app.get("/", (req, res) => {
  res.send("Clarity backend running");
});

/* ─────────────────────────────────────────────────────────────
   POST /api/chat — conversation reflections
───────────────────────────────────────────────────────────── */
app.post("/api/chat", async (req, res) => {
  console.log("Clarity API request received: /api/chat");

  const { messages } = req.body;

  if (!messages || !Array.isArray(messages)) {
    return res.status(400).json({ error: "messages missing or invalid" });
  }

  try {
    const reply = await callOpenAI(messages);
    res.json({ reply });
  } catch (err) {
    console.error("/api/chat error:", err.message);
    res.status(500).json({ reply: "AI request failed" });
  }
});

/* ─────────────────────────────────────────────────────────────
   POST /api/analyze — final analysis, always returns JSON
───────────────────────────────────────────────────────────── */
app.post("/api/analyze", async (req, res) => {
  console.log("Clarity API request received: /api/analyze");

  const { messages } = req.body;

  if (!messages || !Array.isArray(messages)) {
    return res.status(400).json({ error: "messages missing or invalid" });
  }

  try {
    const raw = await callOpenAI(messages, { jsonMode: true });
    const parsed = JSON.parse(raw);
    res.json(parsed);
  } catch (err) {
    console.error("/api/analyze error:", err.message);
    res.status(500).json({ error: "Analysis failed", detail: err.message });
  }
});

/* Start server */
const PORT = process.env.PORT || 8080;

app.listen(PORT, "0.0.0.0", () => {
  console.log(`Server running on port ${PORT}`);
});
