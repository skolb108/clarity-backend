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

/* Health check */
app.get("/", (req, res) => {
  res.send("Clarity backend running");
});

/* Chat endpoint */
app.post("/api/chat", async (req, res) => {

  try {

    const { messages } = req.body;

    if (!messages) {
      return res.status(400).json({
        error: "messages missing"
      });
    }

    const completion = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: messages
    });

    const reply = completion.choices[0].message.content;

    res.json({ reply });

  } catch (error) {

    console.error("OpenAI error:", error);

    res.status(500).json({
      reply: "AI request failed"
    });

  }

});

/* Start server */

const PORT = process.env.PORT || 8080;

app.listen(PORT, "0.0.0.0", () => {
  console.log(`Server running on port ${PORT}`);
});