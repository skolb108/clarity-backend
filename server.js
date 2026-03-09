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

/* Handle OPTIONS preflight */
app.options("*", cors());

/* OpenAI */
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
});

/* Root */
app.get("/", (req, res) => {
  res.status(200).send("Clarity backend running");
});

/* Chat endpoint */
app.post("/api/chat", async (req, res) => {

  try {

    const { messages } = req.body;

    const completion = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages
    });

    const reply = completion.choices[0].message.content;

    res.json({ reply });

  } catch (error) {

    console.error(error);

    res.status(500).json({
      reply: "AI request failed"
    });

  }

});

/* Start server */
const PORT = process.env.PORT;

app.listen(PORT, "0.0.0.0", () => {
  console.log("Server listening on", PORT);
});