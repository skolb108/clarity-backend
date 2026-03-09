import express from "express";
import cors from "cors";
import OpenAI from "openai";
import dotenv from "dotenv";

dotenv.config();

const app = express();

/* CORS CONFIG */
app.use(cors({
  origin: "*",
  methods: ["GET", "POST", "OPTIONS"],
  allowedHeaders: ["Content-Type"]
}));

app.use(express.json());

/* OpenAI */
const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
});

/* Health Check */
app.get("/", (req, res) => {
  res.send("Clarity Server läuft 🚀");
});

/* Chat Endpoint */
app.post("/api/chat", async (req, res) => {

  try {

    console.log("API request received");

    const { messages } = req.body;

    if (!messages) {
      return res.status(400).json({
        error: "Messages missing"
      });
    }

    const completion = await client.chat.completions.create({
      model: "gpt-4o-mini",
      messages: messages,
      temperature: 0.7
    });

    const reply = completion.choices[0].message.content;

    res.json({
      reply
    });

  } catch (error) {

    console.error("OpenAI error:", error);

    res.status(500).json({
      reply: "Entschuldige, ich konnte gerade keine Antwort generieren."
    });

  }

});

/* Start Server */
const PORT = process.env.PORT || 3000;

app.listen(PORT, () => {
  console.log(`Clarity Server läuft auf Port ${PORT}`);
});