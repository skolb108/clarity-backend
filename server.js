import express from "express";
import cors from "cors";
import OpenAI from "openai";
import dotenv from "dotenv";

dotenv.config();

const app = express();

/* Middleware */
app.use(cors());
app.use(express.json());

/* OpenAI Client */
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
});

/* Health Check */
app.get("/", (req, res) => {
  res.send("Clarity backend running");
});

/* Chat Endpoint */
app.post("/api/chat", async (req, res) => {

  try {

    const { messages } = req.body;

    if (!messages) {
      return res.status(400).json({ error: "Messages missing" });
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

/* Start Server */

const PORT = process.env.PORT || 3000;

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});