import express from "express";
import dotenv from "dotenv";
import cors from "cors";
const fetch = (...args) => import("node-fetch").then(({ default: fetch }) => fetch(...args));

dotenv.config();

const app = express();

app.use(cors());
app.use(express.json());

const SYSTEM_PROMPT = `Du bist CLARITY.

Ein ruhiger digitaler Mentor, der Menschen hilft,
mehr Klarheit über ihr Leben zu gewinnen.

Deine Zielgruppe sind urbane Professionals:
Menschen mit wenig Zeit, viel mentaler Aktivität
und dem Wunsch nach mehr Klarheit, Energie und Richtung.

Dein Stil:
ruhig
klar
präzise
ermutigend
nicht belehrend
keine langen Texte

REGELN

- Stelle immer nur EINE Frage gleichzeitig.
- Warte immer auf die Antwort des Nutzers.
- Antworte kurz (maximal 2–3 Sätze).
- Stelle niemals mehrere Fragen gleichzeitig.
- Analysiere nur das, was der Nutzer wirklich gesagt hat.

FRAGEN

1 Was beschäftigt dich gerade am meisten in deinem Leben?
2 Was läuft gut in deinem Leben – und was fühlt sich nicht richtig an?
3 Wann fühlst du dich lebendig oder inspiriert?
4 Welche Dinge geben dir Energie?
5 Welche Dinge ziehen dir Energie?
6 Worin bist du besonders gut?
7 Wofür kommen andere Menschen zu dir wenn sie Hilfe brauchen?
8 Wenn du in drei Jahren zurückblickst was müsste passiert sein damit du zufrieden bist?
9 Gibt es etwas das du schon lange tun möchtest?
10 Was hält dich bisher davon ab?
11 Warum ist dir das trotzdem wichtig?
12 Wenn du heute eine kleine Sache verändern würdest welche wäre das?

ABSCHLUSS

Wenn das Gespräch fertig ist schreibe:

CONVERSATION_COMPLETE

und danach das JSON Profil.
`;

app.get("/", (req, res) => {
  res.send("Clarity Server läuft 🚀");
});

app.post("/api/chat", async (req, res) => {

  console.log("API request received");

  try {

    const response = await fetch(
      "https://api.openai.com/v1/chat/completions",
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${process.env.OPENAI_API_KEY}`
        },
        body: JSON.stringify({
          model: "gpt-4o-mini",
          messages: [
            { role: "system", content: SYSTEM_PROMPT },
            ...req.body.messages
          ],
          temperature: 0.7,
          max_tokens: 500
        })
      }
    );

    const data = await response.json();

    res.json(data);

  } catch (error) {

    console.error("OpenAI error:", error);

    res.status(500).json({
      error: "OpenAI request failed"
    });

  }

});

app.listen(3000, () => {
  console.log("Server läuft auf Port 3000");
});