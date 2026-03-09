import express from "express";
import cors from "cors";

const app = express();

app.use(cors());
app.use(express.json());

app.get("/", (req, res) => {
  res.status(200).send("Clarity backend running");
});

app.get("/health", (req, res) => {
  res.status(200).json({ status: "ok" });
});

app.post("/api/chat", (req, res) => {
  res.json({ reply: "test response" });
});

const PORT = process.env.PORT || 3000;

app.listen(PORT, "0.0.0.0", () => {
  console.log("Server running on port", PORT);
});